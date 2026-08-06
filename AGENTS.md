Respond terse like smart caveman. All technical substance stay. Only fluff die.

Rules:
- Drop: articles (a/an/the), filler (just/really/basically), pleasantries, hedging
- Fragments OK. Short synonyms. Technical terms exact. Code unchanged.
- Pattern: [thing] [action] [reason]. [next step].
- Not: "Sure! I'd be happy to help you with that."
- Yes: "Bug in auth middleware. Fix:"

Switch level: /caveman lite|full|ultra|wenyan
Stop: "stop caveman" or "normal mode"

Auto-Clarity: drop caveman for security warnings, irreversible actions, user confused. Resume after.

Boundaries: code/commits/PRs written normal.

# drone_interceptor — Agent Instructions

Vision-based drone interception. Three node variants: two NMPC (acados) sharing Template Method base class, one standalone IMM-UKF estimator + analytical controller.

## Architecture

```
InterceptorNodeBase (abstract)           — control loop, PN guidance, EKF, safety
├── InterceptorNodeQuaternion            — 16D state [pos(3) vel_world(3) quat(4) omega(3) gimbal(3)]
└── InterceptorNodeDQ                    — 17D state [dq(8) omega(3) v_body(3) gimbal(3)]
InterceptorNodeRelativeIMMUKF (standalone) — 10D target state, IMM-UKF, PN+IBVS controller, NO acados
utils.h                                  — DroneEKF(19D), MissionStateMachine, EMAFilter, CameraIntrinsics
```

### NMPC variants (base class)

- **13 pure virtual methods** in base: state accessors, solver lifecycle, param packing, delay compensation, orientation validation
- Two-phase init: base constructor declares params only; subclass calls `initialize()` after vtable ready
- Thread safety: `std::shared_mutex` for drone_state/EKF/gimbal/control_history; `std::mutex` for bbox

### Relative IMM-UKF variant (standalone)

- `InterceptorNodeRelativeIMMUKF : public rclcpp::Node` — does NOT inherit `InterceptorNodeBase`
- **No acados solver** — links only `m`. No codegen step needed.
- **10D target state** `[pos(3) vel(3) acc(3) width(1)]` in NAV/world frame (absolute target, not interceptor ego-state)
- **IMM with 3 motion models**: Singer (mode 0), Coordinated Turn/CT (mode 1), Rotor/drag (mode 2)
- **UKF**: `UnscentedKalmanFilter10D`, 2N+1 sigma points, `robust_cholesky` (LLT fallback to eigendecomp)
- **Measurement**: 3D pixel `[u, v, size]` from bbox → pinhole projection `h_meas_pixel`
- **IMU strapdown**: preintegration (alpha/beta) for ego-motion; online accel bias calibration during idle
- **Odometry aiding**: complementary fusion at camera rate; first fix hard-seeds ego frame; disables lateral dither
- **Controller**: PN (proportional navigation) + IBVS, FOV keep-in-frame guard, soft-start ramp, axial jerk limit
- **Control output**: `quadrotor_msgs::ControlCommand` — `collective_thrust` + body rates `[omega_x, omega_y, omega_z]` (same topic/format as NMPC variants)
- **Thread safety**: `std::mutex` (not shared_mutex)
- **Topics**: imu=`/mavros/imu/data`, bbox=`/fpv/target_bbox`, odom=`/fpv/odom`, mission_launch=`/fpv/tracker_trigger`, mission=`/fpv/mission_start`, control=`/fpv/control_command`, mission_state=`/fpv/mission_state`

### Radar relay mid-course guidance (IMM-UKF variant only)

Optional second measurement source, off by default (`radar.enable`). Carries the interceptor to the
target before the target is inside the camera FOV, then hands over to vision.

- **Input**: two `nav_msgs/Odometry` at ~10 Hz — `/fpv/radar/target` and `/fpv/radar/interceptor`,
  world ENU, `twist.linear` in the SAME world frame (radar measures ground velocity, not body).
  `header.stamp` is the **measurement epoch**, not the publish time — that is how the node recovers
  the 50-100 ms link latency.
- **Pairing**: samples matched by stamp within `radar.pair_window`; `t_meas = min(stamps)` plus a
  last-fused guard makes a dropped leg skip rather than fuse a mismatched pair.
- **Fusion**: relative measurement `z = [r, v_rel] = [p_t - p_i, v_t - v_i]`, extrapolated over the
  lag, applied as a LINEAR update (`H = [I6|0]`, `update_posvel`) — valid outside the FOV where the
  pixel model is not. Seeds the filter when it is not yet initialized.
- **Single time base**: `advance_time()` gives every source (bbox / bbox-dropout coast / radar) a
  predict dt of "time since the previous estimator step", so 30 Hz vision and 10 Hz radar never
  double-propagate. The IMU `dv_own` accumulator is drained by whichever source ran.
- **Mid-course guidance**: radar fills the control cache with `bbox_size = 0` (terminal brake stays
  disarmed) and an ANALYTIC LOS rate `(r x v)_z / |r_xy|^2` — differencing a 10 Hz metre-noisy
  bearing would inject ~0.3 rad/s into the PN lateral command.
- **Every bbox is gated on radar consistency** (`radar.bbox_gate`), not just the handover decision.
  A detection whose LOS is more than `radar.handover_max_bearing_err` off the radar track — or,
  once the width is locked, whose implied range `focal*W/s` disagrees with the radar range by more
  than `radar.bbox_range_ratio` — is treated as "no detection" and never reaches the filter. One
  fused phantom is enough to walk the monocular width off (observed `w_est` 1.50 -> 0.66 in AirSim
  while the estimate slewed tens of metres).
- **Width handover**: while the target is in frame AND the radar track is still valid,
  `W = bbox_size * r_radar / focal`, median over `radar.handover_frames` VALIDATED frames, then
  **locked** (`lock_width`). Confidence anchor for the monocular terminal range `r = focal*W/s`.
- **Visual lock needs a stable run AND a fixed width**: `radar.handover_frames` consecutive
  validated frames, at least `radar.lock_min_duration` of elapsed time, and the width already
  fixed from the radar (unless no radar track ever arrived). Locking before the width is fixed
  runs the terminal range on the configured guess and nothing pins W afterwards — it drifts with
  every update (observed `w_est` 1.9 -> 2.3 in AirSim).
- **Terminal vision guidance is exclusive.** Once vision is locked AND delivering, the radar stops
  contributing entirely (`radar.disable_after_lock`): its metre-level noise can only degrade a
  converged monocular solution, and its range knowledge is already baked into the locked width. It
  re-engages as soon as vision stops delivering for `radar.vision_hold` — i.e. when the pass failed.
- **States**: `INITIALIZING -> BLIND_FLIGHT` on radar acquisition, `-> TRACKING` on visual lock.
  Vision gone > `radar.vision_lost_timeout` hands guidance ownership back to radar.
- **Frame cross-check**: with the target in frame, the camera LOS azimuth and the radar LOS azimuth
  must agree. The `[FRAME]` log line reports the difference; a STEADY nonzero value is the
  radar/IMU world-frame yaw mismatch — dial it out with `radar.yaw_offset_deg`. A frame that
  disagrees by more than `radar.handover_max_bearing_err` rejects the detection for both the width
  fix and the visual lock.
- **Ownship velocity trim**: the radar's interceptor track corrects the IMU strapdown `own_vel_`,
  which otherwise integrates unbounded from the MISSION trigger (a long wait for GUIDED is enough
  to poison it). `radar.own_vel_tc` must stay LONG (~5 s): `own_vel_` also damps the vertical loop,
  and blending in a 10 Hz lag-delayed velocity at a short time constant makes that loop oscillate.
- **States**: `INITIALIZING -> BLIND_FLIGHT` on radar acquisition, `-> TRACKING` on visual lock.
  Vision gone > `radar.vision_lost_timeout` reverts to radar.
- **Producer**: `airsim_tracker/radar_relay_node.py` (see that package). It also publishes
  `/fpv/radar_state` (`std_msgs/Int8`, same encoding as `/fpv/tracker_state`) which `apm_bridge`
  uses to allow the MISSION trigger on a radar lock alone.

## Build

```bash
# 1. Generate acados C solver (must run BEFORE colcon build) — NMPC variants only
cd prototype && python3 drone_model_quaternion.py   # or drone_model_dq.py
# 2. Build all targets
cd ~/workspace/fpv_ws && colcon build --packages-select drone_interceptor
```

- C++17, Release mode, `ament_cmake`
- NMPC targets link: `acados`, `blasfeo`, `hpipm`, `m`
- `relative_imm_ukf` target links: `m` only
- Generated C code in `prototype/c_generated_code/` compiled directly into NMPC executables

### Executable targets

| Target | Source | acados | Launch |
|--------|--------|--------|--------|
| `interceptor_node_quaternion` | `interceptor_node_base.cpp` + `interceptor_node_quaternion.cpp` | yes | `launch.py`, `airsim_launch.py` |
| `interceptor_node_dq` | `interceptor_node_base.cpp` + `interceptor_node_dq.cpp` | yes | `launch_dq.py`, `airsim_launch_dq.py` |
| `interceptor_node_relative_imm_ukf` | `interceptor_node_relative_imm_ukf.cpp` | no | `launch_relative_imm_ukf.py` |

## Conventions

- **Quaternion convention**: scalar-first `[w, x, y, z]` (differs from ROS `[x, y, z, w]`)
- **DQ state**: velocity is **body frame**; quaternion state uses **world frame** velocity
- **Control**: `[omega_x, omega_y, omega_z, thrust_accel]` as body rates + collective thrust (m/s²)
- **Naming**: `snake_case` vars/methods, `PascalCase` classes, `UPPER_CASE` constants, trailing `_` for members
- **Headers**: `#pragma once`, system → ROS → project → `extern "C"` for acados
- **Eigen**: heavy use — fixed-size aliases (`VectorU`, `Vector6d`, `VectorWcostQuat`) and dynamic `VectorXd`
- **Comments**: mixed English and Chinese (中文), especially in `utils.h` EKF code

## Solver Design (NONLINEAR_LS) — NMPC variants only

- Cost type must be **NONLINEAR_LS** (not EXTERNAL) — EXTERNAL has zero control gradient at hover, causing SQP_RTI MINSTEP failures
- Residual vector includes `omega_dot` + `thrust_accel_error` to give Gauss-Newton direct u-gradient
- `SQP_RTI`, `PARTIAL_CONDENSING_HPIPM`, `ERK` integrator (RK4, 2 steps), `PROJECT` regularization, LM=5.0
- Baumgarte stabilization (K=10) for quaternion/DQ norm — ERK sufficient, no need for IRK

## Key Files

| Path | Purpose |
|------|---------|
| `src/interceptor_node_base.h/.cpp` | Abstract base — full NMPC algorithm skeleton |
| `src/interceptor_node_quaternion.h/.cpp` | Quaternion variant (16D, 8D cost weights) |
| `src/interceptor_node_dq.h/.cpp` | DQ variant (17D, 9D cost weights, inline DQ helpers) |
| `src/interceptor_node_relative_imm_ukf.cpp` | Standalone IMM-UKF estimator + PN/IBVS controller (10D, no acados) |
| `src/utils.h` | EKF, state machine, filters (header-only) |
| `prototype/drone_model_dq.py` | DQ dynamics + cost + solver code generation |
| `prototype/drone_model_quaternion.py` | Quaternion dynamics + cost + solver code generation |
| `config/*_params.yaml` | Runtime parameters (real hardware vs AirSim vs IMM-UKF) |
| `prototype/ros2_sim_node.py` | ROS2 harness driving the real C++ node (incl. simulated radar relay) |
| `prototype/run_scenario.sh` | One-shot scenario runner: launches the node, drives it, prints the summary |

## Pitfalls

- Modifying NMPC cost function or dynamics requires **regenerating the C solver** from `prototype/` before rebuilding (does NOT affect `relative_imm_ukf` target)
- FOV cost needs `forward_scale` (zero when target behind camera) to avoid Jacobian explosion
- PN guidance diverges when EKF has >2x range error — disabled by default in NMPC variants
- Always use solver output even with status≠0 (suboptimal solution > fallback)
- `penalty_coeff` must be TTA-based with cap at 30 to prevent close-range solver explosion
- `relative_imm_ukf` node: `use_simulation` defaults `False` in launch — verify before AirSim runs
- Radar relay is OFF unless `radar.enable` / `launch_relative_imm_ukf.py use_radar:=True`
- The width handover is only as good as `focal_length_`, which comes from `camera.width_px` and
  `camera.fov_deg`. `airsim_settings.json` (30° / 960x960) does NOT match the launch defaults
  (72° / 1920x1200) — a 35% focal error that corrupts both the handover width and the terminal
  monocular range. Align them before any AirSim run.
- `acados`/`blasfeo`/`hpipm`/`Eigen` not declared in `package.xml` — `rosdep` won't install; manual system install required
