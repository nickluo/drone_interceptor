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

Vision-based drone interception. Single ROS2 node: relative IMM-UKF target estimator + analytical PN/IBVS controller. (The former acados NMPC variants — quaternion and dual-quaternion — were removed; history is in git.)

## Architecture

```
InterceptorNodeRelativeIMMUKF : public rclcpp::Node  — 10D target state, IMM-UKF, PN+IBVS controller, NO acados
utils.h                       — MissionStateMachine + fixed-size Eigen aliases (header-only)
guidance_limits.h             — bounded axial speed / terminal speed floor / vision-range helpers
imu_strapdown.h               — gyro attitude integration
camera_pose_history.h         — timestamped camera orientation ring buffer (delay compensation)
```

### Relative IMM-UKF node

- `InterceptorNodeRelativeIMMUKF : public rclcpp::Node` — fully standalone, no base class
- **No acados solver** — links only `m`. No codegen step needed.
- **10D target state** `[pos(3) vel(3) acc(3) width(1)]` — RELATIVE state (target − ownship) in a
  gravity-aligned world frame G; ownship motion enters as an exogenous IMU-preintegrated input
- **IMM with 3 motion models**: Singer (mode 0), Coordinated Turn/CT (mode 1), Rotor/drag (mode 2)
- **UKF**: `UnscentedKalmanFilter10D`, 2N+1 sigma points, `robust_cholesky` (LLT fallback to eigendecomp)
- **Measurement**: 3D pixel `[u, v, size]` from bbox → pinhole projection `h_meas_pixel`
- **IMU strapdown**: preintegration (alpha/beta) for ego-motion; online accel bias calibration during idle
- **Odometry-free**: ego-motion from IMU only; the controller's lateral observability dither provides the
  parallax baseline that makes monocular depth/scale observable
- **Controller**: PN (proportional navigation) + IBVS, FOV keep-in-frame guard, soft-start ramp, axial jerk limit
- **Observability dither close-gates** (3 conditions, ALL must hold to dither): estimated range >
  `controller.dither_min_range`; bbox < `controller.dither_max_bbox` px (direct observable,
  immune to the (W, range) scale error that blinds the range gate — a large bbox = good size-SNR /
  close target where the dither only disturbs terminal precision; 0.5 s stale-hold across dropout
  frames); and no fresh radar track inside `controller.dither_radar_range` (works post-lock — the
  radar cache keeps updating even after fusion disengages). 0 disables either extra gate.
- **Control output**: `quadrotor_msgs::ControlCommand` — `collective_thrust` + body rates `[omega_x, omega_y, omega_z]`
- **Thread safety**: `std::mutex`
- **Topics** (all names overridable via `topics.*` params): imu=`/mavros/imu/data`, bbox=`/fpv/target_bbox`,
  gimbal=`/fpv/gimbal`, mission_launch=`/fpv/tracker_trigger`, mission=`/fpv/mission_start`,
  control=`/fpv/control_command`, mission_state=`/fpv/mission_state`

### Radar relay mid-course guidance

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
  The anchor is the RAW delay-compensated radar pair range, NOT the estimator blend — pre-lock the
  blend is radar-fused with the scale-degenerate camera size channel and can sit far off along the
  W/r hyperbola (observed blend 19.7 m vs radar ~56 m → W locked 0.58 m for a 1.5 m target); the
  median/MAD window absorbs the raw range's per-sample noise.
- **Post-lock width re-anchor** (`radar.width_reanchor_*`, safety net): with W inflated by k the
  monocular range is inflated by k, blinding every r_est-keyed close-range gate (dither min-range,
  width freeze, size gate, coast, fly-through) — the "thinks it's far, actually close, dither stays
  armed" miss. After the visual lock the radar fusion stays disengaged but the radar cache keeps
  updating, so `r_vis = focal*W_locked/bbox` is continuously ratio-checked against `r_radar`; a
  sustained median disagreement (≥ `width_reanchor_min_frames` frames over `width_reanchor_window`,
  ratio outside `1/width_reanchor_ratio..width_reanchor_ratio`, radar range within
  `[min,max]_range`) re-locks W at `W/ratio` and rescales every IMM filter's position state and
  covariance by `1/ratio` (plus the `min_range_seen_/min_vision_range_/last_good_range_` trackers,
  so the correction cannot false-fire the fly-through capture). Bounded by a cooldown and
  `width_reanchor_max` corrections per mission. Logs `[W-REANCHOR]`.
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
cd ~/workspace/fpv_ws && colcon build --packages-select drone_interceptor
```

- C++17, Release mode, `ament_cmake`
- `interceptor_node_relative_imm_ukf` links: `m` only
- No codegen / no external solver dependencies beyond `Eigen3` (system package)

### Executable target

| Target | Source | Launch |
|--------|--------|--------|
| `interceptor_node_relative_imm_ukf` | `src/interceptor_node_relative_imm_ukf.cpp` | `launch_relative_imm_ukf.py` |

## Conventions

- **Quaternion convention**: scalar-first `[w, x, y, z]` (differs from ROS `[x, y, z, w]`)
- **Control**: `[omega_x, omega_y, omega_z, thrust_accel]` as body rates + collective thrust (m/s²)
- **Naming**: `snake_case` vars/methods, `PascalCase` classes, `UPPER_CASE` constants, trailing `_` for members
- **Headers**: `#pragma once`, system → ROS → project
- **Eigen**: heavy use — fixed-size aliases (`Vector6d`, `Matrix6d`) and dynamic `VectorXd`
- **Comments**: mixed English and Chinese (中文)

## Key Files

| Path | Purpose |
|------|---------|
| `src/interceptor_node_relative_imm_ukf.cpp` | Standalone IMM-UKF estimator + PN/IBVS controller (10D) |
| `src/utils.h` | MissionStateMachine, fixed-size Eigen aliases (header-only) |
| `src/guidance_limits.h` | bounded axial speed / terminal speed floor / vision-range helpers (header-only) |
| `src/imu_strapdown.h` | gyro attitude integration (header-only) |
| `src/camera_pose_history.h` | timestamped camera orientation buffer (header-only) |
| `config/relative_imm_ukf_params.yaml` | Runtime parameters (Camera8 SITL profile) |
| `prototype/ros2_sim_node.py` | ROS2 harness driving the real C++ node (incl. simulated radar relay) |
| `prototype/sim_test_relative_imm_ukf.py` | Pure-Python mirror of the C++ estimator + controller |
| `prototype/run_scenario.sh` | One-shot scenario runner: launches the node, drives it, prints the summary |

## Pitfalls

- `use_simulation` defaults `False` in the launch — verify before sim runs
- Radar relay is OFF unless `radar.enable` / `launch_relative_imm_ukf.py use_radar:=True`
- The width handover is only as good as `focal_length_`, which comes from `camera.width_px` and
  `camera.fov_deg`. `airsim_settings.json` (30° / 960x960) does NOT match the launch defaults
  (1536x960 / 72°) — a focal error corrupts both the handover width and the terminal
  monocular range. Align them before any AirSim run.
- `Eigen` is not declared in `package.xml` — rosdep won't install it; manual system install required
