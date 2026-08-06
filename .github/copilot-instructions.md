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

# drone_interceptor — Copilot Instructions

Vision-based drone interception. Three node variants: two NMPC (acados) sharing Template Method base class, one standalone IMM-UKF estimator + analytical controller. See `AGENTS.md` for full reference.

## Architecture

```
InterceptorNodeBase (abstract)           — control loop, PN guidance, EKF, safety
├── InterceptorNodeQuaternion            — 16D state [pos(3) vel_world(3) quat(4) omega(3) gimbal(3)]
└── InterceptorNodeDQ                    — 17D state [dq(8) omega(3) v_body(3) gimbal(3)]
InterceptorNodeRelativeIMMUKF (standalone) — 10D target state, IMM-UKF, PN+IBVS controller, NO acados
utils.h                                  — DroneEKF(19D), MissionStateMachine, EMAFilter, CameraIntrinsics
```

### NMPC variants (base class)

- **13 pure virtual methods** in the base: state accessors, solver lifecycle, param packing, delay compensation, orientation validation
- Two-phase init: base constructor declares params only; subclass calls `initialize()` after vtable is ready
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
- **Radar relay (optional, `radar.enable`, off by default)**: `/fpv/radar/target` + `/fpv/radar/interceptor`
  (`nav_msgs/Odometry`, world ENU, `header.stamp` = measurement epoch) differenced into a relative
  `[r, v_rel]` LINEAR update (`update_posvel`) for mid-course guidance outside the camera FOV. On
  visual acquisition the target width is fixed from `bbox_size * r_radar / focal` and locked. See
  `AGENTS.md` for the full contract.

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

## Solver Design (NONLINEAR_LS)

- Cost type must be **NONLINEAR_LS** (not EXTERNAL) — EXTERNAL has zero control gradient at hover, causing SQP_RTI MINSTEP failures
- Residual vector includes `omega_dot` + `thrust_accel_error` to give Gauss-Newton direct u-gradient
- `SQP_RTI`, `PARTIAL_CONDENSING_HPIPM`, `ERK` integrator (RK4, 2 steps), `PROJECT` regularization, LM=5.0
- Baumgarte stabilization (K=10) for quaternion/DQ norm — ERK is sufficient, no need for IRK

## Key Files

| Path | Purpose |
|------|---------|
| `src/interceptor_node_base.h/.cpp` | Abstract base — full algorithm skeleton |
| `src/interceptor_node_quaternion.h/.cpp` | Quaternion variant (16D, 8D cost weights) |
| `src/interceptor_node_dq.h/.cpp` | DQ variant (17D, 9D cost weights, inline DQ helpers) |
| `src/interceptor_node_relative_imm_ukf.cpp` | Standalone IMM-UKF estimator + PN/IBVS controller (10D, no acados) |
| `src/utils.h` | EKF, state machine, filters (header-only) |
| `prototype/drone_model_dq.py` | DQ dynamics + cost + solver code generation |
| `prototype/drone_model_quaternion.py` | Quaternion dynamics + cost + solver code generation |
| `config/*_params.yaml` | Runtime parameters (real hardware vs AirSim vs IMM-UKF) |

## Pitfalls

- Modifying cost function or dynamics requires **regenerating the C solver** from `prototype/` before rebuilding (does NOT affect `relative_imm_ukf` target)
- FOV cost needs `forward_scale` (zero when target behind camera) to avoid Jacobian explosion
- PN guidance diverges when EKF has >2x range error — disabled by default
- Always use solver output even with status≠0 (suboptimal solution > fallback)
- `penalty_coeff` must be TTA-based with cap at 30 to prevent close-range solver explosion
- `relative_imm_ukf` node: `use_simulation` defaults `False` in launch — verify before AirSim runs
- Radar relay is OFF unless `radar.enable` / `launch_relative_imm_ukf.py use_radar:=True`
- The width handover rides on `focal_length_` (from `camera.width_px` + `camera.fov_deg`);
  `airsim_settings.json` (30° / 960x960) does NOT match the launch defaults (72° / 1920x1200)
- `acados`/`blasfeo`/`hpipm`/`Eigen` not declared in `package.xml` — `rosdep` won't install; manual system install required
