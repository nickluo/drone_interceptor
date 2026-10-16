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

Vision-based drone interception. Single ROS2 node: relative IMM-UKF target estimator + analytical PN/IBVS controller. See `AGENTS.md` for full reference.

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
- **IMM with 3 motion models**: Singer (mode 0), Coordinated Turn/CT (mode 1), Rotor — pure accel-relaxation, no drag term (mode 2)
- **UKF**: `UnscentedKalmanFilter10D`, 2N+1 sigma points, `robust_cholesky` (LLT fallback to eigendecomp)
- **Measurement**: 3D pixel `[u, v, size]` from bbox → pinhole projection `h_meas_pixel`
- **IMU strapdown**: preintegration (alpha/beta) for ego-motion; online accel bias calibration during idle
- **Odometry-free**: ego-motion from IMU only; the controller's lateral observability dither provides the
  parallax baseline that makes monocular depth/scale observable
- **Controller**: PN (proportional navigation) + IBVS, FOV keep-in-frame guard, soft-start ramp, axial jerk limit
- **Control output**: `quadrotor_msgs::ControlCommand` — `collective_thrust` + body rates `[omega_x, omega_y, omega_z]`
- **Thread safety**: `std::mutex`
- **Topics** (all names overridable via `topics.*` params): imu=`/mavros/imu/data`, bbox=`/fpv/target_bbox`,
  gimbal=`/fpv/gimbal`, mission_launch=`/fpv/tracker_trigger`, mission=`/fpv/mission_start`,
  control=`/fpv/control_command`, mission_state=`/fpv/mission_state`
- **Radar relay (optional, `radar.enable`, off by default)**: `/fpv/radar/target` + `/fpv/radar/interceptor`
  (`nav_msgs/Odometry`, world ENU, `header.stamp` = measurement epoch) differenced into a relative
  `[r, v_rel]` LINEAR update (`update_posvel`) for mid-course guidance outside the camera FOV. On
  visual acquisition the target width is fixed from `bbox_size * r_radar / focal` and locked. See
  `AGENTS.md` for the full contract.

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
| `src/relative_imm_ukf.h` | IMM-UKF estimator stack: `UnscentedKalmanFilter10D`, `RobustIMMEstimator10D`, `RelativeIMMUKFEstimator` (header-only) |
| `src/advanced_hybrid_controller.h` | `AdvancedHybridController` — PN/IBVS guidance + thrust allocation (header-only) |
| `src/interceptor_node_relative_imm_ukf.h` | Node class declaration (members, nested cache/sample structs) |
| `src/interceptor_node_relative_imm_ukf.cpp` | Node method definitions + `main()` |
| `src/utils.h` | MissionStateMachine, fixed-size Eigen aliases, `clamp` (header-only) |
| `src/guidance_limits.h` | bounded axial speed / terminal speed floor / vision-range helpers (header-only) |
| `src/imu_strapdown.h` | gyro attitude integration (header-only) |
| `src/camera_pose_history.h` | timestamped camera orientation buffer (header-only) |
| `config/relative_imm_ukf_params.yaml` | Runtime parameters (Camera8 SITL profile) |
| `prototype/ros2_sim_node.py` | ROS2 harness driving the real C++ node (incl. simulated radar relay) |
| `prototype/run_scenario.sh` | One-shot scenario runner: launches the node, drives it, prints the summary |

## Pitfalls

- `use_simulation` defaults `False` in the launch — verify before sim runs
- Radar relay is OFF unless `radar.enable` / `launch_relative_imm_ukf.py use_radar:=True`
- The width handover rides on `focal_length_` (from `camera.width_px` + `camera.fov_deg`);
  `airsim_settings.json` (30° / 960x960) does NOT match the launch defaults (1536x960 / 72°)
- `Eigen` is not declared in `package.xml` — rosdep won't install it; manual system install required
