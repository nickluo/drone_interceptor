# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Vision-based drone interception. A ROS2 (ament_cmake, C++17) package that runs a pursuer UAV to intercept a moving target using onboard vision (a 2D detection bbox). One node: a relative IMM-UKF target estimator with an analytical PN/IBVS controller — no acados, no codegen. (The former acados NMPC variants — quaternion and dual-quaternion — were removed; history is in git.)

`AGENTS.md` (mirrored in `.github/copilot-instructions.md`) is the authoritative, detailed reference for architecture, conventions, estimator design, and pitfalls. Read it for depth; this file gives the actionable summary and the exact commands.

## Build & Run

**Workspace root is `/home/nick/workspace/fpv_ws`** (this package lives at `src/drone_interceptor`).

### Manual dependencies (not in package.xml)

`Eigen3` must be installed on the system yourself — it is NOT declared in `package.xml`, so `rosdep` will not install it. `quadrotor_msgs` (the `ControlCommand` type) is also an external package dependency.

### Build

No codegen step — build directly:

```bash
cd /home/nick/workspace/fpv_ws
colcon build --packages-select drone_interceptor
source install/setup.bash
```

### Run

```bash
ros2 launch drone_interceptor launch_relative_imm_ukf.py   # IMM-UKF estimator + PN/IBVS controller

# IMM-UKF with radar relay mid-course guidance (needs airsim_tracker/radar_launch.py running)
ros2 launch drone_interceptor launch_relative_imm_ukf.py use_radar:=True
```

The launch loads `config/relative_imm_ukf_params.yaml`. Note: `use_simulation` defaults to `False` in the launch — verify it before sim runs.

## Testing

- `colcon test --packages-select drone_interceptor` runs two C++ regression binaries wired in CMake: `guidance_limits_test` (axial-speed / terminal-floor / vision-range math) and `camera_pose_history_test` (timestamped pose buffer).
- `test/relative_imm_ukf_topics_test.py` — ROS graph regression (topics, params). Run with `python3 -m unittest` in the sourced workspace.
- Algorithm verification lives in `prototype/`:
  - `prototype/sim_test_relative_imm_ukf.py` — pure-Python mirror of the C++ IMM-UKF + PN/IBVS controller run against simplified quadrotor dynamics + camera model. This is the de-facto test for the estimator/controller.
  - `prototype/ros2_sim_node.py` — ROS2 simulation harness that publishes `/mavros/imu/data` + `/fpv/gimbal` + `/fpv/target_bbox` and consumes `/fpv/control_command`; runs scenarios (nominal, crossing, maneuvering, dropout, noise, wrong_w) producing the `*.csv`/`*_summary.txt`/`*.png` results in `prototype/`. It also simulates the radar relay (`radar_relay`, `radar_far`, `radar_maneuver`, `radar_only` scenarios, or `--radar` on any scenario) with the same noise/latency model as the AirSim radar node.
  - `prototype/run_scenario.sh <scenario>` — one-shot runner: launches the C++ node (auto-enabling `use_radar` for `radar*` scenarios), drives it, tears both down, prints the summary.

## Architecture (big picture)

```text
interceptor_node_relative_imm_ukf.h    — node class declaration (members, nested cache/sample structs)
interceptor_node_relative_imm_ukf.cpp  — node method definitions + main()
relative_imm_ukf.h                     — UnscentedKalmanFilter10D + RobustIMMEstimator10D + RelativeIMMUKFEstimator
advanced_hybrid_controller.h           — AdvancedHybridController (PN/IBVS guidance + thrust allocation)
utils.h                                — MissionStateMachine, Eigen aliases, clamp
guidance_limits.h                      — bounded axial speed / terminal speed floor / vision-range helpers
imu_strapdown.h                        — gyro attitude integration
camera_pose_history.h                  — timestamped camera orientation ring buffer (delay compensation)
```

- **Estimator stack** (`relative_imm_ukf.h`, header-only): 10D relative state `[r(3) v_rel(3) a_t(3) W(1)]` in a gravity-aligned world frame; IMM over 3 motion models (Singer, coordinated turn, rotor — accel-relaxation, no drag), each a UKF with robust Cholesky; monocular pinhole `[u, v, size]` measurements plus linear radar `[r, v_rel]` updates; ego-motion from IMU strapdown preintegration (odometry-free).
- **Controller** (`advanced_hybrid_controller.h`, header-only): PN + IBVS with FOV keep-in-frame guard, axial jerk limit, terminal speed floor, close-range lateral decay.
- **Node** (`interceptor_node_relative_imm_ukf.{h,cpp}`): ROS wiring, parameter load/validate, radar relay pairing/handover/re-anchor, mission state machine, capture/brake/coast logic, control timer.
- **Radar relay** (optional, `radar.enable`, off by default): `/fpv/radar/target` + `/fpv/radar/interceptor` (`nav_msgs/Odometry`, world ENU, `header.stamp` = measurement epoch) are differenced into a relative `[r, v_rel]` linear UKF update, giving mid-course guidance while the target is still outside the camera FOV. On visual acquisition the target width is fixed from `bbox_size * r_radar / focal` and locked, anchoring the monocular terminal range. Producer is `airsim_tracker/radar_relay_node.py`. Full contract in `AGENTS.md`.
- **Control flow**: vision bbox → relative-state IMM-UKF → guidance (PN mid-course, IBVS/FOV-guarded terminal) → `quadrotor_msgs::ControlCommand` = `collective_thrust` + body rates `[omega_x, omega_y, omega_z]`, published on `/fpv/control_command` at ~50 Hz.

ROS2 topics (overridable via `topics.*` params): subs `/mavros/imu/data`, `/fpv/target_bbox`, `/fpv/gimbal`, `/fpv/tracker_trigger` (Header, frame_id `MISSION`/`OFF`); pubs `/fpv/control_command`, `/fpv/mission_state`, `/mavros/statustext/send`.

## Conventions that cause silent bugs if ignored

- **Quaternion order is scalar-first `[w, x, y, z]`** — NOT ROS's `[x, y, z, w]`. Mixing these up silently corrupts attitude.
- **Camera FLU vs estimator RDF frames**: `/fpv/gimbal` orientation is q_C_W in the FLU camera frame (x=fwd, y=left, z=up); the estimator works in the RDF optical frame (x=right, y=down, z=fwd). Conversion happens at intake — don't mix them.
- **Control vector** is `[omega_x, omega_y, omega_z, thrust_accel]` (body rates + collective thrust in m/s²).
- **`focal_length_` derives from `camera.width_px` + `camera.fov_deg`** — the width handover and terminal monocular range are only as good as it. `airsim_settings.json` must match these before any AirSim run.
- Headers order: `#pragma once`, then system → ROS → project. Naming: `snake_case` members/methods, `PascalCase` classes, `UPPER_CASE` constants, trailing `_` for members. Comments are intentionally mixed English/中文.

## Communication preference

The user prefers terse, low-fluff responses ("caveman" style — drop articles/filler/pleasantries, fragments OK, technical terms exact). This applies to prose only; **code, commits, and PRs are written normally.** Drop the terse style for security warnings, irreversible actions, or when the user seems confused. (Source: `AGENTS.md` header.)

## Key references

- `AGENTS.md` — full architecture, conventions, estimator design, radar-relay contract, and pitfalls table. Read this for anything not covered above.
- `prototype/sim_test_relative_imm_ukf.py` — the verified Python mirror the C++ estimator follows.
