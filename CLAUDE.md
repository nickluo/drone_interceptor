# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Vision-based drone interception. A ROS2 (ament_cmake, C++17) package that runs a pursuer UAV to intercept a moving target using onboard vision (a 2D detection bbox). Three interchangeable node variants share the same control-command interface but differ in state representation and algorithm.

`AGENTS.md` (mirrored in `.github/copilot-instructions.md`) is the authoritative, detailed reference for architecture, conventions, solver design, and pitfalls. Read it for depth; this file gives the actionable summary and the exact commands.

## Build & Run

**Workspace root is `/media/nick/data/workspace/fpv_ws`** (this package lives at `src/drone_interceptor`).

### Manual dependencies (not in package.xml)

`acados`, `blasfeo`, `hpipm`, and `Eigen3` must be installed on the system yourself — they are NOT declared in `package.xml`, so `rosdep` will not install them. `quadrotor_msgs` (the `ControlCommand` type) is also an external package dependency.

### Build — NMPC variants require a codegen step first

The quaternion and DQ (dual-quaternion) variants compile generated ACADOS C code from `prototype/c_generated_code/`. **You must regenerate that code before `colcon build` whenever you change NMPC dynamics or cost**, or the build will fail / use stale code:

```bash
cd /media/nick/data/workspace/fpv_ws/src/drone_interceptor/prototype
python3 drone_model_quaternion.py    # for interceptor_node_quaternion
python3 drone_model_dq.py            # for interceptor_node_dq
```

Then build from the workspace root:

```bash
cd /media/nick/data/workspace/fpv_ws
colcon build --packages-select drone_interceptor
source install/setup.bash
```

The `interceptor_node_relative_imm_ukf` target does NOT use acados (links only `libm`) — it builds without any codegen step.

### Run

```bash
ros2 launch drone_interceptor launch.py                    # quaternion node, real hardware
ros2 launch drone_interceptor launch_dq.py                 # DQ node, real hardware
ros2 launch drone_interceptor launch_relative_imm_ukf.py   # IMM-UKF estimator + PN/IBVS controller

ros2 launch drone_interceptor airsim_launch.py             # quaternion node, AirSim sim params
ros2 launch drone_interceptor airsim_launch_dq.py          # DQ node, AirSim sim params
```

Each launch loads a YAML from `config/` (`*_params.yaml`). AirSim configs differ mainly in `execution_delay` (10 ms vs 50 ms on hardware). Note: `use_simulation` defaults to `False` in the IMM-UKF launch — verify it before AirSim runs.

## Testing

There is **no formal unit-test suite** — `ament_lint_common` is declared but no gtest/pytest targets are wired in CMake. Algorithm verification lives in `prototype/`:

- `prototype/sim_test_relative_imm_ukf.py` — pure-Python mirror of the C++ IMM-UKF + PN/IBVS controller run against simplified quadrotor dynamics + camera model. This is the de-facto test for the estimator/controller.
- `prototype/ros2_sim_node.py` — ROS2 simulation harness that publishes `/fpv/odom` + `/fpv/target_bbox` and consumes `/fpv/control_command`; runs scenarios (nominal, crossing, maneuvering, dropout, noise, wrong_w) producing the `*.csv`/`*_summary.txt`/`*.png` results in `prototype/`.
- `prototype/solver_acmpc_trt_dq.py` — TensorRT prototype of a learned AC-MPC DQ-NMPC policy (separate exploration track, not compiled into the C++ nodes).

There is no linter/CI config actually enabled in CMake.

## Architecture (big picture)

```
InterceptorNodeBase (abstract, Template Method)   — control loop, PN terminal guidance, DroneEKF, safety
├── InterceptorNodeQuaternion   16D [pos(3) vel_world(3) quat(4) omega(3) gimbal(3)]   — acados NMPC
└── InterceptorNodeDQ           17D [dq(8) omega_body(3) v_body(3) gimbal(3)]          — acados NMPC

InterceptorNodeRelativeIMMUKF (standalone rclcpp::Node, NOT a base subclass)
   10D target state [pos(3) vel(3) acc(3) width(1)], IMM(3 models)-UKF, PN+IBVS controller, NO acados
```

- **NMPC variants**: `InterceptorNodeBase` (in `src/interceptor_node_base.{h,cpp}`) owns the full algorithm skeleton via 13 pure virtual methods; the two subclasses supply state-representation-specific accessors and the acados solver binding. Two-phase init: the base constructor only declares params; the subclass calls `initialize()` after the vtable is ready.
- **IMM-UKF variant** (`src/interceptor_node_relative_imm_ukf.cpp`) is fully standalone — no `InterceptorNodeBase`, no acados. It estimates the **absolute target** state (the NMPC variants estimate target state relative to the interceptor's ego-state via `DroneEKF` in `utils.h`).
- **Control flow**: vision bbox → state estimation (DroneEKF 19D, or IMM-UKF 10D) → guidance (NMPC at range, switching to True Proportional Navigation for terminal intercept) → `quadrotor_msgs::ControlCommand` = `collective_thrust` + body rates `[omega_x, omega_y, omega_z]`, published on `/fpv/control_command` at ~50 Hz.

ROS2 topics are identical across variants so any node drives the same low-level flight controller: subs `/fpv/odom`, `/fpv/target_bbox`, `/mavros/imu/data` (IMM-UKF), `/fpv/gimbal` (NMPC hardware), `/fpv/tracker_trigger`; pubs `/fpv/control_command`, `/fpv/mission_state`, `/mavros/statustext/send`. `utils.h` (header-only) holds `DroneEKF`, `MissionStateMachine`, `EMAFilter`, `CameraIntrinsics`.

## Conventions that cause silent bugs if ignored

- **Quaternion order is scalar-first `[w, x, y, z]`** — NOT ROS's `[x, y, z, w]`. Mixing these up silently corrupts attitude.
- **Velocity frame differs by variant**: DQ state velocity is **body frame**; quaternion state velocity is **world frame**.
- **Control vector** is `[omega_x, omega_y, omega_z, thrust_accel]` (body rates + collective thrust in m/s²).
- **NMPC cost type must be `NONLINEAR_LS`, not `EXTERNAL`** — EXTERNAL has a zero control gradient at hover and causes SQP_RTI MINSTEP failures.
- **Regenerate the acados C solver after any NMPC dynamics/cost change**, then rebuild.
- Headers order: `#pragma once`, then system → ROS → project → `extern "C"` for acados. Naming: `snake_case` members/methods, `PascalCase` classes, `UPPER_CASE` constants, trailing `_` for members. Comments are intentionally mixed English/中文.

## Communication preference

The user prefers terse, low-fluff responses ("caveman" style — drop articles/filler/pleasantries, fragments OK, technical terms exact). This applies to prose only; **code, commits, and PRs are written normally.** Drop the terse style for security warnings, irreversible actions, or when the user seems confused. (Source: `AGENTS.md` header.)

## Key references

- `AGENTS.md` — full architecture, conventions, solver design, and pitfalls table. Read this for anything not covered above.
- `README_QUATERNION_NODE.md` — quaternion node math, hard attitude constraints, performance notes.
- `prototype/HARD_CONSTRAINT_IMPLEMENTATION.md`, `prototype/README_quaternion.md` — constraint/quaternion math notes.
