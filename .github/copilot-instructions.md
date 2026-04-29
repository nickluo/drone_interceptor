# drone_interceptor — Copilot Instructions

Vision-based drone interception using NMPC. Two solver variants (quaternion 16D, dual-quaternion 17D) share a Template Method base class.

## Architecture

```
InterceptorNodeBase (abstract)       — control loop, PN guidance, EKF, safety
├── InterceptorNodeQuaternion        — 16D state [pos(3) vel_world(3) quat(4) omega(3) gimbal(3)]
└── InterceptorNodeDQ                — 17D state [dq(8) omega(3) v_body(3) gimbal(3)]
utils.h                              — DroneEKF(19D), MissionStateMachine, EMAFilter, CameraIntrinsics
```

- **13 pure virtual methods** in the base: state accessors, solver lifecycle, param packing, delay compensation, orientation validation
- Two-phase init: base constructor declares params only; subclass calls `initialize()` after vtable is ready
- Thread safety: `std::shared_mutex` for drone_state/EKF/gimbal/control_history; `std::mutex` for bbox

## Build

```bash
# 1. Generate acados C solver (must run BEFORE colcon build)
cd prototype && python3 drone_model_quaternion.py   # or drone_model_dq.py
# 2. Build
cd ~/workspace/fpv_ws && colcon build --packages-select drone_interceptor
```

- C++17, Release mode, `ament_cmake`
- Links: `acados`, `blasfeo`, `hpipm`, `m`
- Generated C code in `prototype/c_generated_code/` compiled directly into executables

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
| `src/utils.h` | EKF, state machine, filters (header-only) |
| `prototype/drone_model_dq.py` | DQ dynamics + cost + solver code generation |
| `prototype/drone_model_quaternion.py` | Quaternion dynamics + cost + solver code generation |
| `config/*_params.yaml` | Runtime parameters (real hardware vs AirSim) |

## Pitfalls

- Modifying cost function or dynamics requires **regenerating the C solver** from `prototype/` before rebuilding
- FOV cost needs `forward_scale` (zero when target behind camera) to avoid Jacobian explosion
- PN guidance diverges when EKF has >2x range error — disabled by default
- Always use solver output even with status≠0 (suboptimal solution > fallback)
- `penalty_coeff` must be TTA-based with cap at 30 to prevent close-range solver explosion
