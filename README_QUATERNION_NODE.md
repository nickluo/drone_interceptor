# Quaternion-based Drone Interceptor Node

## Overview

This document describes the quaternion-based ROS2 C++ implementation of the drone interceptor, corresponding to the `drone_model_quaternion.py` Python implementation.

## Key Features

### 1. Quaternion State Representation
- **State Vector**: `[px, py, pz, vx, vy, vz, qw, qx, qy, qz]` (10 dimensions)
- **Quaternion Convention**: Scalar-first `[qw, qx, qy, qz]`
- **Benefits**: Avoids gimbal lock, provides smooth attitude representation

### 2. Hard Attitude Constraints
- **Mathematical Basis**: `cos(θ) = 1 - 2*(qx² + qy²)` for tilt angle
- **Constraint Implementation**: Box constraints on `qx` and `qy` components
- **Safety Bounds**: `qx, qy ∈ [-√((1-cos(θ_max))/2), √((1-cos(θ_max))/2)]`

### 3. Control Architecture
- **Control Input**: `[wx, wy, wz, az]` (body angular velocities + thrust acceleration)
- **Output**: Body rates for low-level flight controller
- **Safety Features**: Attitude limiting, rate limiting, execution delay compensation

## Implementation Details

### File Structure
```
src/interceptor_node_quaternion.cpp    # Main ROS2 node implementation
config/interceptor_params_quaternion.yaml    # Parameter configuration
launch/launch_quaternion.py            # Launch file for quaternion node
```

### Core Components

#### 1. Quaternion Helper Functions
```cpp
Eigen::Vector4d euler_to_quaternion(double roll, double pitch, double yaw)
Eigen::Vector3d quaternion_to_euler(const Eigen::Vector4d& quat)
Eigen::Vector4d normalize_quaternion(const Eigen::Vector4d& quat)
Eigen::Matrix3d quaternion_to_rotation_matrix(const Eigen::Vector4d& quat)
```

#### 2. Safety Functions
```cpp
Eigen::Vector3d apply_attitude_safety_limit_quat(const Vector10d& drone_state, 
                                                  const Eigen::Vector3d& body_rates, 
                                                  double att_limit)
```

#### 3. ACADOS Integration
- **Solver**: `drone_interceptor_mpc_quat`
- **Generated Files**: Located in `prototype/c_generated_code/`
- **Functions**: 
  - `drone_interceptor_mpc_quat_acados_create_capsule()`
  - `drone_interceptor_mpc_quat_acados_solve()`
  - `drone_interceptor_mpc_quat_acados_update_params()`

## Building and Running

### Prerequisites
1. ROS2 (tested on Humble)
2. ACADOS library
3. Eigen3
4. Generated quaternion ACADOS solver files

### Build Instructions
```bash
cd /path/to/your/workspace
colcon build --packages-select drone_interceptor
source install/setup.bash
```

### Running the Node
```bash
# Simulation mode
ros2 run drone_interceptor interceptor_node_quaternion --ros-args -p use_simulation:=true

# Real mode with custom config
ros2 launch drone_interceptor launch_quaternion.py

# With custom parameters
ros2 run drone_interceptor interceptor_node_quaternion --ros-args --params-file config/interceptor_params_quaternion.yaml
```

## Parameters

### Key Parameters
- `limits.att_limit`: Attitude limit in radians (default: π/4 = 45°)
- `mpc.n_horizon`: MPC horizon length (default: 30)
- `mpc.tf`: Prediction time horizon in seconds (default: 1.0)
- `physics.g`: Gravitational acceleration (default: 9.81)
- `cost.w_los`: Line-of-sight error weight (default: 100.0)

### Hard Constraint Configuration
The attitude constraints are automatically configured based on `limits.att_limit`:
```cpp
max_tilt = att_limit;  // e.g., π/4 radians
cos_max_tilt = cos(max_tilt);
tilt_constraint_bound = (1 - cos_max_tilt) / 2;
qxy_max = sqrt(tilt_constraint_bound);
// Box constraints: qx, qy ∈ [-qxy_max, qxy_max]
```

## Comparison with Euler Implementation

| Aspect | Euler Implementation | Quaternion Implementation |
|--------|---------------------|---------------------------|
| State Size | 9D (pos, vel, euler) | 10D (pos, vel, quat) |
| Singularities | Gimbal lock at ±90° pitch | None |
| Attitude Constraints | Euler angle limits | Quaternion component limits |
| Computational Cost | Lower | Slightly higher |
| Robustness | Limited | High |

## Debug and Troubleshooting

### Common Issues
1. **Segmentation Fault**: Check ACADOS solver generation and library paths
2. **Constraint Violations**: Verify attitude limit parameters
3. **Poor Performance**: Tune cost function weights

### Debug Output
The node provides extensive logging:
```bash
# Enable debug logging
ros2 run drone_interceptor interceptor_node_quaternion --ros-args --log-level DEBUG
```

### Key Log Messages
- Quaternion state values and tilt angles
- Constraint violation reports
- ACADOS solver status
- Control output values

## Performance Characteristics

### Computational Performance
- **ACADOS Solve Time**: Typically 50-200 μs per iteration
- **Update Rate**: 50 Hz (dt = 0.02s for N=30, Tf=1.0)
- **Memory Usage**: ~10-20 MB for solver

### Constraint Enforcement
- **Hard Constraints**: Perfect enforcement (zero violations)
- **Attitude Safety**: Tilt angles maintained within specified limits
- **Control Limits**: Body rates and thrust bounded appropriately

## Future Enhancements

1. **Adaptive Horizon**: Dynamic N_horizon based on target distance
2. **Multi-Target**: Support for multiple target interception
3. **Wind Compensation**: Disturbance rejection improvements
4. **Real-time Tuning**: Dynamic parameter adjustment interface

## References

- ACADOS Documentation: https://docs.acados.org/
- Quaternion Mathematics: Kuipers, "Quaternions and Rotation Sequences"
- MPC Theory: Rawlings, Mayne, Diehl, "Model Predictive Control"
