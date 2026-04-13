"""
基于对偶四元数 NMPC (DQ-NMPC) 的无人机目标拦截仿真。

替代 solver_flat_quaternion.py 中基于四元数的MPC,
使用 DQ-NMPC 在 SE(3) 李群上统一处理位姿跟踪与目标拦截任务。

主要改进:
  1. 对偶四元数统一表示位姿, 消除欧拉角万向节锁
  2. 李代数对数映射代价函数, 旋转+平移耦合优化
  3. 角速度+推力加速度控制接口 (与四元数模型一致)
  4. 保留原有感知管线 (相机模型, Bearing-Angle EKF)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from exp_filter import EMAFilter
from camera_model import CameraModel, PerceptionSim
from target_ekf import TargetEKF
from drone_model_dq import (
    DroneModelDQ,
    pose_to_dq_np, dq_to_pose_np, normalize_dq_np,
    quat_product_np
)

SHOW_RESULTS = True # 是否显示仿真结果图表

# ==============================================================================
# 物理常量 (Physics Constants)
# ==============================================================================
MASS = 4.0            # 无人机质量 (kg)
G = 9.8065            # 重力加速度 (m/s^2)
A_MAX = 20.0          # 最大加速度 (m/s^2) → F_max = A_MAX * MASS
A_MIN = 1.0           # 最小加速度 (m/s^2) → F_min = A_MIN * MASS
DRAG_COEFF = 0.1625   # 线性空气阻力系数
ATT_LIMIT = np.pi / 5 # 姿态角限制

# 惯性参数
IXX = 0.03  # kg*m^2
IYY = 0.03
IZZ = 0.06
TAU_MAX = 2.0  # 最大力矩 (Nm)
ROLL_PITCH_RATE_MAX = 2.0 * np.pi  # 最大滚转/俶仰角速度 (rad/s)
YAW_RATE_MAX = 1.0 * np.pi         # 最大偏航角速度 (rad/s)
RATE_TC = 0.02                     # 角速度一阶响应时间常数 (s)

# ==============================================================================
# MPC 控制器参数
# ==============================================================================
CTRL_FREQ_HZ = 50
TF = 0.8              # 预测时域 (s)
N_HORIZON = int(TF * CTRL_FREQ_HZ)
V_DESIRED = 18.0      # 期望接近速度 (m/s)

# ==============================================================================
# 仿真设置
# ==============================================================================
SIM_TIME = 15.0
INTERCEPTION_DISTANCE = 0.5

# ==============================================================================
# 感知参数
# ==============================================================================
TRACKING_DISTANCE = 0.0  # 期望跟踪距离 (m)
PERCEPTION_DELAY = 0.08
BBOX_POS_STD_DEV = 3.0
BBOX_SIZE_STD_DEV = 3.0
TARGET_PHYSICAL_SIZE_TRUTH = 0.5
TARGET_WIDTH_INITIAL_EST = 1.0

# ==============================================================================
# 相机参数
# ==============================================================================
CAM_FOV_DEG = 30.0
CAM_FOCAL_LENGTH_PHYSICAL = 16.0
CAM_PIXEL_SIZE = 3.45
CAM_FOCAL_LENGTH = CAM_FOCAL_LENGTH_PHYSICAL * 1000 / CAM_PIXEL_SIZE

# ==============================================================================
# 云台参数
# ==============================================================================
GIMBAL_RP_COEFF = 12.0
GIMBAL_Y_COEFF = 10.0
GIMBAL_DISTURBANCE_COEFF = 0.5

# ==============================================================================
# DQ-NMPC 代价权重
# NOTE: effective ω_dot cost = w_control_tau / rp_rate_max² × ω_dot²
#   w_control_tau=0.5 → ~0.013×ω_dot², comparable to quat variant (0.01×ω_dot²)
#   Too-weak regularization causes oscillation via ctrl_coeff_ model mismatch.
# ==============================================================================
W_LOS = 100.0             # 相机帧LOS误差权重 (increased: prioritize boresight alignment)
W_ZEM = 10.0              # 零效果脱靶量(ZEM)权重 (reduced: less aggressive lateral correction)
W_CLOSING = 0.3           # 接近速度权重 (v6: 2× v5, 3× quat — stronger closing incentive)
W_ALIGN = 5.0             # 速度对齐权重 (v7c: reverted from 10, competes with closing in DQ body-frame)
W_CONTROL_F = 0.001       # 推力加速度正则化权重
W_CONTROL_TAU = 1.0       # 角速度命令正则化权重 (v6: halved, allow faster rotation)
W_TILT = 0.01             # 倾斜角惩罚权重 (v6: 5× reduction, allow tilting for accel)
W_OMEGA = 0.001           # 角速度正则化权重 (v7: match quaternion rates penalty, was 0.05)
W_FOV = 10.0              # 相机FOV栏栅代价权重 (v4: soft, no snap-back oscillation)

# 权重向量 (直接值, EXTERNAL代价类型直接使用)
W_COST = np.array([
    W_LOS, W_ZEM,
    W_CLOSING, W_ALIGN,
    W_CONTROL_F, W_CONTROL_TAU,
    W_TILT, W_OMEGA, W_FOV])

# ==============================================================================
# PN 导引律参数 (当前禁用: EKF位置误差大时PN发散)
# ==============================================================================
PN_ENABLED = False
PN_NAV_CONSTANT = 5.0
PN_SWITCH_DIST = 20.0
PN_FOV_SWITCH_DEG = 8.0

# ==============================================================================
# 初始条件
# ==============================================================================
DRONE_INITIAL_STATE = np.array([
    0.0, 0.0, 10.0,      # Position (p_x, p_y, p_z)
    0.0, 0.0, 0.0,       # Velocity (v_x, v_y, v_z)
    0.0, 0.0, 0.0,       # Attitude (phi, theta, psi)
    0.0, 0.0, 0.0,       # Omega (wx, wy, wz)
    0.0, 0.0, 0.0])      # Gimbal (r, p, y)
TARGET_INITIAL_POS = np.array([100.0, 20.0, 15.0])
TARGET_VELOCITY = np.array([-5.0, 5.0, 0.5])


# ==============================================================================
# 模型常量字典
# ==============================================================================
MODEL_CONSTANTS = {
    'G': G, 'MASS': MASS,
    'IXX': IXX, 'IYY': IYY, 'IZZ': IZZ,
    'TAU_MAX': TAU_MAX,
    'ROLL_PITCH_RATE_MAX': ROLL_PITCH_RATE_MAX,
    'YAW_RATE_MAX': YAW_RATE_MAX,
    'RATE_TC': RATE_TC,
    'A_MIN': A_MIN, 'A_MAX': A_MAX,
    'ATT_LIMIT': ATT_LIMIT,
    'N_HORIZON': N_HORIZON, 'TF': TF,
    'DRONE_INITIAL_STATE': DRONE_INITIAL_STATE,
    'CAM_FOV_DEG': CAM_FOV_DEG,
}


# ==============================================================================
# 状态转换工具
# ==============================================================================

def dq_state_to_euler_state(dq_state):
    """DQ状态 → 欧拉角状态 (用于感知管线和记录)
    输入: [dq(8), omega(3), v_body(3), gimbal(3)] = 17维
    输出: [pos(3), vel_world(3), euler(3), omega(3), gimbal(3)] = 15维
    """
    dq = dq_state[0:8]
    omega = dq_state[8:11]
    v_body = dq_state[11:14]
    gimbal = dq_state[14:17]

    q, pos = dq_to_pose_np(dq)
    euler = DroneModelDQ.quaternion_to_euler(q)
    R_mat = DroneModelDQ.quaternion_to_rotation_matrix(q)
    v_world = R_mat @ v_body

    return np.hstack([pos, v_world, euler, omega, gimbal])


def euler_state_to_dq_state(euler_state):
    """欧拉角状态 → DQ状态
    输入: [pos(3), vel_world(3), euler(3), omega(3), gimbal(3)] = 15维
    输出: [dq(8), omega(3), v_body(3), gimbal(3)] = 17维
    """
    pos = euler_state[0:3]
    v_world = euler_state[3:6]
    euler = euler_state[6:9]
    omega = euler_state[9:12]
    gimbal = euler_state[12:15]

    q = DroneModelDQ.euler_to_quaternion(euler[0], euler[1], euler[2])
    dq = pose_to_dq_np(q, pos)
    R_mat = DroneModelDQ.quaternion_to_rotation_matrix(q)
    v_body = R_mat.T @ v_world

    return np.hstack([dq, omega, v_body, gimbal])


# ==============================================================================
# PN 导引律 (适配DQ状态)
# ==============================================================================

def calculate_pn_controls_dq(dq_state, target_pos, target_vel, dt, mass):
    """
    3D比例导引, 输出 [u_wx, u_wy, u_wz, u_az]
    使用更稳健的姿态控制器, 降低增益防止数值发散。
    """
    dq = dq_state[0:8]
    omega = dq_state[8:11]
    v_body = dq_state[11:14]

    q, pos = dq_to_pose_np(dq)
    R_mat = DroneModelDQ.quaternion_to_rotation_matrix(q)
    vel = R_mat @ v_body  # 世界速度

    # 相对运动
    r_vec = target_pos - pos
    v_rel = vel - target_vel
    dist = np.linalg.norm(r_vec) + 1e-6
    r_unit = r_vec / dist

    # LOS角速率
    omega_los = np.cross(r_vec, -v_rel) / (dist * dist)

    # 接近速度 (添加下限防止近失后PN发散)
    vc_raw = np.dot(v_rel, r_unit)
    vc = max(vc_raw, 3.0)

    # PN加速度
    a_pn = PN_NAV_CONSTANT * vc * np.cross(omega_los, r_unit)
    a_des = a_pn + np.array([0.0, 0.0, G])

    # 期望推力方向
    thrust_accel = np.linalg.norm(a_des)
    z_b_des = a_des / (thrust_accel + 1e-6)
    thrust_accel = np.clip(thrust_accel, A_MIN, A_MAX)
    F_cmd = thrust_accel * mass

    # 期望偏航: 指向目标
    r_horiz = np.linalg.norm(r_vec[0:2])
    if r_horiz > 0.1:
        los_yaw = np.arctan2(r_vec[1], r_vec[0])
        x_c = np.array([np.cos(los_yaw), np.sin(los_yaw), 0.0])
    else:
        curr_yaw = DroneModelDQ.quaternion_to_euler(q)[2]
        x_c = np.array([np.cos(curr_yaw), np.sin(curr_yaw), 0.0])

    y_b_des = np.cross(z_b_des, x_c)
    if np.linalg.norm(y_b_des) < 1e-4:
        y_b_des = np.cross(z_b_des, np.array([1, 0, 0]))
    y_b_des /= np.linalg.norm(y_b_des)
    x_b_des = np.cross(y_b_des, z_b_des)
    R_des = np.column_stack((x_b_des, y_b_des, z_b_des))

    # 姿态误差 → 期望角速度 → 力矩
    R_err = R_mat.T @ R_des
    tr = np.trace(R_err)
    theta = np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0))

    if abs(theta) < 1e-6:
        omega_cmd = np.zeros(3)
    else:
        axis = (1 / (2 * np.sin(theta) + 1e-8)) * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]])
        omega_ff = R_mat.T @ omega_los
        # 降低增益: 20.0 (从40.0) 防止Euler积分发散
        omega_cmd = 20.0 * theta * axis + omega_ff

    # 力矩 = J * (Kp * omega_error + Kd * gyro_term)
    J = np.diag([IXX, IYY, IZZ])
    Kp_torque = 40.0  # 从80降低到40
    omega_error = omega_cmd - omega
    tau_cmd = J @ (Kp_torque * omega_error) + np.cross(omega, J @ omega)

    tau_cmd = np.clip(tau_cmd, -TAU_MAX, TAU_MAX)
    return np.array([omega_cmd[0], omega_cmd[1], omega_cmd[2], thrust_accel])


def _dq_state_derivative(dq_state, u_ctrl, mass):
    """计算DQ状态的时间导数 (用于RK4积分)"""
    dq = dq_state[0:8]
    omega = dq_state[8:11]
    v_body = dq_state[11:14]
    gimbal = dq_state[14:17]

    q = dq[0:4]
    R_mat = DroneModelDQ.quaternion_to_rotation_matrix(q)

    u_omega = u_ctrl[0:3]   # 期望角速度
    u_az = u_ctrl[3]        # 推力加速度
    e3 = np.array([0, 0, 1.0])

    # 角加速度 (一阶跟踪)
    omega_dot = np.array([
        (u_omega[0] - omega[0]) / RATE_TC,
        (u_omega[1] - omega[1]) / RATE_TC,
        (u_omega[2] - omega[2]) / (RATE_TC * 2)])

    # 线加速度 (机体坐标系)
    g_body = R_mat.T @ np.array([0, 0, G])
    v_dot = u_az * e3 + np.cross(v_body, omega) - g_body - DRAG_COEFF * v_body

    # 四元数导数
    wx, wy, wz = omega
    omega_mat = np.array([
        [0, -wx, -wy, -wz],
        [wx,  0,  wz, -wy],
        [wy, -wz,  0,  wx],
        [wz,  wy, -wx,  0]])
    q_dot = 0.5 * omega_mat @ q

    # DQ导数: dual部分需要角速度 + 速度两项
    # d_dot = 0.5 * (Omega(omega) @ d + H_r_plus(q) @ [0, v_body])
    d = dq[4:8]
    omega_mat_d = np.array([
        [0, -wx, -wy, -wz],
        [wx,  0,  wz, -wy],
        [wy, -wz,  0,  wx],
        [wz,  wy, -wx,  0]])
    # H+(qr) 矩阵: qr ⊗ [0, v_body]
    H_r = np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1],  q[0], -q[3],  q[2]],
        [q[2],  q[3],  q[0], -q[1]],
        [q[3], -q[2],  q[1],  q[0]]])
    v_quat = np.array([0, v_body[0], v_body[1], v_body[2]])
    d_dot = 0.5 * (omega_mat_d @ d + H_r @ v_quat)

    dq_dot = np.hstack([q_dot, d_dot])

    # 云台衰减
    gimbal_dot = np.array([
        -gimbal[0] * GIMBAL_RP_COEFF,
        -gimbal[1] * GIMBAL_RP_COEFF,
        -gimbal[2] * GIMBAL_Y_COEFF])

    return np.hstack([dq_dot, omega_dot, v_dot, gimbal_dot])


def integrate_dq_state(dq_state, u_ctrl, dt, mass):
    """RK4积分 (代替Euler积分, 数值稳定性更好)"""
    k1 = _dq_state_derivative(dq_state, u_ctrl, mass)
    k2 = _dq_state_derivative(dq_state + 0.5 * dt * k1, u_ctrl, mass)
    k3 = _dq_state_derivative(dq_state + 0.5 * dt * k2, u_ctrl, mass)
    k4 = _dq_state_derivative(dq_state + dt * k3, u_ctrl, mass)

    new_state = dq_state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # 归一化四元数和DQ
    new_state[0:4] = DroneModelDQ.normalize_quaternion(new_state[0:4])
    new_state[0:8] = normalize_dq_np(new_state[0:8])

    return new_state


# ==============================================================================
# 主仿真循环
# ==============================================================================

def main():
    print("[INFO] 构建 DQ-NMPC 模型与求解器...")
    drone_model = DroneModelDQ(MODEL_CONSTANTS)
    acados_solver = drone_model.generate_solver()
    print("[INFO] DQ-NMPC 求解器构建完成。")

    dt = TF / N_HORIZON
    N_sim = int(SIM_TIME / dt)

    # 初始化DQ状态
    x_current_dq = euler_state_to_dq_state(DRONE_INITIAL_STATE)
    target_pos_true = TARGET_INITIAL_POS.copy()
    target_vel_true = TARGET_VELOCITY.copy()

    # 感知管线
    camera = CameraModel(CAM_FOCAL_LENGTH, TARGET_PHYSICAL_SIZE_TRUTH, CAM_FOV_DEG,
                         BBOX_POS_STD_DEV, BBOX_SIZE_STD_DEV)
    estimator = TargetEKF(dt, CAM_FOCAL_LENGTH, BBOX_POS_STD_DEV, BBOX_SIZE_STD_DEV,
                           initial_width_est=TARGET_WIDTH_INITIAL_EST,
                           range_init_bias=0.6,
                           q_accel=3.0, q_W=0.01, alpha_R_inflate=1.0)
    output_filter = EMAFilter(int(N_HORIZON / TF * 0.1), channels=4)
    perceptor = PerceptionSim(camera, PERCEPTION_DELAY, dt)

    target_lost_counter = 0

    # 初始欧拉角状态 (用于记录)
    x_current_euler = dq_state_to_euler_state(x_current_dq)
    history = {
        'x': [x_current_euler],
        'u': [],
        'target_true': [target_pos_true.copy()],
        'target_vel_true': [target_vel_true.copy()],
        'target_est_ekf': [],
        'target_est_raw': [],
        'distance': []
    }

    # 热启动
    u_hover = np.array([0, 0, 0, G])
    for k in range(N_HORIZON):
        acados_solver.set(k, 'x', x_current_dq)
        acados_solver.set(k, 'u', u_hover)
    acados_solver.set(N_HORIZON, 'x', x_current_dq)

    active_controller = "DQ-NMPC"
    print("\n[INFO] 开始 DQ-NMPC 拦截仿真...")
    solve_time_avg = 0.0

    # 死算跟踪状态: 当目标丢失且距离近时, 使用最后好估计做线性外推
    last_good_pos_est = None
    last_good_vel_est = None
    last_good_time = 0.0
    terminal_phase = False  # 终端制导阶段标识

    for i in range(N_sim):
        # -------- 1. 感知 --------
        x_euler_for_perception = dq_state_to_euler_state(x_current_dq)
        full, target_pos_true = perceptor.update(
            x_euler_for_perception, target_pos_true, target_vel_true, history)
        if not full:
            continue
        drone_state_at_meas, meas_bbox, meas_visible = perceptor.get_measure()

        # -------- 2. EKF 预测 --------
        if estimator.is_initialized:
            estimator.predict(dt)

        # -------- 3. EKF 更新 --------
        if meas_visible:
            target_lost_counter = 0
            drone_pos = drone_state_at_meas[0:3]
            psi = drone_state_at_meas[8]
            gr, gp, gy = drone_state_at_meas[12], drone_state_at_meas[13], drone_state_at_meas[14]
            if not estimator.is_initialized:
                estimator.initialize(meas_bbox, drone_pos, psi, gr, gp, gy)
            else:
                estimator.update(meas_bbox, drone_pos, psi, gr, gp, gy)
            raw_pos_est = estimator.get_target_position()
        else:
            target_lost_counter += 1
            raw_pos_est = None
            if estimator.is_initialized:
                q_dbg, pos_dbg = dq_to_pose_np(x_current_dq[0:8])
                dist_dbg = np.linalg.norm(target_pos_true - pos_dbg)
                if target_lost_counter <= 3 or target_lost_counter == int(N_HORIZON * 2.0):
                    print(f"[WARNING] 步骤 {i}: 目标丢失! 盲飞: {target_lost_counter}/{int(N_HORIZON*2.0)}, dist={dist_dbg:.1f}m")

        MAX_BLIND_STEPS = int(N_HORIZON * 2.0)  # 盲飞容忍步数
        if target_lost_counter > MAX_BLIND_STEPS:
            print(f"\n[FAILED] 目标丢失超过 {MAX_BLIND_STEPS*TF/N_HORIZON:.1f}s, 任务失败。")
            break

        # -------- 4. 规划 --------
        if not estimator.is_initialized:
            print(f"[WARNING] 步骤 {i}: EKF未初始化, 悬停等待。")
            u_optimal_raw = np.array([MASS * G, 0, 0, 0])
        else:
            pos_est = estimator.x[0:3]
            vel_est = estimator.x[3:6]

            # 当前无人机位姿
            q_cur, pos_cur = dq_to_pose_np(x_current_dq[0:8])

            # 保存最后好估计 (当目标可见时)
            if meas_visible or target_lost_counter == 0:
                last_good_pos_est = pos_est.copy()
                last_good_vel_est = vel_est.copy()
                last_good_time = i * dt

            # 当近距离丢失目标时, 使用死算外推代替漂移的EKF
            dist_to_est = np.linalg.norm(pos_est - pos_cur)

            if target_lost_counter > 0 and last_good_pos_est is not None:
                # 死算: 从最后好估计做恒速外推
                dt_since_good = i * dt - last_good_time
                pos_est_dr = last_good_pos_est + last_good_vel_est * dt_since_good
                vel_est_dr = last_good_vel_est.copy()

                # 计算死算距离
                dist_to_dr = np.linalg.norm(pos_est_dr - pos_cur)

                # 如果近距离 (< 15m) 且丢失目标, 优先使用死算
                if dist_to_dr < 15.0:
                    pos_est = pos_est_dr
                    vel_est = vel_est_dr
                    if not terminal_phase and dist_to_dr < 10.0:
                        terminal_phase = True
                        print(f"[TERMINAL] 步骤 {i}: 进入终端制导 dist={dist_to_dr:.1f}m (死算)")

            # PN切换检查 (需要同时满足距离和FOV对中条件)
            use_pn = False
            dist_to_target = np.linalg.norm(pos_est - pos_cur)

            if PN_ENABLED:
                if active_controller == "PN":
                    use_pn = True
                else:
                    if dist_to_target < PN_SWITCH_DIST:
                        R_cur = DroneModelDQ.quaternion_to_rotation_matrix(q_cur)
                        r_W = pos_est - pos_cur
                        r_B = R_cur.T @ r_W
                        cos_theta = r_B[0] / (np.linalg.norm(r_B) + 1e-6)
                        theta_deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                        if theta_deg < PN_FOV_SWITCH_DEG:
                            use_pn = True

            if use_pn:
                if active_controller != "PN":
                    print(f"\n[SWITCH] Dist={dist_to_target:.1f}m → 切换到 PN")
                    active_controller = "PN"
                PN_SUBSTEPS = 4
                dt_sub = dt / PN_SUBSTEPS
                for _ in range(PN_SUBSTEPS):
                    u_optimal_raw = calculate_pn_controls_dq(
                        x_current_dq, pos_est, vel_est, dt_sub, MASS)
                    x_current_dq = integrate_dq_state(x_current_dq, u_optimal_raw, dt_sub, MASS)

                # 云台扰动
                omega_cur = x_current_dq[8:11]
                x_current_dq[14] += GIMBAL_DISTURBANCE_COEFF * omega_cur[0] * dt
                x_current_dq[15] += GIMBAL_DISTURBANCE_COEFF * omega_cur[1] * dt
                x_current_dq[16] += GIMBAL_DISTURBANCE_COEFF * omega_cur[2] * dt
            else:
                # ======== DQ-NMPC ========
                acados_solver.set(0, "lbx", x_current_dq)
                acados_solver.set(0, "ubx", x_current_dq)

                # 计算当前距离, 用于自适应参数
                dist_current = dist_to_target

                # 距离自适应期望接近速度: 保持高速直到非常接近
                # 不过度减速: 在最后几米维持足够velocity确保命中
                v_desired_adaptive = V_DESIRED * np.clip(dist_current / 6.0, 0.7, 1.0)

                # 设置每步参数: [target_pos(3), target_vel(3), v_desired, drag_coeff, W(9), gimbal_coeff(2)] = 19
                for k in range(N_HORIZON + 1):
                    t_k = k * dt
                    pred_target_pos = pos_est + t_k * vel_est
                    pred_target_vel = vel_est
                    p_k = np.hstack([
                        pred_target_pos, pred_target_vel,
                        v_desired_adaptive, DRAG_COEFF,
                        W_COST,
                        GIMBAL_RP_COEFF, GIMBAL_Y_COEFF
                    ])
                    acados_solver.set(k, "p", p_k)

                t0 = time.time()
                status = acados_solver.solve()
                solve_dt = time.time() - t0

                if status != 0 and i % 50 == 0:
                    print(f'[WARNING] ACADOS状态 {status} @ 步骤 {i}')

                if i % 50 == 0:
                    print(f'[INFO] 步骤 {i} DQ-NMPC solve: {int(solve_dt*1e6)} us, dist={dist_current:.1f}m, v_des={v_desired_adaptive:.1f}')
                solve_time_avg = solve_time_avg * ((max(i, 1) - 1.0) / max(i, 1)) + solve_dt * 1e6 / max(i, 1)

                u_optimal_raw = acados_solver.get(0, "u")
                x_current_dq = acados_solver.get(1, "x")

                # NaN保护
                if np.any(np.isnan(x_current_dq)):
                    print(f'[ERROR] NaN detected @ 步骤 {i}, 停止仿真。')
                    break

                # 归一化DQ
                x_current_dq[0:8] = normalize_dq_np(x_current_dq[0:8])

                # 云台扰动
                omega_cur = x_current_dq[8:11]
                x_current_dq[14] += GIMBAL_DISTURBANCE_COEFF * omega_cur[0] * dt
                x_current_dq[15] += GIMBAL_DISTURBANCE_COEFF * omega_cur[1] * dt
                x_current_dq[16] += GIMBAL_DISTURBANCE_COEFF * omega_cur[2] * dt

                # 热启动
                for k in range(N_HORIZON - 1):
                    x_next = acados_solver.get(k + 1, "x")
                    x_next[0:8] = normalize_dq_np(x_next[0:8])
                    u_next = acados_solver.get(k + 1, "u")
                    acados_solver.set(k, "x", x_next)
                    acados_solver.set(k, "u", u_next)
                acados_solver.set(N_HORIZON - 1, "u", u_next)
                x_last = acados_solver.get(N_HORIZON, "x")
                x_last[0:8] = normalize_dq_np(x_last[0:8])
                acados_solver.set(N_HORIZON, "x", x_last)

        # -------- 5. 记录 --------
        target_pos_true += target_vel_true * dt

        x_current_euler = dq_state_to_euler_state(x_current_dq)
        history['x'].append(x_current_euler.copy())
        history['u'].append(u_optimal_raw.copy())
        history['target_true'].append(target_pos_true.copy())
        history['target_vel_true'].append(target_vel_true.copy())
        if estimator.is_initialized:
            history['target_est_ekf'].append(estimator.x.copy())
        if raw_pos_est is not None:
            history['target_est_raw'].append(raw_pos_est)

        q_cur, pos_cur = dq_to_pose_np(x_current_dq[0:8])
        distance = np.linalg.norm(target_pos_true - pos_cur)
        history['distance'].append(distance)

        if distance < INTERCEPTION_DISTANCE:
            print(f"\n[SUCCESS] 步骤 {i} 拦截成功! 距离: {distance:.2f}m")
            break

    min_dist = min(history['distance']) if history['distance'] else 999
    if i >= N_sim - 1:
        print(f"\n[INFO] 仿真结束。最终距离: {history['distance'][-1]:.2f}m, 最小距离: {min_dist:.2f}m")
    elif distance >= INTERCEPTION_DISTANCE:
        print(f"[INFO] 最小距离: {min_dist:.2f}m")

    print(f'求解平均耗时: {int(solve_time_avg)} us')
    if SHOW_RESULTS:
        plot_results(history, dt)


# ==============================================================================
# 可视化
# ==============================================================================

def plot_results(hist, dt):
    if not hist['u']:
        print("[ERROR] 无有效数据。")
        return

    hist['x'] = np.array(hist['x'])
    hist['u'] = np.array(hist['u'])
    hist['target_true'] = np.array(hist['target_true'])
    hist['target_vel_true'] = np.array(hist['target_vel_true'])
    hist['target_est_ekf'] = np.array(hist['target_est_ekf']) if hist['target_est_ekf'] else np.array([])
    hist['target_est_raw'] = np.array(hist['target_est_raw']) if hist['target_est_raw'] else np.array([])
    hist['distance'] = np.array(hist['distance'])

    time_x = np.linspace(0, dt * (len(hist['x']) - 1), len(hist['x']))
    time_u = np.linspace(0, dt * (len(hist['u']) - 1), len(hist['u']))

    # --- 3D轨迹 ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(hist['x'][:, 0], hist['x'][:, 1], hist['x'][:, 2],
            label='Drone (DQ-NMPC)', c='b', lw=2)
    ax.plot(hist['target_true'][:, 0], hist['target_true'][:, 1], hist['target_true'][:, 2],
            label='Target True', c='r', ls='--')
    if len(hist['target_est_ekf']) > 0:
        ax.plot(hist['target_est_ekf'][:, 0], hist['target_est_ekf'][:, 1],
                hist['target_est_ekf'][:, 2], label='Target EKF Estimated Traj.', c='g', ls='-')
    if len(hist['target_est_raw']) > 0:
        ax.scatter(hist['target_est_raw'][:, 0], hist['target_est_raw'][:, 1],
                   hist['target_est_raw'][:, 2], label='Target Raw Measurement',
                   c='orange', marker='.', alpha=0.5)
    ax.scatter(*hist['x'][0, 0:3], c='b', s=100, marker='o', label='Drone Start')
    ax.scatter(*hist['target_true'][0], c='r', s=100, marker='x', label='Target Start')
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]'); ax.set_zlabel('Up [m]')
    ax.legend(); ax.set_title('3D Trajectory (DQ-NMPC)'); ax.grid(True); ax.axis('auto')
    plt.show()

    # --- 性能指标 ---
    fig, axs = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    fig.suptitle('DQ-NMPC Interceptor Performance', fontsize=16)

    # 距离
    if len(hist['distance']) > 0:
        axs[0].plot(time_u, hist['distance'], label='Distance to Target')
        axs[0].axhline(y=TRACKING_DISTANCE, color='g', linestyle='--', label=f'Tracking Dist ({TRACKING_DISTANCE} m)')
    axs[0].set_ylabel('Distance [m]'); axs[0].set_title('Distance Over Time')

    # 接近速度和LOS误差
    closing_speed, los_error = [], []
    num_u = len(hist['u'])
    aligned_x = hist['x'][-num_u - 1:-1] if len(hist['x']) > num_u else hist['x'][:num_u]
    aligned_t = hist['target_true'][-num_u - 1:-1] if len(hist['target_true']) > num_u else hist['target_true'][:num_u]
    aligned_tv = hist['target_vel_true'][-num_u - 1:-1] if len(hist['target_vel_true']) > num_u else hist['target_vel_true'][:num_u]

    for j in range(min(num_u, len(aligned_x))):
        dx = aligned_x[j]
        dv, psi = dx[3:6], dx[8]
        tp = aligned_t[j]
        tv = aligned_tv[j]
        los = tp - dx[0:3]
        d = np.linalg.norm(los) + 1e-9
        ln = los / d
        closing_speed.append(np.dot(dv - tv, ln))

        cpsi, spsi = np.cos(psi), np.sin(psi)
        R_WC = np.array([[cpsi, spsi, 0], [-spsi, cpsi, 0], [0, 0, 1]])
        los_C = R_WC @ los
        if d > INTERCEPTION_DISTANCE * 2 and los_C[0] > 1.0:
            los_error.append(los_C[1:3] / los_C[0])
        else:
            los_error.append([0, 0])

    closing_speed = np.array(closing_speed)
    los_error = np.array(los_error)

    if len(closing_speed) > 0:
        t_cs = time_u[:len(closing_speed)]
        axs[1].plot(t_cs, closing_speed, label='Closing Speed')
        axs[1].axhline(y=V_DESIRED, c='r', ls='--', label=f'Desired ({V_DESIRED} m/s)')
    axs[1].set_ylabel('Speed [m/s]'); axs[1].set_title('Closing Speed')

    if len(los_error) > 0:
        t_le = time_u[:len(los_error)]
        axs[2].plot(t_le, los_error[:, 0], label='LOS y-error')
        axs[2].plot(t_le, los_error[:, 1], label='LOS z-error')
    axs[2].set_ylabel('Error'); axs[2].set_title('LOS Error (Camera Frame)')

    # EKF 宽度估计
    if len(hist['target_est_ekf']) > 0 and hist['target_est_ekf'].shape[1] >= 7:
        n_ekf = len(hist['target_est_ekf'])
        t_ekf = time_u[-n_ekf:] if n_ekf <= len(time_u) else time_u
        axs[3].plot(t_ekf, hist['target_est_ekf'][:, 6], 'b-', label='Estimated W (EKF)')
        axs[3].axhline(y=TARGET_PHYSICAL_SIZE_TRUTH, color='r', linestyle='--',
                        label=f'True W ({TARGET_PHYSICAL_SIZE_TRUTH} m)')
        axs[3].axhline(y=TARGET_WIDTH_INITIAL_EST, color='g', linestyle=':',
                        label=f'Initial Guess ({TARGET_WIDTH_INITIAL_EST} m)')
    axs[3].set_ylabel('Width [m]'); axs[3].set_title('Target Width Estimation (Bearing-Angle EKF)')
    axs[3].set_xlabel('Time [s]')

    for ax in axs:
        ax.grid(True); ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

    # --- 状态与控制 ---
    fig, axs = plt.subplots(6, 1, figsize=(15, 20), sharex=True)
    fig.suptitle('States and Control Inputs (DQ-NMPC)', fontsize=16)
    axs[0].plot(time_x, hist['x'][:, 0], label='px'); axs[0].plot(time_x, hist['x'][:, 1], label='py'); axs[0].plot(time_x, hist['x'][:, 2], label='pz'); axs[0].set_ylabel('Position [m]')
    axs[1].plot(time_x, hist['x'][:, 3], label='vx'); axs[1].plot(time_x, hist['x'][:, 4], label='vy'); axs[1].plot(time_x, hist['x'][:, 5], label='vz'); axs[1].set_ylabel('Velocity [m/s]')
    axs[2].plot(time_x, np.rad2deg(hist['x'][:, 6]), label='Roll'); axs[2].plot(time_x, np.rad2deg(hist['x'][:, 7]), label='Pitch'); axs[2].plot(time_x, np.rad2deg(hist['x'][:, 8]), label='Yaw'); axs[2].set_ylabel('Angle [deg]')
    axs[3].plot(time_x, np.rad2deg(hist['x'][:, 12]), label='Gimbal Roll'); axs[3].plot(time_x, np.rad2deg(hist['x'][:, 13]), label='Gimbal Pitch'); axs[3].plot(time_x, np.rad2deg(hist['x'][:, 14]), label='Gimbal Yaw'); axs[3].set_ylabel('Gimbal [deg]')
    if num_u > 0:
        axs[4].plot(time_u, hist['u'][:, 0], label='wx (body-x rate)')
        axs[4].plot(time_u, hist['u'][:, 1], label='wy (body-y rate)')
        axs[4].plot(time_u, hist['u'][:, 2], label='wz (body-z rate)')
        axs[4].set_ylabel('Body Ang. Vel [rad/s]')
        axs[5].plot(time_u, hist['u'][:, 3] * MASS, label='Thrust [N]'); axs[5].set_ylabel('Thrust [N]'); axs[5].set_xlabel('Time [s]')

    for ax in axs:
        ax.grid(True); ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()


if __name__ == "__main__":
    main()
