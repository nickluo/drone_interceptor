import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from exp_filter import EMAFilter
from camera_model import CameraModel, PerceptionSim
from target_ekf import TargetEKF
from drone_model_quaternion import DroneModelQuaternion

# ------------------------------------------------------------------------------
# 仿真参数与常量配置 (Simulation Parameters and Constants)
# ------------------------------------------------------------------------------
# ======== 物理常量 (Physics Constants) ========
MASS = 4.0          # 无人机质量 (kg)
G = 9.8065            # 重力加速度 (m/s^2)
A_MAX = 20.0        # 最大加速度 (m/s^2)
A_MIN = 1.0         # 最小加速度 (m/s^2)
DRAG_COEFF = 0.1625    # 线性空气阻力系数
ROLL_PITCH_RATE_MAX = 2.0 * np.pi # 最大欧拉角速率 (rad/s)
YAW_RATE_MAX = 1.0 * np.pi # 最大欧拉角速率 (rad/s)
ATT_LIMIT = np.pi/5 # 姿态角限制 (Roll & Pitch)

# ======== MPC 控制器参数 (MPC Controller Parameters) ========
CTRL_FREQ_HZ = 50    # 控制器频率 (Hz)
TF = 0.8            # 预测时域总时长 (s)
N_HORIZON = int(TF * CTRL_FREQ_HZ) # 预测时域步数
V_DESIRED = 20.0     # 期望接近速度 (m/s)

# ======== 仿真设置 (Simulation Settings) ========
SIM_TIME = 15.0      # 总仿真时长 (s)
INTERCEPTION_DISTANCE = 0.5 # 成功拦截的距离阈值 (m)
TRACKING_DISTANCE = 0.0 # 期望跟踪距离 (m)
# TRACKING_DISTANCE_ADJ = TRACKING_DISTANCE + 2.0 * np.sqrt(TRACKING_DISTANCE)
TRACKING_GAIN = 1.0 * np.sqrt(TRACKING_DISTANCE / V_DESIRED) # 距离保持控制增益

# ======== 感知模型参数 (Perception Model Parameters) ========
PERCEPTION_DELAY = 0.08  # 感知延迟 (s)
BBOX_POS_STD_DEV = 3.0  # 检测框中心位置的测量噪声标准差 (pixels)
BBOX_SIZE_STD_DEV = 3.0 # 检测框尺寸的测量噪声标准差 (pixels)

# ======== 控制执行延时参数 (Control Execution Delay Parameters) ========
CONTROL_EXECUTION_DELAY = 0.05  # 角速度控制执行延时 (s)
RATE_TC = 0.02  # 角速度一阶响应时间常数 (s)

# ======== 初始条件 (Initial Conditions) ========
DRONE_INITIAL_STATE = np.array([0.0, 0.0, 10.0,     # Position (p_x, p_y, p_z)
                                0.0, 0.0, 0.0,      # Velocity (v_x, v_y, v_z)
                                0.0, 0.0, 0.0,     # Attitude (phi, theta, psi) - 将被转换为四元数
                                0.0, 0.0, 0.0,     # Omega (wx, wy, wz)
                                0.0, 0.0, 0.0])    # Gimbal (r, p, y)
TARGET_INITIAL_POS = np.array([100.0, 20.0, 15.0])
TARGET_VELOCITY = np.array([-10.0, 5.0, 0.5]) # 目标做匀速直线运动
# TARGET_INITIAL_POS = np.array([100.0, 20.0, 15.0])
# TARGET_VELOCITY = np.array([0.0, 0.0, 0.0]) # 目标静止

# ======== 目标与相机物理属性 (Target & Camera Properties) ========
# --- 目标 ---
TARGET_PHYSICAL_SIZE_TRUTH = 0.5  # 目标的真实物理尺寸 (m), 用于相机模型
TARGET_PHYSICAL_SIZE_EST = 0.9    # 状态估计器使用的目标尺寸 (m), 可模拟估计误差

# --- 相机 ---
CAM_FOV_DEG = 30.0  # 相机水平视场角 (degrees)
CAM_FOCAL_LENGTH_PHYSICAL = 16.0 # 物理焦距 (mm)
CAM_PIXEL_SIZE = 3.45            # 像素尺寸 (um)
# f(pixel) = f(mm) * 1000 / pixel_size(um)
CAM_FOCAL_LENGTH = CAM_FOCAL_LENGTH_PHYSICAL * 1000 / CAM_PIXEL_SIZE # 焦距 (in pixels)

# ======== 代价函数 (Cost Function) ========
# --- 权重 ---
W_LOS = 100.0             # 视线误差权重
W_ALIGN = 0.1             # 速度对齐权重
W_CLOSING_SPEED = 0.1     # 接近速度权重
W_RATES = 0.00001        # 角速度控制量权重
W_TILT = 0.000001         # 倾斜角误差权重
W_U_ALPHA = 0.001        # 角速度变化率权重
W_U_THRUST = 0.0001       # 油门控制量权重

# ======== 3D PN 导引律参数 (3D PN Guidance Parameters) ========
PN_ENABLED = True
PN_NAV_CONSTANT = 4.0       # 导航比 (N)
PN_SWITCH_DIST = 15.0       # 切换距离阈值 (m)
PN_FOV_SWITCH_DEG = 15.0    # 切换FOV阈值 (deg, 偏离光轴的角度)
PN_MIN_TIME_GO = 2.0        # 剩余时间阈值 (s)

# ======== 云台参数 (Gimbal Parameters) ========
GIMBAL_RP_COEFF = 12.0
GIMBAL_Y_COEFF = 10.0
GIMBAL_DISTURBANCE_COEFF = 0.5 # 模拟无人机旋转对云台的影响

def convert_euler_to_quaternion_state(euler_state):
    """将欧拉角状态转换为四元数状态"""
    # 输入: [px, py, pz, vx, vy, vz, phi, theta, psi, wx, wy, wz, gr, gp, gy] (15维)
    # 输出: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, gr, gp, gy] (16维)
    pos_vel = euler_state[0:6]
    euler_angles = euler_state[6:9]
    omega = euler_state[9:12]
    gimbal = euler_state[12:15]
    quat = DroneModelQuaternion.euler_to_quaternion(euler_angles[0], euler_angles[1], euler_angles[2])
    return np.hstack([pos_vel, quat, omega, gimbal])

def convert_quaternion_to_euler_state(quat_state):
    """将四元数状态转换为欧拉角状态（用于显示和记录）"""
    # 输入: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, gr, gp, gy] (16维)
    # 输出: [px, py, pz, vx, vy, vz, phi, theta, psi, wx, wy, wz, gr, gp, gy] (15维)
    pos_vel = quat_state[0:6]
    quat = quat_state[6:10]
    omega = quat_state[10:13]
    gimbal = quat_state[13:16]
    euler_angles = DroneModelQuaternion.quaternion_to_euler(quat)
    return np.hstack([pos_vel, euler_angles, omega, gimbal])

def calculate_pn_controls(drone_state, target_pos, target_vel, dt):
    """
    计算基于三维比例导引(3D Proportional Navigation)的控制指令
    
    Args:
        drone_state: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, gr, gp, gy]
        target_pos: [tx, ty, tz]
        target_vel: [tvx, tvy, tvz]
        dt: 时间步长
    
    Returns:
        u_pn: [wx, wy, wz, thrust_accel]
    """
    # 提取状态
    pos = drone_state[0:3]
    vel = drone_state[3:6]
    quat = drone_state[6:10] # qw, qx, qy, qz
    
    # 1. 计算相对运动状态
    r_vec = target_pos - pos
    v_vec = target_vel - vel
    dist = np.linalg.norm(r_vec) + 1e-6
    r_unit = r_vec / dist
    
    # 2. 计算视线转动速率向量 (LOS Rate Vector)
    # Omega = (r x v) / (r . r)
    omega_los = np.cross(r_vec, v_vec) / (dist * dist)
    
    # 3. 计算接近速度 (Closing Speed)
    # Vc = - (r . v) / |r| (标量)
    vc = -np.dot(r_vec, v_vec) / dist
    
    # 4. 计算PN加速度指令 (TPN Acceleration Command)
    # TPN Vector Form: a_cmd = N * Vc * (Omega x r_unit)
    # 方向: 垂直于视线和旋转速率矢量，即在视线转动的平面内，垂直于视线指向修正方向
    # 另一种常用形式: a_cmd = N * (v_rel x Omega) ? 
    # 标准形式推导: a_perp = N * Vc * Omega_scalar. 
    # 矢量方向: Omega x r_unit 是垂直于LOS且在转动平面内的矢量。
    a_pn = PN_NAV_CONSTANT * vc * np.cross(omega_los, r_unit)
    
    # 5. 重力补偿
    # 总期望加速度 = PN指令 + 重力抵消
    a_des = a_pn + np.array([0.0, 0.0, G])
    
    # 6. 计算期望推力方向 (Body Z axis)
    thrust_accel = np.linalg.norm(a_des)
    
    if thrust_accel < 1e-6:
        z_b_des = np.array([0, 0, 1])
    else:
        z_b_des = a_des / thrust_accel
        
    # 限制推力 (可选)
    thrust_accel = np.clip(thrust_accel, A_MIN, A_MAX)
    
    # 7. 计算期望姿态 (Rotation Matrix)
    # 手动计算当前旋转矩阵 R_curr
    qw, qx, qy, qz = quat
    R_curr = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
    ])

    # 我们不仅需要对齐Z轴(推力方向)，还需要确定X轴(偏航方向)
    # 为了减少LOS Azimuth Error (水平视线误差)，我们应主动将机头指向目标
    
    # 计算LOS水平投影方向
    r_vec_horiz_norm = np.linalg.norm(r_vec[0:2])
    if r_vec_horiz_norm > 0.1:
        # 目标在水平面上投影长度足够，使用视线方向作为期望Heading
        los_yaw = np.arctan2(r_vec[1], r_vec[0])
        x_c = np.array([np.cos(los_yaw), np.sin(los_yaw), 0.0])
    else:
        # 如果距离太近或垂直头顶，维持当前偏航
        qw, qx, qy, qz = quat
        curr_yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        x_c = np.array([np.cos(curr_yaw), np.sin(curr_yaw), 0.0])
    
    y_b_des = np.cross(z_b_des, x_c)
    if np.linalg.norm(y_b_des) < 1e-4: # 奇点: z_b_des 与 x_c 平行 (垂直向下或向上)
        y_b_des = np.cross(z_b_des, np.array([1, 0, 0])) # 换个参考
    y_b_des /= np.linalg.norm(y_b_des)
    
    x_b_des = np.cross(y_b_des, z_b_des)
    
    R_des = np.column_stack((x_b_des, y_b_des, z_b_des))
    
    # 8. 姿态控制 (Attitude Control) - 计算角速度指令
    # R_err = R_curr^T * R_des
    R_err = R_curr.T @ R_des
    
    # 将旋转矩阵误差转换为轴角 (Axis-Angle)
    # trace(R) = 1 + 2*cos(theta)
    tr = np.trace(R_err)
    theta = np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0))
    
    if abs(theta) < 1e-6:
        omega_cmd = np.zeros(3)
    else:
        # axis = 1/(2*sin(theta)) * [R(3,2)-R(2,3); R(1,3)-R(3,1); R(2,1)-R(1,2)]
        axis = (1 / (2 * np.sin(theta))) * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])
        # P控制器 + 前馈: omega = Kp * theta * axis + R^T * omega_los
        # 增加前馈项以补偿视线旋转，提高跟踪精度
        omega_ff = R_curr.T @ omega_los
        omega_cmd = 25.0 * theta * axis + omega_ff # Gain Kp=25.0
        
    # 限制角速度
    omega_cmd = np.clip(omega_cmd, 
                        [-ROLL_PITCH_RATE_MAX, -ROLL_PITCH_RATE_MAX, -YAW_RATE_MAX], 
                        [ROLL_PITCH_RATE_MAX, ROLL_PITCH_RATE_MAX, YAW_RATE_MAX])
                        
    return np.hstack([omega_cmd, thrust_accel])

MODEL_CONSTANTS = {
    'G' : G,
    'ROLL_PITCH_RATE_MAX' : ROLL_PITCH_RATE_MAX,
    'YAW_RATE_MAX' : YAW_RATE_MAX,
    'ATT_LIMIT' : ATT_LIMIT,
    'A_MIN' : A_MIN,
    'A_MAX' : A_MAX,
    'N_HORIZON' : N_HORIZON,
    'TF' : TF,
    'DRONE_INITIAL_STATE' : DRONE_INITIAL_STATE
}

W_COST = np.array([np.sqrt(W_LOS),
                    np.sqrt(W_ALIGN),
                    np.sqrt(W_CLOSING_SPEED),
                    np.sqrt(W_RATES),
                    np.sqrt(W_TILT),
                    np.sqrt(W_U_ALPHA),
                    np.sqrt(W_U_THRUST)])

# ------------------------------------------------------------------------------
# 主仿真循环 (Main Simulation Loop)
# ------------------------------------------------------------------------------
def main():

    drone_model = DroneModelQuaternion(MODEL_CONSTANTS)
    acados_solver = drone_model.generate_solver()

    # --- 初始化仿真环境 ---
    dt = TF / N_HORIZON
    N_sim = int(SIM_TIME / dt)
    
    # 将欧拉角初始状态转换为四元数状态
    x_current_quat = convert_euler_to_quaternion_state(DRONE_INITIAL_STATE)
    target_pos_true = TARGET_INITIAL_POS.copy()
    target_vel_true = TARGET_VELOCITY.copy()
    
    camera = CameraModel(CAM_FOCAL_LENGTH, TARGET_PHYSICAL_SIZE_TRUTH, CAM_FOV_DEG, BBOX_POS_STD_DEV, BBOX_SIZE_STD_DEV)
    estimator = TargetEKF(dt, CAM_FOCAL_LENGTH, TARGET_PHYSICAL_SIZE_EST, BBOX_POS_STD_DEV, BBOX_SIZE_STD_DEV)
    output_filter = EMAFilter(int(N_HORIZON/TF*0.1), channels=4)

    perceptor = PerceptionSim(camera, PERCEPTION_DELAY, dt)

    # --- 初始化延迟与盲飞逻辑 ---
    target_lost_counter = 0
    
    # --- 初始化角速度延时补偿变量 ---
    # previous_body_rates = np.array([0.0, 0.0, 0.0])  # 存储前一次机体角速度
    
    # 为了与原版本兼容，我们将四元数状态转换为欧拉角状态用于记录
    x_current_euler = convert_quaternion_to_euler_state(x_current_quat)
    history = {'x': [x_current_euler], 
               'u': [], 
               'target_true': [target_pos_true.copy()], 
               'target_vel_true': [target_vel_true.copy()],
               'target_est_ekf': [], 
               'target_est_raw': [], 
               'distance': []}
    
    # --- 性能改进: 初始化热启动变量 ---
    # 使用合理的初始猜测值（悬停状态）初始化
    u_warm_start = np.tile(np.array([0, 0, 0, G]), (N_HORIZON, 1))
    x_warm_start = np.tile(x_current_quat, (N_HORIZON + 1, 1))

    # --- 性能改进: 设置热启动初始猜测值 ---
    for k in range(N_HORIZON):
        acados_solver.set(k, 'x', x_warm_start[k,:])
        acados_solver.set(k, 'u', u_warm_start[k,:])
    acados_solver.set(N_HORIZON, 'x', x_warm_start[N_HORIZON,:])

    active_controller = "MPC"
    print("\n[INFO] 开始仿真 (使用四元数+EKF, 考虑感知延迟与噪声)...")
    i = 0
    solve_time_avg = 0.0
    for i in range(N_sim):
        # ------------------- 1. 感知 (Perceive) -------------------
        # perceptor需要欧拉角状态，所以转换一下
        x_current_euler_for_perception = convert_quaternion_to_euler_state(x_current_quat)
        full, target_pos_true = perceptor.update(x_current_euler_for_perception, target_pos_true, target_vel_true, history)
        if not full : continue
        drone_state_at_meas, meas_bbox, meas_visible = perceptor.get_measure()

        # ------------------- 2. EKF 预测 (Predict) -------------------
        if estimator.is_initialized:
            estimator.predict(dt)

        # ------------------- 3. EKF 更新 (Update) -------------------
        if meas_visible:
            target_lost_counter = 0 # 目标可见，重置计数器
            drone_pos, psi = drone_state_at_meas[0:3], drone_state_at_meas[8]
            gimbal_r, gimbal_p, gimbal_y = drone_state_at_meas[12], drone_state_at_meas[13], drone_state_at_meas[14]
            
            raw_pos_est = estimator.noisy_bbox_to_pos(meas_bbox, drone_pos, psi, gimbal_r, gimbal_p, gimbal_y)
            if not estimator.is_initialized:
                estimator.initialize(raw_pos_est)
            else:
                estimator.update(meas_bbox, drone_pos, psi, gimbal_r, gimbal_p, gimbal_y)
        else:
            target_lost_counter += 1 # 目标丢失，增加计数器
            raw_pos_est = None
            if estimator.is_initialized:
                print(f"[WARNING] 步骤 {i}: 目标丢失! 盲飞计数: {target_lost_counter}/{N_HORIZON}")

        # --- 检查任务失败条件 ---
        if target_lost_counter > N_HORIZON:
            print(f"\n[FAILED] 目标丢失时间超过 {TF} 秒! 任务失败。")
            break

        # ------------------- 4. 规划 (Plan) -------------------
        if not estimator.is_initialized:
            print(f"[WARNING] 步骤 {i}: EKF尚未初始化, 跳过规划。")
            # 悬停指令：角速度为0，推力加速度等于重力加速度
            u_optimal_raw = np.array([0, 0, 0, G])
            # 保持悬停状态 (位置和速度不变，四元数归一化)
            x_current_quat[3:6] = 0.0  # 速度清零
            x_current_quat[6:10] = DroneModelQuaternion.normalize_quaternion(x_current_quat[6:10])
            x_current_quat[13:16] = 0.0 # 云台归零
        else:
            pos_est = estimator.x[0:3]
            vel_est = estimator.x[3:6]
            
            # --- 检测是否满足切换到 PN 的条件 ---
            use_pn = False
            if PN_ENABLED:
                # 状态机逻辑: 如果已经处于PN模式，则保持PN (Latched)
                if active_controller == "PN":
                    use_pn = True
                else: 
                    # 计算距离
                    dist_to_target = np.linalg.norm(pos_est - x_current_quat[0:3])
                    
                    # 计算目标是否在视场中心附近 (FOV Check)
                    # 修正: 使用当前的系统状态估计(包含云台姿态)来判断，而不是延迟的测量值
                    # 这与C++代码中的逻辑保持一致 (transform to camera frame)
                    target_in_fov_center = False
                    
                    # 1. 计算相对位置矢量 (World Frame)
                    r_W = pos_est - x_current_quat[0:3]
                    dist_est = np.linalg.norm(r_W)
                    
                    if dist_est > 0.1:
                        # 2. 转换到机体坐标系 (Body Frame)
                        qw, qx, qy, qz = x_current_quat[6:10]
                        # R_WB: Body to World
                        R_WB = np.array([
                            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                            [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
                            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
                        ])
                        r_B = R_WB.T @ r_W
                        
                        # 3. 转换到相机/云台坐标系 (Camera Frame)
                        # 假设云台按照 standard aerospace sequence (Yaw->Pitch->Roll) 相对于机体旋转
                        # 或者简单的 3-2-1 欧拉角. 
                        gr, gp, gy = x_current_quat[13:16]
                        cgr, sgr = np.cos(gr), np.sin(gr)
                        cgp, sgp = np.cos(gp), np.sin(gp)
                        cgy, sgy = np.cos(gy), np.sin(gy)
                        
                        # R_BG: Gimbal to Body (Rotation Matrix from Gimbal Frame to Body Frame)
                        # R_z(yaw) * R_y(pitch) * R_x(roll)
                        R_z = np.array([[cgy, -sgy, 0], [sgy, cgy, 0], [0, 0, 1]])
                        R_y = np.array([[cgp, 0, sgp], [0, 1, 0], [-sgp, 0, cgp]])
                        R_x = np.array([[1, 0, 0], [0, cgr, -sgr], [0, sgr, cgr]])
                        R_BG = R_z @ R_y @ R_x
                        
                        r_C = R_BG.T @ r_B
                        
                        # 4. 检查是否在 FOV 中心区域
                        # 假设相机光轴为 Camera Frame 的 X 轴 (Alignment with Body X at zero angles)
                        # 计算视线向量与 X 轴的夹角
                        cos_theta = r_C[0] / np.linalg.norm(r_C)
                        theta_deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                        
                        if theta_deg < PN_FOV_SWITCH_DEG:
                            target_in_fov_center = True
                            
                    # 组合条件: 距离足够近 且 目标在中心区域
                    if (dist_to_target < PN_SWITCH_DIST) and target_in_fov_center:
                        use_pn = True
            
            if use_pn:
                if active_controller == "MPC":
                    print(f"\n[SWITCH] Dist={dist_to_target:.1f}m, Switching to PROPORTIONAL NAVIGATION")
                    active_controller = "PN"
                    
                u_optimal_raw = calculate_pn_controls(x_current_quat, pos_est, vel_est, dt)
                
                # 更新 x_current_quat 用于下一步预测 (简单的 Euler 积分模拟)
                # 注意: 这只是为了闭环仿真的连续性, 实际中 x_current_quat 会在下一帧通过感知更新
                # 这里我们模拟动力学步进
                # 提取
                d_pos = x_current_quat[0:3]
                d_vel = x_current_quat[3:6]
                d_quat = x_current_quat[6:10]
                d_omega = u_optimal_raw[0:3] # 假设角速度完美跟踪
                
                # 位置更新
                d_pos_new = d_pos + d_vel * dt
                
                # 速度更新 (a = R * [0,0,T] - drag - g)
                qw, qx, qy, qz = d_quat
                R_mat = np.array([
                    [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                    [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
                    [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
                ])
                thrust_vec = R_mat @ np.array([0, 0, u_optimal_raw[3]])
                drag_vec = DRAG_COEFF * d_vel
                acc_world = thrust_vec - drag_vec - np.array([0, 0, G])
                d_vel_new = d_vel + acc_world * dt
                
                # 四元数更新 dq = 0.5 * q * omega
                omega_q = np.array([0, d_omega[0], d_omega[1], d_omega[2]])
                # q_dot = 0.5 * q_mult(d_quat, omega_q) 
                # 这里简化: Convert to R, R_new = R * Exp(w*dt), convert back
                # 或者简单的 Euler integration + normalize
                wx, wy, wz = d_omega
                omega_mat = np.array([
                    [0, -wx, -wy, -wz],
                    [wx, 0, wz, -wy],
                    [wy, -wz, 0, wx],
                    [wz, wy, -wx, 0]
                ])
                d_quat_dot = 0.5 * omega_mat @ d_quat
                d_quat_new = d_quat + d_quat_dot * dt
                d_quat_new = DroneModelQuaternion.normalize_quaternion(d_quat_new)
                
                # Gimbal update (Dummy)
                dx_gimbal_new = x_current_quat[13:16] # 保持不变
                
                x_current_quat = np.hstack([d_pos_new, d_vel_new, d_quat_new, d_omega, dx_gimbal_new])
                
            else:
                # MPC Logic
                acados_solver.set(0, "lbx", x_current_quat)
                acados_solver.set(0, "ubx", x_current_quat)
            
                for k in range(N_HORIZON + 1):
                    t_k = k * dt
                    predicted_target_pos_est = pos_est + t_k * vel_est
                    if (TRACKING_DISTANCE > 0.001):
                        # --- 计算期望接近速度 (Tracking Logic) ---
                        x_k = acados_solver.get(k, "x")
                        current_drone_pos = x_k[0:3]
                        dist_to_target = np.linalg.norm(predicted_target_pos_est - current_drone_pos)
                        dist_error = (dist_to_target - TRACKING_DISTANCE) / TRACKING_DISTANCE
                        tracking_coeff = 2.0/(1.0 + np.exp(-TRACKING_GAIN * dist_error)) - 1.0
                        v_exp = tracking_coeff * V_DESIRED
                    else:
                        v_exp = V_DESIRED
                    acados_solver.set(k, "p", np.hstack([predicted_target_pos_est, vel_est, DRAG_COEFF, v_exp, RATE_TC, W_COST, GIMBAL_RP_COEFF, GIMBAL_Y_COEFF]))

                t0 = time.time()
                status = acados_solver.solve()
                if status != 0: 
                    print(f'[WARNING] ACADOS求解器在第 {i} 步返回状态 {status}。')
                    break
                print(f'[INFO] 步骤 {i} acados_solver.solve() {int((time.time()-t0)*1000000)} us')
                solve_time_avg = solve_time_avg * ((i-1.0)/i) + (time.time()-t0)*1000000/i
                
                u_optimal_raw = acados_solver.get(0, "u")
                x_current_quat = acados_solver.get(1, "x")

                # 确保四元数归一化
                x_current_quat[6:10] = DroneModelQuaternion.normalize_quaternion(x_current_quat[6:10])

                # --- 模拟云台受无人机机动影响 ---
                # 简单的干扰模型：云台角速度受到机体角速度的干扰
                wx, wy, wz = x_current_quat[10], x_current_quat[11], x_current_quat[12]
                x_current_quat[13] += GIMBAL_DISTURBANCE_COEFF * wx * dt
                x_current_quat[14] += GIMBAL_DISTURBANCE_COEFF * wy * dt
                x_current_quat[15] += GIMBAL_DISTURBANCE_COEFF * wz * dt

                # --- 性能改进: 为下一次迭代准备热启动数据 ---
                for k in range(N_HORIZON-1):
                    x_next = acados_solver.get(k+1, "x")
                    x_next[6:10] = DroneModelQuaternion.normalize_quaternion(x_next[6:10])
                    u_next = acados_solver.get(k+1, "u")
                    acados_solver.set(k, "x", x_next)
                    acados_solver.set(k, "u", u_next)
                acados_solver.set(N_HORIZON-1, "u", u_next)  # 末尾复制
                x_next = acados_solver.get(N_HORIZON, "x")
                x_next[6:10] = DroneModelQuaternion.normalize_quaternion(x_next[6:10])
                acados_solver.set(N_HORIZON-1, "x", x_next)
            
            # Common control output processing
            u_optimal_filtered = u_optimal_raw
            body_rates = u_optimal_filtered[0:3]
            thrust_accel = u_optimal_filtered[3]
        
        # 应用姿态安全限制防止危险机动
        # body_rates = apply_attitude_safety_limit_quat(x_current_quat, body_rates, ATT_LIMIT)
        
        # 构建实际控制输入（机身角速度 + 推力加速度）
        u_optimal = np.hstack([body_rates, thrust_accel])
        
        # 保存当前机体角速度供下次延时补偿使用
        # previous_body_rates = body_rates.copy()
        
        # 将推力加速度转换为力（牛顿）用于记录
        thrust_force = thrust_accel * MASS
        u_to_log = np.hstack([body_rates, thrust_force])

        # ------------------- 6. 更新环境 & 记录数据 -------------------
        target_pos_true += target_vel_true * dt
        
        # 转换为欧拉角状态用于记录和显示
        x_current_euler = convert_quaternion_to_euler_state(x_current_quat)
        
        history['x'].append(x_current_euler.copy())
        history['u'].append(u_to_log.copy())
        history['target_true'].append(target_pos_true.copy())
        history['target_vel_true'].append(target_vel_true.copy())
        if estimator.is_initialized:
            history['target_est_ekf'].append(estimator.x.copy())
        if raw_pos_est is not None:
            history['target_est_raw'].append(raw_pos_est)
        
        distance = np.linalg.norm(target_pos_true - x_current_quat[0:3])
        history['distance'].append(distance)
        
        if distance < INTERCEPTION_DISTANCE:
            print(f"\n[SUCCESS] 在第 {i} 步成功拦截目标！距离: {distance:.2f} m")
            break
    
    if i >= N_sim - 1: print(f"\n[INFO] 仿真结束。最终距离: {history['distance'][-1]:.2f} m")

    print(f'Solve Average Time {int(solve_time_avg)} us')

    plot_results(history, dt)

def plot_results(hist, dt):
    """绘制仿真结果。"""
    if not hist['u']:
        print("[ERROR] 仿真未能生成有效数据，无法绘图。")
        return

    # 将列表转换为numpy数组以便于索引
    hist['x'] = np.array(hist['x'])
    hist['u'] = np.array(hist['u'])
    hist['target_true'] = np.array(hist['target_true'])
    hist['target_vel_true'] = np.array(hist['target_vel_true'])
    hist['target_est_ekf'] = np.array(hist['target_est_ekf'])
    hist['target_est_raw'] = np.array(hist['target_est_raw'])
    hist['distance'] = np.array(hist['distance'])
    
    time_axis_x = np.linspace(0, dt * (len(hist['x']) - 1), len(hist['x']))
    time_axis_u = np.linspace(0, dt * (len(hist['u']) - 1), len(hist['u']))

    # 3D轨迹图
    fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection='3d')
    ax.plot(hist['x'][:, 0], hist['x'][:, 1], hist['x'][:, 2], label='Drone Trajectory', c='b', lw=2)
    ax.plot(hist['target_true'][:, 0], hist['target_true'][:, 1], hist['target_true'][:, 2], label='Target True Traj.', c='r', ls='--')
    if len(hist['target_est_ekf']) > 0:
        ax.plot(hist['target_est_ekf'][:, 0], hist['target_est_ekf'][:, 1], hist['target_est_ekf'][:, 2], label='Target EKF Estimated Traj.', c='g', ls='-')
    if len(hist['target_est_raw']) > 0:
        ax.scatter(hist['target_est_raw'][:, 0], hist['target_est_raw'][:, 1], hist['target_est_raw'][:, 2], label='Target Raw Measurement', c='orange', marker='.', alpha=0.5)
    ax.scatter(hist['x'][0, 0], hist['x'][0, 1], hist['x'][0, 2], c='b', s=100, marker='o', label='Drone Start')
    ax.scatter(hist['target_true'][0, 0], hist['target_true'][0, 1], hist['target_true'][0, 2], c='r', s=100, marker='x', label='Target Start')
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]'); ax.set_zlabel('Up [m]')
    ax.legend(); ax.set_title('3D Trajectory (Quaternion Version)'); ax.grid(True); ax.axis('auto'); plt.show()
    
    # 性能指标图
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('Interceptor Performance Metrics (Quaternion Version)', fontsize=16)
    if len(hist['distance']) > 0:
        axs[0].plot(time_axis_u, hist['distance'], label='Distance to Target')
        axs[0].axhline(y=TRACKING_DISTANCE, color='g', linestyle='--', label=f'Tracking Dist ({TRACKING_DISTANCE} m)')
    axs[0].set_ylabel('Distance [m]'); axs[0].set_title('Distance Over Time')
    
    # 计算接近速度和LOS误差用于绘图
    closing_speed, los_error = [], []
    if len(hist['u']) > 0:
        # Align history data to the length of control inputs
        num_u_steps = len(hist['u'])
        aligned_x_hist = hist['x'][-num_u_steps-1:-1]
        aligned_target_hist = hist['target_true'][-num_u_steps-1:-1]
        aligned_target_vel_hist = hist['target_vel_true'][-num_u_steps-1:-1]

        for i in range(num_u_steps):
            drone_x_state = aligned_x_hist[i]
            drone_v, psi = drone_x_state[3:6], drone_x_state[8]
            target_pos = aligned_target_hist[i]
            
            los_W = target_pos - drone_x_state[0:3]
            dist = np.linalg.norm(los_W) + 1e-9
            los_norm = los_W / dist
            closing_speed.append(np.dot(drone_v-aligned_target_vel_hist[i], los_norm))
            
            cpsi, spsi = np.cos(psi), np.sin(psi)
            R_W_C = np.array([[cpsi, spsi, 0], [-spsi, cpsi, 0], [0, 0, 1]])
            los_C = R_W_C @ los_W
            # los_error.append(los_C[1:3]/los_C[0])
            los_error.append(los_C[1:3]/los_C[0] if dist > INTERCEPTION_DISTANCE*2 else [0, 0]) # 归一化误差 (y/x, z/x)

    closing_speed, los_error = np.array(closing_speed), np.array(los_error)

    if len(closing_speed) > 0:
        axs[1].plot(time_axis_u, closing_speed, label='Actual Closing Speed (on LOS)')
        axs[1].axhline(y=V_DESIRED, color='r', linestyle='--', label=f'Desired Closing Speed ({V_DESIRED} m/s)')
    axs[1].set_ylabel('Speed [m/s]'); axs[1].set_title('Closing Speed')

    if len(los_error) > 0:
        axs[2].plot(time_axis_u, los_error[:, 0], label='LOS y-error (left/right)')
        axs[2].plot(time_axis_u, los_error[:, 1], label='LOS z-error (up/down)')
    axs[2].set_ylabel('Error'); axs[2].set_title('Line-of-Sight Error in Camera Frame')
    axs[2].set_xlabel('Time [s]')
    
    for ax in axs: ax.grid(True); ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

    # 状态和控制量图
    fig, axs = plt.subplots(6, 1, figsize=(15, 20), sharex=True)
    fig.suptitle('States and Control Inputs (Quaternion Version)', fontsize=16)
    axs[0].plot(time_axis_x, hist['x'][:, 0], label='px'); axs[0].plot(time_axis_x, hist['x'][:, 1], label='py'); axs[0].plot(time_axis_x, hist['x'][:, 2], label='pz'); axs[0].set_ylabel('Position [m]')
    axs[1].plot(time_axis_x, hist['x'][:, 3], label='vx'); axs[1].plot(time_axis_x, hist['x'][:, 4], label='vy'); axs[1].plot(time_axis_x, hist['x'][:, 5], label='vz'); axs[1].set_ylabel('Velocity [m/s]')
    axs[2].plot(time_axis_x, np.rad2deg(hist['x'][:, 6]), label='Roll'); axs[2].plot(time_axis_x, np.rad2deg(hist['x'][:, 7]), label='Pitch'); axs[2].plot(time_axis_x, np.rad2deg(hist['x'][:, 8]), label='Yaw'); axs[2].set_ylabel('Angle [deg]')
    axs[3].plot(time_axis_x, np.rad2deg(hist['x'][:, 12]), label='Gimbal Roll'); axs[3].plot(time_axis_x, np.rad2deg(hist['x'][:, 13]), label='Gimbal Pitch'); axs[3].plot(time_axis_x, np.rad2deg(hist['x'][:, 14]), label='Gimbal Yaw'); axs[3].set_ylabel('Gimbal [deg]')
    if len(hist['u']) > 0:
        axs[4].plot(time_axis_u, hist['u'][:, 0], label='wx (body-x rate)')
        axs[4].plot(time_axis_u, hist['u'][:, 1], label='wy (body-y rate)')
        axs[4].plot(time_axis_u, hist['u'][:, 2], label='wz (body-z rate)')
        axs[4].set_ylabel('Body Ang. Vel [rad/s]')
        axs[5].plot(time_axis_u, hist['u'][:, 3], label='Thrust [N]'); axs[5].set_ylabel('Thrust [N]'); axs[5].set_xlabel('Time [s]')
    for ax in axs: ax.grid(True); ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

if __name__ == "__main__":        
    main()
