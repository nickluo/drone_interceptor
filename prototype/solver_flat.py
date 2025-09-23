import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from exp_filter import EMAFilter
from camera_model import CameraModel, PerceptionSim
from target_ekf import TargetEKF
from drone_model import DroneModel

# ------------------------------------------------------------------------------
# 仿真参数与常量配置 (Simulation Parameters and Constants)
# ------------------------------------------------------------------------------
# ======== 物理常量 (Physics Constants) ========
MASS = 2.0          # 无人机质量 (kg)
G = 9.8065            # 重力加速度 (m/s^2)
T_MAX = 80.0        # 最大推力 (N)
T_MIN = 5.0         # 最小推力 (N)
DRAG_COEFF = 0.325    # 线性空气阻力系数
ROLL_PITCH_RATE_MAX = 4 * np.pi # 最大欧拉角速率 (rad/s)
YAW_RATE_MAX = 2 * np.pi # 最大欧拉角速率 (rad/s)
ATT_LIMIT = np.pi/4 # 姿态角限制 (Roll & Pitch)

# ======== MPC 控制器参数 (MPC Controller Parameters) ========
N_HORIZON = 30       # 预测时域步数
TF = 1.0             # 预测时域总时长 (s)
V_DESIRED = 25.0     # 期望接近速度 (m/s)

# ======== 仿真设置 (Simulation Settings) ========
SIM_TIME = 15.0      # 总仿真时长 (s)
INTERCEPTION_DISTANCE = 1.0 # 成功拦截的距离阈值 (m)

# ======== 感知模型参数 (Perception Model Parameters) ========
PERCEPTION_DELAY = 0.05  # 感知延迟 (s)
BBOX_POS_STD_DEV = 2.0  # 检测框中心位置的测量噪声标准差 (pixels)
BBOX_SIZE_STD_DEV = 5.0 # 检测框尺寸的测量噪声标准差 (pixels)

# ======== 控制执行延时参数 (Control Execution Delay Parameters) ========
CONTROL_EXECUTION_DELAY = 0.1  # 角速度控制执行延时 (s)

# ======== 初始条件 (Initial Conditions) ========
DRONE_INITIAL_STATE = np.array([0.0, 0.0, 10.0,     # Position (p_x, p_y, p_z)
                                0.0, 0.0, 0.0,      # Velocity (v_x, v_y, v_z)
                                0.0, 0.0, 0.0])     # Attitude (phi, theta, psi)
# TARGET_INITIAL_POS = np.array([100.0, 20.0, 15.0])
# TARGET_VELOCITY = np.array([-10.0, 10.0, 2.5]) # 目标做匀速直线运动
TARGET_INITIAL_POS = np.array([100.0, 0.0, 15.0])
TARGET_VELOCITY = np.array([0.0, 0.0, 0.0]) # 目标静止

# ======== 目标与相机物理属性 (Target & Camera Properties) ========
# --- 目标 ---
TARGET_PHYSICAL_SIZE_TRUTH = 0.8  # 目标的真实物理尺寸 (m), 用于相机模型
TARGET_PHYSICAL_SIZE_EST = 1.0    # 状态估计器使用的目标尺寸 (m), 可模拟估计误差

# --- 相机 ---
CAM_FOV_DEG = 40.0  # 相机水平视场角 (degrees)
CAM_FOCAL_LENGTH_PHYSICAL = 12.0 # 物理焦距 (mm)
CAM_PIXEL_SIZE = 3.45            # 像素尺寸 (um)
# f(pixel) = f(mm) * 1000 / pixel_size(um)
CAM_FOCAL_LENGTH = CAM_FOCAL_LENGTH_PHYSICAL * 1000 / CAM_PIXEL_SIZE # 焦距 (in pixels)

# ======== 代价函数 (Cost Function) ========
# --- 权重 ---
W_LOS = 100.0           # 视线误差权重
W_ALIGN = 0.5           # 速度对齐权重
W_CLOSING_SPEED = 1.0   # 接近速度权重
W_U_RATES = 1.0         # 角速度控制量权重
W_U_THRUST = 0.1        # 油门控制量权重

def get_body_rates(phi, theta, w):
    '''Get body-rates from global angle velocity'''
    '''phi, theta as Euler angle Roll and Pitch in ZYX order'''
    '''w as [dphi, dtheta, dpsi]'''
    # 转换为机身角速度
    T_matrix = np.array([[1, 0, -np.sin(theta)], 
                         [0, np.cos(phi), np.sin(phi)*np.cos(theta)], 
                         [0,-np.sin(phi), np.cos(phi)*np.cos(theta)]])
    # body_rates = T_matrix @ np.array([dphi, dtheta, dpsi])
    body_rates = T_matrix @ w
    return body_rates

def predict_future_attitude(x_current, current_body_rates, dt_predict):
    '''Predict future attitude to compensate for control execution delay'''
    # 当前姿态
    phi, theta, psi = x_current[6], x_current[7], x_current[8]
    
    # 获取当前机体角速度（从上一控制周期）
    p, q, r = current_body_rates[0], current_body_rates[1], current_body_rates[2]
    
    # 将当前机体角速度转换为全局角速度，用于姿态积分
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    t_theta = np.tan(theta)
    
    # 避免万向节锁奇点
    if abs(c_theta) < 0.1:
        # 万向节锁附近使用简化预测
        phi_future = phi + p * dt_predict
        theta_future = theta + q * dt_predict
        phi_future = np.clip(phi_future, -ATT_LIMIT, ATT_LIMIT)
        theta_future = np.clip(theta_future, -ATT_LIMIT, ATT_LIMIT)
        return phi_future, theta_future
    
    # 将机体角速度转换为全局角速率
    dphi = p + s_phi * t_theta * q + c_phi * t_theta * r
    dtheta = c_phi * q - s_phi * r
    
    # 使用当前机体角速度预测未来姿态
    phi_future = phi + dphi * dt_predict
    theta_future = theta + dtheta * dt_predict
    
    # 限制预测角度防止发散
    phi_future = np.clip(phi_future, -ATT_LIMIT, ATT_LIMIT)
    theta_future = np.clip(theta_future, -ATT_LIMIT, ATT_LIMIT)
    
    return phi_future, theta_future

def apply_attitude_safety_limit(drone_state, body_rates, att_limit, safety_margin=0.1):
    """应用姿态安全限制，防止危险机动"""
    phi = drone_state[6]   # 当前roll角
    theta = drone_state[7] # 当前pitch角
    
    # 检查当前姿态是否接近限制
    phi_abs = abs(phi)
    theta_abs = abs(theta)
    limit_with_margin = att_limit - safety_margin
    
    safe_body_rates = body_rates.copy()
    
    # 为roll应用保护性缩放
    if phi_abs > limit_with_margin:
        scale_factor = max(0.0, (att_limit - phi_abs) / safety_margin)
        if (phi > 0 and body_rates[0] > 0) or (phi < 0 and body_rates[0] < 0):
            # 机体角速度会进一步增加姿态角 - 缩小它
            safe_body_rates[0] *= scale_factor
    
    # 为pitch应用保护性缩放
    if theta_abs > limit_with_margin:
        scale_factor = max(0.0, (att_limit - theta_abs) / safety_margin)
        if (theta > 0 and body_rates[1] > 0) or (theta < 0 and body_rates[1] < 0):
            # 机体角速度会进一步增加姿态角 - 缩小它
            safe_body_rates[1] *= scale_factor
    
    return safe_body_rates

MODEL_CONSTANTS = {
    'G' : G,
    'ROLL_PITCH_RATE_MAX' : ROLL_PITCH_RATE_MAX,
    'YAW_RATE_MAX' : YAW_RATE_MAX,
    'ATT_LIMIT' : ATT_LIMIT,
    'A_MIN' : T_MIN / MASS,
    'A_MAX' : T_MAX / MASS,
    'N_HORIZON' : N_HORIZON,
    'TF' : TF,
    'DRONE_INITIAL_STATE' : DRONE_INITIAL_STATE
}

W_COST = np.array([np.sqrt(W_LOS),
                    np.sqrt(W_ALIGN),
                    np.sqrt(W_CLOSING_SPEED),
                    np.sqrt(W_U_RATES),
                    np.sqrt(W_U_THRUST)])

# ------------------------------------------------------------------------------
# 主仿真循环 (Main Simulation Loop)
# ------------------------------------------------------------------------------
def main():

    drone_model = DroneModel(MODEL_CONSTANTS)
    acados_solver = drone_model.generate_solver()


    # --- 初始化仿真环境 ---
    dt = TF / N_HORIZON
    N_sim = int(SIM_TIME / dt)
    x_current = DRONE_INITIAL_STATE.copy()
    target_pos_true = TARGET_INITIAL_POS.copy()
    target_vel_true = TARGET_VELOCITY.copy()
    
    camera = CameraModel(CAM_FOCAL_LENGTH, TARGET_PHYSICAL_SIZE_TRUTH, CAM_FOV_DEG, BBOX_POS_STD_DEV, BBOX_SIZE_STD_DEV)
    estimator = TargetEKF(dt, CAM_FOCAL_LENGTH, TARGET_PHYSICAL_SIZE_EST, BBOX_POS_STD_DEV, BBOX_SIZE_STD_DEV)
    output_filter = EMAFilter(int(N_HORIZON/TF*0.2), channels=4)


    perceptor = PerceptionSim(camera, PERCEPTION_DELAY, dt)

    # --- 初始化延迟与盲飞逻辑 ---
    target_lost_counter = 0
    
    # --- 初始化角速度延时补偿变量 ---
    previous_body_rates = np.array([0.0, 0.0, 0.0])  # 存储前一次机体角速度
    
    history = {'x': [x_current], 'u': [], 'target_true': [target_pos_true.copy()], 
               'target_est_ekf': [], 'target_est_raw': [], 'distance': []}
    
    # --- 性能改进: 初始化热启动变量 ---
    # 使用合理的初始猜测值（悬停状态）初始化
    u_warm_start = np.tile(np.array([0, 0, 0, G]), (N_HORIZON, 1))
    x_warm_start = np.tile(x_current, (N_HORIZON + 1, 1))

    print("\n[INFO] 开始仿真 (使用EKF, 考虑感知延迟与噪声)...")
    i = 0
    for i in range(N_sim):
        # ------------------- 1. 感知 (Perceive) -------------------
        full, target_pos_true = perceptor.update(x_current, target_pos_true, target_vel_true, history)
        if not full : continue
        drone_state_at_meas, meas_bbox, meas_visible = perceptor.get_measure()

        # ------------------- 2. EKF 预测 (Predict) -------------------
        if estimator.is_initialized:
            estimator.predict(dt)

        # ------------------- 3. EKF 更新 (Update) -------------------
        if meas_visible:
            target_lost_counter = 0 # 目标可见，重置计数器
            drone_pos, psi = drone_state_at_meas[0:3], drone_state_at_meas[8]
            raw_pos_est = estimator.noisy_bbox_to_pos(meas_bbox, drone_pos, psi)
            if not estimator.is_initialized:
                estimator.initialize(raw_pos_est)
            else:
                estimator.update(meas_bbox, drone_pos, psi)
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
        # if not estimator.is_initialized:
        #     print(f"[WARNING] 步骤 {i}: EKF尚未初始化, 跳过规划。")
        #     # 悬停指令：角速度为0，推力加速度等于重力加速度
        #     u_optimal_raw = np.array([0, 0, 0, G])
        #     # 保持悬停状态
        #     x_current[3:9] = 0.0 
        # else:
        #     pos_est = estimator.x[0:3]
        #     vel_est = estimator.x[3:6]
            
        #     acados_solver.set(0, "lbx", x_current)
        #     acados_solver.set(0, "ubx", x_current)
            
        #     for k in range(N_HORIZON + 1):
        #         t_k = k * dt
        #         predicted_target_pos_est = pos_est + t_k * vel_est
        #         acados_solver.set(k, "p", np.hstack([predicted_target_pos_est, vel_est, DRAG_COEFF, V_DESIRED, W_COST]))

        #     t0 = time.time()
        #     status = acados_solver.solve()
        #     if status != 0: print(f'[WARNING] ACADOS求解器在第 {i} 步返回状态 {status}。')
        #     print(f'acados_solver.solve() {int((time.time()-t0)*1000000)} us')
            
        #     u_optimal_raw = acados_solver.get(0, "u")
        #     x_current = acados_solver.get(1, "x")

        if not estimator.is_initialized:
            print(f"[WARNING] 步骤 {i}: EKF尚未初始化，跳过规划。")
            # 悬停指令：角速度为0，推力加速度等于重力加速度
            u_optimal_raw = np.array([0, 0, 0, G])
            # 保持悬停状态
            x_current[3:9] = 0.0 
        else:
            pos_est = estimator.x[0:3]
            vel_est = estimator.x[3:6]
            
            acados_solver.set(0, "lbx", x_current)
            acados_solver.set(0, "ubx", x_current)
            
            for k in range(N_HORIZON + 1):
                t_k = k * dt
                predicted_target_pos_est = pos_est + t_k * vel_est
                acados_solver.set(k, "p", np.hstack([predicted_target_pos_est, vel_est, DRAG_COEFF, V_DESIRED, W_COST]))

            # --- 性能改进: 设置热启动初始猜测值 ---
            for k in range(N_HORIZON):
                acados_solver.set(k, 'x', x_warm_start[k,:])
                acados_solver.set(k, 'u', u_warm_start[k,:])
            acados_solver.set(N_HORIZON, 'x', x_warm_start[N_HORIZON,:])

            t0 = time.time()
            status = acados_solver.solve()
            if status != 0: print(f'[WARNING] ACADOS求解器在第 {i} 步返回状态 {status}。')
            print(f'acados_solver.solve() {int((time.time()-t0)*1000000)} us')
            
            u_optimal_raw = acados_solver.get(0, "u")
            x_current = acados_solver.get(1, "x")

            # --- 性能改进: 为下一次迭代准备热启动数据 ---
            # 通过将当前解左移一个步长来更新猜测值
            for k in range(N_HORIZON - 1):
                x_warm_start[k,:] = acados_solver.get(k+1, "x")
                u_warm_start[k,:] = acados_solver.get(k+1, "u")
            x_warm_start[N_HORIZON-1,:] = acados_solver.get(N_HORIZON, "x")
            # 对于新出现的最后一个点，简单重复前一个点作为猜测
            x_warm_start[N_HORIZON,:] = acados_solver.get(N_HORIZON, "x")
            u_warm_start[N_HORIZON-1,:] = acados_solver.get(N_HORIZON-1, "u")

        # ------------------- 5. 执行 (Act) & 记录 -------------------
        u_optimal_filtered = output_filter.filter_array(u_optimal_raw)
        
        # 应用角速度执行延时补偿
        # 预测无人机姿态在新机体角速度生效时的位置
        phi_predicted, theta_predicted = predict_future_attitude(x_current, previous_body_rates, CONTROL_EXECUTION_DELAY)
        thrust_accel = u_optimal_filtered[3]

        # 使用预测的未来姿态进行角速度转换，补偿执行延时
        body_rates = get_body_rates(phi_predicted, theta_predicted, u_optimal_filtered[0:3])
        
        # 应用姿态安全限制防止危险机动
        body_rates = apply_attitude_safety_limit(x_current, body_rates, ATT_LIMIT)
        
        # 构建实际控制输入（机身角速度 + 推力加速度）
        u_optimal = np.hstack([body_rates, thrust_accel])
        
        # 保存当前机体角速度供下次延时补偿使用
        previous_body_rates = body_rates.copy()
        
        # 将推力加速度转换为力（牛顿）用于记录
        thrust_force = thrust_accel * MASS
        u_to_log = np.hstack([body_rates, thrust_force])

        # ------------------- 6. 更新环境 & 记录数据 -------------------
        target_pos_true += target_vel_true * dt
        
        history['x'].append(x_current.copy())
        history['u'].append(u_to_log.copy())
        history['target_true'].append(target_pos_true.copy())
        if estimator.is_initialized:
            history['target_est_ekf'].append(estimator.x.copy())
        if raw_pos_est is not None:
            history['target_est_raw'].append(raw_pos_est)
        
        distance = np.linalg.norm(target_pos_true - x_current[0:3])
        history['distance'].append(distance)
        
        if distance < INTERCEPTION_DISTANCE:
            print(f"\n[SUCCESS] 在第 {i} 步成功拦截目标！距离: {distance:.2f} m")
            break
    
    if i >= N_sim - 1: print(f"\n[INFO] 仿真结束。最终距离: {history['distance'][-1]:.2f} m")

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
    ax.legend(); ax.set_title('3D Trajectory'); ax.grid(True); ax.axis('equal'); plt.show()
    
    # 性能指标图
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('Interceptor Performance Metrics', fontsize=16)
    if len(hist['distance']) > 0:
        axs[0].plot(time_axis_u, hist['distance'], label='Distance to Target')
    axs[0].set_ylabel('Distance [m]'); axs[0].set_title('Distance Over Time')
    
    # 计算接近速度和LOS误差用于绘图
    closing_speed, los_error = [], []
    if len(hist['u']) > 0:
        # Align history data to the length of control inputs
        num_u_steps = len(hist['u'])
        aligned_x_hist = hist['x'][-num_u_steps-1:-1]
        aligned_target_hist = hist['target_true'][-num_u_steps-1:-1]

        for i in range(num_u_steps):
            drone_x_state = aligned_x_hist[i]
            drone_v, psi = drone_x_state[3:6], drone_x_state[8]
            target_pos = aligned_target_hist[i]
            
            los_W = target_pos - drone_x_state[0:3]
            dist = np.linalg.norm(los_W) + 1e-9
            los_norm = los_W / dist
            closing_speed.append(np.dot(drone_v, los_norm))
            
            cpsi, spsi = np.cos(psi), np.sin(psi)
            R_W_C = np.array([[cpsi, spsi, 0], [-spsi, cpsi, 0], [0, 0, 1]])
            los_C = R_W_C @ los_W
            los_error.append(los_C[1:3])

    closing_speed, los_error = np.array(closing_speed), np.array(los_error)

    if len(closing_speed) > 0:
        axs[1].plot(time_axis_u, closing_speed, label='Actual Closing Speed (on LOS)')
        axs[1].axhline(y=V_DESIRED, color='r', linestyle='--', label=f'Desired Closing Speed ({V_DESIRED} m/s)')
    axs[1].set_ylabel('Speed [m/s]'); axs[1].set_title('Closing Speed')

    if len(los_error) > 0:
        axs[2].plot(time_axis_u, los_error[:, 0], label='LOS y-error (left/right)')
        axs[2].plot(time_axis_u, los_error[:, 1], label='LOS z-error (up/down)')
    axs[2].set_ylabel('Error [m]'); axs[2].set_title('Line-of-Sight Error in Camera Frame')
    axs[2].set_xlabel('Time [s]')
    
    for ax in axs: ax.grid(True); ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

    # 状态和控制量图
    fig, axs = plt.subplots(5, 1, figsize=(15, 18), sharex=True)
    fig.suptitle('States and Control Inputs', fontsize=16)
    axs[0].plot(time_axis_x, hist['x'][:, 0], label='px'); axs[0].plot(time_axis_x, hist['x'][:, 1], label='py'); axs[0].plot(time_axis_x, hist['x'][:, 2], label='pz'); axs[0].set_ylabel('Position [m]')
    axs[1].plot(time_axis_x, hist['x'][:, 3], label='vx'); axs[1].plot(time_axis_x, hist['x'][:, 4], label='vy'); axs[1].plot(time_axis_x, hist['x'][:, 5], label='vz'); axs[1].set_ylabel('Velocity [m/s]')
    axs[2].plot(time_axis_x, np.rad2deg(hist['x'][:, 6]), label='Roll'); axs[2].plot(time_axis_x, np.rad2deg(hist['x'][:, 7]), label='Pitch'); axs[2].plot(time_axis_x, np.rad2deg(hist['x'][:, 8]), label='Yaw'); axs[2].set_ylabel('Angle [deg]')
    if len(hist['u']) > 0:
        axs[3].plot(time_axis_u, hist['u'][:, 0], label='p (body-x rate)')
        axs[3].plot(time_axis_u, hist['u'][:, 1], label='q (body-y rate)')
        axs[3].plot(time_axis_u, hist['u'][:, 2], label='r (body-z rate)')
        axs[3].set_ylabel('Body Ang. Vel [rad/s]')
        axs[4].plot(time_axis_u, hist['u'][:, 3], label='Thrust [N]'); axs[4].set_ylabel('Thrust [N]'); axs[4].set_xlabel('Time [s]')
    for ax in axs: ax.grid(True); ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

if __name__ == "__main__":        
    main()