import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import casadi as ca
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Dict

# ------------------------------------------------------------------------------
# 1. 仿真参数与常量配置 (Simulation Parameters and Constants)
# ------------------------------------------------------------------------------
# ======== 物理常量 (Physics Constants) ========
MASS = 1.0  # 无人机质量 (kg)
G = 9.81    # 重力加速度 (m/s^2)
T_MAX = 20.0 # 最大推力 (N)
T_MIN = 0.0  # 最小推力 (N)
DRAG_COEFF = 0.1 # 线性空气阻力系数
EULER_RATE_MAX = 4 * np.pi # 最大欧拉角速率 (rad/s)
ATT_LIMIT = np.pi/4 # 姿态角限制 (Roll & Pitch)

# ======== MPC 控制器参数 (MPC Controller Parameters) ========
N_HORIZON = 50       # 预测时域步数
TF = 1.0             # 预测时域总时长 (s)
V_DESIRED = 25.0     # 期望接近速度 (m/s)

# ======== 仿真设置 (Simulation Settings) ========
SIM_TIME = 15.0      # 总仿真时长 (s)
INTERCEPTION_DISTANCE = 0.5 # 成功拦截的距离阈值 (m)

# ======== 初始条件 (Initial Conditions) ========
DRONE_INITIAL_STATE = np.array([0.0, 0.0, 10.0,     # Position
                                0.0, 0.0, 0.0,      # Velocity
                                0.0, 0.0, 0.0])     # Attitude RPY
TARGET_INITIAL_POS = np.array([100.0, 20.0, 15.0])
# TARGET_VELOCITY = np.array([0.0, 0.0, 0.0])
TARGET_VELOCITY = np.array([-10.0, 5.0, 2.5])

# ======== 目标与相机物理属性 (Target & Camera Properties) ========
# --- 目标 ---
TARGET_PHYSICAL_SIZE_TRUTH = 0.8  # 目标的真实物理尺寸 (m), 用于相机模型
TARGET_PHYSICAL_SIZE_EST = 1.2    # 状态估计器使用的目标尺寸 (m), 可模拟估计误差

# --- 相机 ---
CAM_FOV_DEG = 40.0  # 相机水平视场角 (degrees)
CAM_FOCAL_LENGTH_PHYSICAL = 12.0 # 物理焦距 (mm)
CAM_PIXEL_SIZE = 3.45            # 像素尺寸 (um)
# f(pixel) = f(mm) * 1000 / pixel_size(um)
CAM_FOCAL_LENGTH = CAM_FOCAL_LENGTH_PHYSICAL * 1000 / CAM_PIXEL_SIZE # 焦距 (in pixels)

# ------------------------------------------------------------------------------
# 2. 模拟相机与状态估计器 (Camera Model & State Estimator)
# ------------------------------------------------------------------------------
class CameraModel:
    """模拟相机,根据3D位置生成2D检测框信息,并检查FOV"""
    def __init__(self, focal_length, target_size, fov_deg):
        self.f = focal_length
        self.target_size = target_size
        self.max_angle_rad = np.deg2rad(fov_deg / 2.0)

    def project(self, drone_state, target_pos) -> Tuple[Dict[str, float], bool]:
        """
        将3D世界中的目标投影到相机平面。
        返回: (bbox字典, is_visible布尔值)
        """
        drone_pos, psi = drone_state[0:3], drone_state[8]
        los_W = target_pos - drone_pos
        
        cpsi, spsi = np.cos(psi), np.sin(psi)
        R_W_C = np.array([[cpsi, spsi, 0], [-spsi, cpsi, 0], [0, 0, 1]])
        los_C = R_W_C @ los_W
        
        dist_cam_x = los_C[0] # 目标在相机坐标系下的深度/X轴距离
        
        # 检查目标是否在相机前方
        if dist_cam_x <= 0.1:
            return {'u': 0, 'v': 0, 'size': 0}, False

        # 检查目标是否在FOV内
        angle_to_center = np.arctan2(np.sqrt(los_C[1]**2 + los_C[2]**2), dist_cam_x)
        if angle_to_center > self.max_angle_rad:
            return {'u': 0, 'v': 0, 'size': 0}, False

        # 计算检测框信息
        pixel_size = self.f * self.target_size / dist_cam_x
        bbox = {
            'size': pixel_size,
            'u': int(self.f * los_C[1] / dist_cam_x), # 在相机坐标系下的Y轴位置
            'v': int(self.f * los_C[2] / dist_cam_x)  # 在相机坐标系下的Z轴位置
        }
        # print(f'bbox = {bbox}')
        
        return bbox, True

class TargetEstimator:
    """根据相机检测框信息，估计目标的三维状态。"""
    def __init__(self, focal_length, target_size, dt):
        self.f = focal_length
        self.target_size = target_size
        self.dt = dt
        self.last_pos_est = None

    def estimate(self, bbox, drone_state) -> Tuple[np.ndarray, np.ndarray]:
        """从检测框估计目标位置和速度。"""
        drone_pos, psi = drone_state[0:3], drone_state[8]

        # 1. 估计距离
        distance = 0
        if bbox['size'] > 1e-6:
            pixel_size = bbox['size']
            distance = self.f * self.target_size / pixel_size
        else: # 如果面积为0，无法估计
            return self.last_pos_est if self.last_pos_est is not None else (drone_pos, np.zeros(3))

        # 2. 估计方向 (LOS)
        los_C_vec = np.array([self.f, bbox['u'], bbox['v']])
        los_norm = np.linalg.norm(los_C_vec)
        los_C_dir = los_C_vec / (los_norm + 1e-9)
        # los_C_dir = np.array([distance, bbox['u'], bbox['v']])
        # los_C_dir /= (np.linalg.norm(los_C_dir) + 1e-9)

        # 3. 重构三维位置
        cpsi, spsi = np.cos(psi), np.sin(psi)
        R_C_W = np.array([[cpsi, -spsi, 0], [spsi, cpsi, 0], [0, 0, 1]])
        los_W_dir = R_C_W @ los_C_dir
        pos_est = drone_pos + distance * los_W_dir

        # 4. 估计速度 (通过差分)
        vel_est = np.zeros(3)
        if self.last_pos_est is not None:
            vel_est = (pos_est - self.last_pos_est) / self.dt
        
        self.last_pos_est = pos_est.copy()
        
        return pos_est, vel_est

# ------------------------------------------------------------------------------
# 3. 定义无人机模型 (ACADOS Drone Model)
# ------------------------------------------------------------------------------
def drone_model() -> AcadosModel:
    """定义ACADOS所需的无人机动力学模型和代价函数。"""
    model_name = 'drone_interceptor_mpc'
    x = ca.SX.sym('x', 9)       # type: ignore
    u = ca.SX.sym('u', 4)       # type: ignore
    params = ca.SX.sym('p', 6)  # type: ignore
    _, _, _, vx, vy, vz, phi, theta, psi = [x[i] for i in range(9)]
    dphi, dtheta, dpsi, T_norm = [u[i] for i in range(4)]
    
    pos_dot = ca.vertcat(vx, vy, vz)
    euler_dot = ca.vertcat(dphi, dtheta, dpsi)
    
    cphi, sphi = ca.cos(phi), ca.sin(phi)
    cth, sth = ca.cos(theta), ca.sin(theta)
    cpsi, spsi = ca.cos(psi), ca.sin(psi)
    
    R_B_W = ca.vertcat(
        ca.horzcat(cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi),
        ca.horzcat(spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi),
        ca.horzcat(-sth, cth*sphi, cth*cphi)
    )


    T = T_norm * (T_MAX - T_MIN) + T_MIN
    thrust_W = R_B_W @ ca.vertcat(0, 0, T)
    v_dot = (1/MASS)*thrust_W - ca.vertcat(DRAG_COEFF*vx, DRAG_COEFF*vy, DRAG_COEFF*vz) - ca.vertcat(0,0,G)
    f_expl = ca.vertcat(pos_dot, v_dot, euler_dot)

    # ======== 代价函数 (Cost Function) ========
    W_los = 100.0 
    W_align = 5.0
    W_closing_speed = 0.5
    W_u_rates = 0.01
    W_u_thrust = 0.1
    
    drone_pos = x[0:3]
    target_pos = params[0:3]
    los_W = target_pos-drone_pos

    # yaw = ca.arctan2(spsi*cth, cpsi*cth)
    # cp, sp = ca.cos(yaw), ca.sin(yaw)
    # R_W_C = ca.vertcat(
    #     ca.horzcat(cp,sp,0), 
    #     ca.horzcat(-sp,cp,0), 
    #     ca.horzcat(0,0,1))
    R_W_C = ca.vertcat(ca.horzcat(cpsi,spsi,0), ca.horzcat(-spsi,cpsi,0), ca.horzcat(0,0,1))
    los_C = R_W_C @ los_W
    los_error_y = los_C[1]
    los_error_z = los_C[2]
    
    drone_vel_W = x[3:6]
    speed_sq = drone_vel_W.T@drone_vel_W
    dist_sq = los_W.T@los_W
    v_norm = drone_vel_W/ca.sqrt(speed_sq+1e-6)
    los_norm = los_W/ca.sqrt(dist_sq+1e-6)
    alignment_error = 1 - ca.dot(v_norm, los_norm)

    # 期望接近速度误差 (定义为无人机与目标在相机X轴上的相对速度)
    cam_x_axis = R_W_C[0,:].T
    closing_speed_cam_x = ca.dot(drone_vel_W, cam_x_axis) - ca.dot(params[3:6], cam_x_axis)
    closing_speed_error = V_DESIRED - closing_speed_cam_x

    T_hover =  (MASS*G - T_MIN) / (T_MAX - T_MIN) 

    cost_expr_y = ca.vertcat(ca.sqrt(W_los)*los_error_y, ca.sqrt(W_los)*los_error_z,
                             ca.sqrt(W_align)*alignment_error, ca.sqrt(W_closing_speed)*closing_speed_error,
                             ca.sqrt(W_u_rates)*dphi, ca.sqrt(W_u_rates)*dtheta, ca.sqrt(W_u_rates)*dpsi,
                             ca.sqrt(W_u_thrust)*(T_norm - T_hover))
    cost_expr_y_e = ca.vertcat(ca.sqrt(W_los*10)*los_error_y, ca.sqrt(W_los*10)*los_error_z,
                               ca.sqrt(W_align*5)*alignment_error, ca.sqrt(W_closing_speed*5)*closing_speed_error)
    
    model=AcadosModel()
    model.x=x; model.u=u; model.p=params; model.f_expl_expr=f_expl
    model.cost_y_expr=cost_expr_y; model.cost_y_expr_e=cost_expr_y_e; model.name=model_name
    return model

# ------------------------------------------------------------------------------
# 4. 设置 OCP (Optimal Control Problem)
# ------------------------------------------------------------------------------
def setup_ocp(model: AcadosModel) -> AcadosOcp:
    """配置并返回ACADOS OCP对象。"""
    ocp = AcadosOcp()
    ocp.model = model
    
    ocp.cost.cost_type='NONLINEAR_LS'
    ocp.cost.cost_type_e='NONLINEAR_LS'
    ocp.cost.W=np.eye(model.cost_y_expr.shape[0])       # type: ignore
    ocp.cost.W_e=np.eye(model.cost_y_expr_e.shape[0])   # type: ignore
    ocp.cost.yref=np.zeros((model.cost_y_expr.shape[0],))       # type: ignore
    ocp.cost.yref_e=np.zeros((model.cost_y_expr_e.shape[0],))   # type: ignore

    ocp.constraints.idxbu=np.arange(4)
    ocp.constraints.lbu=np.array([-EULER_RATE_MAX]*3+[0.0])
    ocp.constraints.ubu=np.array([EULER_RATE_MAX]*3+[1.0])
    ocp.constraints.idxbx=np.array([6,7])
    ocp.constraints.lbx=np.array([-ATT_LIMIT]*2)
    ocp.constraints.ubx=np.array([ATT_LIMIT]*2)
    ocp.constraints.x0=DRONE_INITIAL_STATE
    ocp.parameter_values=np.zeros(model.p.shape) # type: ignore

    ocp.solver_options.qp_solver='PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx='GAUSS_NEWTON'
    ocp.solver_options.integrator_type='ERK'
    ocp.solver_options.nlp_solver_type='SQP_RTI'
    ocp.solver_options.N_horizon = N_HORIZON
    ocp.solver_options.tf=TF
    return ocp

class EMAFilter:
    """
    指数移动平均滤波器 (Exponential Moving Average Filter)
    
    参数:
    n: 窗口大小, 用于计算alpha值
    channels: 滤波器通道数, 默认为1
    """
    
    def __init__(self, n: int, channels: int = 1):
        if n <= 0:
            raise ValueError("n must be a positive integer")
        if channels <= 0:
            raise ValueError("channels must be a positive integer")
            
        self.alpha = 2.0 / (n + 1)
        self.data = np.zeros(channels)
        self.initialized = np.full(channels, False)
        self.channels = channels
    
    def reset(self) -> None:
        """重置滤波器状态"""
        self.data.fill(0)
        self.initialized.fill(False)
    
    def __getitem__(self, index: int) -> float:
        """获取指定通道的当前滤波值"""
        if index < 0 or index >= self.channels:
            raise IndexError(f"Index {index} out of range for {self.channels} channels")
        return self.data[index]
    
    def is_initialized(self, index: int = 0) -> bool:
        """检查指定通道是否已初始化"""
        if index < 0 or index >= self.channels:
            raise IndexError(f"Index {index} out of range for {self.channels} channels")
        return self.initialized[index]
    
    def filter(self, value: float, index: int) -> float:
        # 处理单个值
        if index is None:
            index = 0
            
        if index < 0 or index >= self.channels:
            raise IndexError(f"Index {index} out of range for {self.channels} channels")
            
        if not self.initialized[index]:
            self.data[index] = value
            self.initialized[index] = True
        else:
            self.data[index] = self.alpha * value + (1 - self.alpha) * self.data[index]
            
        return self.data[index]
    
    def filter_array(self, values: np.ndarray) -> np.ndarray:
        # 处理数组
        if values.shape[0] != self.channels:
            raise ValueError(f"Input array size {values.shape[0]} does not match filter channels {self.channels}")
            
        result = np.zeros(self.channels)
        for i in range(self.channels):
            result[i] = self.filter(values[i], i)
        return result
    
# ------------------------------------------------------------------------------
# 5. 主仿真循环 (Main Simulation Loop)
# ------------------------------------------------------------------------------
def main():
    """主函数：初始化并运行仿真。"""
    output_filter = EMAFilter(int(N_HORIZON/TF*0.1), channels=4)

    drone_model_acados = drone_model()
    ocp = setup_ocp(drone_model_acados)
    acados_generated_path = 'c_generated_code'
    if os.path.exists(acados_generated_path): shutil.rmtree(acados_generated_path)
    json_file = 'acados_ocp_' + drone_model_acados.name + '.json' # type: ignore
    acados_solver = AcadosOcpSolver(ocp, json_file=json_file)

    dt = TF / N_HORIZON; N_sim = int(SIM_TIME / dt)
    x_current = DRONE_INITIAL_STATE.copy()
    target_pos_true = TARGET_INITIAL_POS.copy()
    target_vel_true = TARGET_VELOCITY.copy()
    
    camera = CameraModel(CAM_FOCAL_LENGTH, TARGET_PHYSICAL_SIZE_TRUTH, CAM_FOV_DEG)
    estimator = TargetEstimator(CAM_FOCAL_LENGTH, TARGET_PHYSICAL_SIZE_EST, dt)
    
    history = {'x': [x_current], 
               'u': [], 
               'target_true': [target_pos_true.copy()], 
               'target_est': [],
               'distance': []}

    # *** 新增：用于处理目标丢失的变量 ***
    target_lost_counter = 0
    last_known_pos_est = np.array([0.0, 0.0, 0.0])
    last_known_vel_est = np.array([0.0, 0.0, 0.0])

    print("\n[INFO] 开始仿真 (基于模拟视觉估计)...")
    i = 0
    for i in range(N_sim):
        # ------------------- 感知 (Perceive) -------------------
        bbox, is_visible = camera.project(x_current, target_pos_true)
        
        # ------------------- 估计 (Estimate) -------------------
        if is_visible:
            # 如果目标可见，重置丢失计数器，并用当前观测更新估计
            if target_lost_counter > 0:
                print(f"[INFO] 在第 {i} 步重新捕获目标！")
            target_lost_counter = 0
            pos_est, vel_est = estimator.estimate(bbox, x_current)
            # 保存这次有效的估计，以备后续丢失时使用
            last_known_pos_est, last_known_vel_est = pos_est, vel_est
        else:
            # 如果目标不可见，增加丢失计数器
            target_lost_counter += 1
            print(f"[WARNING] 第 {i} 步目标丢失! 盲飞计数: {target_lost_counter}/{N_HORIZON}")
            # 使用最后一次的有效估计进行“盲飞”预测
            last_known_pos_est += last_known_vel_est * dt
            pos_est, vel_est = last_known_pos_est, last_known_vel_est

        # 检查任务失败条件：丢失时间超过一个预测时域 TF
        if target_lost_counter > N_HORIZON*2:
            print(f"\n[FAILED] 目标丢失时间超过 {TF*2} 秒! 任务失败。")
            break
        
        # ------------------- 规划 (Plan) -------------------
        acados_solver.set(0, "lbx", x_current); acados_solver.set(0, "ubx", x_current)
        for k in range(N_HORIZON + 1):
            t_k = k * dt
            # MPC使用当前（或最后已知）的估计状态进行轨迹预测
            predicted_target_pos_est = pos_est + t_k * vel_est
            acados_solver.set(k, "p", np.hstack([predicted_target_pos_est, vel_est]))

        status = acados_solver.solve()
        if status != 0: print(f'[WARNING] ACADOS求解器在第 {i} 步返回状态 {status}。')

        # ------------------- 执行 (Act) -------------------
        u_optimal = acados_solver.get(0, "u")
        u_optimal = output_filter.filter_array(u_optimal)
        x_current = acados_solver.get(1, "x")
        
        # ------------------- 更新环境 (Update Environment) -------------------
        target_pos_true += target_vel_true * dt
        
        # ------------------- 记录数据 (Log Data) -------------------
        history['x'].append(x_current.copy())
        history['u'].append(u_optimal.copy())
        history['target_true'].append(target_pos_true.copy())
        history['target_est'].append(pos_est.copy())
        distance = np.linalg.norm(target_pos_true - x_current[0:3])
        history['distance'].append(distance)
        
        # 检查拦截条件
        if distance < INTERCEPTION_DISTANCE:
            print(f"\n[SUCCESS] 在第 {i} 步成功拦截目标！距离: {distance:.2f} m")
            break
    
    if i == N_sim - 1: print(f"\n[INFO] 仿真结束。最终距离: {history['distance'][-1]:.2f} m")

    plot_results(history, dt)

def plot_results(hist, dt):
    """绘制仿真结果。"""
    if not hist['u']:
        print("[ERROR] 仿真未能生成有效数据，无法绘图。")
        return

    for key in hist: hist[key] = np.array(hist[key])
    
    num_steps = len(hist['u'])
    time_axis = np.linspace(0, dt * num_steps, num_steps + 1)
    time_axis_u = np.linspace(0, dt * num_steps, num_steps)

    # 3D轨迹图
    fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection='3d')
    ax.plot(hist['x'][:, 0], hist['x'][:, 1], hist['x'][:, 2], label='Drone Trajectory', c='b', lw=2)
    ax.plot(hist['target_true'][:, 0], hist['target_true'][:, 1], hist['target_true'][:, 2], label='Target True Traj.', c='r', ls='--')
    ax.plot(hist['target_est'][:, 0], hist['target_est'][:, 1], hist['target_est'][:, 2], label='Target Estimated Traj.', c='g', ls=':')
    ax.scatter(hist['x'][0, 0], hist['x'][0, 1], hist['x'][0, 2], c='b', s=100, marker='o', label='Drone Start')
    ax.scatter(hist['target_true'][0, 0], hist['target_true'][0, 1], hist['target_true'][0, 2], c='r', s=100, marker='x', label='Target Start')
    ax.scatter(hist['target_est'][0, 0], hist['target_est'][0, 1], hist['target_est'][0, 2], c='g', s=100, marker='x', label='Target Estimated Start')

    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]'); ax.set_zlabel('Up [m]')
    ax.legend(); ax.set_title('3D Trajectory'); ax.grid(True); ax.axis('equal'); plt.show()
    
    # 性能指标图
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('Interceptor Performance Metrics', fontsize=16)
    axs[0].plot(time_axis_u, hist['distance'], label='Distance to Target')
    axs[0].set_ylabel('Distance [m]')
    axs[0].set_title('Distance Over Time')
    
    # 计算接近速度和LOS误差用于绘图分析
    closing_speed, los_error, align_error = [], [], []
    for i in range(num_steps):
        drone_v, psi = hist['x'][i+1, 3:6], hist['x'][i+1, 8]
        los_W = hist['target_true'][i+1] - hist['x'][i+1, 0:3]
        dist = np.linalg.norm(los_W) + 1e-9
        los_norm = los_W / dist
        closing_speed.append(np.dot(drone_v, los_norm))
        align_error.append(1 - np.dot(drone_v / (np.linalg.norm(drone_v) + 1e-9), los_norm))
        cpsi, spsi = np.cos(psi), np.sin(psi)
        R_W_C = np.array([[cpsi, spsi, 0], [-spsi, cpsi, 0], [0, 0, 1]])
        los_C = R_W_C @ los_W
        los_error.append(los_C[1:3])
    
    closing_speed, los_error, align_error = np.array(closing_speed), np.array(los_error), np.array(align_error)

    axs[1].plot(time_axis_u, closing_speed, label='Actual Closing Speed (LOS)')
    axs[1].axhline(y=V_DESIRED, color='r', linestyle='--', label=f'Desired Closing Speed ({V_DESIRED} m/s)')
    axs[1].set_ylabel('Speed [m/s]'); axs[1].set_title('Closing Speed')
    
    axs[2].plot(time_axis_u, los_error[:, 0], label='LOS y-error (left/right)')
    axs[2].plot(time_axis_u, los_error[:, 1], label='LOS z-error (up/down)')
    axs[2].set_ylabel('Error [m]')
    axs[2].set_title('Line-of-Sight Error in Camera Frame')
    axs[2].set_xlabel('Time [s]')
    for ax in axs: ax.grid(True); ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

    # 状态和控制量图
    fig, axs = plt.subplots(5, 1, figsize=(15, 18), sharex=True)
    fig.suptitle('States and Control Inputs', fontsize=16)
    axs[0].plot(time_axis, hist['x'][:, 0], label='px (East)')
    axs[0].plot(time_axis, hist['x'][:, 1], label='py (North)')
    axs[0].plot(time_axis, hist['x'][:, 2], label='pz (Up)')
    axs[0].set_ylabel('Position [m]')
    axs[1].plot(time_axis, hist['x'][:, 3], label='vx')
    axs[1].plot(time_axis, hist['x'][:, 4], label='vy')
    axs[1].plot(time_axis, hist['x'][:, 5], label='vz')
    axs[1].set_ylabel('Velocity [m/s]')
    axs[2].plot(time_axis, np.rad2deg(hist['x'][:, 6]), label='Roll (phi)')
    axs[2].plot(time_axis, np.rad2deg(hist['x'][:, 7]), label='Pitch (theta)')
    axs[2].plot(time_axis, np.rad2deg(hist['x'][:, 8]), label='Yaw (psi)')
    axs[2].set_ylabel('Angle [degrees]')
    axs[3].plot(time_axis_u, hist['u'][:, 0], label='dphi')
    axs[3].plot(time_axis_u, hist['u'][:, 1], label='dtheta')
    axs[3].plot(time_axis_u, hist['u'][:, 2], label='dpsi')
    axs[3].set_ylabel('Angular Velocity [rad/s]')
    axs[4].plot(time_axis_u, hist['u'][:, 3], label='Normalized Thrust', linestyle='--')
    axs[4].set_xlabel('Time [s]')
    axs[4].set_ylabel('Normalized Thrust')
    for ax in axs: ax.grid(True); ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

if __name__ == "__main__":        
    main()
