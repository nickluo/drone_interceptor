import math
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import casadi as ca
import os
import shutil

# ------------------------------------------------------------------------------
# 定义无人机模型 - 使用四元数版本 (ACADOS Drone Model with Quaternions)
# ------------------------------------------------------------------------------
class DroneModelQuaternion:
    def __init__(self, constants, los_only=False):
        G = constants['G']    # 重力加速度 (m/s^2)
        self.constants = constants
        """定义ACADOS所需的无人机动力学模型和代价函数 - 四元数版本。"""
        model_name = 'drone_interceptor_mpc_quat'
        
        # --- 状态和控制变量 ---
        # 其中 q = [qw, qx, qy, qz] 是四元数 (scalar-first convention)
        x = ca.SX.sym('x', 16)      # type: ignore # 状态: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, gimbal_r, gimbal_p, gimbal_y]
        u = ca.SX.sym('u', 4)       # type: ignore # 控制: [u_wx, u_wy, u_wz, thrust] (机体角速度 + 推力加速度)
        params = ca.SX.sym('p', 18) # type: ignore # 参数: [p_target, v_target, drag_coeff, v_desired, rate_tc, w_cost, gimbal_omega_coeff]

        drag_coeff = params[6]
        v_desired = params[7]
        rate_tc = params[8]
        gimbal_RP_coeff = params[16]
        gimbal_Y_coeff = params[17]

        # --- 权重 ---
        W_los = params[9]  # 视线误差权重
        W_align = params[10]  # 速度对齐权重
        W_closing_speed = params[11]  # 接近速度权重
        W_rates = params[12]        # 角速度控制量权重
        W_tilt = params[13]         # 倾斜角误差权重
        W_u_alpha = params[14]      # 角速度变化率权重
        W_u_thrust = params[15]     # 油门控制量权重
        
        # --- 提取状态变量 ---
        pos = x[0:3]        # 位置
        vel = x[3:6]        # 速度  
        quat = x[6:10]      # 四元数 [qw, qx, qy, qz]
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        wx, wy, wz = x[10], x[11], x[12]  # 机体角速度 [wx, wy, wz]
        gimbal_r = x[13]  # 云台滚转角
        gimbal_p = x[14]  # 云台俯仰角
        gimbal_y = x[15]  # 云台相对偏航角

        # --- 提取控制变量 ---
        u_wx, u_wy, u_wz, u_az = u[0], u[1], u[2], u[3]  # 机体角速度和推力加速度
        
        # --- 动力学模型 ---
        
        # 1. 位置动力学: dp/dt = v
        pos_dot = vel
        
        # 2. 四元数到旋转矩阵的转换
        # R_B_W: 从机体系到世界系的旋转矩阵
        # R_B_W = ca.vertcat(
        #     ca.horzcat(1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)),
        #     ca.horzcat(2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)),
        #     ca.horzcat(2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2))
        # )
        
        # 3. 速度动力学: dv/dt = R_B_W * [0; 0; az] - drag_coeff*v - [0; 0; G]
        # thrust_accel_W = R_B_W @ ca.vertcat(0, 0, u_az)
        thrust_accel_W = u_az * ca.vertcat(
            2*(qx*qz+qw*qy), 
            2*(qy*qz-qw*qx), 
            1-2*(qx**2+qy**2))
        vel_dot = thrust_accel_W - drag_coeff * vel - ca.vertcat(0, 0, G)
        
        # 4. 四元数动力学: dq/dt = 0.5 * Omega(omega) * q
        # 其中 Omega(omega) 是角速度的四元数矩阵形式
        omega_quat_matrix = ca.vertcat(
            ca.horzcat(0,  -wx, -wy, -wz),
            ca.horzcat(wx,  0,   wz, -wy),
            ca.horzcat(wy, -wz,  0,   wx),
            ca.horzcat(wz,  wy, -wx,  0))
        # quat_dot = 0.5 * omega_quat_matrix @ quat
        alpha_baumgarte = 1.0  # 稳定化系数
        quat_norm_sq = quat.T @ quat
        quat_dot = 0.5 * omega_quat_matrix @ quat - alpha_baumgarte * (quat_norm_sq - 1) * quat

        omega_dot = ca.vertcat(
            (u_wx - wx) / rate_tc, 
            (u_wy - wy) / rate_tc,
            (u_wz - wz) / (rate_tc * 2) )  # 角速度变化率
        
        gimbal_dot = ca.vertcat(
            -gimbal_r * gimbal_RP_coeff, 
            -gimbal_p * gimbal_RP_coeff, 
            -gimbal_y * gimbal_Y_coeff
        )  # 云台角速度
        
        # 完整的显式动力学方程
        f_expl = ca.vertcat(pos_dot, vel_dot, quat_dot, omega_dot, gimbal_dot)

        # ======== 代价函数 (Cost Function) ========

        # --- 误差项计算 ---
        drone_pos = pos
        target_pos = params[0:3]
        target_vel = params[3:6]
        los_W = target_pos - drone_pos

        # 1. 视线误差 (LOS Error) - 需要从四元数提取偏航角
        # 从四元数提取偏航角分量用于构造相机变换矩阵
        # 从四元数计算cos(psi)和sin(psi)
        cpsi = 1 - 2*(qy**2 + qz**2)  # cos(psi) 从四元数
        spsi = 2*(qw*qz + qx*qy)      # sin(psi) 从四元数

        c1 = ca.cos(gimbal_y)
        s1 = ca.sin(gimbal_y)
        c2 = ca.cos(gimbal_r)
        s2 = ca.sin(gimbal_r)
        c3 = ca.cos(gimbal_p)
        s3 = ca.sin(gimbal_p)
        
        # Gimbal(YPR Order in FLU)到Camera的旋转矩阵
        R_G_C = ca.vertcat(
            ca.horzcat( c1*c3-s1*s2*s3, s1*c3+c1*s2*s3, -c2*s3), 
            ca.horzcat(-s1*c2,          c1*c2,           s2), 
            ca.horzcat( c1*s3+s1*s2*c3, s1*s3-c1*s2*c3,  c2*c3)
        )
        # 世界到Gimbal的旋转矩阵 (仅偏航)
        R_W_G = ca.vertcat(
            ca.horzcat(cpsi, spsi, 0), 
            ca.horzcat(-spsi, cpsi, 0), 
            ca.horzcat(0, 0, 1)
        ) 
        los_C =  R_G_C @ R_W_G @ los_W
        # 处理los_C[0]可能为负值的情况（目标在相机后方）
        # 使用平滑的绝对值函数: sqrt(x^2 + eps) 近似 |x|
        los_C_x_safe = ca.fmax(ca.fabs(los_C[0]), 1.0)   # 防止除零，最小距离1.0m
        
        # 惩罚系数：基于预计到达时间(TTA)动态调整
        # 当TTA < 2s时，惩罚系数从10开始指数增长
        # 公式: z_coeff = 10 * (1 + exp(2 - TTA)) 当TTA<2时快速增大
        dist_to_target = ca.sqrt(los_W.T @ los_W + 1e-6)
        relative_vel_W = vel - target_vel
        relative_vel_sq = relative_vel_W.T @ relative_vel_W
        relative_vel = ca.sqrt(relative_vel_sq + 1e-6)
        time_to_arrival = dist_to_target / relative_vel
        penalty_coeff = 1.0 + 10.0 * ca.exp(2.0 - time_to_arrival)
        
        los_error = W_los * penalty_coeff * ca.vertcat(
            los_C[1] / los_C_x_safe, 
            los_C[2] / los_C_x_safe)
        
        # 2. 速度对齐误差 (Alignment Error)
        # drone_vel_W = vel
        # speed_sq = drone_vel_W.T @ drone_vel_W
        # speed = ca.sqrt(speed_sq + 1e-6)
        # dist_sq = los_W.T @ los_W
        # dist = ca.sqrt(dist_sq + 1e-6)
        # v_norm = drone_vel_W / speed
        # los_norm = los_W / dist
        
        v_norm = relative_vel_W / relative_vel
        los_norm = los_W / dist_to_target

        # coeff_align = ca.tanh(dist*0.25)
        # alignment_error = 1 - ca.dot(v_norm, los_norm)
        # alignment_error = ca.acos(ca.dot(v_norm, los_norm)) / ca.pi  # 使用夹角作为误差度量
        # alignment_error = W_align * coeff_align * ca.acos(ca.dot(v_norm, los_norm)) / ca.pi
        alignment_error = W_align *(1 - ca.dot(v_norm, los_norm))

        # 3. 期望接近速度误差 (Closing Speed Error)
        # cam_x_axis_W = ca.vertcat(cpsi, -spsi, 0)  # 相机X轴在世界坐标系中的表示
        # R_W_B: 从世界系到机体系的旋转矩阵
        # R_W_B = ca.vertcat(
        #     ca.horzcat(1-2*(qy**2+qz**2), 2*(qx*qy+qw*qz), 2*(qx*qz-qw*qy)),
        #     ca.horzcat(2*(qx*qy-qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz+qw*qx)),
        #     ca.horzcat(2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx), 1-2*(qx**2+qy**2))
        # )
        # relative_vel_W = drone_vel_W - params[3:6]
        # relative_vel_C = R_W_B @ relative_vel_W
        # closing_speed_error = W_closing_speed * (1 - relative_vel_C[0] / v_desired)

        # closing_speed_cam_x = ca.dot(relative_vel_W, cam_x_axis_W)
        closing_speed_x = ca.dot(relative_vel_W, los_norm)
        radial_speed_sq = relative_vel_sq - closing_speed_x**2
        # closing_speed_error = v_desired - closing_speed_cam_x
        # closing_speed_error = 1 - closing_speed_cam_x / v_desired
        # closing_speed_error = W_closing_speed * (1 - closing_speed_x / v_desired)
        
        # 当接近速度超过期望速度时(差值为负)，增加惩罚系数
        # 使用 1 + max(0, (closing_speed_x - v_desired) / v_desired) 作为超速惩罚
        speed_diff = v_desired - closing_speed_x
        # overspeed_penalty = 1.0 + 100.0 * ca.fmax(0, -speed_diff / v_desired)  # 超速时惩罚系数增大
        overspeed_penalty = 1.0 + 10.0 / (1.0 + ca.exp(20.0 * speed_diff / v_desired))  # 使用平滑函数避免不连续
        closing_speed_error = W_closing_speed * overspeed_penalty * speed_diff
        radial_speed_error = W_closing_speed * radial_speed_sq
        

        # 4. 控制量误差 (Control Effort)
        # 悬停时推力加速度应等于重力加速度
        # thrust_accel_error = u_az / G - 1
        thrust_accel_error = W_u_thrust * (u_az / G - 1)

        # 5. 角速度软约束 (新增) - 限制角速度大小以防止过激机动
        # 角速度软约束上限 (rad/s)
        rp_rate_constraint = self.constants['ROLL_PITCH_RATE_MAX']
        y_rate_constraint = self.constants['YAW_RATE_MAX']

        rates_constraint = W_rates * ca.vertcat(
            wx/rp_rate_constraint,
            wy/rp_rate_constraint,
            wz/y_rate_constraint
        )

        # wx_change = ca.fmin(1.0, ca.fabs(u_wx/(wx + 1e-6) - 1))
        # wy_change = ca.fmin(1.0, ca.fabs(u_wy/(wy + 1e-6) - 1))
        # wz_change = ca.fmin(1.0, ca.fabs(u_wz/(wz + 1e-6) - 1))

        # w_change = W_u_rates_change * ca.vertcat(
        #     ca.fmin(1.0, ca.fabs(u_wx/(wx + 1e-6) - 1)),
        #     ca.fmin(1.0, ca.fabs(u_wy/(wy + 1e-6) - 1)),
        #     ca.fmin(1.0, ca.fabs(u_wz/(wz + 1e-6) - 1)) * 2.0) # 偏航角速度变化惩罚更重
        
        u_alpha = W_u_alpha * omega_dot

        # 6. 姿态倾斜角限制 (使用四元数直接计算，避免万向节锁)
        # 根据四元数计算无人机机体Z轴与世界Z轴的夹角
        # 机体Z轴在世界坐标系中的表示: k_w = [2(xz+wy), 2(yz-wx), 1-2(x²+y²)]
        # 与世界Z轴[0,0,1]的夹角余弦值: cos(θ) = 1-2(x²+y²)
        cos_tilt_constraint = ca.cos(self.constants['ATT_LIMIT'] * 0.5)  # 最大允许倾斜角的余弦值

        # cos_tilt_angle = 1 - 2*(qx**2 + qy**2)  # 当前倾斜角的余弦值
        # tilt_error = W_tilt * ((1-cos_tilt_angle) / (1-cos_tilt_constraint))
        tilt_ratio = 2*(qx**2 + qy**2) / (1-cos_tilt_constraint)
        # tilt_error = W_tilt * tilt_ratio      # Original linear penalty
        # tilt_error = W_tilt * ca.fmax(0, tilt_ratio - 0.8) ** 2  # 只在接近边界时惩罚
        # # 或
        tilt_error = W_tilt * tilt_ratio * ca.exp(tilt_ratio - 1)  # 指数增长


        cost_expr_y = ca.vertcat(
            los_error, 
            alignment_error, 
            closing_speed_error,
            radial_speed_error,
            tilt_error,
            rates_constraint,
            u_alpha,
            thrust_accel_error
        )

        cost_expr_y_e = ca.vertcat(
            los_error, 
            alignment_error, 
            closing_speed_error,
            radial_speed_error,
            tilt_error,
            rates_constraint
        )
        
        # --- 创建ACADOS模型 ---
        self.model = AcadosModel()
        self.model.x = x
        self.model.u = u
        self.model.p = params
        self.model.f_expl_expr = f_expl
        self.model.cost_y_expr = cost_expr_y
        self.model.cost_y_expr_e = cost_expr_y_e
        self.model.name = model_name

    # ------------------------------------------------------------------------------
    # 设置 OCP (Optimal Control Problem)
    # ------------------------------------------------------------------------------
    def setup_ocp(self) -> AcadosOcp:
        """配置并返回ACADOS OCP对象。"""
        ocp = AcadosOcp('/usr/local')
        ocp.model = self.model
        
        # --- 代价函数配置 ---
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.cost.W = np.eye(self.model.cost_y_expr.shape[0])       # type: ignore
        # ocp.cost.W_e = np.eye(self.model.cost_y_expr_e.shape[0])   # type: ignore
        ocp.cost.W_e = np.diag([np.sqrt(10), np.sqrt(10), np.sqrt(5), np.sqrt(5), np.sqrt(5), np.sqrt(10),np.sqrt(5), np.sqrt(5), np.sqrt(5)])
        ocp.cost.yref = np.zeros((self.model.cost_y_expr.shape[0],))       # type: ignore
        ocp.cost.yref_e = np.zeros((self.model.cost_y_expr_e.shape[0],))   # type: ignore

        # --- 约束配置 ---
        # 控制量约束 (角速度 + 推力加速度)
        ocp.constraints.idxbu = np.arange(4)
        rp_rate_max = self.constants['ROLL_PITCH_RATE_MAX']
        y_rate_max = self.constants['YAW_RATE_MAX']
        a_min = self.constants['A_MIN']
        a_max = self.constants['A_MAX']
        ocp.constraints.lbu = np.array([-rp_rate_max, -rp_rate_max, -y_rate_max, a_min])
        ocp.constraints.ubu = np.array([ rp_rate_max,  rp_rate_max,  y_rate_max, a_max])
        
        # 状态约束 - 姿态角硬约束
        # 基于四元数的倾斜角约束：qx² + qy² ≤ (1-cos(θ_max))/2
        max_tilt = self.constants['ATT_LIMIT']
        cos_max_tilt = np.cos(max_tilt)
        tilt_constraint_bound = (1 - cos_max_tilt) / 2
        
        # 设置状态约束：约束qx和qy的取值范围
        # 状态向量: [px, py, pz, vx, vy, vz, qw, qx, qy, qz]
        # 索引:     [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9]
        ocp.constraints.idxbx = np.array([7, 8, 10, 11, 12])  # qx和qy的索引
        qxy_max = np.sqrt(tilt_constraint_bound)
        ocp.constraints.lbx = np.array([-qxy_max, -qxy_max, -rp_rate_max, -rp_rate_max, -y_rate_max])    # qx和qy的下界
        ocp.constraints.ubx = np.array([ qxy_max,  qxy_max,  rp_rate_max,  rp_rate_max,  y_rate_max])    # qx和qy的上界
        
        # print(f"[INFO] 姿态硬约束设置: qx²+qy² ≤ {tilt_constraint_bound:.4f}")
        # print(f"[INFO] qx, qy 范围: ±{qxy_max:.3f} (对应倾斜角 ≤ {np.rad2deg(max_tilt):.1f}°)")
        
        # 初始条件 - 需要转换欧拉角到四元数
        euler_initial = self.constants['DRONE_INITIAL_STATE'][6:9]  # [phi, theta, psi]
        quat_initial = self.euler_to_quaternion(euler_initial[0], euler_initial[1], euler_initial[2])
        omega_initial = self.constants['DRONE_INITIAL_STATE'][9:12]  # [wx, wy, wz]
        gimbal_initial = self.constants['DRONE_INITIAL_STATE'][12:15]  # [gr, gp, gy]

        x0_quat = np.hstack([
            self.constants['DRONE_INITIAL_STATE'][0:6],  # 位置和速度
            quat_initial,  # 四元数
            omega_initial,  # 角速度
            gimbal_initial  # 云台角度
        ])
        ocp.constraints.x0 = x0_quat
        ocp.parameter_values = np.zeros(self.model.p.shape) # type: ignore

        # --- 求解器配置 ---
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        # ocp.solver_options.sim_method_num_stages = 2 # 减少积分级数以加速 (RK2)
        # ocp.solver_options.sim_method_num_steps = 1  # 每个控制间隔仅一步积分
        ocp.solver_options.qp_solver_cond_N = 5 # 部分冷凝视界
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.N_horizon = self.constants['N_HORIZON']
        ocp.solver_options.tf = self.constants['TF']
        ocp.solver_options.ext_fun_compile_flags = '-O3'
        ocp.solver_options.nlp_solver_max_iter = 1 # SQP_RTI 只需要1次迭代
        ocp.solver_options.print_level = 0

        # 对于高频控制(50Hz+)，可以考虑：
        ocp.solver_options.qp_solver_warm_start = 2  # 热启动级别
        ocp.solver_options.regularize_method = 'PROJECT'  # QP正则化
        ocp.solver_options.globalization = 'FIXED_STEP'  # RTI固定步长

        # 如果四元数动力学需要更高精度：
        ocp.solver_options.sim_method_num_stages = 4  # RK4
        ocp.solver_options.sim_method_num_steps = 2   # 每步2次积分
        return ocp

    def generate_solver(self)->AcadosOcpSolver:
        # --- 初始化ACADOS求解器 ---
        ocp = self.setup_ocp()
        # acados_generated_path = 'c_generated_code'
        # if os.path.exists(acados_generated_path): shutil.rmtree(acados_generated_path)
        json_file = 'acados_ocp_' + self.model.name + '.json' # type: ignore
        acados_solver = AcadosOcpSolver(ocp, json_file=json_file)
        return acados_solver
    
    # ------------------------------------------------------------------------------
    # 辅助函数 (Helper Functions)
    # ------------------------------------------------------------------------------
    
    @staticmethod
    def euler_to_quaternion(phi, theta, psi):
        """将欧拉角转换为四元数 (scalar-first convention: [qw, qx, qy, qz])"""
        # ZYX 欧拉角顺序 (Yaw-Pitch-Roll)
        cy = np.cos(psi * 0.5)
        sy = np.sin(psi * 0.5)
        cp = np.cos(theta * 0.5)
        sp = np.sin(theta * 0.5)
        cr = np.cos(phi * 0.5)
        sr = np.sin(phi * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qw, qx, qy, qz])
    
    @staticmethod
    def quaternion_to_euler(quat):
        """将四元数转换为欧拉角 (ZYX顺序)"""
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        
        # Roll (phi)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        phi = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (theta)
        sinp = 2 * (qw * qy - qz * qx)
        if np.abs(sinp) >= 1:
            theta = np.copysign(np.pi / 2, sinp)  # 使用 90 度当万向节锁时
        else:
            theta = np.arcsin(sinp)
        
        # Yaw (psi)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        psi = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([phi, theta, psi])
    
    @staticmethod
    def normalize_quaternion(quat):
        """归一化四元数"""
        norm = np.linalg.norm(quat)
        if norm > 1e-8:
            return quat / norm
        else:
            # 如果四元数接近零，返回单位四元数
            return np.array([1.0, 0.0, 0.0, 0.0])
    
    @staticmethod
    def quaternion_to_rotation_matrix(quat):
        """将四元数转换为旋转矩阵"""
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
        ])
        
        return R
