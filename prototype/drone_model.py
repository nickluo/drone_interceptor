import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import casadi as ca
import os
import shutil

# ------------------------------------------------------------------------------
# 定义无人机模型 (ACADOS Drone Model)
# ------------------------------------------------------------------------------
class DroneModel:
    def __init__(self, constants):
        G = constants['G']    # 重力加速度 (m/s^2)
        self.constants = constants
        """定义ACADOS所需的无人机动力学模型和代价函数。"""
        model_name = 'drone_interceptor_mpc'
        # --- 状态和控制变量 ---
        x = ca.SX.sym('x', 9)       # type: ignore # 状态: [p, v, e]
        u = ca.SX.sym('u', 4)       # type: ignore # 控制: [w_phi, w_theta, w_psi, a_z]
        params = ca.SX.sym('p', 13)  # type: ignore # 参数: [p_target, v_target, drag_coeff, v_desired, w_cost]

        drag_coeff = params[6]
        v_desired = params[7]
        
        # --- 动力学模型 ---
        phi, theta, psi = x[6], x[7], x[8]
        dphi, dtheta, dpsi, a_z = [u[i] for i in range(4)] # a_z 是机体Z轴推力加速度
        
        pos_dot = x[3:6]
        euler_dot = ca.vertcat(dphi, dtheta, dpsi)
        
        cphi, sphi = ca.cos(phi), ca.sin(phi)
        cth, sth = ca.cos(theta), ca.sin(theta)
        cpsi, spsi = ca.cos(psi), ca.sin(psi)
        
        # 从机体系到世界系的旋转矩阵
        R_B_W = ca.vertcat(
            ca.horzcat(cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi),
            ca.horzcat(spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi),
            ca.horzcat(-sth, cth*sphi, cth*cphi)
        )

        # 推力加速度在世界坐标系的分量
        thrust_accel_W = R_B_W @ ca.vertcat(0, 0, a_z)
        # 速度导数，不再依赖质量MASS
        v_dot = thrust_accel_W - drag_coeff*pos_dot - ca.vertcat(0,0,G)
        
        # 完整的显式动力学方程
        f_expl = ca.vertcat(pos_dot, v_dot, euler_dot)

        # ======== 代价函数 (Cost Function) ========
        # --- 权重 ---
        W_los = params[8]            # 视线误差权重
        W_align = params[9]          # 速度对齐权重
        W_closing_speed = params[10]  # 接近速度权重
        W_u_rates = params[11]        # 角速度控制量权重
        W_u_thrust = params[12]      # 油门控制量权重
        
        # --- 误差项计算 ---
        drone_pos = x[0:3]
        target_pos = params[0:3]
        los_W = target_pos - drone_pos

        # 1. 视线误差 (LOS Error)
        R_W_C = ca.vertcat(ca.horzcat(cpsi,spsi,0), ca.horzcat(-spsi,cpsi,0), ca.horzcat(0,0,1))
        los_C = R_W_C @ los_W
        los_error_y = los_C[1]
        los_error_z = los_C[2]
        
        # 2. 速度对齐误差 (Alignment Error)
        drone_vel_W = x[3:6]
        speed_sq = drone_vel_W.T @ drone_vel_W
        dist_sq = los_W.T @ los_W
        v_norm = drone_vel_W / ca.sqrt(speed_sq + 1e-6)
        los_norm = los_W / ca.sqrt(dist_sq + 1e-6)
        alignment_error = 1 - ca.dot(v_norm, los_norm)

        # 3. 期望接近速度误差 (Closing Speed Error)
        cam_x_axis_W = R_W_C[0,:].T
        relative_vel_W = drone_vel_W - params[3:6]
        closing_speed_cam_x = ca.dot(relative_vel_W, cam_x_axis_W)
        closing_speed_error = v_desired - closing_speed_cam_x

        # 4. 控制量误差 (Control Effort)
        # 悬停时推力加速度应等于重力加速度
        thrust_accel_error = a_z - G

        # --- 构造代价向量 y ---
        # 非线性最小二乘代价: cost = y^T * W * y
        cost_expr_y = ca.vertcat(
            W_los * los_error_y, 
            W_los * los_error_z,
            W_align * alignment_error, 
            W_closing_speed * closing_speed_error,
            W_u_rates * dphi, 
            W_u_rates * dtheta, 
            W_u_rates * dpsi,
            W_u_thrust * thrust_accel_error
        )
        # 终端代价 (Terminal Cost)，在最后一步权重更高
        cost_expr_y_e = ca.vertcat(
            W_los * ca.sqrt(10) * los_error_y, 
            W_los * ca.sqrt(10) * los_error_z,
            W_align * ca.sqrt(5) * alignment_error, 
            W_closing_speed * ca.sqrt(5) * closing_speed_error
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
        ocp.cost.W_e = np.eye(self.model.cost_y_expr_e.shape[0])   # type: ignore
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
        # 状态约束 (Roll, Pitch)
        ocp.constraints.idxbx = np.array([6, 7])
        att_limit = self.constants['ATT_LIMIT']
        ocp.constraints.lbx = np.array([-att_limit] * 2)
        ocp.constraints.ubx = np.array([ att_limit] * 2)
        
        ocp.constraints.x0 = self.constants['DRONE_INITIAL_STATE']
        ocp.parameter_values = np.zeros(self.model.p.shape) # type: ignore

        # --- 求解器配置 ---
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.N_horizon = self.constants['N_HORIZON']
        ocp.solver_options.tf = self.constants['TF']
        ocp.solver_options.ext_fun_compile_flags = '-O3'
        ocp.solver_options.nlp_solver_max_iter = 50
        return ocp

    def generate_solver(self)->AcadosOcpSolver:
        # --- 初始化ACADOS求解器 ---
        ocp = self.setup_ocp()
        # acados_generated_path = 'c_generated_code'
        # if os.path.exists(acados_generated_path): shutil.rmtree(acados_generated_path)
        json_file = 'acados_ocp_' + self.model.name + '.json' # type: ignore
        acados_solver = AcadosOcpSolver(ocp, json_file=json_file)
        return acados_solver