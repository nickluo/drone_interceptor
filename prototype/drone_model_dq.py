"""
基于对偶四元数 (Dual Quaternion) 的无人机目标拦截 NMPC 模型。

使用对偶四元数统一表示位姿 (旋转+平移), 结合拦截特定代价函数,
替代原有四元数MPC方法。

状态: [dq(8), omega_body(3), v_body(3), gimbal(3)] = 17维
  - dq: 对偶四元数 [qw, qx, qy, qz, dw, dx, dy, dz] 编码位置+姿态
  - omega_body: 机体角速度 [wx, wy, wz]
  - v_body: 机体线速度 [vx, vy, vz] (机体坐标系)
  - gimbal: 云台角 [roll, pitch, yaw]

控制: [u_wx, u_wy, u_wz, u_az] = 4维
  - u_wx, u_wy, u_wz: 期望机体角速度 (rad/s)
  - u_az: 推力加速度 (m/s^2)

参考: dq_nmpc 包的对偶四元数运动学与刚体动力学
"""

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import casadi as ca
import os


# ==============================================================================
# 对偶四元数数学工具 (CasADi MX 版本)
# ==============================================================================

def quat_product(p, q):
    """Hamilton乘积: p ⊗ q, scalar-first [w,x,y,z]"""
    H_p = ca.vertcat(
        ca.horzcat(p[0], -p[1], -p[2], -p[3]),
        ca.horzcat(p[1],  p[0], -p[3],  p[2]),
        ca.horzcat(p[2],  p[3],  p[0], -p[1]),
        ca.horzcat(p[3], -p[2],  p[1],  p[0]))
    return H_p @ q


def quat_conjugate(q):
    """四元数共轭"""
    return ca.vertcat(q[0], -q[1], -q[2], -q[3])


def rotate_vector(q, v):
    """用四元数旋转向量 (机体→世界): v' = q ⊗ [0,v] ⊗ q*"""
    v_quat = ca.vertcat(0, v)
    return quat_product(quat_product(q, v_quat), quat_conjugate(q))[1:4]


def rotate_vector_inverse(q, v):
    """反向旋转 (世界→机体): v' = q* ⊗ [0,v] ⊗ q"""
    v_quat = ca.vertcat(0, v)
    return quat_product(quat_product(quat_conjugate(q), v_quat), q)[1:4]


def dq_conj(dq):
    """对偶四元数共轭 (单位DQ的逆): Q* = [qr*, qd*]"""
    return ca.vertcat(
        dq[0], -dq[1], -dq[2], -dq[3],
        dq[4], -dq[5], -dq[6], -dq[7])


def dq_product(dq1, dq2):
    """对偶四元数乘积: Q1 ⊗ Q2"""
    q1r, q1d = dq1[0:4], dq1[4:8]
    q2r, q2d = dq2[0:4], dq2[4:8]
    Qr = quat_product(q1r, q2r)
    Qd = quat_product(q1r, q2d) + quat_product(q1d, q2r)
    return ca.vertcat(Qr, Qd)


def dq_from_pose(q, t):
    """从位姿构造对偶四元数: q为旋转四元数[4], t为平移向量[3]
    约定: d = 0.5 * [0,t] ⊗ q"""
    t_quat = ca.vertcat(0, t)
    d = 0.5 * quat_product(t_quat, q)
    return ca.vertcat(q, d)


def dq_get_translation(dq):
    """从对偶四元数提取平移向量: t = 2 * d ⊗ q*"""
    q = dq[0:4]
    d = dq[4:8]
    t_quat = 2 * quat_product(d, quat_conjugate(q))
    return t_quat[1:4]


def dq_get_quat(dq):
    """从对偶四元数提取旋转四元数"""
    return dq[0:4]


def dq_error(dq_desired, dq_current):
    """对偶四元数误差: Q_e = Q_d* ⊗ Q"""
    return dq_product(dq_conj(dq_desired), dq_current)


def dq_log_map(dq_e):
    """对偶四元数对数映射 (李代数)
    返回8维向量: [0, rot_err(3), 0, trans_err(3)]
    
    rot_err ~ angle/2 * axis (旋转误差, 约等于轴角表示的一半)
    trans_err ~ 0.5 * translation_error (平移误差的一半)
    """
    qr = dq_e[0:4]
    qd = dq_e[4:8]

    # 旋转部分
    vec = qr[1:4]
    norm_vec = ca.sqrt(ca.dot(vec, vec) + 1e-12)
    angle = 2 * ca.atan2(norm_vec, qr[0])
    axis_scaled = 0.5 * angle * vec / norm_vec

    # 平移部分: t_e = 2 * qd ⊗ qr*
    qr_conj = quat_conjugate(qr)
    t_e_quat = 2 * quat_product(qd, qr_conj)
    trans_scaled = 0.5 * t_e_quat[1:4]

    return ca.vertcat(0, axis_scaled, 0, trans_scaled)


def dq_kinematic_eq(dq, twist):
    """对偶四元数运动学方程: dQ/dt = 0.5 * H+(Q) * twist_dq + Baumgarte修正
    
    twist = [omega_body(3), v_body(3)]
    """
    qr = dq[0:4]
    qd = dq[4:8]

    # Baumgarte 稳定化 (保持单位范数)
    K_quat = 10
    norm_r = ca.norm_2(qr)
    quat_error = 1 - norm_r
    aux = ca.vertcat(qr * (K_quat * quat_error), qd * 0)

    # H+(Q) 矩阵
    H_r_plus = ca.vertcat(
        ca.horzcat(qr[0], -qr[1], -qr[2], -qr[3]),
        ca.horzcat(qr[1],  qr[0], -qr[3],  qr[2]),
        ca.horzcat(qr[2],  qr[3],  qr[0], -qr[1]),
        ca.horzcat(qr[3], -qr[2],  qr[1],  qr[0]))

    H_d_plus = ca.vertcat(
        ca.horzcat(qd[0], -qd[1], -qd[2], -qd[3]),
        ca.horzcat(qd[1],  qd[0], -qd[3],  qd[2]),
        ca.horzcat(qd[2],  qd[3],  qd[0], -qd[1]),
        ca.horzcat(qd[3], -qd[2],  qd[1],  qd[0]))

    zeros = ca.DM.zeros(4, 4)
    H_plus = ca.vertcat(
        ca.horzcat(H_r_plus, zeros),
        ca.horzcat(H_d_plus, H_r_plus))

    # Twist → 对偶四元数形式: [0, omega; 0, v]
    twist_dq = ca.vertcat(0, twist[0], twist[1], twist[2],
                          0, twist[3], twist[4], twist[5])

    dq_dot = 0.5 * (H_plus @ twist_dq) + aux
    return dq_dot


# ==============================================================================
# NumPy 版本工具 (用于状态转换)
# ==============================================================================

def quat_product_np(p, q):
    """NumPy四元数乘积"""
    return np.array([
        p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3],
        p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2],
        p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1],
        p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]])


def pose_to_dq_np(q, t):
    """NumPy: 位姿→对偶四元数"""
    t_q = np.array([0.0, t[0], t[1], t[2]])
    d = 0.5 * quat_product_np(t_q, q)
    return np.hstack([q, d])


def dq_to_pose_np(dq):
    """NumPy: 对偶四元数→位姿(quaternion, translation)"""
    q = dq[0:4]
    d = dq[4:8]
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    t_q = 2 * quat_product_np(d, q_conj)
    return q, t_q[1:4]


def normalize_dq_np(dq):
    """归一化对偶四元数"""
    q = dq[0:4].copy()
    d = dq[4:8].copy()
    norm_q = np.linalg.norm(q)
    if norm_q < 1e-8:
        return np.array([1., 0., 0., 0., 0., 0., 0., 0.])
    q = q / norm_q
    d = d / norm_q
    # 修正dual部分: 确保 Re(q)*Re(d).T = 0 (单位DQ约束)
    qd_dot = np.dot(q, d)
    d = d - qd_dot * q
    return np.hstack([q, d])


# ==============================================================================
# DroneModelDQ 模型类
# ==============================================================================

class DroneModelDQ:
    """基于对偶四元数的无人机拦截NMPC模型"""

    def __init__(self, constants):
        self.constants = constants
        model_name = 'drone_interceptor_dq'

        G = constants['G']
        mass = constants['MASS']
        rate_tc = constants.get('RATE_TC', 0.02)
        rp_rate_max = constants.get('ROLL_PITCH_RATE_MAX', 2.0 * np.pi)
        y_rate_max = constants.get('YAW_RATE_MAX', 1.0 * np.pi)

        # =================== 状态 (17维) ===================
        x = ca.SX.sym('x', 17)
        dq = x[0:8]          # 对偶四元数
        q  = dq[0:4]         # 旋转四元数部分
        omega = x[8:11]      # 机体角速度
        v_body = x[11:14]    # 机体线速度
        gimbal_r = x[14]
        gimbal_p = x[15]
        gimbal_y = x[16]

        # =================== 控制 (4维) ===================
        u = ca.SX.sym('u', 4)
        u_omega = u[0:3]     # 期望机体角速度 [u_wx, u_wy, u_wz] (rad/s)
        u_az = u[3]          # 推力加速度 (m/s^2)

        # =================== 参数 (19维) ===================
        params = ca.SX.sym('p', 19)
        target_pos = params[0:3]
        target_vel = params[3:6]
        v_desired = params[6]
        drag_coeff = params[7]
        # 代价权重
        w_los = params[8]         # 相机帧LOS误差权重
        w_zem = params[9]         # 零效果脱靶量(ZEM)权重
        w_closing = params[10]
        w_align = params[11]
        w_control_F = params[12]
        w_control_tau = params[13]
        w_tilt = params[14]
        w_omega = params[15]
        w_fov = params[16]        # 相机FOV栅栏代价权重
        gimbal_RP_coeff = params[17]
        gimbal_Y_coeff = params[18]

        # =================== 动力学 ===================

        e3 = ca.vertcat(0, 0, 1)

        # 1. 对偶四元数运动学
        twist = ca.vertcat(omega, v_body)
        dq_dot = dq_kinematic_eq(dq, twist)

        # 2. 角速度动力学 (一阶跟踪, 与四元数模型一致)
        omega_dot = ca.vertcat(
            (u_omega[0] - omega[0]) / rate_tc,
            (u_omega[1] - omega[1]) / rate_tc,
            (u_omega[2] - omega[2]) / (rate_tc * 2))  # 偏航时间常数2x

        # 3. 机体坐标系线速度动力学
        # dv/dt = u_az * e3 + v × omega - R^T*g - drag*v
        g_world = ca.vertcat(0, 0, G)
        g_body = rotate_vector_inverse(q, g_world)
        v_dot = u_az * e3 + ca.cross(v_body, omega) - g_body - drag_coeff * v_body

        # 4. 云台被动衰减
        gimbal_dot = ca.vertcat(
            -gimbal_r * gimbal_RP_coeff,
            -gimbal_p * gimbal_RP_coeff,
            -gimbal_y * gimbal_Y_coeff)

        f_expl = ca.vertcat(dq_dot, omega_dot, v_dot, gimbal_dot)

        # =================== 代价函数 (拦截 - 仿照原型MPC结构) ===================

        # 从DQ提取世界坐标
        drone_pos = dq_get_translation(dq)
        v_world = rotate_vector(q, v_body)

        # LOS (视线) 相关量
        los_W = target_pos - drone_pos
        dist_sq = ca.dot(los_W, los_W) + 1e-6
        dist = ca.sqrt(dist_sq)
        los_norm = los_W / dist

        # 相对运动
        rel_vel = v_world - target_vel
        rel_speed_sq = ca.dot(rel_vel, rel_vel) + 1e-6
        rel_speed = ca.sqrt(rel_speed_sq)

        # 接近速度 (沿LOS的相对速度分量)
        v_closing = ca.dot(rel_vel, los_norm)

        # 动态惩罚系数: 基于到达时间 (TTA), 限制上限
        time_to_arrival = dist / ca.fmax(v_closing, 0.5)
        penalty_coeff = 1.0 + ca.fmin(10.0 * ca.exp(2.0 - time_to_arrival), 30.0)

        # 从四元数提取偏航分量
        cpsi = 1 - 2*(q[2]**2 + q[3]**2)  # cos(yaw)
        spsi = 2*(q[0]*q[3] + q[1]*q[2])  # sin(yaw)

        # ---------- 1. 相机帧LOS误差 (Camera-frame LOS Error) ----------
        # 稳定云台相机: R_W_G (仅偏航) + R_G_C (云台角度)
        c1 = ca.cos(gimbal_y); s1 = ca.sin(gimbal_y)
        c2 = ca.cos(gimbal_r); s2 = ca.sin(gimbal_r)
        c3 = ca.cos(gimbal_p); s3 = ca.sin(gimbal_p)

        # Gimbal -> Camera 旋转矩阵
        R_G_C = ca.vertcat(
            ca.horzcat( c1*c3-s1*s2*s3, s1*c3+c1*s2*s3, -c2*s3),
            ca.horzcat(-s1*c2,           c1*c2,            s2),
            ca.horzcat( c1*s3+s1*s2*c3, s1*s3-c1*s2*c3,   c2*c3))

        # World -> Gimbal 旋转矩阵 (仅偏航)
        R_W_G = ca.vertcat(
            ca.horzcat(cpsi, spsi, 0),
            ca.horzcat(-spsi, cpsi, 0),
            ca.horzcat(0, 0, 1))

        los_C = R_G_C @ R_W_G @ los_W
        los_C_x_safe = ca.fmax(ca.fabs(los_C[0]), 1.0)

        # ---------- 2. 零效果脱靶量 (ZEM - Zero Effort Miss) ----------
        tgo_raw = dist / ca.fmax(v_closing, 0.5)
        tgo = ca.fmin(tgo_raw, 3.0)

        drone_vel_along_los = ca.dot(v_world, los_norm) * los_norm
        drone_v_perp = v_world - drone_vel_along_los

        target_vel_along_los = ca.dot(target_vel, los_norm) * los_norm
        target_v_perp = target_vel - target_vel_along_los
        target_vel_trust = ca.fmin(tgo / 2.0, 1.0)

        zem = (drone_v_perp - target_vel_trust * target_v_perp) * tgo
        # 使用较小的归一化分母, 近距离保持梯度
        zem_normalized = zem / (dist + 0.5)

        # ---------- 3. 相机FOV栅栏 ----------
        los_G_x = cpsi * los_W[0] + spsi * los_W[1]
        los_G_y = -spsi * los_W[0] + cpsi * los_W[1]
        los_G_z = los_W[2]
        dist_cam_fwd = ca.fmax(los_G_x, 0.1)
        angle_off = ca.sqrt(los_G_y**2 + los_G_z**2) / dist_cam_fwd
        fov_half = np.deg2rad(constants.get('CAM_FOV_DEG', 30.0) / 2.0)
        fov_ratio = angle_off / fov_half
        fov_dist_scale = dist_sq / (dist_sq + 15.0**2)

        # ---------- 4. 速度对齐 ----------
        v_norm = rel_vel / rel_speed
        alignment = ca.dot(v_norm, los_norm)

        # ---------- 5. 倾斜角 ----------
        cos_tilt_limit = ca.cos(constants['ATT_LIMIT'])
        tilt_ratio = 2 * (q[1]**2 + q[2]**2) / (1 - cos_tilt_limit + 1e-8)

        # =================== 代价函数 (EXTERNAL, 精确Hessian) ===================

        # 距离自适应因子: 近距离降低横向修正 (ZEM)
        # 原因: Bearing-Angle EKF的位置误差在近距离时约2-5m,
        # ZEM在近距离用penalty_coeff放大横向修正会导致无人机飞向错误方向
        zem_dist_scale = dist_sq / (dist_sq + 10.0**2)  # 10m处50%, 3m处8%

        # 1. 相机帧LOS误差 (保留penalty_coeff: 始终需要跟踪目标)
        los_cost = w_los * penalty_coeff * (
            (los_C[1] / los_C_x_safe)**2 + (los_C[2] / los_C_x_safe)**2)

        # 2. ZEM (近距离衰减: 避免基于不准确估计的强横向修正)
        zem_cost = w_zem * zem_dist_scale * penalty_coeff * ca.dot(zem_normalized, zem_normalized)

        # 3. FOV栅栏 (近距离减弱, 远距离主导)
        fov_cost = w_fov * fov_dist_scale * fov_ratio**4

        # 4. 接近速度误差
        closing_cost = w_closing * (v_desired - v_closing)**2

        # 5. 速度对齐
        align_cost = w_align * (1 - alignment)**2

        # 6. 控制正则化
        u_hover = ca.vertcat(0, 0, 0, G)
        u_error = u - u_hover
        R_u = ca.diag(ca.vertcat(
            w_control_tau / rp_rate_max**2,
            w_control_tau / rp_rate_max**2,
            w_control_tau / y_rate_max**2,
            w_control_F / G**2))
        control_cost = u_error.T @ R_u @ u_error

        # 7. 倾斜角软约束
        tilt_cost = w_tilt * tilt_ratio * ca.exp(tilt_ratio - 1)

        # 8. 角速度正则化
        omega_cost = w_omega * ca.dot(omega, omega)

        # 总代价
        cost_stage = (los_cost + zem_cost + fov_cost + closing_cost
                      + align_cost + control_cost + tilt_cost + omega_cost)
        # 终端代价: 强调拦截精度
        cost_terminal = (3 * (los_cost + zem_cost + fov_cost)
                         + 2 * (closing_cost + align_cost)
                         + tilt_cost + omega_cost)

        # =================== 构建 ACADOS 模型 ===================
        self.model = AcadosModel()
        self.model.x = x
        self.model.u = u
        self.model.p = params
        self.model.f_expl_expr = f_expl
        self.model.name = model_name

        # EXTERNAL代价 (提供精确Hessian, 近距离收敛性更好)
        self.model.cost_expr_ext_cost = cost_stage
        self.model.cost_expr_ext_cost_e = cost_terminal

    # ------------------------------------------------------------------
    # OCP 配置
    # ------------------------------------------------------------------
    def setup_ocp(self) -> AcadosOcp:
        ocp = AcadosOcp()
        ocp.model = self.model

        # 代价类型: EXTERNAL (精确Hessian + MIRROR正则化)
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # 参数默认值
        ocp.parameter_values = np.zeros(self.model.p.size()[0])

        # 控制约束 [u_wx, u_wy, u_wz, u_az]
        rp_rate_max = self.constants.get('ROLL_PITCH_RATE_MAX', 2.0 * np.pi)
        y_rate_max = self.constants.get('YAW_RATE_MAX', 1.0 * np.pi)
        a_min = self.constants['A_MIN']
        a_max = self.constants['A_MAX']

        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.lbu = np.array([-rp_rate_max, -rp_rate_max, -y_rate_max, a_min])
        ocp.constraints.ubu = np.array([ rp_rate_max,  rp_rate_max,  y_rate_max, a_max])

        # 状态约束: 限制qx, qy (倾斜角硬约束) + 角速度约束
        max_tilt = self.constants['ATT_LIMIT']
        qxy_max = np.sqrt((1 - np.cos(max_tilt)) / 2)
        ocp.constraints.idxbx = np.array([1, 2, 8, 9, 10])
        ocp.constraints.lbx = np.array([-qxy_max, -qxy_max, -rp_rate_max, -rp_rate_max, -y_rate_max])
        ocp.constraints.ubx = np.array([ qxy_max,  qxy_max,  rp_rate_max,  rp_rate_max,  y_rate_max])

        # 初始条件
        x0 = self._get_initial_state()
        ocp.constraints.x0 = x0

        # 求解器配置
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.N_horizon = self.constants['N_HORIZON']
        ocp.solver_options.tf = self.constants['TF']
        ocp.solver_options.ext_fun_compile_flags = '-O3'
        ocp.solver_options.nlp_solver_max_iter = 1
        ocp.solver_options.print_level = 0

        # 热启动与正则化
        ocp.solver_options.qp_solver_cond_N = 5
        ocp.solver_options.qp_solver_warm_start = 2
        ocp.solver_options.regularize_method = 'MIRROR'
        ocp.solver_options.globalization = 'FIXED_STEP'
        ocp.solver_options.levenberg_marquardt = 5.0

        # ERK积分器设置 (显式RK4, Baumgarte稳定化保证DQ范数)
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 2

        return ocp

    def generate_solver(self) -> AcadosOcpSolver:
        ocp = self.setup_ocp()
        json_file = 'acados_ocp_' + self.model.name + '.json'
        return AcadosOcpSolver(ocp, json_file=json_file)

    def _get_initial_state(self):
        ds = self.constants['DRONE_INITIAL_STATE']
        pos = ds[0:3]
        euler = ds[6:9]
        omega = ds[9:12]
        gimbal = ds[12:15]

        q = self.euler_to_quaternion(euler[0], euler[1], euler[2])
        dq_val = pose_to_dq_np(q, pos)

        # 世界速度 → 机体速度
        v_world = ds[3:6]
        R_mat = self.quaternion_to_rotation_matrix(q)
        v_body = R_mat.T @ v_world

        return np.hstack([dq_val, omega, v_body, gimbal])

    # ------------------------------------------------------------------
    # 静态辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def euler_to_quaternion(phi, theta, psi):
        """欧拉角 → 四元数 [qw, qx, qy, qz]"""
        cy, sy = np.cos(psi * 0.5), np.sin(psi * 0.5)
        cp, sp = np.cos(theta * 0.5), np.sin(theta * 0.5)
        cr, sr = np.cos(phi * 0.5), np.sin(phi * 0.5)
        return np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy])

    @staticmethod
    def quaternion_to_euler(q):
        """四元数 → 欧拉角 [phi, theta, psi]"""
        qw, qx, qy, qz = q
        phi = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        sinp = 2*(qw*qy - qz*qx)
        theta = np.arcsin(np.clip(sinp, -1, 1))
        psi = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        return np.array([phi, theta, psi])

    @staticmethod
    def normalize_quaternion(q):
        norm = np.linalg.norm(q)
        return q / norm if norm > 1e-8 else np.array([1., 0., 0., 0.])

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        qw, qx, qy, qz = q
        return np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]])
