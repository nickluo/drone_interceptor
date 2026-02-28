import numpy as np

class TargetEKF:
    """
    基于扩展卡尔曼滤波器 (EKF) 的 3D 目标跟踪器 — Bearing-Angle 方法

    采用恒速运动 (CV) 模型 + 目标宽度 W 自适应估计,
    替代传统方法中对目标尺寸的固定假设，提高测距精度与鲁棒性。

    状态向量: [px, py, pz, vx, vy, vz, W]^T (7维)
      - px, py, pz: 目标绝对位置 (世界坐标系)
      - vx, vy, vz: 目标绝对速度
      - W: 目标物理宽度 (自适应在线估计)

    测量向量: [方位角(azimuth), 俯仰角(elevation), 视角(visual_angle)]^T (3维)
      - 方位角和俯仰角提供方向约束
      - 视角 α = 2·arctan(W/(2·d)) 与 W 联合估计提供距离可观测性

    参考: vs1_3d.py 中的 TargetTrackingEKF_3D bearing_angle 测量模型
    """
    def __init__(self, dt, cam_focal_length, pos_dev, size_dev,
                 initial_width_est,
                 range_init_bias=0.5,
                 q_accel=1.0, q_W=0.002, alpha_R_inflate=2.0):
        self.focal_length = cam_focal_length
        self.dt_nominal = dt

        # 从像素噪声推导角度域测量噪声
        self.sigma_az = pos_dev / cam_focal_length
        self.sigma_el = pos_dev / cam_focal_length
        self.sigma_alpha = size_dev / cam_focal_length

        # 状态维度: [px, py, pz, vx, vy, vz, W]
        self.n = 7
        self.x = np.zeros(self.n)

        # 初始状态协方差
        # 使用 range_init_bias 后, 初始距离误差显著降低 (约 σ=30m 量级),
        # P_pos 相应设为 900;
        # P_W=4 (σ=2m) — 覆盖 W_init 可能偏离真值 0.5~2 倍的不确定性
        self.P = np.diag([
            900.0, 900.0, 900.0,    # 位置 σ≈30m
            100.0, 100.0, 100.0,    # 速度 σ=10m/s
            4.0                     # 宽度 σ=2m
        ])

        # 运动学过程噪声参数 (白噪声加速度谱密度)
        # Q 矩阵在 predict() 中按照运动学 CV 模型构建,
        # 包含位置-速度交叉协方差项, 物理上更准确
        self.q_accel = q_accel   # (m²/s⁴)/Hz — 吸收未建模加速度
        self.q_W = q_W           # (m²/s) — W 随机游走

        # 测量噪声协方差
        # 视角测量噪声额外膨胀, 补偿 bbox 尺寸估计与线性化误差
        self.R = np.diag([
            self.sigma_az**2,
            self.sigma_el**2,
            self.sigma_alpha**2 * alpha_R_inflate
        ])

        self.initial_width_est = initial_width_est
        self.range_init_bias = range_init_bias
        self.is_initialized = False
    
    def initialize(self, bbox, drone_pos, psi, gimbal_r, gimbal_p, gimbal_y):
        """使用首次观测初始化滤波器"""
        initial_pos_est = drone_pos
        if bbox['size'] > 1e-6:
            # range_init_bias < 1 打破 W/d 耦合: 初始位置偏近, 状态 W 保持 W_init,
            # 使 α_pred > α_meas → 创新为负 → EKF 驱动 W 向下修正
            distance = self.range_init_bias * self.focal_length * self.initial_width_est / bbox['size']
            R_WC = self._compute_R_WC(psi, gimbal_r, gimbal_p, gimbal_y)
            R_CW = R_WC.T
            los_C_dir = np.array([1, bbox['u'] / self.focal_length, bbox['v'] / self.focal_length])
            los_W_dir = R_CW @ (los_C_dir / np.linalg.norm(los_C_dir))
            initial_pos_est += distance * los_W_dir

        self.x[0:3] = initial_pos_est
        self.x[3:6] = np.zeros(3)   # 速度初始化为零
        self.x[6] = self.initial_width_est
        self.is_initialized = True
        print(f"[INFO] Bearing-Angle EKF 初始化: pos={initial_pos_est}, W_init={self.x[6]:.2f}m")

    def predict(self, dt):
        """EKF 预测步骤 (恒速 CV 模型, 运动学过程噪声)"""
        F = np.eye(self.n)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # 运动学过程噪声: 白噪声加速度积分
        # 位置-速度交叉协方差保证物理一致性, 避免独立对角 Q 导致的滤波器不稳定
        qa = self.q_accel
        dt2 = dt * dt
        dt3 = dt2 * dt

        Q = np.zeros((self.n, self.n))
        for i in range(3):
            Q[i, i] = qa * dt3 / 3.0        # 位置方差
            Q[i, i+3] = qa * dt2 / 2.0      # 位置-速度协方差
            Q[i+3, i] = qa * dt2 / 2.0      # 速度-位置协方差
            Q[i+3, i+3] = qa * dt            # 速度方差
        Q[6, 6] = self.q_W * dt              # W 随机游走

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z_bbox, drone_pos, psi, gimbal_r, gimbal_p, gimbal_y):
        """
        EKF 更新步骤 — Bearing-Angle 测量模型

        Args:
            z_bbox: 含 'azimuth', 'elevation', 'visual_angle' 键的测量字典
            drone_pos: 无人机位置 [x, y, z]
            psi: 无人机偏航角
            gimbal_r, gimbal_p, gimbal_y: 云台欧拉角
        """
        p_target = self.x[0:3]
        W = self.x[6]

        # 计算世界坐标系到相机坐标系的旋转矩阵
        R_WC = self._compute_R_WC(psi, gimbal_r, gimbal_p, gimbal_y)

        # 计算相机坐标系下的视线向量
        los_W = p_target - drone_pos
        los_C = R_WC @ los_W
        x_c, y_c, z_c = los_C[0], los_C[1], los_C[2]

        # 目标在相机后方 → 跳过更新
        if x_c < 1.0:
            return

        # 奇异性保护: 距离过近时推开状态 (参考 vs1_3d.py)
        r = np.linalg.norm(los_W)
        if r < 5.0:
            scale = 5.0 / (r + 1e-5)
            self.x[0:3] = drone_pos + los_W * scale
            los_W = self.x[0:3] - drone_pos
            los_C = R_WC @ los_W
            x_c, y_c, z_c = los_C[0], los_C[1], los_C[2]

        # === 预测的测量值 ===
        az_pred = np.arctan2(-z_c, x_c)                     # 方位角
        el_pred = np.arctan2(-y_c, x_c)                     # 俯仰角
        alpha_pred = 2.0 * np.arctan(W / (2.0 * x_c))      # 视角

        z_pred = np.array([az_pred, el_pred, alpha_pred])
        z_meas = np.array([z_bbox['azimuth'], z_bbox['elevation'], z_bbox['visual_angle']])

        # 计算残差并规范化角度到 [-π, π]
        y = z_meas - z_pred
        y[0] = (y[0] + np.pi) % (2 * np.pi) - np.pi
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi

        # === 3D 雅可比矩阵 H (3×7) ===
        H = np.zeros((3, self.n))

        # 1. 方位角 azimuth = arctan2(-z_c, x_c)
        den_az = x_c**2 + z_c**2
        if den_az < 1e-10: den_az = 1e-10
        daz_dlosC = np.array([z_c / den_az, 0.0, -x_c / den_az])
        H[0, 0:3] = daz_dlosC @ R_WC

        # 2. 俯仰角 elevation = arctan2(-y_c, x_c)
        den_el = x_c**2 + y_c**2
        if den_el < 1e-10: den_el = 1e-10
        del_dlosC = np.array([y_c / den_el, -x_c / den_el, 0.0])
        H[1, 0:3] = del_dlosC @ R_WC

        # 3. 视角 alpha = 2·arctan(W/(2·x_c))
        den_alpha = 4.0 * x_c**2 + W**2
        H[2, 0:3] = (-4.0 * W / den_alpha) * R_WC[0, :]   # d(α)/d(p)
        H[2, 6] = 4.0 * x_c / den_alpha                     # d(α)/d(W)

        # === 卡尔曼增益与状态更新 ===
        S = H @ self.P @ H.T + self.R

        # 创新门检测: 拒绝线性化误差过大的异常更新 (chi² 阈值, 3-DOF)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        chi2 = y @ S_inv @ y
        if chi2 > 20.0:  # 99.99% 门限, 仅拒绝极端异常值
            return

        K = self.P @ H.T @ S_inv

        self.x = self.x + K @ y

        # Joseph 形式更新协方差 (数值更稳定)
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # 物理约束: 目标宽度 W ≥ 0.1m
        if self.x[6] < 0.1:
            self.x[6] = 0.1
    
    def get_target_position(self):
        """返回当前估计的目标位置"""
        return self.x[0:3]

    def _compute_R_WC(self, psi, gimbal_r, gimbal_p, gimbal_y):
        """计算世界坐标系到相机坐标系的旋转矩阵 R_WC = R_G_C · R_W_G"""
        cpsi, spsi = np.cos(psi), np.sin(psi)
        R_W_G = np.array([[cpsi, spsi, 0], [-spsi, cpsi, 0], [0, 0, 1]])

        c1, s1 = np.cos(gimbal_y), np.sin(gimbal_y)
        c2, s2 = np.cos(gimbal_r), np.sin(gimbal_r)
        c3, s3 = np.cos(gimbal_p), np.sin(gimbal_p)

        R_G_C = np.array([
            [c1*c3 - s1*s2*s3, s1*c3 + c1*s2*s3, -c2*s3],
            [-s1*c2,            c1*c2,              s2],
            [c1*s3 + s1*s2*c3, s1*s3 - c1*s2*c3,   c2*c3]
        ])

        return R_G_C @ R_W_G

