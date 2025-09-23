import numpy as np

class TargetEKF:
    """使用扩展卡尔曼滤波器估计目标状态（位置和速度）。"""
    def __init__(self, dt, cam_focal_length, target_size_est, pos_dev, size_dev):
        self.focal_length = cam_focal_length
        self.target_size_est = target_size_est
        # 状态向量 [px, py, pz, vx, vy, vz]
        self.x = np.zeros(6)
        # 状态协方差矩阵
        self.P = np.eye(6) * 100
        # 状态转移矩阵 (恒定速度模型)
        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3) * dt
        # 过程噪声协方差 (假设速度在每个方向上都有随机加速度)
        q_pos = 0.05 * dt**3 / 3
        q_vel = 0.05 * dt
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
        # 测量噪声协方差
        self.R = np.diag([pos_dev**2, pos_dev**2, size_dev**2])
        self.is_initialized = False

    def initialize(self, initial_pos_est):
        """使用首次观测初始化滤波器"""
        self.x[0:3] = initial_pos_est
        self.x[3:6] = np.zeros(3)
        self.is_initialized = True
        print(f"[INFO] EKF 初始化成功，初始位置: {initial_pos_est}")

    def predict(self, dt):
        """EKF 预测步骤"""
        self.F[0:3, 3:6] = np.eye(3) * dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q


    def update(self, z, drone_pos, psi):
        """EKF 更新步骤"""
        # --- 计算雅可比矩阵 H ---
        # 测量函数 h(x) 将目标状态 x 映射到测量空间 z = [u, v, size]
        # H = dh/dx
        p_target = self.x[0:3]

        los_W = p_target - drone_pos
        cpsi, spsi = np.cos(psi), np.sin(psi)
        R_W_C = np.array([[cpsi, spsi, 0], [-spsi, cpsi, 0], [0, 0, 1]])
        los_C = R_W_C @ los_W
        
        x, y, z_c = los_C[0], los_C[1], los_C[2]
        
        if x < 1e-3: return # 避免除以零
        
        # 预测的测量值
        z_pred = np.array([
            self.focal_length * y / x,
            self.focal_length * z_c / x,
            self.focal_length * self.target_size_est / x
        ])
        
        # 计算雅可比矩阵 H (dh/d(p_target))
        # R_W_C 的行向量
        r1 = R_W_C[0, :]; r2 = R_W_C[1, :]; r3 = R_W_C[2, :]
        
        # 对 los_C 的分量求 p_target 的偏导
        dx_dp = r1; dy_dp = r2; dz_dp = r3
        
        # 对 z_pred 的分量求 p_target 的偏导 (链式法则)
        du_dp = (self.focal_length / x) * dy_dp - (self.focal_length * y / x**2) * dx_dp
        dv_dp = (self.focal_length / x) * dz_dp - (self.focal_length * z_c / x**2) * dx_dp
        dsize_dp = -(self.focal_length * self.target_size_est / x**2) * dx_dp
        
        H_pos = np.vstack([du_dp, dv_dp, dsize_dp])
        H = np.hstack([H_pos, np.zeros((3, 3))]) # 速度部分偏导为0

        # --- 卡尔曼增益、状态和协方差更新 ---
        y_residual = np.array([z['u'], z['v'], z['size']]) - z_pred
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y_residual
        self.P = (np.eye(6) - K @ H) @ self.P

    def noisy_bbox_to_pos(self, bbox, drone_pos, psi):
        """辅助函数:将单个带噪bbox转换为3D位置估计(用于绘图)"""
        if bbox['size'] < 1e-6: 
            return drone_pos
        distance = self.focal_length * self.target_size_est / bbox['size']
        cpsi, spsi = np.cos(psi), np.sin(psi)
        R_C_W = np.array([[cpsi, -spsi, 0], [spsi, cpsi, 0], [0, 0, 1]])
        los_C_dir = np.array([1, bbox['u']/self.focal_length, bbox['v']/self.focal_length])
        los_W_dir = R_C_W @ (los_C_dir / np.linalg.norm(los_C_dir))
        return drone_pos + distance * los_W_dir

