#pragma once

#include <Eigen/Dense>

// Define fixed-size vector types for performance
using Vector5d = Eigen::Matrix<double, 5, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Vector9d = Eigen::Matrix<double, 9, 1>;
using Vector13d = Eigen::Matrix<double, 13, 1>;
using Vector16d = Eigen::Matrix<double, 16, 1>;

// ============================================================================
// EMAFilter Class (from exp_filter.py)
// ============================================================================
class EMAFilter {
public:
    EMAFilter(int n, int channels) : channels_(channels) {
        if (n <= 0 || channels <= 0) {
            throw std::invalid_argument("n and channels must be positive");
        }
        alpha_ = 2.0 / (n + 1);
        data_ = Eigen::VectorXd::Zero(channels);
        initialized_ = std::vector<uint8_t>(channels, false);
    }

    void reset()
    {
        initialized_ = std::vector<uint8_t>(channels_, false);
    }

    Eigen::VectorXd filter(const Eigen::VectorXd& values) {
        if (values.size() != channels_) {
            throw std::runtime_error("Input array size does not match filter channels");
        }
        for (int i = 0; i < channels_; ++i) {
            if (!initialized_[i]) {
                data_[i] = values[i];
                initialized_[i] = true;
            } else {
                data_[i] = alpha_ * values[i] + (1.0 - alpha_) * data_[i];
            }
        }
        return data_;
    }

private:
    double alpha_;
    int channels_;
    Eigen::VectorXd data_;
    std::vector<uint8_t> initialized_;
};

// ============================================================================
// TargetEKF Class (from target_ekf.py)
// ============================================================================
class TargetEKF {
public:
    struct Bbox {
        double u, v, size;
    };

    TargetEKF(double dt, double focal_length, double target_size_est, double pos_dev, double size_dev)
        : focal_length_(focal_length), target_size_est_(target_size_est), is_initialized_(false) {
        
        // 状态向量 [px, py, pz, vx, vy, vz]
        x_ = Vector6d::Zero();
        
        // 状态协方差矩阵
        P_ = Eigen::MatrixXd::Identity(6, 6) * 100.0;
        
        // 状态转移矩阵 (将在 predict 中随 dt 更新)
        F_ = Eigen::MatrixXd::Identity(6, 6);
        F_.block<3,3>(0, 3) = Eigen::Matrix3d::Identity() * dt;

        // 过程噪声协方差 (基于连续白噪声加速度模型)
        double q_pos = 0.05 * std::pow(dt, 3) / 3.0;
        double q_vel = 0.05 * dt;
        Q_ = Eigen::MatrixXd::Zero(6, 6);
        Q_.diagonal() << q_pos, q_pos, q_pos, q_vel, q_vel, q_vel;

        // 测量噪声协方差 (u, v, size)
        R_ = Eigen::Matrix3d::Zero();
        R_.diagonal() << pos_dev * pos_dev, pos_dev * pos_dev, size_dev * size_dev;
    }

    void initialize(const Eigen::Vector3d& initial_pos_est) {
        x_.head<3>() = initial_pos_est;
        x_.tail<3>() = Eigen::Vector3d::Zero(); // 初始速度为0
        is_initialized_ = true;
    }

    void predict(double dt) {
        if (!is_initialized_) return;
        
        // 更新状态转移矩阵
        F_.block<3,3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
        
        // 预测状态
        x_ = F_ * x_;
        
        // 预测协方差
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    void update(const Bbox& z, const Eigen::Vector3d& drone_pos, double psi) {
        if (!is_initialized_) return;
        
        // --- 计算雅可比矩阵 H ---
        Eigen::Vector3d p_target = x_.head<3>();
        Eigen::Vector3d los_W = p_target - drone_pos;

        double cpsi = std::cos(psi);
        double spsi = std::sin(psi);
        Eigen::Matrix3d R_W_C;
        R_W_C <<  cpsi, spsi, 0,
                 -spsi, cpsi, 0,
                     0,    0, 1;
        
        Eigen::Vector3d los_C = R_W_C * los_W;
        double x_c = los_C(0), y_c = los_C(1), z_c = los_C(2);

        if (x_c < 1e-3) return; // 避免除以零

        // 预测的测量值
        Eigen::Vector3d z_pred;
        z_pred << focal_length_ * y_c / x_c,
                  focal_length_ * z_c / x_c,
                  focal_length_ * target_size_est_ / x_c;

        // 计算雅可比矩阵 H (dh/d(p_target))
        Eigen::Vector3d r1 = R_W_C.row(0);
        Eigen::Vector3d r2 = R_W_C.row(1);
        Eigen::Vector3d r3 = R_W_C.row(2);

        Eigen::Vector3d dx_dp = r1;
        Eigen::Vector3d dy_dp = r2;
        Eigen::Vector3d dz_dp = r3;

        Eigen::RowVector3d du_dp = (focal_length_ / x_c) * dy_dp.transpose() - (focal_length_ * y_c / (x_c * x_c)) * dx_dp.transpose();
        Eigen::RowVector3d dv_dp = (focal_length_ / x_c) * dz_dp.transpose() - (focal_length_ * z_c / (x_c * x_c)) * dx_dp.transpose();
        Eigen::RowVector3d dsize_dp = -(focal_length_ * target_size_est_ / (x_c * x_c)) * dx_dp.transpose();

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 6);
        H.block<1,3>(0,0) = du_dp;
        H.block<1,3>(1,0) = dv_dp;
        H.block<1,3>(2,0) = dsize_dp;

        // --- 卡尔曼增益、状态和协方差更新 ---
        Eigen::Vector3d y_residual;
        y_residual << -z.u, -z.v, z.size;
        y_residual -= z_pred;

        Eigen::Matrix3d S = H * P_ * H.transpose() + R_;
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
        
        x_ = x_ + K * y_residual;
        P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H) * P_;
    }
    
    Eigen::Vector3d noisy_bbox_to_pos(const Bbox& bbox, const Eigen::Vector3d& drone_pos, double psi) {
        // 参考 Python 版本的注释
        // cpsi, spsi = np.cos(psi), np.sin(psi)
        // R_W_C = np.array([[cpsi, spsi, 0], [-spsi, cpsi, 0], [0, 0, 1]])
        // # 反推距离
        // est_dist_cam_x = self.f * self.target_size / bbox['size_x']
        // est_los_C_y = -bbox['u'] * est_dist_cam_x / self.f
        // est_los_C_z = -bbox['v'] * est_dist_cam_x / self.f
        // est_los_C = np.array([est_dist_cam_x, est_los_C_y, est_los_C_z])

        // # 转回世界坐标系
        // R_C_W = R_W_C.T
        // est_los_W = R_C_W @ est_los_C
        // est_target_pos = drone_pos + est_los_W

        // if (bbox.size < 1e-6) {
        //     return drone_pos;
        // }
        // printf("drone_pos: [%f, %f, %f], psi: %f, bbox: [u=%f, v=%f, size=%f]\n", 
        //        drone_pos.x(), drone_pos.y(), drone_pos.z(), psi, bbox.u, bbox.v, bbox.size);
        // printf("focal_length_: %f, target_size_est_: %f\n", focal_length_, target_size_est_);

        double est_dist_cam_x = focal_length_ * target_size_est_ / bbox.size;
        double cpsi = std::cos(psi), spsi = std::sin(psi);
        Eigen::Matrix3d R_C_W;
        R_C_W << cpsi, -spsi, 0,
                 spsi,  cpsi, 0,
                    0,     0, 1;
        double est_los_C_y = -bbox.u * est_dist_cam_x / focal_length_;
        double est_los_C_z = -bbox.v * est_dist_cam_x / focal_length_;
        Eigen::Vector3d est_los_C(est_dist_cam_x, est_los_C_y, est_los_C_z);
        Eigen::Vector3d est_los_W = R_C_W * est_los_C;
        Eigen::Vector3d est_target_pos = drone_pos + est_los_W;
        return est_target_pos;
    }

    bool is_initialized() const { return is_initialized_; }
    const Vector6d& get_state() const { return x_; }

private:
    Vector6d x_;
    Eigen::MatrixXd P_;
    Eigen::MatrixXd F_;
    Eigen::MatrixXd Q_;
    Eigen::Matrix3d R_;
    double focal_length_;
    double target_size_est_;
    bool is_initialized_;
};

// Helper to convert Eigen quaternion to Euler angles
Eigen::Vector3d to_euler_angles(const Eigen::Quaterniond& q) {
    Eigen::Vector3d angles;
    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q.w() * q.x() + q.y() * q.z());
    double cosr_cosp = 1 - 2 * (q.x() * q.x() + q.y() * q.y());
    angles.x() = std::atan2(sinr_cosp, cosr_cosp);
    // pitch (y-axis rotation)
    double sinp = 2 * (q.w() * q.y() - q.z() * q.x());
    if (std::abs(sinp) >= 1)
        angles.y() = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        angles.y() = std::asin(sinp);
    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
    double cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
    angles.z() = std::atan2(siny_cosp, cosy_cosp);
    return angles;
}

// Helper to convert Euler angles to quaternion (scalar-first convention: [qw, qx, qy, qz])
Eigen::Vector4d euler_to_quaternion(double phi, double theta, double psi) {
    // ZYX Euler angle order (Yaw-Pitch-Roll)
    double cy = std::cos(psi * 0.5);
    double sy = std::sin(psi * 0.5);
    double cp = std::cos(theta * 0.5);
    double sp = std::sin(theta * 0.5);
    double cr = std::cos(phi * 0.5);
    double sr = std::sin(phi * 0.5);
    Eigen::Vector4d quat;
    quat(0) = cr * cp * cy + sr * sp * sy; // qw
    quat(1) = sr * cp * cy - cr * sp * sy; // qx
    quat(2) = cr * sp * cy + sr * cp * sy; // qy
    quat(3) = cr * cp * sy - sr * sp * cy; // qz
    return quat;
}

// Helper to normalize quaternion
Eigen::Vector4d normalize_quaternion(const Eigen::Vector4d& quat) {
    double norm = quat.norm();
    if (norm > 1e-8) {
        return quat / norm;
    } else {
        // If quaternion is near zero, return identity quaternion
        return Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    }
}

// Helper to convert quaternion to rotation matrix
Eigen::Matrix3d quaternion_to_rotation_matrix(const Eigen::Vector4d& quat) {
    double qw = quat(0), qx = quat(1), qy = quat(2), qz = quat(3);
    Eigen::Matrix3d R;
    R << 1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy),
         2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx),
         2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy);
    return R;
}

// Helper to get yaw angle from quaternion
double get_yaw_from_quaternion(const Eigen::Vector4d& quat) {
    double qw = quat(0), qx = quat(1), qy = quat(2), qz = quat(3);
    double siny_cosp = 2 * (qw * qz + qx * qy);
    double cosy_cosp = 1 - 2 * (qy * qy + qz * qz);
    return std::atan2(siny_cosp, cosy_cosp);
}

// 四元数积分函数
Eigen::Vector4d integrate_quaternion(const Eigen::Vector4d& quat, const Eigen::Vector3d& body_rates, double dt) {
    // 使用四元数微分方程进行积分
    double p = body_rates(0), q = body_rates(1), r = body_rates(2);
    
    // 四元数导数矩阵
    Eigen::Matrix4d omega_matrix;
    omega_matrix <<  0, -p, -q, -r,
                        p,  0,  r, -q,
                        q, -r,  0,  p,
                        r,  q, -p,  0;
    
    // 一阶积分近似
    Eigen::Vector4d quat_dot = 0.5 * omega_matrix * quat;
    return quat + quat_dot * dt;
}
