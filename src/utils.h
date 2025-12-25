#pragma once

#include <cmath>
#include <atomic>
#include <algorithm>
#include <stdexcept>
#include <Eigen/Dense>

// Define fixed-size vector/matrix types for performance
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3x6d = Eigen::Matrix<double, 3, 6>;
using RowVector3d = Eigen::Matrix<double, 1, 3>;

// ============================================================================
// 任务状态枚举和线程安全状态机
// ============================================================================
enum class MissionState : uint8_t {
    IDLE,           // 空闲状态，等待任务启动
    INITIALIZING,   // 初始化中，等待传感器数据
    TRACKING,       // 跟踪目标中
    BLIND_FLIGHT,   // 目标丢失，盲飞状态
    FAILSAFE,       // 故障安全模式
    COMPLETED,      // 任务完成
    ABORTED         // 任务中止
};

// 求解器恢复动作
enum class SolverRecoveryAction : uint8_t {
    CONTINUE,       // 继续使用上一次解
    RESET_WARM,     // 重置热启动
    FAILSAFE,       // 进入failsafe模式
    ABORT           // 终止任务
};

class MissionStateMachine {
public:
    MissionStateMachine() : state_(MissionState::IDLE) {}
    
    // 原子地获取当前状态
    MissionState get() const { 
        return state_.load(std::memory_order_acquire); 
    }
    
    // 原子地设置状态
    void set(MissionState new_state) { 
        state_.store(new_state, std::memory_order_release); 
    }
    
    // 尝试从from状态转换到to状态，成功返回true
    bool transition(MissionState from, MissionState to) {
        MissionState expected = from;
        return state_.compare_exchange_strong(expected, to, 
            std::memory_order_acq_rel, std::memory_order_acquire);
    }
    
    // 检查是否可以进行控制
    bool can_control() const {
        MissionState s = get();
        return s == MissionState::TRACKING || s == MissionState::BLIND_FLIGHT;
    }
    
    // 检查任务是否活跃
    bool is_active() const {
        MissionState s = get();
        return s == MissionState::INITIALIZING || 
               s == MissionState::TRACKING || 
               s == MissionState::BLIND_FLIGHT;
    }
    
    // 检查是否已终止
    bool is_terminated() const {
        MissionState s = get();
        return s == MissionState::COMPLETED || 
               s == MissionState::ABORTED || 
               s == MissionState::IDLE;
    }
    
    // 获取状态名称（用于日志）
    static const char* state_name(MissionState s) {
        switch (s) {
            case MissionState::IDLE: return "IDLE";
            case MissionState::INITIALIZING: return "INITIALIZING";
            case MissionState::TRACKING: return "TRACKING";
            case MissionState::BLIND_FLIGHT: return "BLIND_FLIGHT";
            case MissionState::FAILSAFE: return "FAILSAFE";
            case MissionState::COMPLETED: return "COMPLETED";
            case MissionState::ABORTED: return "ABORTED";
            default: return "UNKNOWN";
        }
    }

private:
    std::atomic<MissionState> state_;
};

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

    TargetEKF(double dt, double focal_length, double target_size_est, double pos_dev, double size_dev,
              double perspective_compensation = 1.0)
        : focal_length_(focal_length), target_size_est_(target_size_est), is_initialized_(false),
          consecutive_update_failures_(0), max_covariance_threshold_(1e6),
          perspective_compensation_(perspective_compensation) {
        
        // 状态向量 [px, py, pz, vx, vy, vz]
        x_ = Vector6d::Zero();
        
        // 状态协方差矩阵 (使用固定大小矩阵)
        P_ = Matrix6d::Identity() * 100.0;
        
        // 状态转移矩阵 (将在 predict 中随 dt 更新)
        F_ = Matrix6d::Identity();
        F_.block<3,3>(0, 3) = Eigen::Matrix3d::Identity() * dt;

        // 过程噪声协方差 (基于连续白噪声加速度模型)
        double q_pos = 0.05 * std::pow(dt, 3) / 3.0;
        double q_vel = 0.05 * dt;
        Q_ = Matrix6d::Zero();
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

    // void update(const Bbox& z, const Eigen::Vector3d& drone_pos, double psi) {
    //     if (!is_initialized_) return;
        
    //     // --- 计算雅可比矩阵 H ---
    //     Eigen::Vector3d p_target = x_.head<3>();
    //     Eigen::Vector3d los_W = p_target - drone_pos;

    //     double cpsi = std::cos(psi);
    //     double spsi = std::sin(psi);
    //     Eigen::Matrix3d R_W_C;
    //     R_W_C <<  cpsi, spsi, 0,
    //              -spsi, cpsi, 0,
    //                  0,    0, 1;
        
    //     Eigen::Vector3d los_C = R_W_C * los_W;
    //     double x_c = los_C(0), y_c = los_C(1), z_c = los_C(2);

    //     if (x_c < 1e-3) return; // 避免除以零

    //     // 预测的测量值
    //     Eigen::Vector3d z_pred;
    //     z_pred << focal_length_ * y_c / x_c,
    //               focal_length_ * z_c / x_c,
    //               focal_length_ * target_size_est_ / x_c;

    //     // 计算雅可比矩阵 H (dh/d(p_target))
    //     Eigen::Vector3d r1 = R_W_C.row(0);
    //     Eigen::Vector3d r2 = R_W_C.row(1);
    //     Eigen::Vector3d r3 = R_W_C.row(2);

    //     Eigen::Vector3d dx_dp = r1;
    //     Eigen::Vector3d dy_dp = r2;
    //     Eigen::Vector3d dz_dp = r3;

    //     Eigen::RowVector3d du_dp = (focal_length_ / x_c) * dy_dp.transpose() - (focal_length_ * y_c / (x_c * x_c)) * dx_dp.transpose();
    //     Eigen::RowVector3d dv_dp = (focal_length_ / x_c) * dz_dp.transpose() - (focal_length_ * z_c / (x_c * x_c)) * dx_dp.transpose();
    //     Eigen::RowVector3d dsize_dp = -(focal_length_ * target_size_est_ / (x_c * x_c)) * dx_dp.transpose();

    //     Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 6);
    //     H.block<1,3>(0,0) = du_dp;
    //     H.block<1,3>(1,0) = dv_dp;
    //     H.block<1,3>(2,0) = dsize_dp;

    //     // --- 卡尔曼增益、状态和协方差更新 ---
    //     Eigen::Vector3d y_residual;
    //     y_residual << -z.u, -z.v, z.size;
    //     y_residual -= z_pred;

    //     Eigen::Matrix3d S = H * P_ * H.transpose() + R_;
    //     Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
        
    //     x_ = x_ + K * y_residual;
    //     P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H) * P_;
    // }

    void update(const Bbox& z, const Eigen::Vector3d& drone_pos, Eigen::Quaterniond q_C_W) {
        if (!is_initialized_) return;
        
        // --- 计算雅可比矩阵 H ---
        Eigen::Vector3d p_target = x_.head<3>();
        Eigen::Vector3d los_W = p_target - drone_pos;

        Eigen::Matrix3d R_W_C = q_C_W.toRotationMatrix().transpose();
        
        Eigen::Vector3d los_C = R_W_C * los_W;
        double x_c = los_C(0), y_c = los_C(1), z_c = los_C(2);

        if (x_c < 1e-3) {
            consecutive_update_failures_++;
            return; // 避免除以零
        }

        // 预测的测量值
        Eigen::Vector3d z_pred;
        z_pred << focal_length_ * y_c / x_c,
                  focal_length_ * z_c / x_c,
                  focal_length_ * target_size_est_ / x_c;

        // 计算雅可比矩阵 H (dh/d(p_target)) - 使用固定大小矩阵
        Eigen::Vector3d r1 = R_W_C.row(0);
        Eigen::Vector3d r2 = R_W_C.row(1);
        Eigen::Vector3d r3 = R_W_C.row(2);

        double x_c_sq = x_c * x_c;
        RowVector3d du_dp = (focal_length_ / x_c) * r2.transpose() - (focal_length_ * y_c / x_c_sq) * r1.transpose();
        RowVector3d dv_dp = (focal_length_ / x_c) * r3.transpose() - (focal_length_ * z_c / x_c_sq) * r1.transpose();
        RowVector3d dsize_dp = -(focal_length_ * target_size_est_ / x_c_sq) * r1.transpose();

        Matrix3x6d H = Matrix3x6d::Zero();
        H.block<1,3>(0,0) = du_dp;
        H.block<1,3>(1,0) = dv_dp;
        H.block<1,3>(2,0) = dsize_dp;

        // --- 卡尔曼增益、状态和协方差更新 ---
        // 对测量值应用透视补偿
        double v_sq = z.v * z.v;
        double f_sq = focal_length_ * focal_length_;
        double v_compensated = z.v - perspective_compensation_ * target_size_est_ * z.v * focal_length_ / (f_sq + v_sq) / (focal_length_ * target_size_est_ / z.size);
        
        Eigen::Vector3d y_residual;
        y_residual << -z.u, -v_compensated, z.size;
        y_residual -= z_pred;

        // 使用LLT分解替代矩阵求逆，提高数值稳定性
        Eigen::Matrix3d S = H * P_ * H.transpose() + R_;
        Eigen::LLT<Eigen::Matrix3d> llt_S(S);
        if (llt_S.info() != Eigen::Success) {
            // S矩阵不正定，跳过更新
            consecutive_update_failures_++;
            return;
        }
        
        // K = P * H^T * S^{-1}，通过求解线性系统实现
        Eigen::Matrix<double, 6, 3> K = P_ * H.transpose() * llt_S.solve(Eigen::Matrix3d::Identity());
        
        x_ = x_ + K * y_residual;
        P_ = (Matrix6d::Identity() - K * H) * P_;
        
        // 确保P保持对称正定
        P_ = (P_ + P_.transpose()) * 0.5;
        
        consecutive_update_failures_ = 0;
    }

    Eigen::Vector3d noisy_bbox_to_pos(const Bbox& bbox, const Eigen::Vector3d& drone_pos, Eigen::Quaterniond q_C_W) {
        double est_dist_cam_x = focal_length_ * target_size_est_ / bbox.size;
        Eigen::Matrix3d R_C_W = q_C_W.toRotationMatrix();
        double est_los_C_y = -bbox.u * est_dist_cam_x / focal_length_;
        double est_los_C_z = -bbox.v * est_dist_cam_x / focal_length_;
        
        // 透视补偿：当从斜向观察目标时，检测框中心会偏向可见部分的中心
        // 理论上球形目标偏移约为半径(0.5)，但实测使用直径(1.0)效果更好
        // 可能原因：检测器边界框算法特性、目标非理想球形、系统性偏差等
        // 补偿量 = factor * target_size * sin(θ) * cos(θ) = factor * target_size * v * f / (f² + v²)
        double v_sq = bbox.v * bbox.v;
        double f_sq = focal_length_ * focal_length_;
        double z_compensation = perspective_compensation_ * target_size_est_ * bbox.v * focal_length_ / (f_sq + v_sq);
        est_los_C_z -= z_compensation;
        
        Eigen::Vector3d est_los_C(est_dist_cam_x, est_los_C_y, est_los_C_z);
        Eigen::Vector3d est_los_W = R_C_W * est_los_C;
        Eigen::Vector3d est_target_pos = drone_pos + est_los_W;
        return est_target_pos;
    }

    bool is_initialized() const { return is_initialized_; }
    const Vector6d& get_state() const { return x_; }
    
    // 检查协方差矩阵是否健康（未发散）
    bool check_covariance_health() const {
        double max_var = P_.diagonal().maxCoeff();
        if (max_var > max_covariance_threshold_) {
            return false;  // 协方差过大，滤波器可能发散
        }
        // 检测正定性（检查最小特征值）
        Eigen::SelfAdjointEigenSolver<Matrix6d> solver(P_, Eigen::EigenvaluesOnly);
        if (solver.eigenvalues().minCoeff() < 0) {
            return false;  // 协方差矩阵不正定
        }
        return true;
    }
    
    // 重置协方差矩阵（当发散时调用）
    void reset_covariance() {
        P_ = Matrix6d::Identity() * 100.0;
        consecutive_update_failures_ = 0;
    }
    
    // 获取连续更新失败次数
    int get_consecutive_failures() const { return consecutive_update_failures_; }

private:
    Vector6d x_;
    Matrix6d P_;
    Matrix6d F_;
    Matrix6d Q_;
    Eigen::Matrix3d R_;
    double focal_length_;
    double target_size_est_;
    double perspective_compensation_;  // 透视补偿因子 (0.5=半径, 1.0=直径)
    bool is_initialized_;
    int consecutive_update_failures_;
    double max_covariance_threshold_;
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
