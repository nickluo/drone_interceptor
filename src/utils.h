#pragma once

#include <cmath>
#include <atomic>
#include <algorithm>
#include <stdexcept>
#include <deque>
#include <array>
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
    TERMINAL_GUIDANCE, // 末制导 (PN)
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
        return s == MissionState::TRACKING || s == MissionState::BLIND_FLIGHT || s == MissionState::TERMINAL_GUIDANCE;
    }
    
    // 检查任务是否活跃
    bool is_active() const {
        MissionState s = get();
        return s == MissionState::INITIALIZING || 
               s == MissionState::TRACKING || 
               s == MissionState::BLIND_FLIGHT || s == MissionState::TERMINAL_GUIDANCE;
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
            case MissionState::TERMINAL_GUIDANCE: return "TERMINAL_GUIDANCE";
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
// EMAFilter Class (from exp_filter.py) - 向量版本
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
// EMAFilterTemplate - 模板版本，用于固定通道数的EMA滤波
// ============================================================================
template<std::uint32_t N = 1>
class EMAFilterTemplate {
public:
    explicit EMAFilterTemplate(int n) : alpha_(2.0 / (n + 1)) {
        reset();
    }
    
    void reset() {
        data_.fill(0);
        initialized_.fill(false);
    }
    
    double operator[](int index) const {
        return data_[index];
    }
    
    bool check(int index) const {
        return initialized_[index];
    }
    
    double filter(double val, int index = 0) {
        if (!initialized_[index]) {
            data_[index] = val;
            initialized_[index] = true;
        } else {
            data_[index] = alpha_ * val + (1.0 - alpha_) * data_[index];
        }
        return data_[index];
    }

private:
    double alpha_;
    std::array<double, N> data_;
    std::array<bool, N> initialized_;
};

// ============================================================================
// 用于存储相机内参的结构体
// ============================================================================
struct CameraIntrinsics {
    double fx, fy, px, py;
};

// ============================================================================
// DroneEKF Class - 6-State CV Model in World Frame (Stable design)
// State: [x, y, z, vx, vy, vz] (World Frame) - Same as TargetEKF
// Measurement: [cx, cy, w, h] (Image Plane)
// ============================================================================
class DroneEKF {
public:
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    using Matrix6d = Eigen::Matrix<double, 6, 6>;
    using Matrix4x6d = Eigen::Matrix<double, 4, 6>;
    using Matrix6x4d = Eigen::Matrix<double, 6, 4>;

    struct Bbox {
        double cx, cy, w, h;
    };

    struct LOS {
        double azimuth, elevation, azimuth_rate, elevation_rate;
    };

    struct StateHistory {
        int64_t timestamp;
        Vector6d state;
        Matrix6d covariance;
        Eigen::Vector3d imu_input;
    };

    /**
     * @brief Constructor
     */
    DroneEKF(const CameraIntrinsics& intrinsics,
             const Eigen::Vector2d& process_noise_std,
             const Eigen::Vector2d& measurement_noise_std,
             int imu_freq,
             double estimated_detection_latency,
             double target_size_est = 0.5,
             double perspective_compensation = 1.0)
        : intrinsics_(intrinsics),
          max_buffer_size_(static_cast<size_t>(imu_freq * estimated_detection_latency * 2)),
          dt_(1000000 / imu_freq),
          ema_filter_(imu_freq / 10),
          target_size_est_(target_size_est),
          perspective_compensation_(perspective_compensation),
          is_initialized_(false),
          last_timestamp_(-1)
    {
        if (max_buffer_size_ < 10) max_buffer_size_ = 10;

        x_.setZero();

        // CV model: Use position and velocity noise only
        double std_pos = process_noise_std(0);
        double std_vel = process_noise_std(1);

        Vector6d q_diag;
        q_diag << std_pos*std_pos, std_pos*std_pos, std_pos*std_pos,
                  std_vel*std_vel, std_vel*std_vel, std_vel*std_vel;
        Q_base_ = q_diag.asDiagonal();
        P_ = Q_base_ * 100.0;

        double std_meas_pos = measurement_noise_std(0);
        double std_meas_size = measurement_noise_std(1);
        Eigen::Vector4d r_diag;
        r_diag << std_meas_pos*std_meas_pos, std_meas_pos*std_meas_pos,
                  std_meas_size*std_meas_size, std_meas_size*std_meas_size;
        R_ = r_diag.asDiagonal();
    }

    /**
     * @brief Predict step (World Frame CV Model)
     */
    void predict(int64_t current_time_us, const Eigen::Vector3d& imu_omega_flu) {
        if (!is_initialized_) return;

        if (current_time_us > 0) {
            if (last_timestamp_ > 0) {
                dt_ = current_time_us - last_timestamp_;
            }
            last_timestamp_ = current_time_us;
        }

        const double dt = dt_ * 1e-6;

        // F matrix for CV model: x_{k+1} = F * x_k
        Matrix6d F = Matrix6d::Identity();
        F(0,3) = dt; F(1,4) = dt; F(2,5) = dt;

        x_ = F * x_;
        
        // Discrete-time process noise (based on continuous white noise acceleration)
        // Q = integral of F * G * Q_c * G^T * F^T
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        Matrix6d Q = Matrix6d::Zero();
        
        // Position variance contribution
        Q(0,0) = dt3/3.0; Q(1,1) = dt3/3.0; Q(2,2) = dt3/3.0;
        // Velocity variance contribution  
        Q(3,3) = dt; Q(4,4) = dt; Q(5,5) = dt;
        // Cross-correlation
        Q(0,3) = dt2/2.0; Q(1,4) = dt2/2.0; Q(2,5) = dt2/2.0;
        Q(3,0) = dt2/2.0; Q(4,1) = dt2/2.0; Q(5,2) = dt2/2.0;
        
        // Scale by base process noise
        double q_scale = Q_base_(3,3); // velocity variance as scale
        Q *= q_scale;
        
        P_ = F * P_ * F.transpose() + Q;

        if (current_time_us > 0) {
            history_buffer_.push_back({current_time_us, x_, P_, imu_omega_flu});
            if (history_buffer_.size() > max_buffer_size_) {
                history_buffer_.pop_front();
            }
        }
    }

    /**
     * @brief Update step with 2D Bbox and 3D Drone State
     */
    void update(int64_t ts, const Bbox& measurement, 
                const Eigen::Vector3d& drone_pos, 
                const Eigen::Quaterniond& q_C_W) {
        
        last_drone_pos_ = drone_pos;
        last_q_C_W_ = q_C_W;

        if (!is_initialized_) {
            initialize(measurement, drone_pos, q_C_W);
            return;
        }

        auto it = std::min_element(history_buffer_.begin(), history_buffer_.end(),
            [&](const StateHistory& a, const StateHistory& b) {
                return std::abs(a.timestamp - ts) < std::abs(b.timestamp - ts);
            });

        if (it != history_buffer_.end()) {
            // Rewind to closest historical state
            x_ = it->state;
            P_ = it->covariance;
            
            // Update with measurement
            update_internal(measurement, drone_pos, q_C_W);

            // Replay predictions from that point
            for (auto replay_it = std::next(it); replay_it != history_buffer_.end(); ++replay_it) {
                dt_ = replay_it->timestamp - std::prev(replay_it)->timestamp;
                predict(-1.0, replay_it->imu_input);
            }
        } else {
            update_internal(measurement, drone_pos, q_C_W);
        }
    }
    
    void update(int64_t ts, const Bbox& measurement) {
        update(ts, measurement, last_drone_pos_, last_q_C_W_);
    }

    Bbox get_bbox() const {
        return project_to_bbox(x_, last_drone_pos_, last_q_C_W_);
    }
    
    Bbox project_to_bbox(const Vector6d& state, 
                         const Eigen::Vector3d& drone_pos, 
                         const Eigen::Quaterniond& q_C_W) const {
        Bbox b;
        Eigen::Vector3d P_W = state.head<3>();
        Eigen::Vector3d P_rel_W = P_W - drone_pos;
        Eigen::Vector3d P_C = q_C_W.inverse() * P_rel_W;
        
        if (P_C.x() < 0.1) P_C.x() = 0.1;

        b.cx = intrinsics_.px - P_C.y() * intrinsics_.fx / P_C.x();
        b.cy = intrinsics_.py - P_C.z() * intrinsics_.fy / P_C.x();
        b.w = (intrinsics_.fx * target_size_est_) / P_C.x();
        b.h = (intrinsics_.fy * target_size_est_) / P_C.x();
        return b;
    }

    bool is_healthy() const {
        return !x_.hasNaN() && P_.diagonal().maxCoeff() < 1e6;
    }
    
    Vector6d get_state() const { return x_; }
    
    bool is_initialized() const { return is_initialized_; }
    
    void reset() {
        is_initialized_ = false;
        x_.setZero();
        P_ = Q_base_ * 100;
        history_buffer_.clear();
        ema_filter_.reset();
    }

    struct WorldState {
        Eigen::Vector3d pos;
        Eigen::Vector3d vel;
    };

    WorldState get_est_world_state(const Eigen::Vector3d& drone_pos, 
                                   const Eigen::Vector3d& drone_vel,
                                   const Eigen::Quaterniond& q_C_W,
                                   const Eigen::Vector3d& omega_body) const {
        WorldState result;
        result.pos = x_.head<3>();
        result.vel = x_.tail<3>();
        return result;
    }
    
    LOS state_to_los(bool compensate_rotation = true) {
        Eigen::Vector3d P_rel = x_.head<3>() - last_drone_pos_;
        Eigen::Vector3d P_C = last_q_C_W_.inverse() * P_rel; 
        
        double norm_x = P_C.x();
        double norm_y = P_C.y();
        double norm_z = P_C.z();
        LOS l;
        l.azimuth = std::atan2(norm_y, norm_x);
        l.elevation = std::atan2(norm_z, norm_x);
        l.azimuth_rate = 0;
        l.elevation_rate = 0;
        return l;
    }

private:
    void initialize(const Bbox& z, const Eigen::Vector3d& drone_pos, const Eigen::Quaterniond& q_C_W) {
        Eigen::Vector3d P_W = noisy_bbox_to_pos_init(z, drone_pos, q_C_W);
        x_.setZero();
        x_.head<3>() = P_W;
        // Initialize velocity to zero
        x_.tail<3>().setZero();
        is_initialized_ = true;
        
        last_drone_pos_ = drone_pos;
        last_q_C_W_ = q_C_W;
    }

    void update_internal(const Bbox& z_meas, 
                         const Eigen::Vector3d& drone_pos, 
                         const Eigen::Quaterniond& q_C_W) {
        
        // 1. Predict Measurement z_pred
        Eigen::Vector3d P_W = x_.head<3>();
        Eigen::Vector3d P_rel_W = P_W - drone_pos;
        Eigen::Matrix3d R_W_to_C = q_C_W.toRotationMatrix().transpose();
        Eigen::Vector3d P_C = R_W_to_C * P_rel_W;
        
        double xc = P_C.x();
        double yc = P_C.y();
        double zc = P_C.z();
        
        if (xc < 0.1) return; // Behind camera

        Eigen::Vector4d z_pred;
        z_pred(0) = intrinsics_.px - yc * intrinsics_.fx / xc;
        z_pred(1) = intrinsics_.py - zc * intrinsics_.fy / xc;
        z_pred(2) = intrinsics_.fx * target_size_est_ / xc;
        z_pred(3) = intrinsics_.fy * target_size_est_ / xc;

        // 2. Jacobian H (4x6)
        double xc2 = xc * xc;
        
        Eigen::Matrix<double, 4, 3> J_cam;
        // du/dP_C
        J_cam.row(0) << yc * intrinsics_.fx / xc2, -intrinsics_.fx / xc, 0.0;
        // dv/dP_C
        J_cam.row(1) << zc * intrinsics_.fy / xc2, 0.0, -intrinsics_.fy / xc;
        // dw/dP_C
        J_cam.row(2) << -intrinsics_.fx * target_size_est_ / xc2, 0.0, 0.0;
        // dh/dP_C
        J_cam.row(3) << -intrinsics_.fy * target_size_est_ / xc2, 0.0, 0.0;
        
        Matrix4x6d H = Matrix4x6d::Zero();
        H.block<4,3>(0,0) = J_cam * R_W_to_C;
        // Velocity states have zero Jacobian (not directly observed)
        
        // 3. Kalman Update
        Eigen::Vector4d z_vec;
        z_vec << z_meas.cx, z_meas.cy, z_meas.w, z_meas.h;
        Eigen::Vector4d y = z_vec - z_pred;
        
        Eigen::Matrix4d S = H * P_ * H.transpose() + R_;
        
        // Use LLT decomposition for numerical stability
        Eigen::LLT<Eigen::Matrix4d> llt_S(S);
        if (llt_S.info() != Eigen::Success) {
            return; // Skip update if S is not positive definite
        }
        
        Matrix6x4d K = P_ * H.transpose() * llt_S.solve(Eigen::Matrix4d::Identity());
        
        x_ = x_ + K * y;
        P_ = (Matrix6d::Identity() - K * H) * P_;
        
        // Ensure P stays symmetric
        P_ = (P_ + P_.transpose()) * 0.5;
    }
    
    Eigen::Vector3d noisy_bbox_to_pos_init(const Bbox& bbox, 
                                           const Eigen::Vector3d& drone_pos, 
                                           const Eigen::Quaterniond& q_C_W) const {
        double bbox_size = std::max(bbox.w, bbox.h);
        if (bbox_size < 1e-6) bbox_size = 1.0;
        double depth = intrinsics_.fx * target_size_est_ / bbox_size;
        
        double u_rel = bbox.cx - intrinsics_.px;
        double v_rel = bbox.cy - intrinsics_.py;
        
        double yc = -u_rel * depth / intrinsics_.fx;
        double zc = -v_rel * depth / intrinsics_.fy;
        double xc = depth;
        
        Eigen::Vector3d P_C(xc, yc, zc);
        return drone_pos + q_C_W * P_C;
    }

    Vector6d x_;
    Matrix6d P_;
    Matrix6d Q_base_;
    Eigen::Matrix4d R_;

    CameraIntrinsics intrinsics_;
    std::deque<StateHistory> history_buffer_;
    size_t max_buffer_size_;

    bool is_initialized_;
    int64_t last_timestamp_;
    int64_t dt_;

    EMAFilterTemplate<2> ema_filter_;
    
    double target_size_est_;
    double perspective_compensation_;
    
    Eigen::Vector3d last_drone_pos_;
    Eigen::Quaterniond last_q_C_W_;
};

// ============================================================================
// TargetEKF Class (保留用于兼容性，建议使用DroneEKF)
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
