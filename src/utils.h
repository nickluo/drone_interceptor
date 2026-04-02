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

    Eigen::VectorXd data() const {
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
// DroneEKF Class - 结合图像空间跟踪和3D位置估计的EKF
// 状态向量 (18维): 
//   图像空间: [cx, cy, w, h, vx_img, vy_img, vw, vh, ax_img, ay_img, aw, ah]
//   3D空间:   [px, py, pz, vx_3d, vy_3d, vz_3d]
// ============================================================================
class DroneEKF {
public:
    // 固定大小类型定义
    using Vector19d = Eigen::Matrix<double, 19, 1>;
    using Matrix19d = Eigen::Matrix<double, 19, 19>;
    using Matrix7x19d = Eigen::Matrix<double, 7, 19>;
    using Vector7d = Eigen::Matrix<double, 7, 1>;
    using Matrix7d = Eigen::Matrix<double, 7, 7>;

    // 检测框结构体
    struct Bbox {
        double cx, cy, w, h;
    };

    // ============================================================================
    // 用于存储历史状态的结构体 (用于延迟测量处理)
    // ============================================================================
    struct StateHistory {
        int64_t timestamp;
        Vector19d state;
        Matrix19d covariance;
        Eigen::Vector3d imu_input;
    };

    /**
     * @brief 构造函数
     * @param intrinsics 相机内参
     * @param process_noise_std 过程噪声标准差 [std_pos_img, std_vel_img, std_accel_img, std_size, std_size_vel, std_accel_size, std_pos_3d, std_vel_3d]
     * @param measurement_noise_std 测量噪声标准差 [std_meas_pos, std_meas_size, std_meas_3d]
     * @param estimated_imu_freq 估计的IMU数据频率 (Hz)
     * @param estimated_detection_latency 估计的检测延迟 (秒)
     * @param target_size_est 目标估计尺寸 (米)
     */
    DroneEKF(const CameraIntrinsics& intrinsics,
             const Eigen::Matrix<double, 9, 1>& process_noise_std,
             const Eigen::Vector3d& measurement_noise_std,
             int estimated_imu_freq,
             double estimated_detection_latency,
             double target_size_est,
             double tau_W = 15.0,
             double alpha_R_inflate = 1.0)
        : intrinsics_(intrinsics),
          max_buffer_size_(static_cast<size_t>(estimated_imu_freq * estimated_detection_latency * 2)),
          dt_(1.0 / estimated_imu_freq),
          target_size_est_(target_size_est),
          tau_W_(tau_W),
          alpha_R_inflate_(alpha_R_inflate),
          ema_filter_(estimated_imu_freq / 20),
          imu_flow_filter_(estimated_imu_freq / 20),
          is_initialized_(false),
          last_timestamp_(-1.0)
    {
        if (max_buffer_size_ < 10) {
            max_buffer_size_ = 10;
        }

        x_.setZero();
        v_imu_cache_.setZero();

        set_covariance(process_noise_std, measurement_noise_std);

        // 初始协方差
        P_ = Q_ * 10.0;
        P_(18, 18) = 4.0; // P_W
        
        // 测量矩阵 H (7x19): 直接观测 [cx, cy, w, h] 和通过投影观测3D位置计算来的 [az, el, alpha]
        // H(0:3, 0:3) 在上面为单位阵
        // H(4:6, 12:18) 在update中动态计算 (非线性)
        H_.setZero();
        H_(0, 0) = 1.0; // cx
        H_(1, 1) = 1.0; // cy
        H_(2, 2) = 1.0; // w
        H_(3, 3) = 1.0; // h
    }

    void set_covariance(const Eigen::Matrix<double, 9, 1>& process_noise_std,
             const Eigen::Vector3d& measurement_noise_std) {
        // 解析过程噪声
        double std_pos_img = process_noise_std(0);
        double std_vel_img = process_noise_std(1);
        double std_accel_img = process_noise_std(2);
        double std_size = process_noise_std(3);
        double std_size_vel = process_noise_std(4);
        double std_accel_size = process_noise_std(5);
        double std_q_W = process_noise_std(6);
        double std_pos_3d = process_noise_std(7);  // 现在解释为 sqrt(q_accel_3d)
        // double std_vel_3d = process_noise_std(8);  // 运动学模型中不再使用

        // 存储运动学过程噪声参数 (在predict中动态构建含交叉协方差的Q矩阵)
        q_accel_3d_ = std_pos_3d * std_pos_3d;  // 加速度谱密度 (m²/s⁴)
        q_W_ = std_q_W * std_q_W;               // W随机游走方差率

        // 构建过程噪声协方差矩阵 Q (19x19)
        // 仅填充图像空间部分; 3D部分在predict()中用运动学模型动态计算
        Vector19d q_diag;
        q_diag << std_pos_img*std_pos_img, std_pos_img*std_pos_img,           // cx, cy
                  std_size*std_size, std_size*std_size,                        // w, h
                  std_vel_img*std_vel_img, std_vel_img*std_vel_img,            // vx_img, vy_img
                  std_size_vel*std_size_vel, std_size_vel*std_size_vel,        // vw, vh
                  std_accel_img*std_accel_img, std_accel_img*std_accel_img,    // ax_img, ay_img
                  std_accel_size*std_accel_size, std_accel_size*std_accel_size,// aw, ah
        // 3D空间: 置零, 在predict()中用运动学CV模型动态填充 (含位置-速度交叉协方差)
                  0.0, 0.0, 0.0,   // px, py, pz
                  0.0, 0.0, 0.0,   // vx, vy, vz
                  0.0;             // W
        Q_ = q_diag.asDiagonal();

        // 测量噪声协方差 R (7x7): [cx, cy, w, h, az, el, alpha]
        double std_meas_pos = measurement_noise_std(0);
        double std_meas_size = measurement_noise_std(1);
        double std_meas_angle = measurement_noise_std(2); // alpha noise, effectively size_dev / f
        Vector7d r_diag;
        double f_avg = (intrinsics_.fx + intrinsics_.fy) / 2.0;
        double sigma_az = std_meas_pos / intrinsics_.fx;
        double sigma_el = std_meas_pos / intrinsics_.fy;
        double sigma_alpha = std_meas_angle / f_avg;
        r_diag << std_meas_pos*std_meas_pos, std_meas_pos*std_meas_pos,
                  std_meas_size*std_meas_size, std_meas_size*std_meas_size,
                  sigma_az*sigma_az, sigma_el*sigma_el, sigma_alpha*sigma_alpha * alpha_R_inflate_; // configurable alpha_R_inflate
        R_ = r_diag.asDiagonal();
    }


    /**
     * @brief EKF预测步骤
     * @param current_time 当前时间戳 (微秒)
     * @param imu_omega 相机角速度 [omega_x, omega_y, omega_z] - 期望FRD坐标系 (X前, Y右, Z下)
     *                  如果输入是FLU坐标系，会自动转换
     * @param is_flu 输入角速度是否为FLU坐标系 (默认true，因为ROS2通常使用FLU)
     */
    void predict(int64_t current_time, const Eigen::Vector3d& imu_omega, bool is_flu = true) {
        if (!is_initialized_) return;

        // 保存云台角速度 (转换到FRD相机坐标系用于LOS rate计算)
        // FLU→FRD: y取反, z取反
        if (is_flu) {
            last_gimbal_omega_frd_ << imu_omega(0), -imu_omega(1), -imu_omega(2);
        } else {
            last_gimbal_omega_frd_ = imu_omega;
        }

        if (current_time > 0) {
            if (last_timestamp_ > 0) {
                const int64_t delta_us = current_time - last_timestamp_;
                if (delta_us <= 0) {
                    return; // 防止非递增时间戳导致负dt
                }
                dt_ = delta_us;
            }
            last_timestamp_ = current_time;
        }

        // 确保dt有效
        const double dt = dt_ / 1e6; // 转换为秒
        const double dt2 = 0.5 * dt * dt;
        const Vector19d x_prev = x_;
        
        // 检查输入有效性
        if (imu_omega.hasNaN() || x_prev.hasNaN()) {
            return;
        }

        Vector19d x_pred;
        
        // --- 图像空间 CA 模型预测 (12维) ---
        // 位置: p = p0 + v*dt + 0.5*a*dt²
        x_pred(0) = x_prev(0) + x_prev(4) * dt + x_prev(8) * dt2;   // cx
        x_pred(1) = x_prev(1) + x_prev(5) * dt + x_prev(9) * dt2;   // cy
        x_pred(2) = x_prev(2) + x_prev(6) * dt + x_prev(10) * dt2;  // w
        x_pred(3) = x_prev(3) + x_prev(7) * dt + x_prev(11) * dt2;  // h

        // 速度: v = v0 + a*dt
        x_pred(4) = x_prev(4) + x_prev(8) * dt;   // vx_img
        x_pred(5) = x_prev(5) + x_prev(9) * dt;   // vy_img
        x_pred(6) = x_prev(6) + x_prev(10) * dt;  // vw
        x_pred(7) = x_prev(7) + x_prev(11) * dt;  // vh

        // 加速度: a = a0 * decay (衰减因子防止加速度状态无限增长)
        // 使用指数衰减，时间常数约300ms，防止CA模型在IMU补偿偏差下积分发散
        const double accel_decay = std::exp(-dt / 0.3);
        x_pred.segment<4>(8) = x_prev.segment<4>(8) * accel_decay;

        // --- IMU运动补偿 ---
    // 输入 imu_omega 是期望在FRD坐标系 (X前/光轴, Y右, Z下)
    // 如果输入是FLU (X前, Y左, Z上)，需要转换: omega_y_FRD = -omega_y_FLU, omega_z_FRD = -omega_z_FLU
    double omega_x = imu_omega(0); // FRD Roll (绕光轴)
    double omega_y = is_flu ? -imu_omega(1) : imu_omega(1); // FRD Pitch (绕右轴)
    double omega_z = is_flu ? -imu_omega(2) : imu_omega(2); // FRD Yaw (绕下轴)
    
    // 光流公式期望标准相机坐标系 (X右, Y下, Z前)
    // 旋转角速度对应关系:
    // w_x (绕相机的X轴旋转, 即Pitch) = omega_y (FRD绕Y轴)
    // w_y (绕相机的Y轴旋转, 即Yaw) = omega_z (FRD绕Z轴)
    // w_z (绕相机的Z轴旋转, 即Roll) = omega_x (FRD绕X轴)
    double w_x = omega_y;
    double w_y = omega_z;
    double w_z = omega_x;

    double cx = x_prev(0);
    double cy = x_prev(1);
    double px = intrinsics_.px;
    double py = intrinsics_.py;
    double fx = intrinsics_.fx;
    double fy = intrinsics_.fy;

    double v_imu_cx = (1.0/fy)*(cx-px)*(cy-py)*w_x 
                    - (fx + (1.0/fx)*(cx-px)*(cx-px))*w_y 
                    + (fx/fy)*(cy-py)*w_z;
    double v_imu_cy = (fy + (1.0/fy)*(cy-py)*(cy-py))*w_x 
                    - (1.0/fx)*(cx-px)*(cy-py)*w_y 
                    - (fy/fx)*(cx-px)*w_z;

    // Filter IMU flow to reduce noise injection
    v_imu_cx = imu_flow_filter_.filter(v_imu_cx, 0);
    v_imu_cy = imu_flow_filter_.filter(v_imu_cy, 1);

    v_imu_cache_ << v_imu_cx, v_imu_cy;
    
    x_pred(0) += v_imu_cx * dt;
    x_pred(1) += v_imu_cy * dt;

    // --- 3D空间 CV 模型预测 (6维) ---
    // 位置: p = p0 + v*dt
    x_pred(12) = x_prev(12) + x_prev(15) * dt;  // px
    x_pred(13) = x_prev(13) + x_prev(16) * dt;  // py
    x_pred(14) = x_prev(14) + x_prev(17) * dt;  // pz

    // 速度: v = v0
    x_pred.segment<3>(15) = x_prev.segment<3>(15);
    
    // Width W: Random walk
    x_pred(18) = x_prev(18);

    // --- 计算雅可比矩阵 F (19x19) ---
    Matrix19d F = Matrix19d::Identity();
    
    // 图像空间 CA 模型
    F(0, 4) = dt;  F(0, 8) = dt2;
    F(1, 5) = dt;  F(1, 9) = dt2;
    F(2, 6) = dt;  F(2, 10) = dt2;
    F(3, 7) = dt;  F(3, 11) = dt2;
    F(4, 8) = dt;
    F(5, 9) = dt;
    F(6, 10) = dt;
    F(7, 11) = dt;

    // IMU补偿的雅可比
    double d_vcx_dcx = (1.0/fy)*(cy-py)*w_x - (2.0/fx)*(cx-px)*w_y;
    double d_vcx_dcy = (1.0/fy)*(cx-px)*w_x + (fx/fy)*w_z;
    double d_vcy_dcx = -(1.0/fx)*(cy-py)*w_y - (fy/fx)*w_z;
    double d_vcy_dcy = (2.0/fy)*(cy-py)*w_x - (1.0/fx)*(cx-px)*w_y;

        F(0, 0) += d_vcx_dcx * dt;
        F(0, 1) += d_vcx_dcy * dt;
        F(1, 0) += d_vcy_dcx * dt;
        F(1, 1) += d_vcy_dcy * dt;

        // 3D空间 CV 模型
        F(12, 15) = dt;
        F(13, 16) = dt;
        F(14, 17) = dt;

        // --- 更新状态和协方差 ---
    x_ = x_pred;
    
    // Safety Clamping for Image Coordinates to prevent Jacobian explosion 
    // when target is tracked far off-screen during extreme maneuvers (>80 deg out)
    x_(0) = std::clamp(x_(0), -3000.0, 4000.0);
    x_(1) = std::clamp(x_(1), -3000.0, 4000.0);
    
    // 构建含运动学交叉协方差的过程噪声矩阵
    // 图像空间: 按dt_ratio线性缩放 (保持原有行为)
    double dt_ratio = dt / 0.01;
    if (dt_ratio < 0.01) dt_ratio = 0.01;
    if (dt_ratio > 10.0) dt_ratio = 10.0;

    Matrix19d Q_full = Q_ * dt_ratio;  // 图像空间部分 (Q_的3D块已置零)

    // 3D运动学CV过程噪声 (白噪声加速度积分, 含位置-速度交叉协方差)
    // Q_3D = q_a * [dt³/3  dt²/2]  ⊗ I₃
    //              [dt²/2  dt   ]
    const double q_dt3_3 = q_accel_3d_ * dt * dt * dt / 3.0;
    const double q_dt2_2 = q_accel_3d_ * dt * dt / 2.0;
    const double q_dt1   = q_accel_3d_ * dt;
    for (int i = 0; i < 3; ++i) {
        Q_full(12 + i, 12 + i) = q_dt3_3;   // 位置方差
        Q_full(12 + i, 15 + i) = q_dt2_2;   // 位置-速度交叉协方差
        Q_full(15 + i, 12 + i) = q_dt2_2;   // 速度-位置交叉协方差
        Q_full(15 + i, 15 + i) = q_dt1;     // 速度方差
    }
    // W随机游走 + tau_W指数衰减: 初期允许W快速收敛, 后期冻结防漂移
    double w_decay = (tau_W_ > 0.0) ? std::exp(-static_cast<double>(update_count_) / tau_W_) : 1.0;
    Q_full(18, 18) = q_W_ * w_decay * dt_ratio;

    P_ = F * P_ * F.transpose() + Q_full;
        
        // 强制对称性
        P_ = 0.5 * (P_ + P_.transpose());

        if (current_time > 0) {
            history_buffer_.push_back({current_time, x_, P_, imu_omega});
            if (history_buffer_.size() > max_buffer_size_) {
                history_buffer_.pop_front();
            }
        }
    }

    /**
     * @brief EKF更新步骤 (使用Bbox结构体)
     * @param current_time 当前时间戳
     * @param bbox 检测框 {cx, cy, w, h}
     * @param drone_pos 无人机位置
     * @param q_C_W 相机到世界的四元数
     */
    void update(int64_t current_time,
                const Bbox& bbox,
                const Eigen::Vector3d& drone_pos,
                const Eigen::Quaterniond& q_C_W) {
        Eigen::Vector4d bbox_vec;
        bbox_vec << bbox.cx, bbox.cy, bbox.w, bbox.h;
        update(current_time, current_time, bbox_vec, drone_pos, q_C_W);
    }

    /**
     * @brief EKF更新步骤
     * @param current_time 当前时间戳
     * @param ts 测量时间戳
     * @param bbox [cx, cy, w, h]
     * @param drone_pos 无人机位置
     * @param q_C_W 相机到世界的四元数
     */
    void update(int64_t current_time, int64_t ts,
                const Eigen::Vector4d& bbox,
                const Eigen::Vector3d& drone_pos,
                const Eigen::Quaterniond& q_C_W) {
        if (!is_initialized_) {
            initialize(bbox, drone_pos, q_C_W);
            return;
        }

        if (history_buffer_.empty()) {
            update_internal(bbox, drone_pos, q_C_W);
            return;
        }

        // 查找历史状态 (时间戳单调递增，使用二分查找加速)
        auto it = std::lower_bound(
            history_buffer_.begin(), history_buffer_.end(), ts,
            [](const StateHistory& entry, int64_t t) { return entry.timestamp < t; });

        auto best_it = it;
        if (it == history_buffer_.end()) {
            best_it = std::prev(history_buffer_.end());
        } else if (it != history_buffer_.begin()) {
            auto prev_it = std::prev(it);
            if (std::llabs(prev_it->timestamp - ts) <= std::llabs(it->timestamp - ts)) {
                best_it = prev_it;
            }
        }

        x_ = best_it->state;
        P_ = best_it->covariance;
        update_internal(bbox, drone_pos, q_C_W);

        // 重放后续IMU输入
        for (auto replay_it = std::next(best_it); replay_it != history_buffer_.end(); ++replay_it) {
            const int64_t delta_us = replay_it->timestamp - std::prev(replay_it)->timestamp;
            if (delta_us <= 0) {
                continue;
            }
            dt_ = delta_us;
            predict(-1.0, replay_it->imu_input);
        }
    }

    /** @brief 获取图像空间状态 [cx, cy, w, h] */
    Eigen::Vector4d getImageState() const {
        return x_.head<4>();
    }

    /** @brief 获取3D位置状态 [px, py, pz] */
    Eigen::Vector3d get3DPosition() const {
        return x_.segment<3>(12);
    }

    /** @brief 获取3D速度状态 [vx, vy, vz] */
    Eigen::Vector3d get3DVelocity() const {
        return x_.segment<3>(15);
    }

    /** @brief 获取完整6D状态 [px, py, pz, vx, vy, vz] */
    Vector6d get3DState() const {
        Vector6d state;
        state.head<3>() = x_.segment<3>(12);
        state.tail<3>() = x_.segment<3>(15);
        return state;
    }

    /** @brief 获取完整19D状态向量 */
    const Vector19d& getFullState() const {
        return x_;
    }

    /**
     * @brief 获取估计的物理宽度
     */
    double getEstimatedWidth() const {
        return x_(18);
    }

    /** @brief 获取IMU光流补偿缓存 */
    Eigen::Vector2d getImuFlowCache() const {
        return v_imu_cache_;
    }

    /**
     * @brief 将状态转换为视线角和角速度 (惯性系)
     * @param compensate_rotation 是否使用惯性系LOS角速度 (加上云台角速度)
     * @return [azimuth, elevation, azimuth_rate, elevation_rate] (弧度, 弧度/秒)
     *
     * 原理: 惯性系LOS角速度 = 相机坐标系LOS角变化率 + 云台角速度
     * 相机坐标系LOS角变化率通过EKF位置状态的有限差分计算, 避免使用
     * 被前向运动透视流(translational flow)污染的EKF速度状态.
     */
    Eigen::Vector4d stateToLosAngles(bool compensate_rotation = false) {
        double cx = x_(0);
        double cy = x_(1);

        double norm_x = (cx - intrinsics_.px) / intrinsics_.fx;
        double norm_y = (cy - intrinsics_.py) / intrinsics_.fy;

        double azimuth = std::atan(norm_x);
        double elevation = std::atan(norm_y);

        // 使用有限差分计算相机坐标系内的LOS角变化率
        // 这避免了EKF速度状态(vx_img)被前向运动的translational flow污染的问题
        double azimuth_rate = 0.0;
        double elevation_rate = 0.0;
        double dt_sec = dt_ / 1e6;

        if (los_angle_initialized_ && dt_sec > 0.001 && dt_sec < 0.1) {
            // 相机坐标系内的LOS角变化率 (目标在画面中的角速度)
            double camera_az_rate = (azimuth - prev_azimuth_) / dt_sec;
            double camera_el_rate = (elevation - prev_elevation_) / dt_sec;

            if (compensate_rotation) {
                // 惯性系LOS角速度 = 相机系变化率 + 云台角速度
                // 云台FRD坐标系: yaw(omega_z_FRD) 对应 azimuth, pitch(omega_y_FRD) 对应 elevation
                // 注意符号: 云台向右转(yaw>0)时, 目标在画面中向左移(azimuth减小)
                // 所以: 惯性系azimuth_rate = camera_rate + gimbal_yaw_rate
                azimuth_rate = camera_az_rate + last_gimbal_omega_frd_(2);   // + omega_z_FRD (yaw)
                elevation_rate = camera_el_rate + last_gimbal_omega_frd_(1); // + omega_y_FRD (pitch)
            } else {
                // 相机坐标系LOS角速度 (不加云台旋转)
                azimuth_rate = camera_az_rate;
                elevation_rate = camera_el_rate;
            }
        }

        prev_azimuth_ = azimuth;
        prev_elevation_ = elevation;
        los_angle_initialized_ = true;

        azimuth_rate = ema_filter_.filter(azimuth_rate, 0);
        elevation_rate = ema_filter_.filter(elevation_rate, 1);

        Eigen::Vector4d result;
        result << azimuth, elevation, azimuth_rate, elevation_rate;
        return result;
    }
    
    /** @brief 兼容性接口：检查滤波器是否已初始化 */
    bool is_initialized() const { return is_initialized_; }
    
    /** @brief W 是否在最近一次更新中可观测 */
    bool isWidthObservable() const { return w_observable_; }
    
    /** @brief 检查EKF状态是否健康 (无NaN) */
    bool is_healthy() const {
        if (x_.hasNaN() || P_.hasNaN()) return false;
        return checkCovarianceHealth();
    }
    
    /** @brief 获取6D状态 [px, py, pz, vx, vy, vz] - 兼容TargetEKF接口 */
    Vector6d get_state() const {
        return get3DState();
    }

    /** @brief 重置滤波器 */
    void reset() {
        is_initialized_ = false;
        x_.setZero();
        P_ = Q_ * 10.0;
        history_buffer_.clear();
        ema_filter_.reset();
        imu_flow_filter_.reset();
        los_angle_initialized_ = false;
        last_gimbal_omega_frd_.setZero();
        alpha_innov_ema_ = 0.0;
        update_count_ = 0;
    }

    /** @brief 检查协方差矩阵健康状态 */
    bool checkCovarianceHealth() const {
        double max_var = P_.diagonal().maxCoeff();
        if (max_var > 1e6) return false;
        Eigen::SelfAdjointEigenSolver<Matrix19d> solver(P_, Eigen::EigenvaluesOnly);
        return solver.eigenvalues().minCoeff() >= 0;
    }

private:
    /**
     * @brief 使用第一个测量值初始化滤波器
     * @param bbox_measurement [cx, cy, w, h]
     * @param drone_pos 无人机位置 (世界坐标系)
     * @param q_C_W 相机到世界的四元数
     */
    void initialize(const Eigen::Vector4d& bbox_measurement,
                    const Eigen::Vector3d& drone_pos,
                    const Eigen::Quaterniond& q_C_W) {
        x_.setZero();
        
        // 检查bbox有效性
        double bbox_w = bbox_measurement(2);
        double bbox_h = bbox_measurement(3);
        if (bbox_w < 1.0 || bbox_h < 1.0 || std::isnan(bbox_w) || std::isnan(bbox_h)) {
            // 无效的bbox，不初始化
            return;
        }
        
        // 图像空间初始化
        x_.head<4>() = bbox_measurement;
        
        // 3D位置初始化 - 从bbox估计 (使用较大的边)
        // 使用与TargetEKF相同的相机坐标系约定:
        // 相机坐标系: X前, Y右, Z下
        // 图像坐标系: u = bbox.cx - px (正=右), v = bbox.cy - py (正=下)
        double bbox_size = std::max(bbox_w, bbox_h);
        // range_init_bias=0.5: 故意将初始距离偏近, 打破 W/d 不可观测歧义
        // 使得 alpha_pred > alpha_meas, 驱动 EKF 向下修正 W
        // (参考 Python target_ekf.py 的 range_init_bias 机制)
        constexpr double range_init_bias = 0.5;
        double est_dist_cam_x = range_init_bias * intrinsics_.fx * target_size_est_ / bbox_size;
        
        // 限制估计距离在合理范围
        est_dist_cam_x = std::clamp(est_dist_cam_x, 1.0, 1000.0);
        
        // 计算图像偏移 (相对主点)
        double u = bbox_measurement(0) - intrinsics_.px;  // 水平偏移
        double v = bbox_measurement(1) - intrinsics_.py;  // 垂直偏移
        
        // 相机系LOS向量 (注意负号 - 匹配FLU中Y为左, Z为上)
        double est_los_C_y = -u * est_dist_cam_x / intrinsics_.fx;
        double est_los_C_z = -v * est_dist_cam_x / intrinsics_.fy;
        Eigen::Vector3d los_C(est_dist_cam_x, est_los_C_y, est_los_C_z);
        
        Eigen::Matrix3d R_C_W = q_C_W.toRotationMatrix();
        Eigen::Vector3d target_pos_W = drone_pos + R_C_W * los_C;
        
        // 检查结果有效性
        if (target_pos_W.hasNaN()) {
            return;
        }
        
        x_.segment<3>(12) = target_pos_W;
        x_.segment<3>(15).setZero();
        x_(18) = target_size_est_;
        
        // 重置协方差
        P_ = Q_ * 10.0;
        
        // 为3D距离/速度/宽度赋予巨大的初始不确定度
        P_(12, 12) = 400.0; // px
        P_(13, 13) = 400.0; // py
        P_(14, 14) = 400.0; // pz
        
        P_(15, 15) = 25.0;  // vx
        P_(16, 16) = 25.0;  // vy
        P_(17, 17) = 25.0;  // vz
        
        P_(18, 18) = 5.0;   // W
        
        is_initialized_ = true;
        last_timestamp_ = -1.0;
        history_buffer_.clear();
    }

    /**
     * @brief 内部更新函数
     * @param bbox 测量边界框 [cx, cy, w, h]
     * @param drone_pos 无人机位置
     * @param q_C_W 相机到世界的四元数
     */
    void update_internal(const Eigen::Vector4d& bbox,
                         const Eigen::Vector3d& drone_pos,
                         const Eigen::Quaterniond& q_C_W) {
        // 检查输入有效性
        if (bbox.hasNaN() || drone_pos.hasNaN() || x_.hasNaN()) {
            return;
        }
        
        // 检查bbox尺寸有效性
        if (bbox(2) < 1.0 || bbox(3) < 1.0) {
            return;
        }
        
        // --- 计算3D投影测量和雅可比 ---
        Eigen::Vector3d p_target = x_.segment<3>(12);
        double W = x_(18);
        Eigen::Vector3d los_W = p_target - drone_pos;
        Eigen::Matrix3d R_W_C = q_C_W.toRotationMatrix().transpose();
        Eigen::Vector3d los_C = R_W_C * los_W;
        
        double x_c = los_C(0), y_c = los_C(1), z_c = los_C(2);

        if (x_c < 1.0) return; // 避免除零，目标必须在相机前方
        
        // 奇异性保护
        double r = los_W.norm();
        if (r < 2.0) {
            double scale = 5.0 / (r + 1e-5);
            x_.segment<3>(12) = drone_pos + los_W * scale;
            los_W = x_.segment<3>(12) - drone_pos;
            los_C = R_W_C * los_W;
            x_c = los_C(0); y_c = los_C(1); z_c = los_C(2);
        }

        // 预测的 Bearing-Angle 测量 (azimuth, elevation, visual_angle)
        // 相机坐标系(FLU): X前, Y左, Z上
        // 目标向右 (Y反向) -> 图像cx>px (向右) -> 期望az>0, 故 az = arctan2(-y_c, x_c)
        // 目标向下 (Z反向) -> 图像cy>py (向下) -> 期望el>0, 故 el = arctan2(-z_c, x_c)
        double az_pred = std::atan2(-y_c, x_c);
        double el_pred = std::atan2(-z_c, x_c);
        double alpha_pred = 2.0 * std::atan(W / (2.0 * x_c));

        // 从bbox计算实际测量的 az, el, alpha
        double bbox_u = bbox(0) - intrinsics_.px;
        double bbox_v = bbox(1) - intrinsics_.py;
        double az_meas = std::atan2(bbox_u, intrinsics_.fx);
        double el_meas = std::atan2(bbox_v, intrinsics_.fy);
        
        double size_meas = std::max(bbox(2), bbox(3));
        double f_avg = (intrinsics_.fx + intrinsics_.fy) / 2.0;
        double alpha_meas = 2.0 * std::atan(size_meas / (2.0 * f_avg));

        // 构建完整测量向量 [cx, cy, w, h, az, el, alpha]
        Vector7d z_full;
        z_full << bbox(0), bbox(1), bbox(2), bbox(3),
                  az_meas, el_meas, alpha_meas;

        // 预测测量
        Vector7d z_pred;
        z_pred << x_(0), x_(1), x_(2), x_(3),
                  az_pred, el_pred, alpha_pred;

        // 计算雅可比 H (7x19)
        static const Matrix7x19d H_base = [] {
            Matrix7x19d m = Matrix7x19d::Zero();
            m(0, 0) = 1.0; m(1, 1) = 1.0; m(2, 2) = 1.0; m(3, 3) = 1.0;
            return m;
        }();
        Matrix7x19d H = H_base;

        // 3D投影雅可比 (d[az,el,alpha]/d[px,py,pz,W])
        double den_az = x_c * x_c + y_c * y_c;
        if (den_az < 1e-10) den_az = 1e-10;
        Eigen::RowVector3d daz_dlosC(y_c / den_az, -x_c / den_az, 0.0);
        H.block<1, 3>(4, 12) = daz_dlosC * R_W_C;

        double den_el = x_c * x_c + z_c * z_c;
        if (den_el < 1e-10) den_el = 1e-10;
        Eigen::RowVector3d del_dlosC(z_c / den_el, 0.0, -x_c / den_el);
        H.block<1, 3>(5, 12) = del_dlosC * R_W_C;

        double den_alpha = 4.0 * x_c * x_c + W * W;
        H.block<1, 3>(6, 12) = (-4.0 * W / den_alpha) * R_W_C.row(0);
        H(6, 18) = 4.0 * x_c / den_alpha;

        // --- 卡尔曼更新 ---
        Vector7d y = z_full - z_pred;
        y(4) = std::remainder(y(4), 2.0 * M_PI);
        y(5) = std::remainder(y(5), 2.0 * M_PI);
        
        // 限制残差大小防止发散
        for (int i = 0; i < 7; ++i) {
            y(i) = std::clamp(y(i), -500.0, 500.0);
        }
        
        Eigen::Matrix<double, 19, 7> PH_T = P_ * H.transpose();
        Matrix7d S = H * PH_T + R_;

        // 使用LLT分解替代行列式检查
        Eigen::LLT<Matrix7d> llt_S(S);
        if (llt_S.info() != Eigen::Success) {
            return; // S矩阵不正定
        }

        // χ² 创新门检测: 拒绝线性化误差过大的异常更新
        // 7自由度χ²分布, 阈值24.3对应99.9%置信水平
        Vector7d S_inv_y = llt_S.solve(y);
        double chi2 = y.dot(S_inv_y);
        if (chi2 > 24.3) {
            return;
        }

        Eigen::Matrix<double, 19, 7> K = PH_T * llt_S.solve(Matrix7d::Identity());
        
        // 检查K是否有效
        if (K.hasNaN()) {
            return;
        }

        // ============================================================
        // W 可观测度评估 — 组合评分机制
        //
        // Test 1: α 方差分解 (随机不确定度检测)
        //   W 方差贡献占比 ρ = σ²_W / (σ²_pos + σ²_R)
        //   ρ < 0.5 → 位置 + 噪声主导, W 不可观
        //
        // Test 2: α 创新系统性偏差检测 (CV模型滞后检测)
        //   跟踪 α 创新的指数加权均值
        //   如果 |EMA(α_innov)| > 2·√(S_αα), 说明模型有系统性偏差
        //   (CV模型滞后导致的位置偏差), 此时冻结 W
        //
        // 两个条件必须同时通过才更新 W
        // ============================================================
        const double H_alpha_W = H(6, 18);
        const Eigen::RowVector3d H_alpha_p = H.block<1, 3>(6, 12);
        
        const double alpha_var_from_pos = (H_alpha_p * P_.block<3, 3>(12, 12) * H_alpha_p.transpose())(0, 0);
        const double alpha_var_from_W   = H_alpha_W * H_alpha_W * P_(18, 18);
        const double alpha_var_noise    = R_(6, 6);
        
        const double w_contribution_ratio = alpha_var_from_W / (alpha_var_from_pos + alpha_var_noise + 1e-20);
        const bool variance_test_pass = (w_contribution_ratio > 0.5);

        // Test 2: α 创新偏差跟踪
        const double alpha_innov = y(6);                  // 本次 α 创新
        const double alpha_innov_std = std::sqrt(S(6, 6)); // 创新的理论标准差
        // EMA (alpha ≈ 0.05, 满帧约20次更新稳定)
        alpha_innov_ema_ = 0.05 * alpha_innov + 0.95 * alpha_innov_ema_;
        // 归一化偏差: |EMA| / σ, 超过 2 表示 95% 置信下存在系统偏差
        const double normalized_bias = std::abs(alpha_innov_ema_) / (alpha_innov_std + 1e-20);
        const bool bias_test_pass = (normalized_bias < 2.0);
        
        w_observable_ = variance_test_pass && bias_test_pass;

        if (!w_observable_) {
            // 冻结 W: 将 K 的第 18 行置零
            K.row(18).setZero();
        }
        
        x_ = x_ + K * y;

        // Joseph形式协方差更新
        static const Matrix19d I = Matrix19d::Identity();
        Matrix19d I_KH = I - K * H;
        P_ = I_KH * P_ * I_KH.transpose() + K * R_ * K.transpose();
        P_ = 0.5 * (P_ + P_.transpose());
        
        // 物理约束: 目标宽度 W >= 0.1m
        if (x_(18) < 0.1) x_(18) = 0.1;

        // 更新计数 (用于tau_W衰减)
        update_count_++;

        // 最终NaN检查
        if (x_.hasNaN() || P_.hasNaN()) {
            // 恢复到更新前状态并重置
            reset();
        }
    }

    // 状态向量 (19维)
    Vector19d x_;
    // 协方差矩阵
    Matrix19d P_;
    // 过程噪声协方差
    Matrix19d Q_;
    // 测量噪声协方差
    Matrix7d R_;
    // 测量矩阵
    Matrix7x19d H_;

    // 相机参数
    CameraIntrinsics intrinsics_;
    double target_size_est_;

    // 历史缓冲区
    std::deque<StateHistory> history_buffer_;
    size_t max_buffer_size_;

    // 状态标志
    bool is_initialized_;
    int64_t last_timestamp_;
    int64_t dt_;

    // IMU缓存
    Eigen::Vector2d v_imu_cache_;
    Eigen::Vector3d last_gimbal_omega_frd_ = Eigen::Vector3d::Zero(); // 云台角速度 (FRD)

    // LOS角有限差分缓存
    double prev_azimuth_ = 0.0;
    double prev_elevation_ = 0.0;
    bool los_angle_initialized_ = false;

    // 3D运动学过程噪声参数 (在predict中动态构建含交叉协方差的Q矩阵)
    double q_accel_3d_ = 1.0;     // 加速度谱密度 (m²/s⁴), 对齐Python target_ekf q_accel=1.0
    double q_W_ = 2.5e-5;        // W随机游走方差率, 对齐Python q_W≈0.002 级

    // W估计参数 (对齐Python TargetEKF: tau_W=15, alpha_R_inflate=1.0)
    double tau_W_ = 15.0;          // W指数衰减时间常数 (>0启用, 0=禁用)
    double alpha_R_inflate_ = 1.0; // alpha测量噪声膨胀因子
    int update_count_ = 0;         // 更新计数 (用于tau_W衰减)

    // W 可观测度标志和创新偏差跟踪
    bool w_observable_ = true;
    double alpha_innov_ema_ = 0.0;  // α 创新的指数加权均值 (检测系统偏差)

    // EMA滤波器
    EMAFilterTemplate<2> ema_filter_;
    EMAFilterTemplate<2> imu_flow_filter_;
};

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
