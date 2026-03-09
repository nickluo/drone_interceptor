#include <chrono>
#include <cmath>

#include <Eigen/Dense>

#include "interceptor_node_quaternion.h"

// Control limits constants
// Refer to prototype/solver_flat_quaternion.py for corresponding limits
namespace ControlLimits {
    constexpr double ROLL_PITCH_RATE_MAX = 2.0 * M_PI;  // rad/s
    constexpr double YAW_RATE_MAX = 1.0 * M_PI;         // rad/s
    constexpr double THRUST_MIN = 1.0;                  // m/s^2
    constexpr double THRUST_MAX = 20.0;                 // m/s^2
};

using namespace std::chrono_literals;

InterceptorNodeQuaternion::InterceptorNodeQuaternion() 
    : Node("interceptor_node_quaternion") 
{
    declare_parameters();
    load_parameters();

    // 使用四元数版本的ACADOS求解器
    acados_capsule_ = drone_interceptor_mpc_quat_acados_create_capsule();
    int status = drone_interceptor_mpc_quat_acados_create(acados_capsule_);
    if (status) {
        RCLCPP_ERROR(this->get_logger(), "acados_create() returned status %d. Exiting.", status);
        // throw std::runtime_error("Failed to create ACADOS solver");
    }
    nlp_config_ = drone_interceptor_mpc_quat_acados_get_nlp_config(acados_capsule_);
    nlp_dims_ = drone_interceptor_mpc_quat_acados_get_nlp_dims(acados_capsule_);
    nlp_in_ = drone_interceptor_mpc_quat_acados_get_nlp_in(acados_capsule_);
    nlp_out_ = drone_interceptor_mpc_quat_acados_get_nlp_out(acados_capsule_);

    setup_ekf();
    int filter_n = static_cast<int>(RATE_TC_ * 2.0 / dt_);
    if (filter_n < 5) filter_n = 5;
    output_filter_ = std::make_unique<EMAFilter>(filter_n, 4);
    
    // 初始化 PN 加速度滤波器 (Smoothing constant N=2, channels=2) 降低延迟，提高对机动目标的响应速度
    pn_accel_filter_ = std::make_unique<EMAFilter>(2, 2);

    // 初始化四元数状态 - drone_state_ 已在声明时初始化有效四元数
    drone_state_ = VectorX::Zero();
    drone_state_(6) = 1.0;  // 四元数 w 分量初始化为1
    x_warm_start_.resize(N_HORIZON_ + 1, drone_state_);  // 直接使用有效初始状态
    u_warm_start_.resize(N_HORIZON_, VectorU::Zero());
    
    // 初始化控制输入为悬停状态
    for (int i = 0; i < N_HORIZON_; ++i) u_warm_start_[i] << 0, 0, 0, G_;

    sub_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    timer_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    
    control_pub_ = this->create_publisher<quadrotor_msgs::msg::ControlCommand>(
        "/fpv/control_command", 
        rclcpp::SensorDataQoS());
    mission_state_pub_ = this->create_publisher<std_msgs::msg::Bool>(
        "/fpv/mission_state", 10);

    status_publisher_ = std::make_unique<StatusPublisher>(*this, 2);

    if (use_simulation_) {
        RCLCPP_INFO(this->get_logger(), "Running in SIMULATION mode with QUATERNION dynamics.");
        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = sub_callback_group_;
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/fpv/odom", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(), 
            std::bind(&InterceptorNodeQuaternion::odom_callback, this, std::placeholders::_1), sub_options);
        timer_boot_count_ = 0;         
        timer_ = this->create_wall_timer(std::chrono::duration<double>(dt_), 
            std::bind(&InterceptorNodeQuaternion::simulation_loop, this),
            timer_callback_group_);
        // 使用状态机管理任务状态
        mission_state_.set(MissionState::INITIALIZING);
    } else {
        RCLCPP_INFO(this->get_logger(), "Running in REAL mode with QUATERNION dynamics.");
        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = sub_callback_group_;
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/fpv/odom", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(), 
            std::bind(&InterceptorNodeQuaternion::odom_callback, this, std::placeholders::_1), sub_options);
        gimbal_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/fpv/gimbal", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(), 
            std::bind(&InterceptorNodeQuaternion::gimbal_callback, this, std::placeholders::_1), sub_options);
        bbox_sub_ = this->create_subscription<vision_msgs::msg::Detection2D>(
            "/fpv/target_bbox", rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(), 
            std::bind(&InterceptorNodeQuaternion::bbox_callback, this, std::placeholders::_1), sub_options);
        mission_launch_sub_ = this->create_subscription<std_msgs::msg::Header>(
            "/fpv/tracker_trigger", 10, 
            std::bind(&InterceptorNodeQuaternion::mission_launch_callback, this, std::placeholders::_1));
    }
}
    
InterceptorNodeQuaternion::~InterceptorNodeQuaternion() 
{
    if (timer_) {
        timer_->cancel();
    }
    if (acados_capsule_) {
        drone_interceptor_mpc_quat_acados_free(acados_capsule_);
        drone_interceptor_mpc_quat_acados_free_capsule(acados_capsule_);
    }
}

void InterceptorNodeQuaternion::load_parameters() 
{
    this->get_parameter("use_simulation", use_simulation_);
    this->get_parameter("physics.g", G_);
    
    int ctrl_rate = 50;
    this->get_parameter("mpc.ctrl_rate", ctrl_rate);
    double TF = 1.0;
    this->get_parameter("mpc.tf", TF);
    N_HORIZON_ = static_cast<int>(TF * ctrl_rate);
    dt_ = TF / N_HORIZON_;


    this->get_parameter("mpc.v_desired", V_DESIRED_);
    this->get_parameter("mpc.tracking_distance", TRACKING_DISTANCE_);
    TRACKING_GAIN_ = 1.0 * std::sqrt(TRACKING_DISTANCE_ / V_DESIRED_);
    this->get_parameter("physics.drag_coeff", DRAG_COEFF_);
    this->get_parameter("perception.bbox_pos_std_dev", bbox_pos_std_dev_);
    this->get_parameter("perception.bbox_size_std_dev", bbox_size_std_dev_);
    this->get_parameter("perception.target_physical_size_est", target_physical_size_est_);
    this->get_parameter("perception.perspective_compensation", perspective_compensation_);
    this->get_parameter("perception.detection_latency", detection_latency_);
    this->get_parameter("control.execution_delay", control_execution_delay_);
    this->get_parameter("control.rate_tc", RATE_TC_);
    this->get_parameter("control.rate_control_enabled", rate_control_enabled_);
    // 自适应计算 ctrl_coeff_: 一阶系统在采样周期 dt_ 内的响应比例
    ctrl_coeff_ = 1.0 - std::exp(-dt_ / RATE_TC_);
    double max_history_time_factor;
    this->get_parameter("control.max_history_time_factor", max_history_time_factor);
    max_history_time_ = control_execution_delay_ * max_history_time_factor;
    this->get_parameter("limits.att_warning_deg", att_limit_);
    
    this->get_parameter("gimbal.rp_coeff", gimbal_rp_coeff_);
    this->get_parameter("gimbal.y_coeff", gimbal_y_coeff_);

    // 相机参数
    double cam_focal_mm, cam_pixel_um;
    int cam_width_px, cam_height_px;
    int native_width_px;
    double fov_deg;
    this->get_parameter("camera.fov_deg", fov_deg);
    this->get_parameter("camera.native_width_px", native_width_px);
    this->get_parameter("camera.width_px", cam_width_px);
    this->get_parameter("camera.height_px", cam_height_px);
    this->get_parameter("camera.focal_length_mm", cam_focal_mm);
    this->get_parameter("camera.pixel_size_um", cam_pixel_um);
    double cam_focal_length;
    if (fov_deg > 0.0) {
        cam_focal_length = cam_width_px * 0.5 / tan(fov_deg * M_PI / 180.0 / 2.0);
    } else {
        double scale_factor = static_cast<double>(cam_width_px) / static_cast<double>(native_width_px);
        cam_focal_length = cam_focal_mm * 1000.0 / cam_pixel_um * scale_factor;
    }
    cam_intrinsics_.fx = cam_focal_length;
    cam_intrinsics_.fy = cam_focal_length;
    cam_intrinsics_.px = cam_width_px / 2.0;
    cam_intrinsics_.py = cam_height_px / 2.0;

    if (use_simulation_)
    {
        std::vector<double> target_pos_vec, target_vel_vec;
        this->get_parameter("sim.initial_conditions.target_pos", target_pos_vec);
        this->get_parameter("sim.initial_conditions.target_vel", target_vel_vec);
        target_pos_true_ = Eigen::Map<Eigen::Vector3d>(target_pos_vec.data());
        target_vel_true_ = Eigen::Map<Eigen::Vector3d>(target_vel_vec.data());
        this->get_parameter("sim.interception_dist", interception_dist_);
    }

    // 代价函数权重
    std::vector<double> cost_weights_vec(8);
    this->get_parameter("cost.w_los", cost_weights_vec[0]);
    this->get_parameter("cost.w_align", cost_weights_vec[1]);
    this->get_parameter("cost.w_closing_speed", cost_weights_vec[2]);
    this->get_parameter("cost.w_rates", cost_weights_vec[3]);
    this->get_parameter("cost.w_tilt", cost_weights_vec[4]);
    this->get_parameter("cost.w_u_alpha", cost_weights_vec[5]);
    this->get_parameter("cost.w_u_thrust", cost_weights_vec[6]);
    this->get_parameter("cost.w_zem", cost_weights_vec[7]);
    
    for (int i=0; i < 8; ++i) w_cost_(i) = std::sqrt(cost_weights_vec[i]);
    if (TRACKING_DISTANCE_ > 0.001) {
        auto coeff = std::sqrt(10);
        auto los_coeff = std::sqrt(TRACKING_DISTANCE_ * 0.1);
        if (los_coeff > coeff) los_coeff = coeff;
        w_cost_(0) /= los_coeff; // 减小视线误差权重以促进保持距离跟踪的稳定性
        w_cost_(2) *= coeff; // 增加闭合速度权重以促进保持一定距离的跟踪
    }

    // PN Parameters    
    this->get_parameter("pn.enabled", pn_enabled_);
    this->get_parameter("pn.nav_constant", pn_nav_constant_);
    this->get_parameter("pn.switch_dist", pn_switch_dist_);
    this->get_parameter("pn.fov_switch_deg", pn_fov_switch_deg_);
    this->get_parameter("pn.max_accel_xy", pn_max_accel_xy_);
    this->get_parameter("pn.max_accel_z", pn_max_accel_z_);
}

void InterceptorNodeQuaternion::setup_ekf()
{
    // 创建DroneEKF - 使用图像坐标系的12维CA模型 + 3D位置估计 (19D)
    
    // 9维过程噪声: [std_pos_img, std_vel_img, std_accel_img, std_size, std_size_vel, std_accel_size, std_q_W, std_pos_3d, std_vel_3d]
    Eigen::Matrix<double, 9, 1> process_noise;
    process_noise << bbox_pos_std_dev_ * 0.1,   // std_pos_img
                    bbox_pos_std_dev_ * 0.5,   // std_vel_img
                    bbox_pos_std_dev_ * 0.25,  // std_accel_img
                    bbox_size_std_dev_ * 0.1,  // std_size
                    bbox_size_std_dev_ * 0.5,  // std_size_vel
                    bbox_size_std_dev_ * 0.25, // std_accel_size
                    0.002,                           // std_q_W → q_W=4e-6 (对齐vs1_3d.py级别, 収敛后W更稳定)
                    1.0,                             // sqrt(q_accel_3d) → 运动学CV加速度谱密度=1.0
                    0.0;                             // std_vel_3d (运动学模型中不再使用)
    
    // 3维测量噪声: [std_meas_pos, std_meas_size, std_meas_angle]
    Eigen::Vector3d measurement_noise;
    measurement_noise << bbox_pos_std_dev_, bbox_size_std_dev_, bbox_size_std_dev_ * 0.1;
    
    const int predicted_imu_freq = 100;
    ekf_ = std::make_unique<DroneEKF>(cam_intrinsics_, process_noise, measurement_noise,
                                        predicted_imu_freq, detection_latency_,
                                        target_physical_size_est_, perspective_compensation_);
}

void InterceptorNodeQuaternion::mission_launch_callback(const std_msgs::msg::Header::SharedPtr msg)
{
    // 中止任务
    if (msg->frame_id == "OFF" && mission_state_.is_active()) {
        RCLCPP_INFO(this->get_logger(), "Mission abort received! Stopping control...");
        status_publisher_->publish(
            mavros_msgs::msg::StatusText::WARNING, 
            1, "Mission aborted. Stopping...", true); 
        transition_to_state(MissionState::ABORTED, "User abort command");
        stop();
        return;
    }
    
    // 任务已启动或定时器已运行，忽略
    if (!mission_state_.is_terminated() || timer_)
        return;
        
    if (msg->frame_id == "MISSION") {
        RCLCPP_INFO(this->get_logger(), "Mission launch received! Starting control...");
        status_publisher_->publish(
            mavros_msgs::msg::StatusText::INFO, 
            1, "Mission launched. Starting...", true);
        // 重置状态
        {
            std::unique_lock<std::shared_mutex> lock(mtx_ekf_);  // 写入使用独占锁
            setup_ekf();
        }
        {
            std::unique_lock<std::shared_mutex> lock(mtx_drone_state_);  // 写入使用独占锁
            drone_state_.setZero();
            drone_state_(6) = 1.0;  // 设置有效的单位四元数 w=1
        }
        {
            std::lock_guard<std::mutex> lock(mtx_bbox_);
            current_bbox_ = vision_msgs::msg::Detection2D();
            is_visible_ = false;
        }
        
        output_filter_->reset();
        pn_accel_filter_->reset();
        control_history_.clear();

        odom_received_ = false;
        gimbal_received_ = false;

        timer_boot_count_ = 0;
        target_lost_counter_ = 0;
        solver_failure_counter_ = 0;
        consecutive_solver_failures_ = 0;
        
        // Reset guidance state
        pn_active_ = 0; // PN 初始状态设为关闭

        // 重置性能监控变量，确保每次任务启动行为一致
        solve_count_ = 0;
        avg_solve_time_us_ = 0.0;
        ekf_health_check_counter_ = 0;
        
        // 初始化状态设置 - 使用有效四元数状态
        for (int i = 0; i <= N_HORIZON_; ++i) x_warm_start_[i] = drone_state_;
        for (int i = 0; i < N_HORIZON_; ++i) u_warm_start_[i] << 0, 0, 0, G_;

        // 使用状态机设置状态
        mission_state_.set(MissionState::INITIALIZING);
        mission_state_pub_->publish(std::make_unique<std_msgs::msg::Bool>(std_msgs::msg::Bool().set__data(true)));
    }
}

void InterceptorNodeQuaternion::gimbal_callback(const sensor_msgs::msg::Imu::SharedPtr msg) 
{
    if (!mission_state_.is_active()) 
        return;
    gimbal_received_ = true;
    {
        std::unique_lock<std::shared_mutex> lock(mtx_gimbal_);  // 写入使用独占锁
        current_gimbal_orientation_ = Eigen::Quaterniond(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
    }
    {
        std::unique_lock<std::shared_mutex> lock(mtx_drone_state_);  // 写入使用独占锁
        // 利用协方差字段存储欧拉角 (r, p, y) for MPC use
        drone_state_.tail<3>() << Eigen::Vector3d(msg->orientation_covariance[0], msg->orientation_covariance[1], msg->orientation_covariance[2]);
    }
    {
        std::unique_lock<std::shared_mutex> lock(mtx_ekf_);  // 可能重置，使用独占锁
        if (ekf_->is_initialized()) {
            // 使用angular_velocity字段获取角速度 (FLU坐标系: roll, pitch, yaw)
            Eigen::Vector3d current_gimbal_rate;
            current_gimbal_rate << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
            // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            //     "Gimbal IMU (FLU): roll=%.3f pitch=%.3f yaw=%.3f rad/s (%.1f %.1f %.1f deg/s)",
            //     current_gimbal_rate(0), current_gimbal_rate(1), current_gimbal_rate(2),
            //     current_gimbal_rate(0) * 180.0/M_PI, current_gimbal_rate(1) * 180.0/M_PI, current_gimbal_rate(2) * 180.0/M_PI);
            ekf_->predict(rclcpp::Time(msg->header.stamp).nanoseconds() / 1000, current_gimbal_rate);
        }
    }
}
    
void InterceptorNodeQuaternion::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) 
{
    if (!mission_state_.is_active()) 
        return;
    Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                         msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    
    // Check for NaNs in odometry
    if (std::isnan(q.w()) || std::isnan(q.x()) || std::isnan(q.y()) || std::isnan(q.z()) ||
        std::isnan(msg->pose.pose.position.x) || std::isnan(msg->pose.pose.position.y) || std::isnan(msg->pose.pose.position.z)) {
        RCLCPP_ERROR(this->get_logger(), "NaN detected in odometry! Ignoring.");
        return;
    }

    // Eigen::Vector4d quat_normalized(q.w(), q.x(), q.y(), q.z());
    // 转换机体系速度到世界系速度
    Eigen::Vector3d vel_body(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);
    Eigen::Vector3d vel_world = q * vel_body;
    {
        std::unique_lock<std::shared_mutex> lock(mtx_drone_state_);  // 写入使用独占锁
        // 提取位置
        drone_state_.head<3>() << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
        // 提取速度
        drone_state_.segment<3>(3) = vel_world;
        // 提取四元数
        drone_state_.segment<4>(6) << q.w(), q.x(), q.y(), q.z();
        // 提取角速度
        drone_state_.segment<3>(10) << msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z;

        if (!gimbal_received_)
        {
            drone_state_.tail<3>() << Eigen::Vector3d::Zero();
        }
    }

    if (!gimbal_received_) {
        std::unique_lock<std::shared_mutex> lock(mtx_ekf_);  // 可能重置，使用独占锁
        if (ekf_->is_initialized()) {
            // Horizontal Gimbal Assumption: Camera does not roll/pitch with body.
            // We calculate world angular velocity and only pass the Z component (Yaw rate)
            Eigen::Vector3d w_body(msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z);
            Eigen::Vector3d w_world = q * w_body;
            
            // Pass [0, 0, yaw_rate] to EKF because camera is horizontal
            ekf_->predict(rclcpp::Time(msg->header.stamp).nanoseconds() / 1000, Eigen::Vector3d(0.0, 0.0, w_world.z()));
        }
    }

    update_control_history(rclcpp::Time(msg->header.stamp).seconds(), Eigen::Vector3d(
        msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z));

    if (mission_state_.is_terminated()) 
        return;
    
    if (!odom_received_)
    {
        {
            std::unique_lock<std::shared_mutex> lock(mtx_drone_state_);  // 写入使用独占锁
            for (int i = 0; i <= N_HORIZON_; ++i)
                x_warm_start_[i] = drone_state_;
        }
        for (int i = 0; i < N_HORIZON_; ++i) 
            u_warm_start_[i] << msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z, G_;
        odom_received_ = true;
    }
}

void InterceptorNodeQuaternion::bbox_callback(const vision_msgs::msg::Detection2D::SharedPtr msg) 
{ 
    if (!odom_received_ || !mission_state_.is_active())
        return;
    
    // 创建DroneEKF的Bbox结构 (使用像素坐标)
    DroneEKF::Bbox bbox;
    bbox.cx = msg->bbox.center.position.x;
    bbox.cy = msg->bbox.center.position.y;
    bbox.w = msg->bbox.size_x;
    bbox.h = msg->bbox.size_y;
    
    bool frame_valid = msg->header.frame_id != "";
    
    // ============================================
    // Bbox跳变检测：拒绝突然的位置/尺寸跳变
    // ============================================
    if (frame_valid && last_bbox_valid_) {
        // 计算中心点距离跳变
        double dx = bbox.cx - last_bbox_.cx;
        double dy = bbox.cy - last_bbox_.cy;
        double center_jump = std::sqrt(dx*dx + dy*dy);
        
        // 计算尺寸变化比例
        double size_ratio_w = (last_bbox_.w > 0) ? bbox.w / last_bbox_.w : 1.0;
        double size_ratio_h = (last_bbox_.h > 0) ? bbox.h / last_bbox_.h : 1.0;
        
        // 跳变阈值：最大允许150像素位置跳变，2倍尺寸变化
        const double max_center_jump = cam_intrinsics_.px * 0.3; // 约15%图像宽度
        const double max_size_ratio = 2.0;
        const double min_size_ratio = 0.5;
        
        bool position_jump = center_jump > max_center_jump;
        bool size_jump = size_ratio_w > max_size_ratio || size_ratio_w < min_size_ratio ||
                         size_ratio_h > max_size_ratio || size_ratio_h < min_size_ratio;
        
        if (position_jump || size_jump) {
            bbox_jump_counter_++;
            RCLCPP_WARN(this->get_logger(), 
                "Bbox jump detected! center_jump=%.1f, size_ratio=(%.2f, %.2f), counter=%d",
                center_jump, size_ratio_w, size_ratio_h, bbox_jump_counter_);
            
            // 连续3次跳变后重置跟踪状态
            if (bbox_jump_counter_ >= 3) {
                RCLCPP_ERROR(this->get_logger(), "Bbox jump persistent, resetting last_bbox");
                last_bbox_valid_ = false;
                bbox_jump_counter_ = 0;
            }
            // 拒绝此次检测，不更新EKF
            return;
        } else {
            bbox_jump_counter_ = 0;  // 重置跳变计数
        }
    }
    
    // 更新上一帧bbox用于跳变检测
    last_bbox_ = bbox;
    last_bbox_valid_ = frame_valid;
    
    {
        std::lock_guard<std::mutex> lock(mtx_bbox_);
        current_bbox_ = *msg;
        is_visible_ = frame_valid;
    }

    if (is_visible_)
    {
        // RCLCPP_INFO(this->get_logger(), "Received bbox: cx=%.2f, cy=%.2f, w=%.2f, h=%.2f", 
        //     bbox.cx, bbox.cy, bbox.w, bbox.h);

        int64_t detection_time = rclcpp::Time(msg->header.stamp).nanoseconds() / 1000;
        
        // 获取当前无人机状态和云台姿态 (Moved up for DroneEKF 3D update)
        VectorX curr_drone_state_for_ekf;
        {
           std::shared_lock<std::shared_mutex> lock(mtx_drone_state_);
           curr_drone_state_for_ekf = drone_state_;
        }
        
        Eigen::Quaterniond q_C_W;
        if (gimbal_received_) {
            std::shared_lock<std::shared_mutex> lock(mtx_gimbal_);
            q_C_W = current_gimbal_orientation_;
        } else {
            // Horizontal Gimbal: Only Yaw follows drone
            double psi = get_yaw_from_quaternion(curr_drone_state_for_ekf.segment<4>(6));
            Eigen::AngleAxisd yaw_rot(psi, Eigen::Vector3d::UnitZ());
            q_C_W = Eigen::Quaterniond(yaw_rot);
        }
        
        // ============================================
        // 直接从检测框计算LOS角度和角速率 (相机系)
        // ============================================
        detection_ratio_ = std::max(bbox.w / cam_intrinsics_.px , bbox.h / cam_intrinsics_.py);
        
        if (detection_ratio_ < 1.0) // 避免过大的bbox导致过大的噪声估计
        {
            std::unique_lock<std::shared_mutex> ekf_lock(mtx_ekf_);
            if (detection_ratio_ > 0.1) {
                const double coeff = std::exp((detection_ratio_ - 0.1) * 2); // 指数加权以增强近距离检测的影响
                const double bbox_pos_std_dev = coeff * bbox_pos_std_dev_;
                const double bbox_size_std_dev = coeff * bbox_size_std_dev_;
                const double target_physical_size_est = coeff * target_physical_size_est_;
                
                Eigen::Matrix<double, 9, 1> process_noise;
                // 接近阶段增大q_accel_3d: CV模型在机动时跟踪滞后,
                // 增大3D过程噪声使EKF将alpha残差归因于位置而非W
                const double q_accel_approach = std::max(1.0, coeff * coeff);
                
                process_noise << bbox_pos_std_dev * 0.1,   // std_pos_img
                                bbox_pos_std_dev * 0.5,   // std_vel_img
                                bbox_pos_std_dev * 0.25,  // std_accel_img
                                bbox_size_std_dev  * 0.1,  // std_size
                                bbox_size_std_dev * 0.5,  // std_size_vel
                                bbox_size_std_dev * 0.25, // std_accel_size
                                0.002,                    // std_q_W → q_W=4e-6 (对齐vs1_3d.py级别)
                                std::sqrt(q_accel_approach), // sqrt(q_accel_3d): 接近时增大以匹配机动动态
                                0.0;                      // std_vel_3d (运动学模型中不再使用)
                
                Eigen::Vector3d measurement_noise;
                // 注意: alpha噪声使用原始(未膨胀)值, 近距离alpha信号SNR更高不应降权
                measurement_noise << bbox_pos_std_dev, bbox_size_std_dev, bbox_size_std_dev_ * 0.1;
                
                ekf_->set_covariance(process_noise, measurement_noise);
            }
            // 执行update
            ekf_->update(detection_time, bbox, curr_drone_state_for_ekf.head<3>(), q_C_W);
        }
    }
    
    if (!timer_ && msg->header.frame_id != "" && mission_state_.is_active()) {
        timer_boot_count_ = 0;
        timer_ = this->create_wall_timer(std::chrono::duration<double>(dt_), 
            std::bind(&InterceptorNodeQuaternion::real_loop, this),
            timer_callback_group_);
        // 转换到跟踪状态
        mission_state_.transition(MissionState::INITIALIZING, MissionState::TRACKING);
    }
}

void InterceptorNodeQuaternion::failsafe(bool hover) 
{
    auto control_msg = std::make_unique<quadrotor_msgs::msg::ControlCommand>();
    control_msg->header.stamp = this->get_clock()->now();
    control_msg->control_mode = hover ? 
        quadrotor_msgs::msg::ControlCommand::ATTITUDE : 
        quadrotor_msgs::msg::ControlCommand::BODY_RATES;
    control_msg->armed = true;
    control_msg->bodyrates.x = 0;
    control_msg->bodyrates.y = 0;
    control_msg->bodyrates.z = 0;
    control_msg->orientation.w = 1.0;
    control_msg->orientation.x = 0.0;
    control_msg->orientation.y = 0.0;
    control_msg->orientation.z = 0.0;
    control_msg->collective_thrust = G_;
    control_pub_->publish(std::move(control_msg));
}

void InterceptorNodeQuaternion::real_loop() {
    if (!odom_received_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for odometry...");
        return;
    }
    
    // 定期检查EKF健康状态
    check_ekf_health();
    
    Eigen::Vector3d pos_est {0.0, 0.0, 0.0};
    Eigen::Vector3d vel_est {0.0, 0.0, 0.0};
    
    // 获取当前无人机状态和云台姿态
    VectorX curr_drone_state;
    Eigen::Quaterniond gimbal_orientation;
    {
        std::shared_lock<std::shared_mutex> lock(mtx_drone_state_);
        curr_drone_state = drone_state_;
    }
    
    if (gimbal_received_) {
        std::shared_lock<std::shared_mutex> lock(mtx_gimbal_);
        gimbal_orientation = current_gimbal_orientation_;
    } else {
        // Camera on horizontal gimbal (only yaw follows drone)
        double psi = get_yaw_from_quaternion(curr_drone_state.segment<4>(6));
        Eigen::AngleAxisd yaw_rot(psi, Eigen::Vector3d::UnitZ());
        gimbal_orientation = Eigen::Quaterniond(yaw_rot);
    }
    

    {
        std::shared_lock<std::shared_mutex> lock(mtx_ekf_);  // 只读取状态，使用共享锁
        if (ekf_->is_initialized())
        {
            // 不在这里调用 predict - 回调函数 (gimbal_callback/odom_callback) 已经在做这件事
            // 重复调用 predict 会导致状态被多次传播，引起速度发散
            
            // 直接获取世界系下的状态
            Eigen::VectorXd est_state = ekf_->get_state();
            pos_est = est_state.head<3>();
            vel_est = est_state.tail<3>();
            
            // 检查EKF状态是否有效
            if (!ekf_->is_healthy()) {
                RCLCPP_ERROR(this->get_logger(), "NaN in EKF state, reinitializing...");
                // 需要升级到独占锁来重置
                lock.unlock();
                std::unique_lock<std::shared_mutex> write_lock(mtx_ekf_);
                ekf_->reset();
                return;
            }
        }
        else
            return; // 等待初始化
    }
    run_control_cycle(pos_est, vel_est);
}

void InterceptorNodeQuaternion::simulation_loop() 
{
    if (!odom_received_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for odometry...");
        return;
    }
    run_control_cycle(target_pos_true_, target_vel_true_);
    target_pos_true_ += target_vel_true_ * dt_; // 仅用于测试
    double distance = (target_pos_true_ - drone_state_.head<3>()).norm();
    // 节流日志，每500ms输出一次
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, 
        "Distance to target: %.2f m", distance);
    if (distance < interception_dist_) {
        RCLCPP_INFO(this->get_logger(), "[SUCCESS] Target intercepted at distance %.2f m. Stopping simulation.", distance);
        transition_to_state(MissionState::COMPLETED, "Target intercepted");
        stop();
    }
}

void InterceptorNodeQuaternion::stop() 
{
    if (timer_) {
        timer_->cancel();
        timer_ = nullptr;
    }
    // 状态机由调用者设置（COMPLETED 或 ABORTED）
    failsafe(true);
    mission_state_pub_->publish(std::make_unique<std_msgs::msg::Bool>(std_msgs::msg::Bool().set__data(false)));
}

void InterceptorNodeQuaternion::run_control_cycle(const Eigen::Vector3d &pos_est, const Eigen::Vector3d &vel_est) 
{
    VectorX curr_drone_state;
    {
        std::shared_lock<std::shared_mutex> lock(mtx_drone_state_);  // 读取使用共享锁
        curr_drone_state = drone_state_;
    }

    if (is_visible_) {
        target_lost_counter_ = 0;
        // 从盲飞状态恢复到跟踪状态
        mission_state_.transition(MissionState::BLIND_FLIGHT, MissionState::TRACKING);
    } 
    else 
    {
        if (target_lost_counter_ == 0) {
            // 进入盲飞状态
            mission_state_.transition(MissionState::TRACKING, MissionState::BLIND_FLIGHT);
        }
        if (target_lost_counter_ % 10 == 0) {
            status_publisher_->publish(
                mavros_msgs::msg::StatusText::WARNING, 
                1, "Target lost. Blind flight", false);
        }
        target_lost_counter_++;
        // 使用节流日志减少I/O开销
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500, 
            "Target lost! Blind flight count: %d/%d", target_lost_counter_, N_HORIZON_);
    }

    int target_lost_counter_limit = N_HORIZON_ * 2;
    if (pn_active_ > 0) {
        target_lost_counter_limit = N_HORIZON_;
    }

    if (target_lost_counter_ > target_lost_counter_limit) {
        RCLCPP_ERROR(this->get_logger(),
            "Target lost for too long. Aborting.");
        status_publisher_->publish(
            mavros_msgs::msg::StatusText::ERROR, 
            1, "Target lost. Aborting.", true);
        transition_to_state(MissionState::ABORTED, "Target lost too long");
        stop();
        return;
    }
    
    // 验证当前状态
    if (!validate_state(curr_drone_state, "current_state")) {
        failsafe(false);
        return;
    }


    // 检查姿态安全
    if (!check_attitude_safety(curr_drone_state)) {
        // 姿态接近极限，记录但继续（MPC会处理）
    }
    
    // 使用延迟补偿的预测状态设置初始条件约束
    VectorX predicted_state = predict_delayed_state(curr_drone_state, control_execution_delay_);
    // 验证预测状态
    if (!validate_state(predicted_state, "predicted_state")) {
        RCLCPP_ERROR(this->get_logger(), "Invalid predicted state! Resetting solver.");
        failsafe(false);
        output_filter_->reset();
        // Reset warm start
        for (int i = 0; i <= N_HORIZON_; ++i) x_warm_start_[i] = curr_drone_state;
        for (int i = 0; i < N_HORIZON_; ++i) u_warm_start_[i] << 0, 0, 0, G_;
        return;
    }

    // 验证控制输出
    // Handle PN latching state transition
    

    // RCLCPP_INFO(this->get_logger(), "PN Guidance: Enabled=%d, Active=%d", pn_enabled_ ? 1 : 0, pn_active_ ? 1 : 0);
    
    if (pn_enabled_ && pn_active_ == 0 && !std::isnan(pos_est.x()) && !std::isnan(vel_est.x())) {
        // 计算接近速度估计 (用于PN)
        Eigen::Vector3d rel_pos = pos_est - curr_drone_state.head<3>();
        Eigen::Vector3d rel_vel = vel_est - curr_drone_state.segment<3>(3);
        double closing_speed_est = 1.0;  // 用于PN的接近速度估计
        double dist_to_target = rel_pos.norm();
        if (dist_to_target > 1e-3) {
            closing_speed_est = -rel_pos.dot(rel_vel) / dist_to_target;
        }
        
        // 使用DroneEKF的stateToLosAngles获取LOS角度判断是否切换到PN
        bool los_valid = false;
        double azimuth = 0.0, elevation = 0.0;
        Eigen::Vector4d los_state;
        {
            std::shared_lock<std::shared_mutex> lock(mtx_ekf_);
            if (ekf_ && ekf_->is_initialized() && is_visible_) {
                los_state = ekf_->stateToLosAngles(true);
                azimuth = los_state(0);
                elevation = los_state(1);
                los_valid = !std::isnan(azimuth) && !std::isnan(elevation);
            }
        }
        
        if (los_valid) {
            // 计算LOS偏离视轴角度 (相机系)
            double angle_off_boresight = std::sqrt(azimuth * azimuth + elevation * elevation);
            // 收紧FOV阈值，确保在更接近真实的碰撞航线上时才进入PN
            bool target_in_fov_center = (angle_off_boresight < pn_fov_switch_deg_ * M_PI / 180.0);
            
            RCLCPP_INFO(this->get_logger(), "LOS angles (az,el): [%.2f, %.2f] deg, Off-boresight: %.2f deg, In FOV: %d", 
                azimuth * 180.0 / M_PI, elevation * 180.0 / M_PI,
                angle_off_boresight * 180.0 / M_PI, target_in_fov_center ? 1 : 0);
            RCLCPP_INFO(this->get_logger(), "Distance to target: %.2f m, Closing speed: %.2f m/s, Estimated width: %.2f m [W %s]", 
                dist_to_target, closing_speed_est, ekf_->getEstimatedWidth(),
                ekf_->isWidthObservable() ? "OBS" : "FROZEN");
            // Diagnostic: print EKF internal decomposition
            auto full_state = ekf_->getFullState();
            auto v_imu = ekf_->getImuFlowCache();
            RCLCPP_INFO(this->get_logger(), 
                "EKF diag: vx_img=%.1f vy_img=%.1f | v_imu=[%.1f,%.1f] | ax=%.1f ay=%.1f | LOS_rate=[%.2f,%.2f]°/s",
                full_state(4), full_state(5), v_imu(0), v_imu(1),
                full_state(8), full_state(9),
                los_state(2) * 180.0 / M_PI, los_state(3) * 180.0 / M_PI);
                
            // ============================================================
            // 模糊评分PN切换判定
            // 每个条件贡献0~1分, 距离紧迫度提升容忍度
            // ============================================================
            double angle_off_boresight_deg = angle_off_boresight * 180.0 / M_PI;
            double vertical_speed = std::abs(rel_vel.z());
            double max_vertical_speed_for_pn = std::max(2.0, closing_speed_est * 0.5);
            double los_rate_check = std::sqrt(los_state(2)*los_state(2) + 
                                              los_state(3)*los_state(3));
            double los_rate_check_deg = los_rate_check * 180.0 / M_PI;
            
            // ============================================================
            // 距离自适应FOV阈值: 近距离时MPC无法维持紧指向
            // MPC在机动目标上的off-boresight约为: 3-5° @ 10m
            // 因此近距离应放宽FOV要求
            // ============================================================
            double dist_factor = std::clamp((pn_switch_dist_ * 1.5 - dist_to_target) / pn_switch_dist_, 0.0, 1.5);
            double effective_fov_deg = pn_fov_switch_deg_ * (1.0 + dist_factor);  // 2° → 最大5° at close range
            double max_los_rate_deg = 3.0 * effective_fov_deg;  // LOS rate阈值也自适应
            
            // 各因子评分 (0=差, 1=好), 使用平滑sigmoid
            // FOV评分: 使用距离自适应阈值
            double fov_score = 1.0 / (1.0 + std::exp(4.0 * (angle_off_boresight_deg / effective_fov_deg - 1.0)));
            // 垂直速度评分
            double vz_score = 1.0 / (1.0 + std::exp(4.0 * (vertical_speed / max_vertical_speed_for_pn - 1.0)));
            // LOS角速度评分: 使用自适应阈值
            double los_rate_score = 1.0 / (1.0 + std::exp(4.0 * (los_rate_check_deg / max_los_rate_deg - 1.0)));
            
            // 距离紧迫度: 越近越宽容 (在1.5*switch_dist处为1.0, 在switch_dist处为1.33)
            double dist_ratio = dist_to_target / (pn_switch_dist_ * 1.5);
            double urgency = 1.0 + 0.5 * (1.0 - dist_ratio);
            
            // 加权评分 (FOV权重最高)
            double pn_score = (0.5 * fov_score + 0.25 * vz_score + 0.25 * los_rate_score) * urgency;
            const double pn_switch_threshold = 0.60;
            
            // 硬性兜底: 距离 < switch_dist时, 只要目标可见就强制切换
            // 不等到0.5*switch_dist, 因为那时MPC已经严重偏离
            bool force_switch = (dist_to_target < pn_switch_dist_) && is_visible_;
            
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 100,
                "PN score: %.2f (fov=%.2f[eff=%.1f°] vz=%.2f los=%.2f urgency=%.2f) thresh=%.2f %s",
                pn_score, fov_score, effective_fov_deg, vz_score, los_rate_score, urgency, 
                pn_switch_threshold, force_switch ? "[FORCE]" : "");
            
            if ((dist_to_target < pn_switch_dist_ * 1.5 && pn_score >= pn_switch_threshold) || force_switch) {
                pn_vc_initial_ = closing_speed_est;
                pn_active_ = 1; // Latch PN active
                pn_target_vel_valid_ = false;  // 重置APN的目标速度历史
                pn_target_accel_filtered_ = Eigen::Vector3d::Zero();  // 重置APN滤波器
                pn_apn_warmup_counter_ = 0;  // 重置warmup计数器
                RCLCPP_INFO(this->get_logger(), 
                    "Switching to IMAGE-BASED PN (Latch Active, score=%.2f%s), Thrust Efficiency: %.3f", 
                    pn_score, force_switch ? " FORCED" : "", thrust_efficiency_);
            }
        }
    }
    
    VectorU u_optimal_raw;

    if (pn_active_ > 0) {
        // Use Image-Based PN Guidance (直接使用检测框计算的LOS角速率)
        u_optimal_raw = calculate_pn_controls_image_based(curr_drone_state, pos_est, vel_est);
        // Reset solver failure counter as we are not relying on it
        consecutive_solver_failures_ = 0;
    } else {
        // Use MPC
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", predicted_state.data());
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", predicted_state.data());

        VectorP p_vec;
        for (int k = 0; k <= N_HORIZON_; ++k) {
            Eigen::Vector3d predicted_target_pos = pos_est + (k * dt_) * vel_est;
            double tracking_coeff = 1.0;
            if (TRACKING_DISTANCE_ > 0.001) // Tracking logic if distance is set
            {
                VectorX x_k;
                ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "x", x_k.data());
                auto x_k_pos = x_k.head<3>();
                double dist_to_target = (predicted_target_pos - x_k_pos).norm();
                double dist_error = (dist_to_target - TRACKING_DISTANCE_) / TRACKING_DISTANCE_;
                tracking_coeff = 2.0/(1.0 + std::exp(-TRACKING_GAIN_ * dist_error)) - 1.0;
                // RCLCPP_INFO(this->get_logger(), "Step %d: Dist to target: %.2f m, Tracking coeff: %.2f", k, dist_to_target, tracking_coeff);
            }
            p_vec << predicted_target_pos, vel_est, DRAG_COEFF_, tracking_coeff * V_DESIRED_, RATE_TC_, w_cost_, gimbal_rp_coeff_, gimbal_y_coeff_;
            drone_interceptor_mpc_quat_acados_update_params(acados_capsule_, k, p_vec.data(), p_vec.size());
        }

        // 设置热启动
        for (int i = 0; i < N_HORIZON_; ++i) {
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "x", x_warm_start_[i].data());
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "u", u_warm_start_[i].data());
        }
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, N_HORIZON_, "x", x_warm_start_[N_HORIZON_].data());

        // 求解
        auto t_start = std::chrono::high_resolution_clock::now();
        int status = drone_interceptor_mpc_quat_acados_solve(acados_capsule_);
        auto t_end = std::chrono::high_resolution_clock::now();
        
        // 性能监控：计算平均求解时间
        double solve_time_us = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());
        solve_count_++;
        avg_solve_time_us_ = avg_solve_time_us_ + (solve_time_us - avg_solve_time_us_) / solve_count_;
        
        // 节流日志输出，每秒1次
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
            "ACADOS solve: %.0f us (avg: %.0f us)", solve_time_us, avg_solve_time_us_);
        
        if (status != ACADOS_SUCCESS)
        {
            // 使用增强的故障恢复策略
            SolverRecoveryAction action = handle_solver_failure(status);
            
            switch (action) {
                case SolverRecoveryAction::CONTINUE:
                    // 使用上一次的解，继续执行
                    RCLCPP_DEBUG(this->get_logger(), "Solver failed, using previous solution");
                    break;
                    
                case SolverRecoveryAction::RESET_WARM:
                    // 重置热启动并发送failsafe
                    reset_solver_warm_start(curr_drone_state);
                    failsafe(false);
                    output_filter_->reset();
                    return;
                    
                case SolverRecoveryAction::FAILSAFE:
                    // 进入failsafe模式
                    transition_to_state(MissionState::FAILSAFE, "Solver failures");
                    failsafe(true);
                    output_filter_->reset();
                    return;
                    
                case SolverRecoveryAction::ABORT:
                    // 终止任务
                    RCLCPP_ERROR(this->get_logger(), "ACADOS solver failed too many times. Aborting.");
                    status_publisher_->publish(
                        mavros_msgs::msg::StatusText::ERROR, 
                        1, "ACADOS solver failed. Aborting.", true);    
                    transition_to_state(MissionState::ABORTED, "Solver failed");
                    stop();
                    return;
            }
        } else {
            // 求解成功，重置失败计数
            consecutive_solver_failures_ = 0;
            
            // ================================================================
            // 自适应推力效率学习 (在MPC阶段收集数据)
            // 通过比较指令推力和实际加速度来估计推力效率
            // ================================================================
            if (last_thrust_cmd_ > 1.0) {
                // 计算期望的垂直加速度 (假设完美推力效率)
                Eigen::Quaterniond q_curr(curr_drone_state(6), curr_drone_state(7), 
                                          curr_drone_state(8), curr_drone_state(9));
                Eigen::Matrix3d R_B_W_curr = q_curr.toRotationMatrix();
                Eigen::Vector3d thrust_dir_world = R_B_W_curr.col(2);  // 机体Z轴在世界系中的方向
                Eigen::Vector3d expected_accel = thrust_dir_world * last_thrust_cmd_ - Eigen::Vector3d(0, 0, G_);
                
                // 计算实际加速度 (通过速度差分)
                Eigen::Vector3d current_vel = curr_drone_state.segment<3>(3);
                Eigen::Vector3d accel_measured = (current_vel - last_accel_world_) / dt_;
                last_accel_world_ = current_vel;  // 保存当前速度用于下次计算
                
                // 只在非极端机动时更新估计 (避免噪声干扰)
                double expected_accel_z = expected_accel.z();
                double actual_accel_z = accel_measured.z();
                if (std::abs(expected_accel_z) > 0.5 && std::abs(actual_accel_z) > 0.1) {
                    double efficiency_sample = actual_accel_z / expected_accel_z;
                    // 限制合理范围
                    efficiency_sample = std::clamp(efficiency_sample, 0.5, 1.5);
                    // 禁用动态推力估计，因为它在高速追击的大倾角下（阻力大）会严重低估推力，
                    // 导致 PN 阶段推力指令补偿过大（推力乘子高达 1/0.5=2.0），从而让倾角超限(>45度)。
                    // thrust_efficiency_ = 0.9 * thrust_efficiency_ + 0.1 * efficiency_sample;
                }
            }
            last_thrust_cmd_ = u_optimal_raw(3);
        }

        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", u_optimal_raw.data());
    }
    
    // Warm start update
    if (pn_active_ == 0) {
        // 验证控制输出
        if (!validate_control(u_optimal_raw)) {
            RCLCPP_WARN(this->get_logger(), "Invalid control output, applying limits");
        }
        // 应用控制限幅
        u_optimal_raw = apply_control_limits(u_optimal_raw);
        for (int i = 0; i < N_HORIZON_ - 1; ++i) {
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i + 1, "x", x_warm_start_[i].data());
            // 正确的四元数归一化方式
            x_warm_start_[i].segment<4>(6) = normalize_quaternion(x_warm_start_[i].segment<4>(6));
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i + 1, "u", u_warm_start_[i].data());
        }
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, N_HORIZON_, "x", x_warm_start_[N_HORIZON_ - 1].data());
        // 正确的四元数归一化方式
        x_warm_start_[N_HORIZON_ - 1].segment<4>(6) = normalize_quaternion(x_warm_start_[N_HORIZON_ - 1].segment<4>(6));
        x_warm_start_[N_HORIZON_] = x_warm_start_[N_HORIZON_ - 1];
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, N_HORIZON_ - 1, "u", u_warm_start_[N_HORIZON_ - 1].data());
    } 
    // else {
    //     // In PN mode, we should perhaps propagate state simply to keep x_warm_start_ valid if we switch back?
    //     // For now, let's just keep x_warm_start_ as current state replicated
    //     for (int i = 0; i <= N_HORIZON_; ++i) x_warm_start_[i] = curr_drone_state;
    // }

    double thrust = 0;
    Eigen::Vector3d body_rates;

    Eigen::VectorXd u_filtered_dynamic = output_filter_->filter(u_optimal_raw);

    if (timer_boot_count_++ < N_HORIZON_ * 1) {  // 前几次发送悬停控制，等待滤波器稳定
        failsafe(false);  // 发送悬停指令而非不发送任何控制
        return;
    }

    // 提取控制量
    if (pn_active_ == 1 && detection_ratio_ > 1.0) {
        pn_active_ = 2; // 切换到盲飞阶段
        thrust = pn_last_u_pn_(3);
        body_rates << pn_last_u_pn_(0), pn_last_u_pn_(1), pn_last_u_pn_(2);
    }
    else {
        thrust = u_filtered_dynamic(3);
        body_rates << u_filtered_dynamic(0), u_filtered_dynamic(1), u_filtered_dynamic(2);
    }


    quadrotor_msgs::msg::ControlCommand::UniquePtr control_msg = 
        std::make_unique<quadrotor_msgs::msg::ControlCommand>();
    control_msg->header.stamp = this->get_clock()->now();
    control_msg->armed = true;
    control_msg->collective_thrust = thrust;
    if (rate_control_enabled_)
    {
        control_msg->control_mode = quadrotor_msgs::msg::ControlCommand::BODY_RATES;
        // ctrl_coeff_ = 1 - exp(-dt/rate_tc): 一阶系统离散响应系数，补偿MPC模型与实际飞控角速度响应的差异
        control_msg->bodyrates.x = body_rates(0) * ctrl_coeff_;
        control_msg->bodyrates.y = body_rates(1) * ctrl_coeff_;
        control_msg->bodyrates.z = body_rates(2) * ctrl_coeff_;
    }
    else
    {
        control_msg->control_mode = quadrotor_msgs::msg::ControlCommand::ATTITUDE;
        control_msg->expected_execution_time.sec = static_cast<int>(dt_);
        control_msg->expected_execution_time.nanosec = static_cast<int>((dt_ - static_cast<int>(dt_)) * 1e9);
        control_msg->orientation.w = curr_drone_state(6);
        control_msg->orientation.x = curr_drone_state(7);
        control_msg->orientation.y = curr_drone_state(8);
        control_msg->orientation.z = curr_drone_state(9);
        control_msg->bodyrates.x = body_rates(0);
        control_msg->bodyrates.y = body_rates(1);
        control_msg->bodyrates.z = body_rates(2);
    }
    // RCLCPP_INFO(this->get_logger(), "Publishing control: p: %.2f, q: %.2f, r: %.2f, thrust: %.2f", 
    //     body_rates(0), body_rates(1), body_rates(2), thrust);
    control_pub_->publish(std::move(control_msg));
}

// ============================================================================
// Image-Based PN Guidance Implementation (直接使用检测框计算的LOS角速率)
// 
// 比例导引法 (Proportional Navigation, PN) 原理:
// 要求侧向加速度与LOS角速率成正比: a_cmd = N * Vc * LOS_rate
// 其中 N = 导航常数 (通常3-5), Vc = 接近速度
//
// 坐标系约定:
// - 相机系: X前, Y右, Z下 (标准相机坐标系)
// - 机体系 (FLU): X前, Y左, Z上  
// - 世界系 (ENU): X东, Y北, Z上
//
// LOS角度约定 (相机系):
// - azimuth > 0: 目标在相机右边
// - elevation > 0: 目标在相机下方
// - azimuth_rate > 0: 目标向右移动
// - elevation_rate > 0: 目标向下移动
//
// 机体系FLU下的四旋翼动力学 (右手定则):
// - 正roll (绕X轴正向旋转) = 右翼下压/左翼抬起 = 向右倾斜 = 产生向右加速度 (负Y方向)
// - 正pitch (绕Y轴正向旋转) = 机头下压/机尾抬起 = 向前倾斜 = 产生向前加速度 (正X方向)
// - 负pitch = 机头抬起 = 向后倾斜 = 产生向后加速度 (负X方向)
// - 推力沿机体Z轴正向 (向上)
// ============================================================================
VectorU InterceptorNodeQuaternion::calculate_pn_controls_image_based(const VectorX& drone_state, 
    const Eigen::Vector3d &pos_est, const Eigen::Vector3d &vel_est)
{
    VectorU u_pn;
    u_pn << 0.0, 0.0, 0.0, G_;  // 默认悬停
    
    // Blind Mode / Inertial Terminal Phase (State 2)
    // Hold the last valid command to avoid glint error jumps
    if (pn_active_ == 2) {
        return pn_last_u_pn_;
    }
    
    // 从EKF获取LOS角度和角速率 (仅用于日志和偏航控制)
    Eigen::Vector4d los_state;
    {
        std::shared_lock<std::shared_mutex> lock(mtx_ekf_);
        if (!ekf_ || !ekf_->is_initialized()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 100, 
                "PN: EKF not initialized, returning hover");
            return u_pn;
        }
        los_state = ekf_->stateToLosAngles(true);
    }
    
    double azimuth = los_state(0);       // 方位角 (弧度), 正=目标在相机右边
    double elevation = los_state(1);     // 俯仰角 (弧度), 正=目标在相机下方
    double azimuth_rate = los_state(2);  // 方位角速率 (弧度/秒)
    double elevation_rate = los_state(3); // 俯仰角速率 (弧度/秒)
    
    // 检查有效性
    if (std::isnan(azimuth_rate) || std::isnan(elevation_rate)) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 100, 
            "PN: Invalid LOS rates (NaN), returning hover");
        return u_pn;
    }
    
    // ============================================================================
    // Miss检测: 目标偏离视轴过大时进入末段惯性飞行
    // 在追击阶段如果目标大幅偏离视轴，说明已经miss，继续PN只会发散
    // ============================================================================
    double off_boresight_rad = std::sqrt(azimuth * azimuth + elevation * elevation);
    double off_boresight_deg = off_boresight_rad * 180.0 / M_PI;
    if (off_boresight_deg > 30.0 && pn_active_ == 1) {
        pn_active_ = 2;
        pn_last_u_pn_.head<3>() *= 0.3;  // 大幅降低角速度指令
        RCLCPP_WARN(this->get_logger(), 
            "PN: Miss detected! Off-boresight=%.1f° > 30°, switching to terminal coast", 
            off_boresight_deg);
        return pn_last_u_pn_;
    }

    // ============================================================================
    // 3D True Proportional Navigation (TPN) - 直接在世界系中计算
    // 
    // 原理: 使PN加速度垂直于LOS方向，消除LOS角速率
    // 公式: a_cmd = N * Vc * Omega_LOS (向量形式)
    // 其中 Omega_LOS = (r × v_rel) / |r|² 是LOS角速度向量
    // 
    // TPN命令: a_pn = N * Vc * (Omega_LOS × r_hat)
    //        = N * Vc * [(r × v_rel) / |r|² × r_hat]
    //        = N * Vc / |r| * [(r_hat × v_rel) × r_hat]
    //        = N * Vc / |r| * [v_rel - (v_rel · r_hat) * r_hat]  (垂直于r的v_rel分量)
    // ============================================================================
    
    Eigen::Vector3d drone_pos = drone_state.head<3>();
    Eigen::Vector3d drone_vel = drone_state.segment<3>(3);
    
    Eigen::Vector3d rel_pos = pos_est - drone_pos;  // 目标相对无人机位置
    Eigen::Vector3d rel_vel = vel_est - drone_vel;  // 目标相对无人机速度
    
    double dist_to_target = rel_pos.norm();
    if (dist_to_target < 0.1) {
        RCLCPP_WARN(this->get_logger(), "PN: Target too close (%.2fm)", dist_to_target);
        return pn_last_u_pn_;
    }
    
    Eigen::Vector3d los_unit = rel_pos / dist_to_target;  // LOS单位向量
    
    // 接近速度 (正=靠近)
    double closing_speed = -rel_pos.dot(rel_vel) / dist_to_target;
    
    // 确保接近速度为正 (使用合理的下限，避免PN增益放大)
    if (closing_speed < 1.0) {
        closing_speed = 3.0;  // 固定下限，不使用pn_vc_initial_以免过度放大
    }
    
    // 估计拦截时间
    double tti = dist_to_target / closing_speed;
    
    // ============================================================================
    // 末制导切换
    // ============================================================================
    const double terminal_dist_threshold = ekf_->getEstimatedWidth() * 2.5;
    const double terminal_tti_threshold = 0.2;
    if ((dist_to_target < terminal_dist_threshold || tti < terminal_tti_threshold) && pn_active_ == 1) {
        pn_active_ = 2;
        pn_last_u_pn_.head<3>() *= 0.5; // Retain partial correction instead of coasting blind
        RCLCPP_INFO(this->get_logger(), 
            "PN: Entering terminal phase (dist=%.2fm, TTI=%.2fs), coasting with zero angular rates", 
            dist_to_target, tti);
        return pn_last_u_pn_;
    }
    
    // ============================================================================
    // 3D PN加速度计算
    // 
    // LOS角速度向量: Omega = (r × v_rel) / |r|²
    // PN加速度: a_pn = N * Vc * (Omega × los_unit)
    // ============================================================================
    double base_N = pn_nav_constant_;
    
    // 动态调整N值：接近速度极快时缩小N，避免指令过度放大导致的振荡
    double adaptive_N = base_N;
    if (closing_speed > 10.0) {
        adaptive_N = base_N * (10.0 / closing_speed);
    }
    
    // 末段渐弱: TTI < 0.4s 时线性降低N, 防止近距离剧烈振荡
    // 0.4s 对应约4m距离(Vc=10m/s), 此时目标已非常近, 检测噪声大
    // 注意: 对于机动目标，不能过早衰减，否则末段修正能力不足
    if (tti < 0.4) {
        adaptive_N *= std::max(0.3, tti / 0.4);  // 最低降到30%
    }
    
    adaptive_N = std::max(2.0, adaptive_N); // 保证最小制导增益
    
    // ============================================================================
    // 计算LOS角速度向量 (世界系 ENU)
    // ============================================================================
    Eigen::Vector3d omega_los; 
    
    if (gimbal_received_) {
        // current_gimbal_orientation_ (来自于AirSim发布的gimbal)
        // 在 tracker 节点中已经被转换为标准的 ROS ENU 坐标系表示：
        // 它的物理意义是：从相机机体坐标系(FLU: 前左上) 到 AirSim世界系(ENU: 东北天) 的旋转。
        std::shared_lock<std::shared_mutex> gimbal_lock(mtx_gimbal_);
        Eigen::Quaterniond q_FLU_ENU = current_gimbal_orientation_;
        
        // 构造Camera Body (FLU)坐标系下的LOS角速率向量
        // elevation_rate > 0 (目标在画面中向下移) -> 视线绕FLU左轴(Y)做正旋转
        // azimuth_rate > 0 (目标在画面中向右移) -> 视线绕FLU上轴(Z)做负旋转
        Eigen::Vector3d omega_los_FLU(0.0, elevation_rate, -azimuth_rate);
        
        // 转换到 ROS 世界系 (ENU)
        omega_los = q_FLU_ENU.toRotationMatrix() * omega_los_FLU;
    } else {
        // 无云台时的Fallback: 使用无人机的FLU坐标系和偏航角。
        // drone_state 的四元数是从 FLU 到 ENU 的旋转。
        double psi = get_yaw_from_quaternion(drone_state.segment<4>(6));
        Eigen::Matrix3d R_yaw_ENU = Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        
        // 构造FLU机体系(前左上)的LOS角速率向量
        // elevation_rate > 0 (目标向下移) -> 绕左轴(Y)做正旋转
        // azimuth_rate > 0 (目标向右移) -> 绕上轴(Z)做负旋转
        Eigen::Vector3d omega_los_FLU(0.0, elevation_rate, -azimuth_rate);
        
        // 转换到 ROS 世界系 (ENU)
        omega_los = R_yaw_ENU * omega_los_FLU;
    }
    
    // 限制omega_los的大小，避免噪声放大
    const double max_omega_for_pn = 20.0 * M_PI / 180.0;  // 降低上限，减少检测跳变的影响
    if (omega_los.norm() > max_omega_for_pn) {
        omega_los = omega_los.normalized() * max_omega_for_pn;
    }
    
    // 帧间omega_los变化率限制: 防止检测跳变导致的瞬间大修正
    // 允许较大变化率以适应机动目标 (120°/s^2)
    const double max_omega_change_per_step = 120.0 * M_PI / 180.0 * dt_;
    Eigen::Vector3d omega_los_diff = omega_los - pn_last_omega_los_;
    if (omega_los_diff.norm() > max_omega_change_per_step && pn_last_omega_los_.norm() > 1e-6) {
        omega_los = pn_last_omega_los_ + omega_los_diff.normalized() * max_omega_change_per_step;
    }
    pn_last_omega_los_ = omega_los;
    
    // PN加速度 = N * Vc * (omega_los × los_unit)
    // 这个加速度垂直于LOS方向，指向需要转向的方向
    double capped_closing_speed = std::min(closing_speed, 15.0); // 限制最大速度乘子
    Eigen::Vector3d a_pn = adaptive_N * capped_closing_speed * omega_los.cross(los_unit);
    
    // ============================================================================
    // Augmented PN: 目标加速度补偿项
    // 
    // 标准PN对匀速目标有效, 对机动(圆周运动)目标存在系统性LOS角速率残差。
    // APN公式: a_cmd = N*Vc*omega_los×r_hat + (N/3)*a_T_perp  (降低增益以适应噪声)
    // 其中 a_T_perp 是目标加速度在垂直于LOS方向的分量。
    //
    // 通过差分vel_est估计目标加速度，经过EMA滤波后使用。
    // 前0.3s为warmup期，不应用APN，避免初始瀐变冲击。
    // ============================================================================
    if (pn_target_vel_valid_ && !vel_est.hasNaN()) {
        Eigen::Vector3d a_target_raw = (vel_est - pn_last_target_vel_) / dt_;
        
        // 限制目标加速度估计的幅值 (防止噪声)
        const double max_target_accel = 15.0;  // m/s^2, 圆周运动v²/R, 10²/10=10
        if (a_target_raw.norm() > max_target_accel) {
            a_target_raw = a_target_raw.normalized() * max_target_accel;
        }
        
        // EMA滤波: tau=0.3s, 平滑目标加速度估计
        const double tau_accel = 0.3;
        const double alpha_accel = dt_ / (tau_accel + dt_);
        pn_target_accel_filtered_ = (1.0 - alpha_accel) * pn_target_accel_filtered_ + alpha_accel * a_target_raw;
        pn_apn_warmup_counter_++;
        
        // Warmup期结束后才应用APN (约 0.3s = 15帧@50Hz)
        const int apn_warmup_frames = static_cast<int>(0.3 / dt_);
        if (pn_apn_warmup_counter_ >= apn_warmup_frames) {
            // 投影到垂直于LOS的平面
            Eigen::Vector3d a_target_perp = pn_target_accel_filtered_ - 
                pn_target_accel_filtered_.dot(los_unit) * los_unit;
            
            // APN增强项: (N/3) * a_T_perp  (理论値N/2, 因噪声降低为N/3)
            a_pn += (adaptive_N / 3.0) * a_target_perp;
            
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 200,
                "APN: a_filtered=[%.1f,%.1f,%.1f] a_perp=[%.1f,%.1f,%.1f]",
                pn_target_accel_filtered_.x(), pn_target_accel_filtered_.y(), pn_target_accel_filtered_.z(),
                a_target_perp.x(), a_target_perp.y(), a_target_perp.z());
        }
    }
    // 更新目标速度历史
    if (!vel_est.hasNaN()) {
        pn_last_target_vel_ = vel_est;
        pn_target_vel_valid_ = true;
    }
    
    // ============================================================================
    // 添加前向加速度分量 (沿LOS方向)
    // 用于维持或增加接近速度
    // ============================================================================
    // 始终以V_DESIRED_和初始Vc中的较大值作为期望接近速度
    double desired_closing_speed = std::max(pn_vc_initial_, V_DESIRED_);
    double closing_speed_error = desired_closing_speed - closing_speed;
    double ax_forward = std::clamp(1.5 * closing_speed_error, -3.0, 8.0);
    
    // PN优先级: 当PN侧向修正较大时，缩减前向加速度以避免抢占加速度预算
    // 这防止了大前向加速(大倾角)干扰PN的精确制导
    double a_pn_mag = a_pn.norm();
    if (a_pn_mag > 2.0) {
        double pn_priority_scale = std::max(0.2, 2.0 / a_pn_mag);
        ax_forward *= pn_priority_scale;
    }
    
    // 沿LOS方向的加速度 (朝向目标为正)
    Eigen::Vector3d a_forward = ax_forward * los_unit;
    
    // 总PN加速度 (世界系)
    Eigen::Vector3d a_world = a_pn + a_forward;
    
    // ============================================================================
    // 重力补偿保护: 限制a_world.z下限，确保最低安全推力
    // a_total.z = a_world.z + G, 允许俯冲但不允许完全自由落体
    // 限制a_world.z >= -0.85*G, 保证 a_total.z >= 0.15*G (允许积极俯冲)
    // ============================================================================
    a_world.z() = std::max(a_world.z(), -0.85 * G_);
    
    // 限制水平和垂直加速度 (统一缩放以保持向量方向)
    double a_xy = std::sqrt(a_world.x() * a_world.x() + a_world.y() * a_world.y());
    double scale_xy = (a_xy > pn_max_accel_xy_) ? (pn_max_accel_xy_ / a_xy) : 1.0;
    double scale_z = 1.0;
    if (std::abs(a_world.z()) > pn_max_accel_z_) {
        scale_z = pn_max_accel_z_ / std::abs(a_world.z());
    }
    double scale = std::min(scale_xy, scale_z);
    a_world *= scale;
    
    // 重新计算并滤波
    a_xy = std::sqrt(a_world.x() * a_world.x() + a_world.y() * a_world.y());
    Eigen::VectorXd accel_raw(2);
    accel_raw << a_xy, a_world.z();
    Eigen::VectorXd accel_filtered = pn_accel_filter_->filter(accel_raw);
    
    // 保持XY方向不变，应用滤波后的幅值
    if (a_xy > 0.1) {
        double ratio = accel_filtered(0) / a_xy;
        a_world.x() *= ratio;
        a_world.y() *= ratio;
    }
    a_world.z() = accel_filtered(1);
    
    // ============================================================================
    // 推力和姿态计算
    // ============================================================================
    
    // 总加速度 = PN加速度 + 重力补偿
    Eigen::Vector3d a_total = a_world + Eigen::Vector3d(0.0, 0.0, G_);
    
    // 推力大小 (为了达到a_total的实际物理加速度，指令需除以推力效率; 比如效率0.84，需要指令更大)
    double eff = std::clamp(thrust_efficiency_, 0.5, 1.0);
    double thrust = a_total.norm() / eff;
    // 推力下限: 允许低于G以实现俯冲机动，但保持最低安全推力
    thrust = std::clamp(thrust, ControlLimits::THRUST_MIN, ControlLimits::THRUST_MAX);
    
    // 期望的机体Z轴方向 = 推力方向
    Eigen::Vector3d z_body_desired = a_total.normalized();
    if (z_body_desired.hasNaN() || z_body_desired.norm() < 0.5) {
        z_body_desired = Eigen::Vector3d(0.0, 0.0, 1.0);
    }
    
    // ============================================================================
    // 角速度计算
    // ============================================================================
    
    Eigen::Quaterniond q_curr(drone_state(6), drone_state(7), drone_state(8), drone_state(9));
    Eigen::Matrix3d R_B_W = q_curr.toRotationMatrix();
    Eigen::Vector3d z_body_current = R_B_W.col(2);
    
    // 计算旋转误差
    Eigen::Vector3d rotation_axis = z_body_current.cross(z_body_desired);
    double sin_angle = rotation_axis.norm();
    double cos_angle = z_body_current.dot(z_body_desired);

    // 内环姿态跟踪增益: 固定值，与PN导航常数解耦
    // 过高会导致姿态振荡; 过低会导致跟踪滞后
    // 4.0对应~25°误差时产生~1.75 rad/s角速度，响应时间约0.07s
    const double attitude_gain = 4.0;
    const double yaw_gain = 2.5;
    
    Eigen::Vector3d omega_world = Eigen::Vector3d::Zero();
    if (sin_angle > 1e-6) {
        rotation_axis.normalize();
        double angle = std::atan2(sin_angle, cos_angle);
        omega_world = rotation_axis * angle * attitude_gain;
    }
    
    // 转换到机体系
    Eigen::Vector3d omega_body = R_B_W.transpose() * omega_world;
    
    // ============================================================================
    // 偏航控制: 使用LOS水平投影方向
    // 目标: 让无人机机头指向目标
    // ============================================================================
    Eigen::Vector3d los_horizontal = los_unit;
    los_horizontal.z() = 0.0;
    if (los_horizontal.norm() > 0.1) {
        los_horizontal.normalize();
        
        // 当前机头方向 (世界系)
        Eigen::Vector3d x_body_current = R_B_W.col(0);
        Eigen::Vector3d x_body_horizontal = x_body_current;
        x_body_horizontal.z() = 0.0;
        if (x_body_horizontal.norm() > 0.1) {
            x_body_horizontal.normalize();
            
            // 计算偏航误差 (使用叉积的z分量)
            double yaw_error = x_body_horizontal.cross(los_horizontal).z();
            double yaw_rate_cmd = yaw_gain * yaw_error;
            yaw_rate_cmd = std::clamp(yaw_rate_cmd, -ControlLimits::YAW_RATE_MAX, ControlLimits::YAW_RATE_MAX);
            omega_body(2) = yaw_rate_cmd;
        }
    }
    
    // 限制角速度
    omega_body(0) = std::clamp(omega_body(0), -ControlLimits::ROLL_PITCH_RATE_MAX, ControlLimits::ROLL_PITCH_RATE_MAX);
    omega_body(1) = std::clamp(omega_body(1), -ControlLimits::ROLL_PITCH_RATE_MAX, ControlLimits::ROLL_PITCH_RATE_MAX);
    
    // ============================================================================
    // 输出
    // ============================================================================
    u_pn << omega_body(0), omega_body(1), omega_body(2), thrust;
    
    pn_last_u_pn_ = u_pn;
    
    // 调试日志
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 100,
        "PN: az=%.2f° el=%.2f° | az_rate=%.2f°/s el_rate=%.2f°/s | Vc=%.2fm/s dist=%.2fm TTI=%.2fs",
        azimuth * 180.0 / M_PI, elevation * 180.0 / M_PI,
        azimuth_rate * 180.0 / M_PI, elevation_rate * 180.0 / M_PI,
        closing_speed, dist_to_target, tti);
    
    // Calculate world LOS rate magnitude from omega_los for logging
    double omega_los_deg = omega_los.norm() * 180.0 / M_PI;
    
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 100,
        "PN: omega_los=%.2f°/s | a_pn=[%.1f,%.1f,%.1f] a_fwd=%.1f | omega=[%.2f,%.2f,%.2f] | T=%.1f",
        omega_los_deg, a_pn.x(), a_pn.y(), a_pn.z(), ax_forward,
        omega_body(0), omega_body(1), omega_body(2), thrust);
    
    return u_pn;
}

Eigen::Vector3d InterceptorNodeQuaternion::get_history_body_rates(const Eigen::Vector3d &current_body_rates, double delay_time) 
{
    std::shared_lock<std::shared_mutex> lock(mtx_control_history_);
    if (control_history_.empty()) {
        return current_body_rates;  // 如果没有历史数据，返回当前状态
    }
    // 获取延迟时间内的历史控制输入
    Eigen::Vector3d body_rates = Eigen::Vector3d::Zero();
    double current_time = this->get_clock()->now().seconds();
    double history_start_time = current_time - delay_time;

    for (const auto& control_entry : control_history_) {
        if (control_entry.first >= history_start_time) {
            body_rates = control_entry.second;
            break;
        }
    }
    return body_rates;
}

// 延迟补偿状态预测函数
VectorX InterceptorNodeQuaternion::predict_delayed_state(const VectorX& current_state, double delay_time) 
{
    VectorX predicted_state = current_state;

    {
        std::shared_lock<std::shared_mutex> lock(mtx_control_history_);
        if (control_history_.empty() || delay_time < 1e-3)
        {
            return predicted_state;  // 如果没有历史数据，返回当前状态
        }
    }
    
    double current_time = this->get_clock()->now().seconds();
    double prediction_start_time = current_time - delay_time;
    
    Eigen::Vector3d average_body_rates = Eigen::Vector3d::Zero();
    int valid_samples = 0;
    
    // 计算延迟时间窗口内的平均角速度
    // 优化：使用二分查找找到起始位置
    {
        std::shared_lock<std::shared_mutex> lg(mtx_control_history_);
        
        // 使用lower_bound进行二分查找
        auto it = std::lower_bound(
            control_history_.begin(), control_history_.end(),
            prediction_start_time,
            [](const auto& entry, double t) { return entry.first < t; }
        );
        
        // 从找到的位置开始累加
        for (; it != control_history_.end(); ++it) {
            average_body_rates += it->second;
            valid_samples++;
        }
    }
    
    if (valid_samples > 0) {
        average_body_rates /= valid_samples;
    } else {
        // 如果没有有效样本，使用最近的角速度
        average_body_rates = current_state.segment<3>(10);
    }
    
    // 预测姿态变化 (四元数积分)
    Eigen::Vector4d current_quat = current_state.segment<4>(6);
    Eigen::Vector4d predicted_quat = integrate_quaternion(current_quat, average_body_rates, delay_time);
    predicted_state.segment<4>(6) = normalize_quaternion(predicted_quat);
    
    // 简单的位置和速度预测（假设当前速度保持不变）
    Eigen::Vector3d current_vel = current_state.segment<3>(3);
    predicted_state.head<3>() += current_vel * delay_time;
    
    return predicted_state;
}

// 更新控制历史
void InterceptorNodeQuaternion::update_control_history(double current_time, const Eigen::Vector3d& body_rates) {
    std::unique_lock<std::shared_mutex> lock(mtx_control_history_);
    control_history_.emplace_back(current_time, body_rates);
    // 清理过旧的历史数据
    double cutoff_time = current_time - max_history_time_;
    while (!control_history_.empty() && control_history_.front().first < cutoff_time) {
        control_history_.pop_front();
    }
}

// ============================================================================
// 状态验证和安全检查函数
// ============================================================================

bool InterceptorNodeQuaternion::validate_state(const VectorX& state, const char* context) const {
    // 检查NaN
    if (state.hasNaN()) {
        RCLCPP_ERROR(this->get_logger(), "NaN detected in state (%s)", context);
        return false;
    }
    
    // 检查四元数范数
    double quat_norm = state.segment<4>(6).norm();
    if (std::abs(quat_norm - 1.0) > 0.1) {
        RCLCPP_WARN(this->get_logger(), "Quaternion norm deviation: %.4f (%s)", quat_norm, context);
        return false;
    }
    
    // 检查位置合理性 (防止极端值)
    if (state.head<3>().norm() > 10000.0) {
        RCLCPP_ERROR(this->get_logger(), "Position out of range (%s)", context);
        return false;
    }
    
    // 检查速度合理性
    if (state.segment<3>(3).norm() > 100.0) {
        RCLCPP_WARN(this->get_logger(), "Velocity unusually high: %.2f m/s (%s)", 
            state.segment<3>(3).norm(), context);
    }
    
    return true;
}

bool InterceptorNodeQuaternion::validate_control(const VectorU& u) const {
    // 检查NaN
    if (u.hasNaN()) {
        RCLCPP_ERROR(this->get_logger(), "NaN detected in control output");
        return false;
    }
    
    // 检查推力范围
    if (u(3) < ControlLimits::THRUST_MIN * 0.5 || u(3) > ControlLimits::THRUST_MAX * 1.5) {
        RCLCPP_WARN(this->get_logger(), "Thrust out of expected range: %.2f m/s^2", u(3));
        return false;
    }
    
    // 检查角速度范围
    if (std::abs(u(0)) > ControlLimits::ROLL_PITCH_RATE_MAX * 1.5 ||
        std::abs(u(1)) > ControlLimits::ROLL_PITCH_RATE_MAX * 1.5 ||
        std::abs(u(2)) > ControlLimits::YAW_RATE_MAX * 1.5) {
        RCLCPP_WARN(this->get_logger(), "Body rates out of expected range: [%.2f, %.2f, %.2f]",
            u(0), u(1), u(2));
        return false;
    }
    
    return true;
}

VectorU InterceptorNodeQuaternion::apply_control_limits(const VectorU& u_raw) const {
    VectorU u_limited = u_raw;
    
    // 角速度限幅
    u_limited(0) = std::clamp(u_limited(0), -ControlLimits::ROLL_PITCH_RATE_MAX, ControlLimits::ROLL_PITCH_RATE_MAX);
    u_limited(1) = std::clamp(u_limited(1), -ControlLimits::ROLL_PITCH_RATE_MAX, ControlLimits::ROLL_PITCH_RATE_MAX);
    u_limited(2) = std::clamp(u_limited(2), -ControlLimits::YAW_RATE_MAX, ControlLimits::YAW_RATE_MAX);
    
    // 推力限幅
    u_limited(3) = std::clamp(u_limited(3), ControlLimits::THRUST_MIN, ControlLimits::THRUST_MAX);
    
    return u_limited;
}

bool InterceptorNodeQuaternion::check_attitude_safety(const VectorX& state) {
    Eigen::Vector4d quat = state.segment<4>(6);
    double qx = quat(1), qy = quat(2);
    
    // 使用四元数直接计算倾斜角（避免欧拉角奇异性）
    double cos_tilt = 1 - 2*(qx*qx + qy*qy);
    double tilt_angle = std::acos(std::clamp(cos_tilt, -1.0, 1.0)) * 180.0/M_PI;
    
    if (tilt_angle > att_limit_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "Tilt angle %.1f deg exceeds limit %.1f deg", 
            tilt_angle, att_limit_);
        return false;
    }
    return true;
}

void InterceptorNodeQuaternion::check_ekf_health() {
    std::shared_lock<std::shared_mutex> lock(mtx_ekf_);  // 可能重置，使用独占锁
    
    if (!ekf_ || !ekf_->is_initialized()) return;
    
    // 定期检查EKF健康状态 (每50次循环检查一次)
    if (++ekf_health_check_counter_ < 50) return;
    ekf_health_check_counter_ = 0;
    
    // 检查EKF状态是否包含NaN
    if (!ekf_->is_healthy()) {
        RCLCPP_WARN(this->get_logger(), "EKF state contains NaN, resetting...");
        // 需要升级到独占锁来重置
        lock.unlock();
        std::unique_lock<std::shared_mutex> write_lock(mtx_ekf_);
        ekf_->reset();
    }
}

// ============================================================================
// 求解器故障恢复和状态机辅助函数
// ============================================================================

SolverRecoveryAction InterceptorNodeQuaternion::handle_solver_failure(int status) {
    consecutive_solver_failures_++;
    
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
        "ACADOS solver failed with status %d (consecutive: %d)", 
        status, consecutive_solver_failures_);
    
    // 根据连续失败次数决定恢复策略
    if (consecutive_solver_failures_ <= MAX_SOLVER_FAILURES_WARN) {
        // 轻微故障：使用上一次解继续
        return SolverRecoveryAction::CONTINUE;
    }
    else if (consecutive_solver_failures_ <= MAX_SOLVER_FAILURES_FAILSAFE) {
        // 中等故障：重置热启动
        RCLCPP_WARN(this->get_logger(), 
            "Solver failures reached %d, resetting warm start", 
            consecutive_solver_failures_);
        return SolverRecoveryAction::RESET_WARM;
    }
    else if (consecutive_solver_failures_ <= MAX_SOLVER_FAILURES_ABORT) {
        // 严重故障：进入failsafe模式
        RCLCPP_ERROR(this->get_logger(), 
            "Solver failures reached %d, entering failsafe mode", 
            consecutive_solver_failures_);
        status_publisher_->publish(
            mavros_msgs::msg::StatusText::WARNING, 
            1, "Solver issues, failsafe mode", true);
        return SolverRecoveryAction::FAILSAFE;
    }
    else {
        // 致命故障：终止任务
        return SolverRecoveryAction::ABORT;
    }
}

void InterceptorNodeQuaternion::reset_solver_warm_start(const VectorX& current_state) {
    RCLCPP_INFO(this->get_logger(), "Resetting solver warm start...");
    
    for (int i = 0; i <= N_HORIZON_; ++i) {
        x_warm_start_[i] = current_state;
        // 确保四元数归一化
        x_warm_start_[i].segment<4>(6) = normalize_quaternion(x_warm_start_[i].segment<4>(6));
    }
    for (int i = 0; i < N_HORIZON_; ++i) {
        u_warm_start_[i] << 0, 0, 0, G_;
    }
    
    // 部分重置失败计数，给予恢复机会
    consecutive_solver_failures_ = consecutive_solver_failures_ / 2;
}

void InterceptorNodeQuaternion::transition_to_state(MissionState new_state, const char* reason) {
    MissionState old_state = mission_state_.get();
    
    if (old_state == new_state) return;
    
    mission_state_.set(new_state);
    
    RCLCPP_INFO(this->get_logger(), "Mission state: %s -> %s (%s)",
        MissionStateMachine::state_name(old_state),
        MissionStateMachine::state_name(new_state),
        reason);
}

// ============================================================================
// StatusPublisher Implementation
// ============================================================================

StatusPublisher::StatusPublisher(rclcpp::Node &node, int hz)
{
    status_pub_ = node.create_publisher<mavros_msgs::msg::StatusText>(
        "/mavros/statustext/send", 10);
    status_timer_ = node.create_wall_timer(std::chrono::milliseconds(1000 / hz), [this](){
        static uint8_t severity;
        static std::string status;

        std::lock_guard<std::mutex> lock(status_mtx_);
        if (status_counter_<=0)
        {
            if (status_queue_.empty()) return;
            auto [level, counter, text] = status_queue_.front();
            severity = level;
            status = text;
            status_counter_ = counter;
            // printf("Publishing status: severity=%d, count=%d, text=%s\n", severity, status_counter_, status.c_str());
            status_queue_.pop();
        }
        if (status_counter_-- > 0)
        {
            // Publish status
            auto msg = std::make_unique<mavros_msgs::msg::StatusText>();
            msg->header.stamp = rclcpp::Clock().now();
            msg->severity = severity; // mavros_msgs::msg::StatusText::WARNING;
            msg->text= status;
            status_pub_->publish(std::move(msg));
        }
    });
}

StatusPublisher::~StatusPublisher()
{
    if (status_timer_)
        status_timer_->cancel();
}

bool StatusPublisher::publish(uint8_t severity, int count,const std::string& text, bool force)
{
    std::lock_guard<std::mutex> lock(status_mtx_);
    if (!force && (status_counter_ > 0 || !status_queue_.empty()))
        return false;
    status_queue_.emplace(severity, count, text);
    return true;
}

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InterceptorNodeQuaternion>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

