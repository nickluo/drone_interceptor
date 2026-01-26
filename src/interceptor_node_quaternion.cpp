#include <iostream>
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

    // 创建DroneEKF - 使用图像坐标系的12维CA模型
    CameraIntrinsics cam_intrinsics;
    cam_intrinsics.fx = cam_focal_length_;
    cam_intrinsics.fy = cam_focal_length_;
    cam_intrinsics.px = cam_width_px_ / 2.0;
    cam_intrinsics.py = cam_height_px_ / 2.0;
    
    Eigen::Vector2d process_noise;
    process_noise << bbox_pos_std_dev_ * 0.1,   // std_pos
                     bbox_pos_std_dev_ * 0.5;   // std_vel
    Eigen::Vector2d measurement_noise;
    measurement_noise << bbox_pos_std_dev_, bbox_size_std_dev_;
    
    const int predicted_imu_freq = 100;
    ekf_ = std::make_unique<DroneEKF>(cam_intrinsics, process_noise, measurement_noise,
                                       predicted_imu_freq, detection_latency_,
                                       target_physical_size_est_, perspective_compensation_);
    ekf_legacy_ = std::make_unique<TargetEKF>(dt_, cam_focal_length_, target_physical_size_est_, 
                                              bbox_pos_std_dev_, bbox_size_std_dev_, perspective_compensation_);
    int filter_n = static_cast<int>(RATE_TC_ * 2.0 / dt_);
    if (filter_n < 5) filter_n = 5;
    output_filter_ = std::make_unique<EMAFilter>(filter_n, 4);
    
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
    // 公式: ctrl_coeff = ctrl_scale * (1 - exp(-dt / rate_tc))
    // ctrl_scale 用于补偿实际飞控与理想一阶模型的差异
    // this->get_parameter("control.ctrl_scale", ctrl_scale);
    ctrl_coeff_ = 1.0 - std::exp(-dt_ / RATE_TC_);
    // ctrl_coeff_ = ctrl_scale * ctrl_coeff_base;
    // RCLCPP_INFO(this->get_logger(), "ctrl_coeff: %.4f (base=%.4f, scale=%.2f, dt=%.4f, rate_tc=%.4f)", 
    //             ctrl_coeff_, ctrl_coeff_base, ctrl_scale, dt_, RATE_TC_);
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
    if (fov_deg > 0.0) {
        cam_width_px_ = cam_width_px;
        cam_height_px_ = cam_height_px;
        cam_focal_length_ = cam_width_px * 0.5 / tan(fov_deg * M_PI / 180.0 / 2.0);
    } else {
        cam_width_px_ = cam_width_px;
        cam_height_px_ = cam_height_px;
        double scale_factor = static_cast<double>(cam_width_px) / static_cast<double>(native_width_px);
        cam_focal_length_ = cam_focal_mm * 1000.0 / cam_pixel_um * scale_factor;
    }

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
    std::vector<double> cost_weights_vec(7);
    this->get_parameter("cost.w_los", cost_weights_vec[0]);
    this->get_parameter("cost.w_align", cost_weights_vec[1]);
    this->get_parameter("cost.w_closing_speed", cost_weights_vec[2]);
    this->get_parameter("cost.w_rates", cost_weights_vec[3]);
    this->get_parameter("cost.w_tilt", cost_weights_vec[4]);
    this->get_parameter("cost.w_u_alpha", cost_weights_vec[5]);
    this->get_parameter("cost.w_u_thrust", cost_weights_vec[6]);
    for (int i=0; i < 7; ++i) w_cost_(i) = std::sqrt(cost_weights_vec[i]);
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
            // 创建DroneEKF - 使用图像坐标系的12维CA模型
            CameraIntrinsics cam_intrinsics;
            cam_intrinsics.fx = cam_focal_length_;
            cam_intrinsics.fy = cam_focal_length_;
            cam_intrinsics.px = cam_width_px_ / 2.0;
            cam_intrinsics.py = cam_height_px_ / 2.0;
            
            // 6-State CV Model: only need position and velocity noise
            Eigen::Vector2d process_noise;
            process_noise << bbox_pos_std_dev_ * 0.1,   // std_pos
                             bbox_pos_std_dev_ * 0.5;   // std_vel
            Eigen::Vector2d measurement_noise;
            measurement_noise << bbox_pos_std_dev_, bbox_size_std_dev_;
            
            const int predicted_imu_freq = 100;
            ekf_ = std::make_unique<DroneEKF>(cam_intrinsics, process_noise, measurement_noise,
                                               predicted_imu_freq, detection_latency_,
                                               target_physical_size_est_, perspective_compensation_);
            ekf_legacy_ = std::make_unique<TargetEKF>(dt_, cam_focal_length_, target_physical_size_est_, 
                                                  bbox_pos_std_dev_, bbox_size_std_dev_, perspective_compensation_);
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
        
        // 重置LOS测量状态
        {
            std::unique_lock<std::shared_mutex> lock(mtx_los_);
            los_angles_.setZero();
            los_angles_prev_.setZero();
            los_rates_.setZero();
            los_measurement_time_ = 0.0;
            los_initialized_ = false;
        }

        output_filter_->reset();
        control_history_.clear();

        odom_received_ = false;
        gimbal_received_ = false;

        timer_boot_count_ = 0;
        target_lost_counter_ = 0;
        solver_failure_counter_ = 0;
        consecutive_solver_failures_ = 0;
        
        // Reset guidance state
        pn_active_ = false;

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
    
    // 检查极端姿态值
    // Eigen::Vector3d euler_angles = to_euler_angles(q); // 仅用于检查
    // if (std::abs(euler_angles(0)) > att_limit_ * M_PI/180.0 || std::abs(euler_angles(1)) > att_limit_ * M_PI/180.0) {
    //     RCLCPP_WARN(this->get_logger(), "Large tilt angle detected!! Roll: %.2f°, Pitch: %.2f°", 
    //                 euler_angles(0) * 180.0/M_PI, euler_angles(1) * 180.0/M_PI);
    // }
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
    {
        std::lock_guard<std::mutex> lock(mtx_bbox_);
        current_bbox_ = *msg;
        is_visible_ = current_bbox_.header.frame_id != "";
    }

    if (is_visible_)
    {
        // 创建DroneEKF的Bbox结构 (使用像素坐标)
        DroneEKF::Bbox bbox;
        bbox.cx = msg->bbox.center.position.x;
        bbox.cy = msg->bbox.center.position.y;
        bbox.w = msg->bbox.size_x;
        bbox.h = msg->bbox.size_y;

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
        std::unique_lock<std::shared_mutex> lock(mtx_ekf_);  // 写入使用独占锁
        // 执行update (New signature for World Frame DroneEKF)
        ekf_->update(detection_time, bbox, curr_drone_state_for_ekf.head<3>(), q_C_W);
        
        // 更新旧的EKF
        TargetEKF::Bbox legacy_bbox;
        legacy_bbox.u = bbox.cx - (cam_width_px_ / 2.0);
        legacy_bbox.v = bbox.cy - (cam_height_px_ / 2.0);
        legacy_bbox.size = std::max(bbox.w, bbox.h);
        
        // 注意：旧TargetEKF使用的是Body-to-World的Quaternion（如果相机固定）
        // 如果使用current_gimbal_orientation_，需要确保它是World-to-Camera还是Body-to-Camera
        // 假设这里我们使用机体姿态（无独立云台）
         if (!ekf_legacy_->is_initialized()) {
            Eigen::Vector3d init_pos = ekf_legacy_->noisy_bbox_to_pos(legacy_bbox, curr_drone_state_for_ekf.head<3>(), q_C_W);
            ekf_legacy_->initialize(init_pos);
        } else {
            ekf_legacy_->update(legacy_bbox, curr_drone_state_for_ekf.head<3>(), q_C_W);
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

    if (target_lost_counter_ > N_HORIZON_ * 2) {
        RCLCPP_ERROR(this->get_logger(),
            "Target lost for too long. Aborting.");
        status_publisher_->publish(
            mavros_msgs::msg::StatusText::ERROR, 
            1, "Target lost. Aborting.", true);
        transition_to_state(MissionState::ABORTED, "Target lost too long");
        stop();
        return;
    }

    // 近距离调试日志：当距离小于5米时打印详细信息
    // double dist_to_target = (pos_est - curr_drone_state.head<3>()).norm();
    // Eigen::Vector3d pos_error = pos_est - curr_drone_state.head<3>();
    // if (dist_to_target < 5.0) {
    //     RCLCPP_INFO(this->get_logger(), 
    //         "CLOSE: dist=%.2fm, z_err=%.2f, visible=%d, vz=%.2f",
    //         dist_to_target, pos_error.z(), is_visible_ ? 1 : 0, 
    //         curr_drone_state(5));  // vz = 垂直速度
    // }
    
    // 验证当前状态
    if (!validate_state(curr_drone_state, "current_state")) {
        failsafe(false);
        return;
    }

    // --- Compare EKF Outputs ---
    if (ekf_legacy_ && ekf_legacy_->is_initialized()) {
         ekf_legacy_->predict(dt_);
         Eigen::VectorXd legacy_state = ekf_legacy_->get_state();
         Eigen::Vector3d pos_legacy = legacy_state.head<3>();
         Eigen::Vector3d vel_legacy = legacy_state.tail<3>();
         
         double pos_diff = (pos_est - pos_legacy).norm();
         double vel_diff = (vel_est - vel_legacy).norm();
         
         // 节流打印比较日志
        //  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
         RCLCPP_INFO(this->get_logger(),
             "EKF Comparison:\n"
             "  DroneEKF Pos: [%.2f, %.2f, %.2f], Vel: [%.2f, %.2f, %.2f]\n"
             "  TargetEKF Pos: [%.2f, %.2f, %.2f], Vel: [%.2f, %.2f, %.2f]\n"
             "  Diff Pos: %.2f m, Vel: %.2f m/s",
             pos_est.x(), pos_est.y(), pos_est.z(), vel_est.x(), vel_est.y(), vel_est.z(),
             pos_legacy.x(), pos_legacy.y(), pos_legacy.z(), vel_legacy.x(), vel_legacy.y(), vel_legacy.z(),
             pos_diff, vel_diff);
         
         // 可以在这里根据情况选择使用原来的EKF
         // pos_est = pos_legacy;
         // vel_est = vel_legacy;
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
    bool should_use_pn = false;
    double closing_speed_est = 0.0;  // 用于PN的接近速度估计

    // RCLCPP_INFO(this->get_logger(), "PN Guidance: Enabled=%d, Active=%d", pn_enabled_ ? 1 : 0, pn_active_ ? 1 : 0);
    
    if (pn_enabled_ && !std::isnan(pos_est.x()) && !std::isnan(vel_est.x())) {
        // 计算接近速度估计 (用于PN)
        Eigen::Vector3d rel_pos = pos_est - curr_drone_state.head<3>();
        Eigen::Vector3d rel_vel = vel_est - curr_drone_state.segment<3>(3);
        double dist_to_target = rel_pos.norm();
        if (dist_to_target > 1e-3) {
            closing_speed_est = -rel_pos.dot(rel_vel) / dist_to_target;
        }
        
        if (pn_active_) {
            should_use_pn = true;
        } else {
            // 使用图像直接测量的LOS角度判断是否切换到PN
            Eigen::Vector2d current_los_angles;
            bool los_valid = false;
            {
                std::shared_lock<std::shared_mutex> lock(mtx_los_);
                if (los_initialized_ && is_visible_) {
                    current_los_angles = los_angles_;
                    los_valid = true;
                }
            }
            
            if (los_valid) {
                // 计算LOS偏离视轴角度 (相机系)
                double angle_off_boresight = std::sqrt(current_los_angles(0)*current_los_angles(0) + 
                                                        current_los_angles(1)*current_los_angles(1));
                bool target_in_fov_center = (std::abs(angle_off_boresight) < pn_fov_switch_deg_ * M_PI / 180.0);
                
                RCLCPP_INFO(this->get_logger(), "LOS angles (az,el): [%.2f, %.2f] deg, Off-boresight: %.2f deg, In FOV: %d", 
                    current_los_angles(0) * 180.0 / M_PI, current_los_angles(1) * 180.0 / M_PI,
                    angle_off_boresight * 180.0 / M_PI, target_in_fov_center ? 1 : 0);
                RCLCPP_INFO(this->get_logger(), "Distance to target: %.2f m, Closing speed: %.2f m/s", 
                    dist_to_target, closing_speed_est);
                    
                if (dist_to_target < pn_switch_dist_ && target_in_fov_center) {
                    should_use_pn = true;
                    if (!pn_active_) {
                        // 首次切换到PN，记录当前body系x轴速度作为期望接近速度
                        Eigen::Vector3d vel_world = curr_drone_state.segment<3>(3);
                        Eigen::Quaterniond q(curr_drone_state(6), curr_drone_state(7), curr_drone_state(8), curr_drone_state(9));  // w,x,y,z
                        Eigen::Matrix3d R_body = q.toRotationMatrix();
                        Eigen::Vector3d vel_body = R_body.transpose() * vel_world;
                        pn_vc_initial_ = std::max(vel_body(0) * 1.5, V_DESIRED_);
                        RCLCPP_INFO(this->get_logger(), "PN Initial Vc_body: %.2f m/s", pn_vc_initial_);
                    }
                    pn_active_ = true;
                    RCLCPP_INFO(this->get_logger(), "Switching to IMAGE-BASED PROPORTIONAL NAVIGATION (Latch Active)");
                }
            }
        }
    }
    
    VectorU u_optimal_raw;

    if (should_use_pn) {
        // Use Image-Based PN Guidance (直接使用检测框计算的LOS角速率)
        u_optimal_raw = calculate_pn_controls_image_based(curr_drone_state);
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
        }

        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", u_optimal_raw.data());
    }

    // 验证控制输出
    if (!validate_control(u_optimal_raw)) {
        RCLCPP_WARN(this->get_logger(), "Invalid control output, applying limits");
    }
    // 应用控制限幅
    VectorU u_optimal_limited = apply_control_limits(u_optimal_raw);
    
    // Warm start update
    if (!should_use_pn) {
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
    } else {
        // In PN mode, we should perhaps propagate state simply to keep x_warm_start_ valid if we switch back?
        // For now, let's just keep x_warm_start_ as current state replicated
        for (int i = 0; i <= N_HORIZON_; ++i) x_warm_start_[i] = curr_drone_state;
    }

    Eigen::VectorXd u_filtered_dynamic = output_filter_->filter(u_optimal_limited);
    
    if (timer_boot_count_++ < N_HORIZON_ * 1) {  // 前几次发送悬停控制，等待滤波器稳定
        failsafe(false);  // 发送悬停指令而非不发送任何控制
        return;
    }

    // 提取控制量
    double thrust = u_filtered_dynamic(3);
    Eigen::Vector3d body_rates(u_filtered_dynamic(0), u_filtered_dynamic(1), u_filtered_dynamic(2));

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
// ============================================================================
VectorU InterceptorNodeQuaternion::calculate_pn_controls_image_based(const VectorX& drone_state)
{
    return {0.0, 0.0, 0.0, G_}; // 占位符，实际实现略
}

Eigen::Vector3d InterceptorNodeQuaternion::get_history_body_rates(const Eigen::Vector3d &current_body_rates, double delay_time) 
{
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

    if (control_history_.empty() || delay_time < 1e-3)
    {
        return predicted_state;  // 如果没有历史数据，返回当前状态
    }
    
    double current_time = this->get_clock()->now().seconds();
    double prediction_start_time = current_time - delay_time;
    
    Eigen::Vector3d average_body_rates = Eigen::Vector3d::Zero();
    int valid_samples = 0;
    
    // 计算延迟时间窗口内的平均角速度
    // 优化：使用二分查找找到起始位置
    {
        std::shared_lock<std::shared_mutex> lg(mtx_drone_state_);  // 读取使用共享锁
        
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
    std::unique_lock<std::shared_mutex> lock(mtx_ekf_);  // 可能重置，使用独占锁
    
    if (!ekf_ || !ekf_->is_initialized()) return;
    
    // 定期检查EKF健康状态 (每50次循环检查一次)
    if (++ekf_health_check_counter_ < 50) return;
    ekf_health_check_counter_ = 0;
    
    // 检查EKF状态是否包含NaN
    if (!ekf_->is_healthy()) {
        RCLCPP_WARN(this->get_logger(), "EKF state contains NaN, resetting...");
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

