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

    // 创建DroneEKF - 使用图像坐标系的12维CA模型 + 3D位置估计
    CameraIntrinsics cam_intrinsics;
    cam_intrinsics.fx = cam_focal_length_;
    cam_intrinsics.fy = cam_focal_length_;
    cam_intrinsics.px = cam_width_px_ / 2.0;
    cam_intrinsics.py = cam_height_px_ / 2.0;
    
    setup_ekf();
    // ekf_legacy_ = std::make_unique<TargetEKF>(dt_, cam_focal_length_, target_physical_size_est_, 
    //                                           bbox_pos_std_dev_, bbox_size_std_dev_, perspective_compensation_);
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

void InterceptorNodeQuaternion::setup_ekf()
{
    // 创建DroneEKF - 使用图像坐标系的12维CA模型 + 3D位置估计
    CameraIntrinsics cam_intrinsics;
    cam_intrinsics.fx = cam_focal_length_;
    cam_intrinsics.fy = cam_focal_length_;
    cam_intrinsics.px = cam_width_px_ / 2.0;
    cam_intrinsics.py = cam_height_px_ / 2.0;
    
    // 8维过程噪声: [std_pos_img, std_vel_img, std_accel_img, std_size, std_size_vel, std_accel_size, std_pos_3d, std_vel_3d]
    Eigen::Matrix<double, 8, 1> process_noise;
    process_noise << bbox_pos_std_dev_ * 0.1,   // std_pos_img
                    bbox_pos_std_dev_ * 0.5,   // std_vel_img
                    bbox_pos_std_dev_ * 0.25,  // std_accel_img
                    bbox_size_std_dev_ * 0.1,  // std_size
                    bbox_size_std_dev_ * 0.5,  // std_size_vel
                    bbox_size_std_dev_ * 0.25, // std_accel_size
                    target_physical_size_est_,         // std_pos_3d (米)
                    target_physical_size_est_ * 2.0;   // std_vel_3d (米/秒)
    // 3维测量噪声: [std_meas_pos, std_meas_size, std_meas_3d]
    Eigen::Vector3d measurement_noise;
    measurement_noise << bbox_pos_std_dev_, bbox_size_std_dev_, target_physical_size_est_;
    
    const int predicted_imu_freq = 100;
    ekf_ = std::make_unique<DroneEKF>(cam_intrinsics, process_noise, measurement_noise,
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
            // ekf_legacy_ = std::make_unique<TargetEKF>(dt_, cam_focal_length_, target_physical_size_est_, 
            //                                       bbox_pos_std_dev_, bbox_size_std_dev_, perspective_compensation_);
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
        //  if (!ekf_legacy_->is_initialized()) {
        //     Eigen::Vector3d init_pos = ekf_legacy_->noisy_bbox_to_pos(legacy_bbox, curr_drone_state_for_ekf.head<3>(), q_C_W);
        //     ekf_legacy_->initialize(init_pos);
        // } else {
        //     ekf_legacy_->update(legacy_bbox, curr_drone_state_for_ekf.head<3>(), q_C_W);
        // }
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
    // if (ekf_legacy_ && ekf_legacy_->is_initialized()) {
    //      ekf_legacy_->predict(dt_);
    //      Eigen::VectorXd legacy_state = ekf_legacy_->get_state();
    //      Eigen::Vector3d pos_legacy = legacy_state.head<3>();
    //      Eigen::Vector3d vel_legacy = legacy_state.tail<3>();
         
    //      double pos_diff = (pos_est - pos_legacy).norm();
    //      double vel_diff = (vel_est - vel_legacy).norm();
         
    //      // 节流打印比较日志
    //     //  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
    //      RCLCPP_INFO(this->get_logger(),
    //          "EKF Comparison:\n"
    //          "  DroneEKF Pos: [%.2f, %.2f, %.2f], Vel: [%.2f, %.2f, %.2f]\n"
    //          "  TargetEKF Pos: [%.2f, %.2f, %.2f], Vel: [%.2f, %.2f, %.2f]\n"
    //          "  Diff Pos: %.2f m, Vel: %.2f m/s",
    //          pos_est.x(), pos_est.y(), pos_est.z(), vel_est.x(), vel_est.y(), vel_est.z(),
    //          pos_legacy.x(), pos_legacy.y(), pos_legacy.z(), vel_legacy.x(), vel_legacy.y(), vel_legacy.z(),
    //          pos_diff, vel_diff);
         
    //      // 可以在这里根据情况选择使用原来的EKF
    //      // pos_est = pos_legacy;
    //      // vel_est = vel_legacy;
    // }

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
            // 使用DroneEKF的stateToLosAngles获取LOS角度判断是否切换到PN
            bool los_valid = false;
            double azimuth = 0.0, elevation = 0.0;
            {
                std::shared_lock<std::shared_mutex> lock(mtx_ekf_);
                if (ekf_ && ekf_->is_initialized() && is_visible_) {
                    Eigen::Vector4d los_state = ekf_->stateToLosAngles(true);
                    azimuth = los_state(0);
                    elevation = los_state(1);
                    los_valid = !std::isnan(azimuth) && !std::isnan(elevation);
                }
            }
            
            if (los_valid) {
                // 计算LOS偏离视轴角度 (相机系)
                double angle_off_boresight = std::sqrt(azimuth * azimuth + elevation * elevation);
                bool target_in_fov_center = (angle_off_boresight < pn_fov_switch_deg_ * M_PI / 180.0);
                
                RCLCPP_INFO(this->get_logger(), "LOS angles (az,el): [%.2f, %.2f] deg, Off-boresight: %.2f deg, In FOV: %d", 
                    azimuth * 180.0 / M_PI, elevation * 180.0 / M_PI,
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
                        // pn_vc_initial_ = std::max(vel_body(0) * 1.5, V_DESIRED_);
                        pn_vc_initial_ = vel_body(0);
                        RCLCPP_INFO(this->get_logger(), "PN Initial Vc_body: %.2f m/s", pn_vc_initial_);
                        
                        // 重置PN推力状态，使用当前MPC的推力作为起点
                        pn_last_thrust_ = G_;  // 从悬停推力开始
                    }
                    pn_active_ = true;
                    RCLCPP_INFO(this->get_logger(), 
                        "Switching to IMAGE-BASED PN (Latch Active), Thrust Efficiency: %.3f", 
                        thrust_efficiency_);
                }
            }
        }
    }
    
    VectorU u_optimal_raw;

    if (should_use_pn) {
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
                    // 指数移动平均更新
                    thrust_efficiency_ = 0.9 * thrust_efficiency_ + 0.1 * efficiency_sample;
                }
            }
            last_thrust_cmd_ = u_optimal_raw(3);
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
    
    // 从EKF获取LOS角度和角速率
    Eigen::Vector4d los_state;
    {
        std::shared_lock<std::shared_mutex> lock(mtx_ekf_);
        if (!ekf_ || !ekf_->is_initialized()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 100, 
                "PN: EKF not initialized, returning hover");
            return u_pn;
        }
        // stateToLosAngles返回 [azimuth, elevation, azimuth_rate, elevation_rate] (弧度, 弧度/秒)
        // compensate_rotation = true: 补偿相机自转，获取惯性系下的LOS角速率
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

    Eigen::Vector3d rel_pos = pos_est - drone_state.head<3>();
    Eigen::Vector3d rel_vel = vel_est - drone_state.segment<3>(3);
    double dist_to_target = rel_pos.norm();
    double closing_speed_est = pn_vc_initial_;
    if (dist_to_target > 1e-3) {
        closing_speed_est = -rel_pos.dot(rel_vel) / dist_to_target;
    }

    // 使用初始记录的接近速度 (在PN激活时锁定)
    double Vc = closing_speed_est;
    if (Vc < V_DESIRED_ * 0.5) {
        Vc = V_DESIRED_;  // 最小接近速度
    }
    
    // ============================================================================
    // 自适应增益PD控制器
    // 
    // 设计原则:
    // 1. 使用二阶系统理论设计增益: ωn (自然频率) 和 ζ (阻尼比)
    // 2. K_p = ωn², K_d = 2*ζ*ωn
    // 3. 自然频率根据估计的接近时间(TTI)自适应调整
    // 4. 保持阻尼比 ζ ≈ 0.7-1.0 确保稳定性
    //
    // TTI估计: 使用目标尺寸变化率
    // 如果目标变大，说明在接近; 变化率越大，越接近
    // ============================================================================
    
    
    // 估计接近时间 (TTI)
    double tti_from_dist = dist_to_target / closing_speed_est;
    double tti = std::clamp(tti_from_dist, 0.3, 5.0);  // 限制在0.3-5秒范围
    
    // ============================================================================
    // 自适应自然频率设计
    // 
    // 原则: 控制器带宽应该随TTI减小而增加
    // 但带宽不能太高，否则会放大噪声和延迟
    //
    // ωn = base_freq / sqrt(tti)  (TTI越小，频率越高)
    // 限制 ωn 在 [1, 4] rad/s 范围内
    // ============================================================================
    double base_omega = 2.0;  // 基础自然频率
    double omega_n = base_omega / std::sqrt(tti);
    omega_n = std::clamp(omega_n, 1.0, 4.0);  // 限制频率范围
    
    // 阻尼比: 略欠阻尼以获得快速响应，但不能太小避免振荡
    double zeta = 0.8;  // 阻尼比
    
    // PD增益计算 (单位: rad/s² per rad, rad/s² per rad/s)
    // 转换为加速度增益需要乘以一个尺度因子
    // 对于小角度: a ≈ distance * ω² * θ
    // 使用 Vc 作为距离代理 (典型末端距离 ≈ Vc * 1秒)
    double gain_scale = Vc;  // 将角度误差转换为加速度的尺度
    
    double K_p_adaptive = omega_n * omega_n * gain_scale;
    double K_d_adaptive = 2.0 * zeta * omega_n * gain_scale;
    
    // 限制增益范围，防止过激
    K_p_adaptive = std::clamp(K_p_adaptive, 10.0, 100.0);
    K_d_adaptive = std::clamp(K_d_adaptive, 5.0, 50.0);
    
    // 水平和垂直可以使用不同增益 (垂直响应通常较慢)
    double K_p_az = K_p_adaptive;
    double K_d_az = K_d_adaptive;
    double K_p_el = K_p_adaptive * 1.2;  // 垂直略高，因为推力响应慢
    double K_d_el = K_d_adaptive * 0.8;  // 垂直阻尼略低
    
    // ============================================================================
    // PD控制加速度计算
    // ============================================================================
    double a_cam_y = K_p_az * azimuth + K_d_az * azimuth_rate;
    double a_cam_z = K_p_el * elevation + K_d_el * elevation_rate;
    
    // 限制加速度指令 (允许更大的垂直机动)
    const double max_accel_h = G_ * 1.0;   // 水平方向限制 (~10 m/s²)
    const double max_accel_v = G_ * 1.0;   // 垂直方向限制 (~10 m/s²)
    a_cam_y = std::clamp(a_cam_y, -max_accel_h, max_accel_h);
    a_cam_z = std::clamp(a_cam_z, -max_accel_v, max_accel_v);
    
    // ============================================================================
    // 获取云台姿态和机体姿态
    // ============================================================================
    Eigen::Quaterniond q_drone(drone_state(6), drone_state(7), drone_state(8), drone_state(9));
    Eigen::Matrix3d R_B_W = q_drone.toRotationMatrix();  // Body to World rotation
    
    // 获取云台姿态 (相机相对于世界系的姿态)
    // 注意: 需要包含从相机FRD到世界ENU的完整变换
    Eigen::Matrix3d R_C_W;  // Camera to World rotation
    bool has_independent_gimbal = gimbal_received_;
    
    if (has_independent_gimbal) {
        // 独立云台: 使用接收的云台姿态
        // 假设云台姿态已经是相机系(FRD)到世界系(ENU)的完整变换
        std::shared_lock<std::shared_mutex> lock(mtx_gimbal_);
        R_C_W = current_gimbal_orientation_.toRotationMatrix();
    } else {
        // 水平云台: 相机只跟随机体的yaw，保持水平
        // 需要两步变换:
        // 1. 相机FRD到"标准"相机系 (在yaw=0时与ENU对齐，但Y/Z翻转)
        // 2. 应用yaw旋转
        
        double yaw = std::atan2(R_B_W(1, 0), R_B_W(0, 0));
        
        // 基础变换: 相机FRD到世界ENU (yaw=0时)
        // Camera X (前) -> World X (东)
        // Camera Y (右) -> World -Y (南, 因为面向东时右边是南)
        // Camera Z (下) -> World -Z (下, ENU中Z向上)
        Eigen::Matrix3d R_FRD_to_ENU;
        R_FRD_to_ENU << 1,  0,  0,
                        0, -1,  0,
                        0,  0, -1;
        
        // Yaw旋转矩阵 (绕世界Z轴)
        Eigen::Matrix3d R_yaw;
        double c = std::cos(yaw);
        double s = std::sin(yaw);
        R_yaw << c, -s, 0,
                 s,  c, 0,
                 0,  0, 1;
        
        // 组合: 先FRD到ENU，再绕Z轴旋转yaw
        R_C_W = R_yaw * R_FRD_to_ENU;
    }
    
    // ============================================================================
    // 将相机系加速度转换到世界系，再转换到机体系
    // 
    // 相机系 (FRD): X前, Y右, Z下
    // 机体系 (FLU): X前, Y左, Z上
    // 世界系 (ENU): X东, Y北, Z上
    // ============================================================================
    
    // 相机系加速度向量 (X前方向不需要加速度，由推力和俯仰提供)
    Eigen::Vector3d a_cam(0.0, a_cam_y, a_cam_z);
    
    // 转换到世界系: a_world = R_C_W * a_cam
    Eigen::Vector3d a_world = R_C_W * a_cam;
    
    // 转换到机体系: a_body = R_W_B * a_world = R_B_W^T * a_world
    Eigen::Vector3d a_body = R_B_W.transpose() * a_world;
    
    // 相机系到机体系的坐标变换 (FRD -> FLU): y_flu = -y_frd, z_flu = -z_frd
    // 但由于我们通过世界系转换，已经正确处理了坐标变换
    
    // ============================================================================
    // 计算所需姿态角 (机体系FLU)
    // 
    // FLU系下 (右手定则):
    // - 向右加速 (a_body_y < 0, 因为Y轴向左) 需要正roll (向右倾斜)
    // - 向前加速 (a_body_x > 0) 需要正pitch (机头下压)
    // - 向上加速 (a_body_z > 0) 需要增加推力
    //
    // 小角度近似:
    // a_body_y ≈ -g * roll  =>  roll ≈ -a_body_y / g
    // a_body_x ≈ g * pitch  =>  pitch ≈ a_body_x / g
    // ============================================================================
    
    // 计算所需滚转角: 需要产生 a_body_y 的加速度
    // 在FLU系中，正roll产生负Y加速度（向右），所以 roll = -a_body_y / g
    double roll_desired = -std::atan2(a_body(1), G_);
    
    // 计算所需俯仰角调整
    // 当前俯仰角 + 垂直加速度补偿
    // 在FLU系中，正pitch产生负X加速度（减速），所以向前加速需要负pitch
    // 对于垂直方向 (Z轴向上)，a_body_z > 0 表示向上加速，需要增加推力
    // a_body_z < 0 表示向下加速，可以通过减小推力或正pitch来实现
    
    // 提取当前欧拉角 (ZYX顺序: yaw-pitch-roll) for FLU
    double roll_current = std::atan2(R_B_W(2, 1), R_B_W(2, 2));
    double pitch_current = -std::asin(std::clamp(R_B_W(2, 0), -1.0, 1.0));
    double yaw_current = std::atan2(R_B_W(1, 0), R_B_W(0, 0));
    
    // 俯仰角: 基于当前俯仰 + 补偿项
    // a_body_x 正表示向前加速需求，需要正pitch (机头下压)
    // 保守处理，减小增益
    double pitch_desired = pitch_current + std::atan2(a_body(0), G_) * 0.3;
    
    // 限制姿态角 (使用自适应限制：远距离时限制更严格)
    // TTI大时使用较小倾斜，TTI小时允许更大倾斜
    double tilt_scale = std::clamp(1.0 / tti, 0.3, 1.0);  // TTI=3s -> 0.33, TTI=0.5s -> 1.0
    const double max_tilt_base = att_limit_ * M_PI / 180.0;
    const double max_tilt = max_tilt_base * tilt_scale;
    roll_desired = std::clamp(roll_desired, -max_tilt, max_tilt);
    pitch_desired = std::clamp(pitch_desired, -max_tilt, max_tilt);
    
    // ============================================================================
    // 角速率计算 (比例控制 + 变化率限制)
    // ============================================================================
    double rate_gain = 2.5;  // 角度误差到角速率的增益 (降低以减少振荡)
    double roll_rate = rate_gain * (roll_desired - roll_current);
    double pitch_rate = rate_gain * (pitch_desired - pitch_current);
    
    // Yaw控制: 保持相机对准目标
    // 需要将相机系的azimuth误差转换到机体系的yaw误差
    double yaw_error;
    if (has_independent_gimbal) {
        // 独立云台: 机体yaw不需要跟踪目标
        yaw_error = 0.0;
    } else {
        // 水平云台: 相机yaw跟随机体，需要机体转向目标
        // azimuth > 0 表示目标在右边，需要向右转 (FLU中负yaw rate)
        // 注意: FLU中正yaw是绕Z轴正向 (向上)，即逆时针 (向左转)
        yaw_error = -azimuth;  // 取负使得目标在右边时向右转
    }
    double yaw_rate = rate_gain * yaw_error * gimbal_y_coeff_;
    (void)yaw_current;
    
    // 限制角速率
    roll_rate = std::clamp(roll_rate, -ControlLimits::ROLL_PITCH_RATE_MAX * 0.5, ControlLimits::ROLL_PITCH_RATE_MAX * 0.5);
    pitch_rate = std::clamp(pitch_rate, -ControlLimits::ROLL_PITCH_RATE_MAX * 0.5, ControlLimits::ROLL_PITCH_RATE_MAX * 0.5);
    yaw_rate = std::clamp(yaw_rate, -ControlLimits::YAW_RATE_MAX * 0.5, ControlLimits::YAW_RATE_MAX * 0.5);
    
    // ============================================================================
    // 推力计算 (FLU系: Z轴向上)
    // 
    // 需要的垂直加速度 a_body_z (向上为正):
    // thrust = g + a_body_z (在无倾斜时)
    // 考虑倾斜补偿: thrust = (g + a_body_z) / cos(tilt)
    //
    // 注意: PN末端导引需要快速响应，不能有太多增益衰减
    // ============================================================================
    double tilt_angle = std::sqrt(roll_desired * roll_desired + pitch_desired * pitch_desired);
    double cos_tilt = std::cos(std::min(tilt_angle, max_tilt));
    
    // 注意: 推力效率 > 1 表示实际加速度比预期大
    // 但为了稳定性，暂时不使用自适应效率，直接使用1.0
    // double eff = std::clamp(thrust_efficiency_, 0.8, 1.2);
    double eff = 1.0;
    
    // 基础推力补偿重力和倾斜
    double thrust_base = G_ / cos_tilt;
    
    // 垂直加速度补偿 - 直接使用计算的加速度，不衰减
    // a_body(2) > 0 = 向上加速需求 -> 增加推力
    // a_body(2) < 0 = 向下加速需求 -> 减小推力
    double thrust_accel = a_body(2);  // 直接使用，不衰减
    
    double thrust_cmd = (thrust_base + thrust_accel) / eff;
    
    // 限制推力变化率 (PN模式需要更快响应)
    // 20 m/s²/s 的变化率，允许在0.25秒内从悬停到最大倾斜
    const double max_thrust_rate = ControlLimits::THRUST_MAX * dt_;
    if (pn_last_thrust_ < 0.1) pn_last_thrust_ = G_;  // 初始化
    double thrust_delta = thrust_cmd - pn_last_thrust_;
    thrust_delta = std::clamp(thrust_delta, -max_thrust_rate, max_thrust_rate);
    double thrust = pn_last_thrust_ + thrust_delta;
    
    thrust = std::clamp(thrust, ControlLimits::THRUST_MIN, ControlLimits::THRUST_MAX);
    pn_last_thrust_ = thrust;
    
    // 组装控制输出 [roll_rate, pitch_rate, yaw_rate, thrust]
    u_pn << roll_rate, pitch_rate, yaw_rate, thrust;
    
    // 调试日志 (包含自适应增益信息)
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 100,
    // RCLCPP_INFO(this->get_logger(),
        "PN: LOS[az=%.1f°, el=%.1f°] Rate[%.1f, %.1f]°/s, "
        "TTI=%.2fs, Kp=%.1f, Kd=%.1f, "
        "a_body[%.1f, %.1f, %.1f], Att[r=%.1f°, p=%.1f°]->[%.1f°, %.1f°], "
        "Cmd[p=%.2f, q=%.2f, r=%.2f, T=%.1f]",
        azimuth * 180.0/M_PI, elevation * 180.0/M_PI,
        azimuth_rate * 180.0/M_PI, elevation_rate * 180.0/M_PI,
        tti, K_p_az, K_d_az,
        a_body(0), a_body(1), a_body(2),
        roll_current * 180.0/M_PI, pitch_current * 180.0/M_PI,
        roll_desired * 180.0/M_PI, pitch_desired * 180.0/M_PI,
        roll_rate, pitch_rate, yaw_rate, thrust);
    
    return u_pn;
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

