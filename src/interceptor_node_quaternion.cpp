#include <iostream>
#include <chrono>
#include <cmath>

// Eigen for linear algebra
#include <Eigen/Dense>

#include "interceptor_node_quaternion.h"

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
    
    ekf_ = std::make_unique<TargetEKF>(dt_, cam_focal_length_, target_physical_size_est_,
                                        bbox_pos_std_dev_, bbox_size_std_dev_);
    output_filter_ = std::make_unique<EMAFilter>(static_cast<int>(N_HORIZON_ / TF_ * 0.1), 4);
    
    // 初始化四元数状态
    x_warm_start_.resize(N_HORIZON_ + 1, Vector13d::Zero());
    u_warm_start_.resize(N_HORIZON_, Eigen::Vector4d::Zero());
    
    // 初始化状态设置
    for (int i = 0; i <= N_HORIZON_; ++i) x_warm_start_[i] = drone_state_;
    for (int i = 0; i < N_HORIZON_; ++i) u_warm_start_[i] << 0, 0, 0, G_;

    sub_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    timer_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    
    control_pub_ = this->create_publisher<quadrotor_msgs::msg::ControlCommand>(
        "/fpv/control_command", 
        rclcpp::SensorDataQoS());

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
    } else {
        RCLCPP_INFO(this->get_logger(), "Running in REAL mode with QUATERNION dynamics.");
        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = sub_callback_group_;
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/fpv/odom", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(), 
            std::bind(&InterceptorNodeQuaternion::odom_callback, this, std::placeholders::_1), sub_options);
        bbox_sub_ = this->create_subscription<vision_msgs::msg::Detection2D>(
            "/fpv/target_bbox", rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(), 
            std::bind(&InterceptorNodeQuaternion::bbox_callback, this, std::placeholders::_1), sub_options);
        mission_complete_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/fpv/mission_complete", 10, 
            std::bind(&InterceptorNodeQuaternion::mission_complete_callback, this, std::placeholders::_1));
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
    this->get_parameter("mpc.n_horizon", N_HORIZON_);
    this->get_parameter("mpc.tf", TF_);
    this->get_parameter("mpc.v_desired", V_DESIRED_);
    this->get_parameter("physics.drag_coeff", DRAG_COEFF_);
    this->get_parameter("perception.bbox_pos_std_dev", bbox_pos_std_dev_);
    this->get_parameter("perception.bbox_size_std_dev", bbox_size_std_dev_);
    this->get_parameter("perception.target_physical_size_est", target_physical_size_est_);
    this->get_parameter("control.execution_delay", control_execution_delay_);
    this->get_parameter("control.rate_tc", RATE_TC_);
    double max_history_time_factor;
    this->get_parameter("control.max_history_time_factor", max_history_time_factor);
    max_history_time_ = control_execution_delay_ * max_history_time_factor;
    this->get_parameter("limits.att_limit", att_limit_);
    dt_ = TF_ / N_HORIZON_;

    // 相机参数
    double cam_focal_mm, cam_pixel_um;
    int cam_width_px;
    double fov_deg;
    this->get_parameter("camera.fov_deg", fov_deg);
    this->get_parameter("camera.width_px", cam_width_px);
    this->get_parameter("camera.focal_length_mm", cam_focal_mm);
    this->get_parameter("camera.pixel_size_um", cam_pixel_um);
    if (cam_width_px > 0) {
        cam_width_px_ = cam_width_px;
        cam_focal_length_ = cam_width_px * 0.5 / tan(fov_deg * M_PI / 180.0 / 2.0);
    } else {
        cam_focal_length_ = cam_focal_mm * 1000.0 / cam_pixel_um;
        cam_width_px_ = floor(2.0 * cam_focal_length_ * tan(fov_deg * M_PI / 180.0 / 2.0));
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
}

void InterceptorNodeQuaternion::mission_complete_callback(const std_msgs::msg::Bool::SharedPtr msg)
{
    if (!is_completed_ && msg->data) {
        RCLCPP_INFO(this->get_logger(), "Mission complete! Stopping control...");
        is_completed_ = true;
        failsafe(true);
        if (timer_) {
            timer_->cancel();
        }
    }
}
    
void InterceptorNodeQuaternion::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) 
{
    Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                            msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    // Eigen::Vector4d quat_normalized(q.w(), q.x(), q.y(), q.z());
    // 转换机体系速度到世界系速度
    Eigen::Vector3d vel_body(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);
    Eigen::Vector3d vel_world = q * vel_body;
    {
        std::lock_guard<std::mutex> lock(mtx_drone_state_);
        // 提取位置
        drone_state_.head<3>() << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
        // 提取四元数
        drone_state_.segment<4>(6) << q.w(), q.x(), q.y(), q.z();
        // 提取速度
        drone_state_.segment<3>(3) = vel_world;
        // 提取角速度
        drone_state_.tail<3>() << msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z;
        update_control_history(rclcpp::Time(msg->header.stamp).seconds(), drone_state_.tail<3>());
    }

    if (is_completed_) 
        return;
    
    // 检查极端姿态值
    Eigen::Vector3d euler_angles = to_euler_angles(q); // 仅用于检查
    if (std::abs(euler_angles(0)) > att_limit_ || std::abs(euler_angles(1)) > att_limit_) {
        RCLCPP_WARN(this->get_logger(), "Large tilt angle detected!! Roll: %.2f°, Pitch: %.2f°", 
                    euler_angles(0) * 180.0/M_PI, euler_angles(1) * 180.0/M_PI);
    }

    // double qx = q.x(), qy = q.y();
    // double cos_tilt_angle = 1 - 2*(qx*qx + qy*qy);
    // double tilt_angle = std::acos(std::clamp(cos_tilt_angle, -1.0, 1.0));
    // if (tilt_angle > att_limit_) 
    // {
    //     printf("Large tilt angle detected! Tilt: %.2f°\n", tilt_angle * 180.0/M_PI);
    // }

    if (!odom_received_)
    {
        for (int i = 0; i <= N_HORIZON_; ++i)
            x_warm_start_[i] = drone_state_;
        for (int i = 0; i < N_HORIZON_; ++i) 
            u_warm_start_[i] << msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z, G_;
        odom_received_ = true;
    }
}

void InterceptorNodeQuaternion::bbox_callback(const vision_msgs::msg::Detection2D::SharedPtr msg) 
{ 
    if (!odom_received_)
        return;
    // vision_msgs::msg::BoundingBox2D bbox = msg->bbox;
    {
        std::lock_guard<std::mutex> lock(mtx_bbox_);
        current_bbox_ = *msg;
        is_visible_ = current_bbox_.header.frame_id != "";
    }

    Vector13d curr_drone_state;
    {
        std::lock_guard<std::mutex> lock(mtx_drone_state_);
        curr_drone_state = drone_state_;
    }

    if (is_visible_)
    {
        TargetEKF::Bbox bbox;
        bbox.u = msg->bbox.center.position.x - cam_width_px_ / 2.0;
        bbox.v = msg->bbox.center.position.y - cam_width_px_ / 2.0;
        bbox.size = msg->bbox.size_x;

        double psi = get_yaw_from_quaternion(curr_drone_state.segment<4>(6));
        std::lock_guard<std::mutex> lock(mtx_ekf_);
        if (ekf_->is_initialized())
            ekf_->update(bbox, curr_drone_state.head<3>(), psi);
        else
        {
            Eigen::Vector3d raw_pos_est = ekf_->noisy_bbox_to_pos(bbox, curr_drone_state.head<3>(), psi);
            ekf_->initialize(raw_pos_est);
        }
    }
    
    if (!timer_ && msg->header.frame_id != "" && !is_completed_) {
        timer_boot_count_ = 0;
        timer_ = this->create_wall_timer(std::chrono::duration<double>(dt_), 
            std::bind(&InterceptorNodeQuaternion::real_loop, this),
            timer_callback_group_);
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
        // printf("Waiting for odometry...\n");
        return;
    }
    Eigen::Vector3d pos_est {0.0, 0.0, 0.0};
    Eigen::Vector3d vel_est {0.0, 0.0, 0.0};
    {
        std::lock_guard<std::mutex> lock(mtx_ekf_);
        if (ekf_->is_initialized())
        {
            ekf_->predict(dt_);
            const auto& ekf_state = ekf_->get_state();
            pos_est = ekf_state.head<3>();
            vel_est = ekf_state.tail<3>();
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
        // printf("Waiting for odometry...\n");
        return;
    }
    run_control_cycle(target_pos_true_, target_vel_true_);
    target_pos_true_ += target_vel_true_ * dt_; // 仅用于测试
    double distance = (target_pos_true_ - drone_state_.head<3>()).norm();
    RCLCPP_INFO(this->get_logger(), "Distance to target: %.2f m", distance);
    if (distance < interception_dist_) {
        RCLCPP_INFO(this->get_logger(), "[SUCCESS] Target intercepted at distance %.2f m. Stopping simulation.", distance);
        // printf("[SUCCESS] Target intercepted at distance %.2f m. Stopping simulation. \n", distance);
        timer_->cancel();
        
        // 悬停控制
        failsafe(true);
    }
}

void InterceptorNodeQuaternion::run_control_cycle(const Eigen::Vector3d &pos_est, const Eigen::Vector3d &vel_est) 
{
    Vector13d curr_drone_state;
    {
        std::lock_guard<std::mutex> lock(mtx_drone_state_);
        curr_drone_state = drone_state_;
    }

    if (is_visible_) {
        target_lost_counter_ = 0;
    } 
    else 
    {
        target_lost_counter_++;
        RCLCPP_WARN(this->get_logger(), "Target lost! Blind flight count: %d/%d", target_lost_counter_, N_HORIZON_);
    }

    if (target_lost_counter_ > N_HORIZON_ * 2) {
        RCLCPP_ERROR(this->get_logger(),
            "Target lost for too long. Aborting.");
        is_completed_ = true;
        failsafe(true);
        if (timer_) 
            timer_->cancel();
        return;
    }
    
    // 使用延迟补偿的预测状态设置初始条件约束
    Vector13d predicted_state = predict_delayed_state(curr_drone_state, control_execution_delay_);
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", predicted_state.data());
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", predicted_state.data());

    Vector16d p_vec;
    for (int k = 0; k <= N_HORIZON_; ++k) {
        Eigen::Vector3d predicted_target_pos = pos_est + (k * dt_) * vel_est;
        p_vec << predicted_target_pos, vel_est, DRAG_COEFF_, V_DESIRED_, RATE_TC_, w_cost_;
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
    RCLCPP_INFO(this->get_logger(), "ACADOS solve time: %ld us", std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());
    // printf("ACADOS solve time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());
    
    if (status != ACADOS_SUCCESS)
    {
        RCLCPP_WARN(this->get_logger(), "ACADOS solver returned status %d.", status);
        // printf("ACADOS solver returned status %d.\n", status);
        
        // 保持控制
        failsafe(false);
        output_filter_->reset();
        for (int i = 0; i < N_HORIZON_; ++i) {
            x_warm_start_[i] = curr_drone_state;
            u_warm_start_[i] = Eigen::Vector4d(0, 0, 0, G_);
        }
        x_warm_start_[N_HORIZON_] = x_warm_start_[N_HORIZON_ - 1];
        return;
    }

    Eigen::Vector4d u_optimal_raw;
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", u_optimal_raw.data());

    // Warm start update
    for (int i = 0; i < N_HORIZON_ - 1; ++i) {
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i + 1, "x", x_warm_start_[i].data());
        auto quat = x_warm_start_[i].segment<4>(6);
        quat = normalize_quaternion(quat);
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i + 1, "u", u_warm_start_[i].data());
    }
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, N_HORIZON_, "x", x_warm_start_[N_HORIZON_ - 1].data());
    auto quat = x_warm_start_[N_HORIZON_ - 1].segment<4>(6);
    quat = normalize_quaternion(quat);
    x_warm_start_[N_HORIZON_] = x_warm_start_[N_HORIZON_ - 1];
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, N_HORIZON_ - 1, "u", u_warm_start_[N_HORIZON_ - 1].data());

    // 输出滤波
    Eigen::VectorXd u_filtered_dynamic = output_filter_->filter(u_optimal_raw);
    
    if (timer_boot_count_ < N_HORIZON_ * 2)  // 前几次不发布控制，等待滤波器稳定
    {
        timer_boot_count_++;
        return;
    }

    // 提取控制量
    double thrust_accel = u_filtered_dynamic(3);
    Eigen::Vector3d body_rates(u_filtered_dynamic(0), u_filtered_dynamic(1), u_filtered_dynamic(2));

    quadrotor_msgs::msg::ControlCommand::UniquePtr control_msg = 
        std::make_unique<quadrotor_msgs::msg::ControlCommand>();
    control_msg->header.stamp = this->get_clock()->now();
    control_msg->control_mode = quadrotor_msgs::msg::ControlCommand::BODY_RATES;
    control_msg->armed = true;
    control_msg->bodyrates.x = body_rates(0); // body p rate
    control_msg->bodyrates.y = body_rates(1); // body q rate
    control_msg->bodyrates.z = body_rates(2); // body r rate
    control_msg->collective_thrust = thrust_accel;
    control_pub_->publish(std::move(control_msg));
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
Vector13d InterceptorNodeQuaternion::predict_delayed_state(const Vector13d& current_state, double delay_time) 
{
    Vector13d predicted_state = current_state;

    if (control_history_.empty() || delay_time < 1e-3)
    {
        return predicted_state;  // 如果没有历史数据，返回当前状态
    }
    
    double current_time = this->get_clock()->now().seconds();
    double prediction_start_time = current_time - delay_time;
    
    // 使用简单的积分来预测状态变化
    // 这里使用常数角速度假设进行积分

    Eigen::Vector3d average_body_rates = Eigen::Vector3d::Zero();
    int valid_samples = 0;
    
    // 计算延迟时间窗口内的平均角速度
    {
        std::lock_guard<std::mutex> lg(mtx_drone_state_);
        for (const auto& control_entry : control_history_) {
            if (control_entry.first >= prediction_start_time) {
                average_body_rates += control_entry.second;
                valid_samples++;
            }
        }
    }
    
    if (valid_samples > 0) {
        average_body_rates /= valid_samples;
    } else {
        // 如果没有有效样本，使用最近的角速度
        average_body_rates = current_state.tail<3>();;
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

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InterceptorNodeQuaternion>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
