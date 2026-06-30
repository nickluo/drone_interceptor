#include <chrono>
#include <cmath>

#include <Eigen/Dense>

#include "interceptor_node_base.h"

using namespace std::chrono_literals;

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
        if (status_counter_ <= 0)
        {
            if (status_queue_.empty()) return;
            auto [level, counter, text] = status_queue_.front();
            severity = level;
            status = text;
            status_counter_ = counter;
            status_queue_.pop();
        }
        if (status_counter_-- > 0)
        {
            auto msg = std::make_unique<mavros_msgs::msg::StatusText>();
            msg->header.stamp = rclcpp::Clock().now();
            msg->severity = severity;
            msg->text = status;
            status_pub_->publish(std::move(msg));
        }
    });
}

StatusPublisher::~StatusPublisher()
{
    if (status_timer_)
        status_timer_->cancel();
}

bool StatusPublisher::publish(uint8_t severity, int count, const std::string& text, bool force)
{
    std::lock_guard<std::mutex> lock(status_mtx_);
    if (!force && (status_counter_ > 0 || !status_queue_.empty()))
        return false;
    status_queue_.emplace(severity, count, text);
    return true;
}

// ============================================================================
// InterceptorNodeBase — Constructor / Destructor
// ============================================================================

InterceptorNodeBase::InterceptorNodeBase(const std::string& node_name)
    : Node(node_name)
{
    // Phase 1 only: declare common parameters
    // Subclass constructor must call initialize() after its own vtable is set
    declare_common_parameters();
}

InterceptorNodeBase::~InterceptorNodeBase()
{
    if (timer_) {
        timer_->cancel();
    }
    // Note: destroy_solver() is NOT called here — it's a pure virtual.
    // Each subclass destructor is responsible for calling destroy_solver().
}

void InterceptorNodeBase::initialize()
{
    // Called by subclass constructor after vtable is ready
    declare_specific_parameters();
    load_common_parameters();
    load_specific_parameters();

    if (!init_solver()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create ACADOS solver. Exiting.");
    }

    setup_ekf();

    int filter_n = static_cast<int>(RATE_TC_ * 2.0 / dt_);
    if (filter_n < 5) filter_n = 5;
    output_filter_ = std::make_unique<EMAFilter>(filter_n, 4);

    // PN filters (N=3: 3D TPN已EKF平滑，轻滤波降低末段相位滞后)
    pn_accel_filter_ = std::make_unique<EMAFilter>(3, 2);
    pn_omega_los_filter_ = std::make_unique<EMAFilter>(3, 3);

    // State initialization
    drone_state_ = make_identity_state();
    x_warm_start_.resize(N_HORIZON_ + 1, drone_state_);
    u_warm_start_.resize(N_HORIZON_, VectorU::Zero());
    for (int i = 0; i < N_HORIZON_; ++i) u_warm_start_[i] << 0, 0, 0, G_;

    // ROS setup
    sub_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    timer_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    control_pub_ = this->create_publisher<quadrotor_msgs::msg::ControlCommand>(
        "/fpv/control_command", rclcpp::SensorDataQoS());
    mission_state_pub_ = this->create_publisher<std_msgs::msg::Bool>(
        "/fpv/mission_state", 10);

    status_publisher_ = std::make_unique<StatusPublisher>(*this, 2);

    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = sub_callback_group_;

    if (use_simulation_) {
        RCLCPP_INFO(this->get_logger(), "Running in SIMULATION mode.");
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/fpv/odom", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
            std::bind(&InterceptorNodeBase::odom_callback, this, std::placeholders::_1), sub_options);
        timer_boot_count_ = 0;
        timer_ = this->create_wall_timer(std::chrono::duration<double>(dt_),
            std::bind(&InterceptorNodeBase::simulation_loop, this),
            timer_callback_group_);
        mission_state_.set(MissionState::INITIALIZING);
    } else {
        RCLCPP_INFO(this->get_logger(), "Running in REAL mode.");
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/fpv/odom", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
            std::bind(&InterceptorNodeBase::odom_callback, this, std::placeholders::_1), sub_options);
        gimbal_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/fpv/gimbal", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
            std::bind(&InterceptorNodeBase::gimbal_callback, this, std::placeholders::_1), sub_options);
        bbox_sub_ = this->create_subscription<vision_msgs::msg::Detection2D>(
            "/fpv/target_bbox", rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(),
            std::bind(&InterceptorNodeBase::bbox_callback, this, std::placeholders::_1), sub_options);
        mission_launch_sub_ = this->create_subscription<std_msgs::msg::Header>(
            "/fpv/tracker_trigger", 10,
            std::bind(&InterceptorNodeBase::mission_launch_callback, this, std::placeholders::_1));
    }
}

// ============================================================================
// Parameter Declaration and Loading
// ============================================================================

void InterceptorNodeBase::declare_common_parameters()
{
    this->declare_parameter<bool>("use_simulation", true);

    this->declare_parameter<double>("physics.g", 9.81);
    this->declare_parameter<double>("physics.drag_coeff", 0.1);
    this->declare_parameter<double>("limits.att_warning_deg", 45.0);
    this->declare_parameter<int>("mpc.ctrl_rate", 50);
    this->declare_parameter<double>("mpc.tf", 1.0);
    this->declare_parameter<double>("mpc.v_desired", 25.0);
    this->declare_parameter<double>("mpc.tracking_distance", 0.0);

    this->declare_parameter<bool>("pn.enabled", true);
    this->declare_parameter<double>("pn.nav_constant", 4.0);
    this->declare_parameter<double>("pn.switch_dist", 15.0);
    this->declare_parameter<double>("pn.fov_switch_deg", 15.0);
    this->declare_parameter<double>("pn.max_accel_xy", 20.0);
    this->declare_parameter<double>("pn.max_accel_z", 20.0);
    this->declare_parameter<double>("pn.drag_ff_coeff", 0.5);
    this->declare_parameter<double>("pn.tau_attitude", 0.4);

    this->declare_parameter<double>("sim.interception_dist", 1.0);
    this->declare_parameter<std::vector<double>>("sim.initial_conditions.target_pos", {100.0, 20.0, 15.0});
    this->declare_parameter<std::vector<double>>("sim.initial_conditions.target_vel", {-10.0, 10.0, 2.5});

    this->declare_parameter<double>("perception.bbox_pos_std_dev", 2.0);
    this->declare_parameter<double>("perception.bbox_size_std_dev", 5.0);
    this->declare_parameter<double>("perception.target_physical_size_est", 1.0);
    this->declare_parameter<double>("perception.detection_latency", 0.08);

    this->declare_parameter<double>("control.execution_delay", 0.05);
    this->declare_parameter<double>("control.rate_tc", 0.05);
    this->declare_parameter<double>("control.max_history_time_factor", 2.0);
    this->declare_parameter<bool>("control.rate_control_enabled", false);

    this->declare_parameter<int>("camera.width_px", 640);
    this->declare_parameter<int>("camera.height_px", 640);
    this->declare_parameter<double>("camera.fov_deg", -1.0);
    this->declare_parameter<int>("camera.native_width_px", 640);
    this->declare_parameter<double>("camera.focal_length_mm", 12.0);
    this->declare_parameter<double>("camera.pixel_size_um", 3.45);

    this->declare_parameter<double>("gimbal.rp_coeff", 1.2);
    this->declare_parameter<double>("gimbal.y_coeff", 1.5);
    this->declare_parameter<bool>("camera.fixed", false);
    this->declare_parameter<double>("camera.pitch_deg", 0.0);
}

void InterceptorNodeBase::load_common_parameters()
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
    this->get_parameter("perception.detection_latency", detection_latency_);
    this->get_parameter("control.execution_delay", control_execution_delay_);
    this->get_parameter("control.rate_tc", RATE_TC_);
    this->get_parameter("control.rate_control_enabled", rate_control_enabled_);
    ctrl_coeff_ = 1.0 - std::exp(-dt_ / RATE_TC_);
    double max_history_time_factor;
    this->get_parameter("control.max_history_time_factor", max_history_time_factor);
    max_history_time_ = control_execution_delay_ * max_history_time_factor;
    this->get_parameter("limits.att_warning_deg", att_limit_);

    this->get_parameter("gimbal.rp_coeff", gimbal_rp_coeff_);
    this->get_parameter("gimbal.y_coeff", gimbal_y_coeff_);
    this->get_parameter("camera.fixed", camera_fixed_);
    this->get_parameter("camera.pitch_deg", camera_pitch_deg_);

    // Camera
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

    if (use_simulation_) {
        std::vector<double> target_pos_vec, target_vel_vec;
        this->get_parameter("sim.initial_conditions.target_pos", target_pos_vec);
        this->get_parameter("sim.initial_conditions.target_vel", target_vel_vec);
        target_pos_true_ = Eigen::Map<Eigen::Vector3d>(target_pos_vec.data());
        target_vel_true_ = Eigen::Map<Eigen::Vector3d>(target_vel_vec.data());
        this->get_parameter("sim.interception_dist", interception_dist_);
    }

    // PN
    this->get_parameter("pn.enabled", pn_enabled_);
    this->get_parameter("pn.nav_constant", pn_nav_constant_);
    this->get_parameter("pn.switch_dist", pn_switch_dist_);
    this->get_parameter("pn.fov_switch_deg", pn_fov_switch_deg_);
    this->get_parameter("pn.max_accel_xy", pn_max_accel_xy_);
    this->get_parameter("pn.max_accel_z", pn_max_accel_z_);
    this->get_parameter("pn.drag_ff_coeff", pn_drag_ff_coeff_);
    this->get_parameter("pn.tau_attitude", pn_tau_attitude_);
}

// ============================================================================
// ROS Callbacks
// ============================================================================

void InterceptorNodeBase::gimbal_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    if (!mission_state_.is_active())
        return;
    gimbal_received_ = true;
    {
        std::unique_lock<std::shared_mutex> lock(mtx_gimbal_);
        current_gimbal_orientation_ = Eigen::Quaterniond(
            msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
    }
    {
        std::unique_lock<std::shared_mutex> lock(mtx_drone_state_);
        // Gimbal euler angles stored in last 3 elements of state
        int n = drone_state_.size();
        drone_state_.segment<3>(n - 3) << Eigen::Vector3d(
            msg->orientation_covariance[0], msg->orientation_covariance[1], msg->orientation_covariance[2]);
    }
    {
        std::unique_lock<std::shared_mutex> lock(mtx_ekf_);
        if (ekf_->is_initialized()) {
            Eigen::Vector3d current_gimbal_rate;
            current_gimbal_rate << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
            ekf_->predict(rclcpp::Time(msg->header.stamp).nanoseconds() / 1000, current_gimbal_rate);
        }
    }
}

void InterceptorNodeBase::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    if (!mission_state_.is_active())
        return;
    Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                         msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);

    if (std::isnan(q.w()) || std::isnan(q.x()) || std::isnan(q.y()) || std::isnan(q.z()) ||
        std::isnan(msg->pose.pose.position.x) || std::isnan(msg->pose.pose.position.y) || std::isnan(msg->pose.pose.position.z)) {
        RCLCPP_ERROR(this->get_logger(), "NaN detected in odometry! Ignoring.");
        return;
    }

    Eigen::Vector3d pos(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    Eigen::Vector3d vel_body(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);
    Eigen::Vector3d omega(msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z);

    {
        std::unique_lock<std::shared_mutex> lock(mtx_drone_state_);
        encode_odom(drone_state_, pos, q, vel_body, omega);
        if (!gimbal_received_) {
            int n = drone_state_.size();
            drone_state_.segment<3>(n - 3).setZero();
        }
    }

    if (camera_fixed_) {
        std::unique_lock<std::shared_mutex> lock(mtx_ekf_);
        if (ekf_->is_initialized()) {
            double pitch_rad = -camera_pitch_deg_ * M_PI / 180.0;
            Eigen::Matrix3d R_B_C;
            R_B_C = Eigen::AngleAxisd(pitch_rad, Eigen::Vector3d::UnitY());
            Eigen::Vector3d w_cam = R_B_C * omega;
            ekf_->predict(rclcpp::Time(msg->header.stamp).nanoseconds() / 1000, w_cam, true);
        }
    } else if (!gimbal_received_) {
        std::unique_lock<std::shared_mutex> lock(mtx_ekf_);
        if (ekf_->is_initialized()) {
            Eigen::Vector3d w_world = q * omega;
            ekf_->predict(rclcpp::Time(msg->header.stamp).nanoseconds() / 1000, Eigen::Vector3d(0.0, 0.0, w_world.z()));
        }
    }

    update_control_history(rclcpp::Time(msg->header.stamp).seconds(), omega);

    if (mission_state_.is_terminated())
        return;

    if (!odom_received_) {
        {
            std::unique_lock<std::shared_mutex> lock(mtx_drone_state_);
            for (int i = 0; i <= N_HORIZON_; ++i)
                x_warm_start_[i] = drone_state_;
        }
        for (int i = 0; i < N_HORIZON_; ++i)
            u_warm_start_[i] << omega.x(), omega.y(), omega.z(), G_;
        odom_received_ = true;
    }
}

void InterceptorNodeBase::bbox_callback(const vision_msgs::msg::Detection2D::SharedPtr msg)
{
    if (!odom_received_ || !mission_state_.is_active())
        return;

    DroneEKF::Bbox bbox;
    bbox.cx = msg->bbox.center.position.x;
    bbox.cy = msg->bbox.center.position.y;
    bbox.w = msg->bbox.size_x;
    bbox.h = msg->bbox.size_y;

    bool frame_valid = msg->header.frame_id != "";

    // Bbox jump detection
    if (frame_valid && last_bbox_valid_) {
        double dx = bbox.cx - last_bbox_.cx;
        double dy = bbox.cy - last_bbox_.cy;
        double center_jump = std::sqrt(dx * dx + dy * dy);
        double size_ratio_w = (last_bbox_.w > 0) ? bbox.w / last_bbox_.w : 1.0;
        double size_ratio_h = (last_bbox_.h > 0) ? bbox.h / last_bbox_.h : 1.0;
        const double max_center_jump = cam_intrinsics_.px * 0.3;
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
            if (bbox_jump_counter_ >= 3) {
                RCLCPP_ERROR(this->get_logger(), "Bbox jump persistent, resetting last_bbox");
                last_bbox_valid_ = false;
                bbox_jump_counter_ = 0;
            }
            return;
        } else {
            bbox_jump_counter_ = 0;
        }
    }

    last_bbox_ = bbox;
    last_bbox_valid_ = frame_valid;

    {
        std::lock_guard<std::mutex> lock(mtx_bbox_);
        current_bbox_ = *msg;
        is_visible_ = frame_valid;
    }

    if (is_visible_) {
        int64_t detection_time = rclcpp::Time(msg->header.stamp).nanoseconds() / 1000;

        Eigen::VectorXd curr_drone_state_for_ekf;
        {
            std::shared_lock<std::shared_mutex> lock(mtx_drone_state_);
            curr_drone_state_for_ekf = drone_state_;
        }

        Eigen::Vector3d drone_pos = get_position(curr_drone_state_for_ekf);
        Eigen::Quaterniond q_C_W;
        if (camera_fixed_) {
            Eigen::Vector4d quat = get_quaternion(curr_drone_state_for_ekf);
            Eigen::Quaterniond q_B_W(quat(0), quat(1), quat(2), quat(3));
            Eigen::Quaterniond q_C_B(Eigen::AngleAxisd(camera_pitch_deg_ * M_PI / 180.0, Eigen::Vector3d::UnitY()));
            q_C_W = q_B_W * q_C_B;
        } else if (gimbal_received_) {
            std::shared_lock<std::shared_mutex> lock(mtx_gimbal_);
            q_C_W = current_gimbal_orientation_;
        } else {
            Eigen::Vector4d quat = get_quaternion(curr_drone_state_for_ekf);
            double psi = get_yaw_from_quaternion(quat);
            Eigen::AngleAxisd yaw_rot(psi, Eigen::Vector3d::UnitZ());
            q_C_W = Eigen::Quaterniond(yaw_rot);
        }

        detection_ratio_ = std::max(bbox.w / cam_intrinsics_.px, bbox.h / cam_intrinsics_.py);

        if (detection_ratio_ < 1.0) {
            std::unique_lock<std::shared_mutex> ekf_lock(mtx_ekf_);
            if (detection_ratio_ > 0.1) {
                const double coeff = std::exp((detection_ratio_ - 0.1) * 2);
                const double bbox_pos_std = coeff * bbox_pos_std_dev_;
                const double bbox_size_std = coeff * bbox_size_std_dev_;
                const double q_accel_approach = std::max(1.0, coeff * coeff);

                Eigen::Matrix<double, 9, 1> process_noise;
                process_noise << bbox_pos_std * 0.1,
                                 bbox_pos_std * 0.5,
                                 bbox_pos_std * 0.25,
                                 bbox_size_std * 0.1,
                                 bbox_size_std * 0.5,
                                 bbox_size_std * 0.25,
                                 ekf_std_q_W_,
                                 std::sqrt(q_accel_approach),
                                 0.0;

                Eigen::Vector3d measurement_noise;
                measurement_noise << bbox_pos_std, bbox_size_std, bbox_size_std_dev_ * 0.1;
                ekf_->set_covariance(process_noise, measurement_noise);
            }
            ekf_->update(detection_time, bbox, drone_pos, q_C_W);
        }
    }

    if (!timer_ && msg->header.frame_id != "" && mission_state_.is_active()) {
        timer_boot_count_ = 0;
        timer_ = this->create_wall_timer(std::chrono::duration<double>(dt_),
            std::bind(&InterceptorNodeBase::real_loop, this),
            timer_callback_group_);
        mission_state_.transition(MissionState::INITIALIZING, MissionState::TRACKING);
    }
}

void InterceptorNodeBase::mission_launch_callback(const std_msgs::msg::Header::SharedPtr msg)
{
    if (msg->frame_id == "OFF" && mission_state_.is_active()) {
        RCLCPP_INFO(this->get_logger(), "Mission abort received! Stopping control...");
        status_publisher_->publish(
            mavros_msgs::msg::StatusText::WARNING,
            1, "Mission aborted. Stopping...", true);
        transition_to_state(MissionState::ABORTED, "User abort command");
        stop();
        return;
    }

    if (!mission_state_.is_terminated() || timer_)
        return;

    if (msg->frame_id == "MISSION") {
        RCLCPP_INFO(this->get_logger(), "Mission launch received! Starting control...");
        status_publisher_->publish(
            mavros_msgs::msg::StatusText::INFO,
            1, "Mission launched. Starting...", true);

        // Recreate acados solver to clear internal dual variables and QP warm start.
        // Without this, residual Lagrange multipliers from the previous mission's
        // terminal high-speed state corrupt SQP_RTI's first QP iteration.
        destroy_solver();
        if (!init_solver()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to recreate ACADOS solver on mission restart.");
            return;
        }

        {
            std::unique_lock<std::shared_mutex> lock(mtx_ekf_);
            setup_ekf();
        }
        {
            std::unique_lock<std::shared_mutex> lock(mtx_drone_state_);
            drone_state_ = make_identity_state();
        }
        {
            std::lock_guard<std::mutex> lock(mtx_bbox_);
            current_bbox_ = vision_msgs::msg::Detection2D();
            is_visible_ = false;
        }

        output_filter_->reset();
        pn_accel_filter_->reset();
        pn_omega_los_filter_->reset();
        control_history_.clear();

        odom_received_ = false;
        gimbal_received_ = false;

        timer_boot_count_ = 0;
        target_lost_counter_ = 0;
        solver_failure_counter_ = 0;
        consecutive_solver_failures_ = 0;

        pn_active_ = 0;
        pn_collision_detected_ = false;
        pn_last_cmd_rates_.setZero();

        solve_count_ = 0;
        avg_solve_time_us_ = 0.0;
        ekf_health_check_counter_ = 0;

        for (int i = 0; i <= N_HORIZON_; ++i) x_warm_start_[i] = drone_state_;
        for (int i = 0; i < N_HORIZON_; ++i) u_warm_start_[i] << 0, 0, 0, G_;

        mission_state_.set(MissionState::INITIALIZING);
        mission_state_pub_->publish(std::make_unique<std_msgs::msg::Bool>(std_msgs::msg::Bool().set__data(true)));
    }
}

// ============================================================================
// Control Loop
// ============================================================================

void InterceptorNodeBase::failsafe(bool hover)
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

void InterceptorNodeBase::real_loop()
{
    if (!odom_received_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for odometry...");
        return;
    }

    check_ekf_health();

    Eigen::Vector3d pos_est{0.0, 0.0, 0.0};
    Eigen::Vector3d vel_est{0.0, 0.0, 0.0};

    Eigen::VectorXd curr_drone_state;
    Eigen::Quaterniond gimbal_orientation;
    {
        std::shared_lock<std::shared_mutex> lock(mtx_drone_state_);
        curr_drone_state = drone_state_;
    }

    if (camera_fixed_) {
        Eigen::Vector4d quat = get_quaternion(curr_drone_state);
        Eigen::Quaterniond q_B_W(quat(0), quat(1), quat(2), quat(3));
        Eigen::Quaterniond q_C_B(Eigen::AngleAxisd(camera_pitch_deg_ * M_PI / 180.0, Eigen::Vector3d::UnitY()));
        gimbal_orientation = q_B_W * q_C_B;
    } else if (gimbal_received_) {
        std::shared_lock<std::shared_mutex> lock(mtx_gimbal_);
        gimbal_orientation = current_gimbal_orientation_;
    } else {
        Eigen::Vector4d quat = get_quaternion(curr_drone_state);
        double psi = get_yaw_from_quaternion(quat);
        Eigen::AngleAxisd yaw_rot(psi, Eigen::Vector3d::UnitZ());
        gimbal_orientation = Eigen::Quaterniond(yaw_rot);
    }

    {
        std::shared_lock<std::shared_mutex> lock(mtx_ekf_);
        if (ekf_->is_initialized()) {
            Eigen::VectorXd est_state = ekf_->get_state();
            pos_est = est_state.head<3>();
            vel_est = est_state.tail<3>();
            if (!ekf_->is_healthy()) {
                RCLCPP_ERROR(this->get_logger(), "NaN in EKF state, reinitializing...");
                lock.unlock();
                std::unique_lock<std::shared_mutex> write_lock(mtx_ekf_);
                ekf_->reset();
                return;
            }
        } else {
            return;
        }
    }
    run_control_cycle(pos_est, vel_est);
}

void InterceptorNodeBase::simulation_loop()
{
    if (!odom_received_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for odometry...");
        return;
    }
    run_control_cycle(target_pos_true_, target_vel_true_);
    target_pos_true_ += target_vel_true_ * dt_;

    Eigen::VectorXd curr_state;
    {
        std::shared_lock<std::shared_mutex> lock(mtx_drone_state_);
        curr_state = drone_state_;
    }
    Eigen::Vector3d drone_pos = get_position(curr_state);
    double distance = (target_pos_true_ - drone_pos).norm();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
        "Distance to target: %.2f m", distance);
    if (distance < interception_dist_) {
        RCLCPP_INFO(this->get_logger(), "[SUCCESS] Target intercepted at distance %.2f m. Stopping simulation.", distance);
        transition_to_state(MissionState::COMPLETED, "Target intercepted");
        stop();
    }
}

void InterceptorNodeBase::stop()
{
    if (timer_) {
        timer_->cancel();
        timer_ = nullptr;
    }
    failsafe(true);
    mission_state_pub_->publish(std::make_unique<std_msgs::msg::Bool>(std_msgs::msg::Bool().set__data(false)));
}

// ============================================================================
// run_control_cycle — the main algorithm skeleton
// ============================================================================

void InterceptorNodeBase::run_control_cycle(const Eigen::Vector3d& pos_est, const Eigen::Vector3d& vel_est)
{
    Eigen::VectorXd curr_drone_state;
    {
        std::shared_lock<std::shared_mutex> lock(mtx_drone_state_);
        curr_drone_state = drone_state_;
    }

    Eigen::Vector3d drone_pos = get_position(curr_drone_state);
    Eigen::Vector3d drone_vel = get_velocity_world(curr_drone_state);

    // Target lost management
    if (is_visible_) {
        target_lost_counter_ = 0;
        mission_state_.transition(MissionState::BLIND_FLIGHT, MissionState::TRACKING);
    } else {
        if (target_lost_counter_ == 0) {
            mission_state_.transition(MissionState::TRACKING, MissionState::BLIND_FLIGHT);
        }
        if (target_lost_counter_ % 10 == 0) {
            status_publisher_->publish(
                mavros_msgs::msg::StatusText::WARNING,
                1, "Target lost. Blind flight", false);
        }
        target_lost_counter_++;
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "Target lost! Blind flight count: %d/%d", target_lost_counter_, N_HORIZON_);
    }

    int target_lost_counter_limit = N_HORIZON_ * 2;
    if (pn_active_ > 0) {
        target_lost_counter_limit = N_HORIZON_;
    }

    if (target_lost_counter_ > target_lost_counter_limit) {
        RCLCPP_ERROR(this->get_logger(), "Target lost for too long. Aborting.");
        status_publisher_->publish(
            mavros_msgs::msg::StatusText::ERROR,
            1, "Target lost. Aborting.", true);
        transition_to_state(MissionState::ABORTED, "Target lost too long");
        stop();
        return;
    }

    if (!validate_state(curr_drone_state, "current_state")) {
        failsafe(false);
        return;
    }

    if (!check_attitude_safety(curr_drone_state)) {
        // Log but continue
    }

    Eigen::VectorXd predicted_state = predict_delayed_state(curr_drone_state, control_execution_delay_);
    if (!validate_state(predicted_state, "predicted_state")) {
        RCLCPP_ERROR(this->get_logger(), "Invalid predicted state! Resetting solver.");
        failsafe(false);
        output_filter_->reset();
        for (int i = 0; i <= N_HORIZON_; ++i) x_warm_start_[i] = curr_drone_state;
        for (int i = 0; i < N_HORIZON_; ++i) u_warm_start_[i] << 0, 0, 0, G_;
        return;
    }

    // PN switching logic
    if (pn_enabled_ && pn_active_ == 0 && !std::isnan(pos_est.x()) && !std::isnan(vel_est.x())) {
        Eigen::Vector3d rel_pos = pos_est - drone_pos;
        Eigen::Vector3d rel_vel = vel_est - drone_vel;
        double closing_speed_est = 1.0;
        double dist_to_target = rel_pos.norm();
        if (dist_to_target > 1e-3) {
            closing_speed_est = -rel_pos.dot(rel_vel) / dist_to_target;
        }

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
            double angle_off_boresight = std::sqrt(azimuth * azimuth + elevation * elevation);

            RCLCPP_INFO(this->get_logger(), "LOS angles (az,el): [%.2f, %.2f] deg, Off-boresight: %.2f deg, In FOV: %d",
                azimuth * 180.0 / M_PI, elevation * 180.0 / M_PI,
                angle_off_boresight * 180.0 / M_PI,
                (angle_off_boresight < pn_fov_switch_deg_ * M_PI / 180.0) ? 1 : 0);
            RCLCPP_INFO(this->get_logger(), "Distance to target: %.2f m, Closing speed: %.2f m/s, Estimated width: %.2f m [W %s]",
                dist_to_target, closing_speed_est, ekf_->getEstimatedWidth(),
                ekf_->isWidthObservable() ? "OBS" : "FROZEN");

            auto full_state = ekf_->getFullState();
            auto v_imu = ekf_->getImuFlowCache();
            RCLCPP_INFO(this->get_logger(),
                "EKF diag: vx_img=%.1f vy_img=%.1f | v_imu=[%.1f,%.1f] | ax=%.1f ay=%.1f | LOS_rate=[%.2f,%.2f]°/s",
                full_state(4), full_state(5), v_imu(0), v_imu(1),
                full_state(8), full_state(9),
                los_state(2) * 180.0 / M_PI, los_state(3) * 180.0 / M_PI);

            // Fuzzy scoring PN switch decision
            double angle_off_boresight_deg = angle_off_boresight * 180.0 / M_PI;
            double vertical_speed = std::abs(rel_vel.z());
            double max_vertical_speed_for_pn = std::max(2.0, closing_speed_est * 0.5);
            double los_rate_check = std::sqrt(los_state(2) * los_state(2) + los_state(3) * los_state(3));
            double los_rate_check_deg = los_rate_check * 180.0 / M_PI;

            double dist_factor = std::clamp((pn_switch_dist_ * 1.5 - dist_to_target) / pn_switch_dist_, 0.0, 1.5);
            double effective_fov_deg = pn_fov_switch_deg_ * (1.0 + dist_factor);
            double max_los_rate_deg = 3.0 * effective_fov_deg;

            double fov_score = 1.0 / (1.0 + std::exp(4.0 * (angle_off_boresight_deg / effective_fov_deg - 1.0)));
            double vz_score = 1.0 / (1.0 + std::exp(4.0 * (vertical_speed / max_vertical_speed_for_pn - 1.0)));
            double los_rate_score = 1.0 / (1.0 + std::exp(4.0 * (los_rate_check_deg / max_los_rate_deg - 1.0)));

            double dist_ratio = dist_to_target / (pn_switch_dist_ * 1.5);
            double urgency = 1.0 + 0.5 * (1.0 - dist_ratio);

            double pn_score = (0.5 * fov_score + 0.25 * vz_score + 0.25 * los_rate_score) * urgency;
            const double pn_switch_threshold = 0.60;

            bool force_switch = (dist_to_target < pn_switch_dist_) && is_visible_;

            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 100,
                "PN score: %.2f (fov=%.2f[eff=%.1f°] vz=%.2f los=%.2f urgency=%.2f) thresh=%.2f %s",
                pn_score, fov_score, effective_fov_deg, vz_score, los_rate_score, urgency,
                pn_switch_threshold, force_switch ? "[FORCE]" : "");

            if ((dist_to_target < pn_switch_dist_ * 1.5 && pn_score >= pn_switch_threshold) || force_switch) {
                pn_vc_initial_ = closing_speed_est;
                pn_active_ = 1;
                pn_target_vel_valid_ = false;
                pn_target_accel_filtered_ = Eigen::Vector3d::Zero();
                pn_apn_warmup_counter_ = 0;
                pn_last_omega_los_ = Eigen::Vector3d::Zero();
                pn_collision_detected_ = false;
                pn_last_cmd_rates_ = Eigen::Vector3d::Zero();
                pn_accel_filter_->reset();
                pn_omega_los_filter_->reset();
                RCLCPP_INFO(this->get_logger(),
                    "Switching to IMAGE-BASED PN (Latch Active, score=%.2f%s), Thrust Efficiency: %.3f",
                    pn_score, force_switch ? " FORCED" : "", thrust_efficiency_);
            }
        }
    }

    // ================================================================
    // Compute control
    // ================================================================

    // Boot phase: skip solver entirely to prevent warm-start divergence
    if (timer_boot_count_++ < N_HORIZON_ * 1) {
        failsafe(false);
        return;
    }
    // One-time reset at boot → active transition
    if (timer_boot_count_ == N_HORIZON_ + 1) {
        RCLCPP_INFO(this->get_logger(), "Boot complete: resetting solver warm start and output filter");
        reset_solver_warm_start(curr_drone_state);
        output_filter_->reset();
    }

    VectorU u_optimal_raw;

    if (pn_active_ > 0) {
        u_optimal_raw = calculate_pn_controls_image_based(curr_drone_state, pos_est, vel_est);
        consecutive_solver_failures_ = 0;

        // Collision detection (gyroscope)
        if (!pn_collision_detected_) {
            Eigen::Vector3d gyro_rates = get_omega_body(curr_drone_state);
            Eigen::Vector3d rate_error = gyro_rates - pn_last_cmd_rates_ * ctrl_coeff_;
            double rate_error_norm = rate_error.norm();
            if (rate_error_norm > 5.0) {
                pn_collision_detected_ = true;
                Eigen::Vector3d vel = get_velocity_world(curr_drone_state);
                RCLCPP_WARN(this->get_logger(),
                    "[COLLISION] Gyro spike! |err|=%.1f rad/s "
                    "gyro=[%.2f,%.2f,%.2f] cmd=[%.2f,%.2f,%.2f] "
                    "vel=[%.1f,%.1f,%.1f]",
                    rate_error_norm,
                    gyro_rates.x(), gyro_rates.y(), gyro_rates.z(),
                    pn_last_cmd_rates_.x() * ctrl_coeff_,
                    pn_last_cmd_rates_.y() * ctrl_coeff_,
                    pn_last_cmd_rates_.z() * ctrl_coeff_,
                    vel.x(), vel.y(), vel.z());
                transition_to_state(MissionState::COMPLETED, "Collision detected via gyro");
                stop();
                status_publisher_->publish(
                    mavros_msgs::msg::StatusText::WARNING,
                    1, "Mission Completed. Stopping...", true);
                return;
            }
            pn_last_cmd_rates_ = Eigen::Vector3d(u_optimal_raw(0), u_optimal_raw(1), u_optimal_raw(2));
        }
    } else {
        // MPC solve
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", predicted_state.data());
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", predicted_state.data());

        pack_and_set_mpc_params(pos_est, vel_est, drone_pos);

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 200,
            "MPC init: pos=[%.1f,%.1f,%.1f] vel=[%.2f,%.2f,%.2f] tgt=[%.1f,%.1f,%.1f] x0=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]",
            drone_pos.x(), drone_pos.y(), drone_pos.z(),
            drone_vel.x(), drone_vel.y(), drone_vel.z(),
            pos_est.x(), pos_est.y(), pos_est.z(),
            predicted_state(0), predicted_state(1), predicted_state(2), predicted_state(3),
            predicted_state(4), predicted_state(5), predicted_state(6), predicted_state(7));

        // Set warm start
        for (int i = 0; i < N_HORIZON_; ++i) {
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "x", x_warm_start_[i].data());
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "u", u_warm_start_[i].data());
        }
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, N_HORIZON_, "x", x_warm_start_[N_HORIZON_].data());

        // Solve
        auto t_start = std::chrono::high_resolution_clock::now();
        int status = solve_step();
        auto t_end = std::chrono::high_resolution_clock::now();

        double solve_time_us = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());
        solve_count_++;
        avg_solve_time_us_ = avg_solve_time_us_ + (solve_time_us - avg_solve_time_us_) / solve_count_;

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "ACADOS solve: %.0f us (avg: %.0f us)", solve_time_us, avg_solve_time_us_);

        if (status != ACADOS_SUCCESS) {
            SolverRecoveryAction action = handle_solver_failure(status);
            switch (action) {
                case SolverRecoveryAction::CONTINUE:
                    RCLCPP_DEBUG(this->get_logger(), "Solver failed, using previous solution");
                    break;
                case SolverRecoveryAction::RESET_WARM:
                    reset_solver_warm_start(curr_drone_state);
                    failsafe(false);
                    output_filter_->reset();
                    return;
                case SolverRecoveryAction::FAILSAFE:
                    transition_to_state(MissionState::FAILSAFE, "Solver failures");
                    failsafe(true);
                    output_filter_->reset();
                    return;
                case SolverRecoveryAction::ABORT:
                    RCLCPP_ERROR(this->get_logger(), "ACADOS solver failed too many times. Aborting.");
                    status_publisher_->publish(
                        mavros_msgs::msg::StatusText::ERROR,
                        1, "ACADOS solver failed. Aborting.", true);
                    transition_to_state(MissionState::ABORTED, "Solver failed");
                    stop();
                    return;
            }
        } else {
            consecutive_solver_failures_ = 0;
            // Thrust efficiency estimation (currently disabled)
            last_thrust_cmd_ = u_optimal_raw(3);
        }

        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", u_optimal_raw.data());

        // Debug: log solver stage-1 state to see if trajectory planning is reasonable
        {
            Eigen::VectorXd x1_debug = Eigen::VectorXd::Zero(state_dim());
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", x1_debug.data());
            Eigen::Vector3d pos1 = get_position(x1_debug);
            Eigen::Vector3d vel1 = get_velocity_world(x1_debug);
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 200,
                "MPC plan: u0=[%.3f,%.3f,%.3f,%.2f] x1_pos=[%.1f,%.1f,%.1f] x1_vel=[%.2f,%.2f,%.2f]",
                u_optimal_raw(0), u_optimal_raw(1), u_optimal_raw(2), u_optimal_raw(3),
                pos1.x(), pos1.y(), pos1.z(), vel1.x(), vel1.y(), vel1.z());
        }
    }

    // Warm start update (MPC only)
    if (pn_active_ == 0) {
        if (!validate_control(u_optimal_raw)) {
            RCLCPP_WARN(this->get_logger(), "Invalid control output, applying limits");
        }
        u_optimal_raw = apply_control_limits(u_optimal_raw);
        for (int i = 0; i < N_HORIZON_ - 1; ++i) {
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i + 1, "x", x_warm_start_[i].data());
            normalize_warm_start_orientation(x_warm_start_[i]);
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i + 1, "u", u_warm_start_[i].data());
        }
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, N_HORIZON_, "x", x_warm_start_[N_HORIZON_ - 1].data());
        normalize_warm_start_orientation(x_warm_start_[N_HORIZON_ - 1]);
        x_warm_start_[N_HORIZON_] = x_warm_start_[N_HORIZON_ - 1];
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, N_HORIZON_ - 1, "u", u_warm_start_[N_HORIZON_ - 1].data());
    }

    // ================================================================
    // Extract and publish control
    // ================================================================
    double thrust = 0;
    Eigen::Vector3d body_rates;

    if (pn_active_ == 1 && detection_ratio_ > 1.0) {
        pn_active_ = 2;
        thrust = pn_last_u_pn_(3);
        body_rates << pn_last_u_pn_(0), pn_last_u_pn_(1), pn_last_u_pn_(2);
    }
    else if (pn_active_ > 0) {
        // PN mode: direct output (bypass MPC EMA filter to avoid phase lag)
        thrust = u_optimal_raw(3);
        body_rates << u_optimal_raw(0), u_optimal_raw(1), u_optimal_raw(2);
    }
    else {
        // MPC mode: EMA filtered output
        Eigen::VectorXd u_filtered_dynamic = output_filter_->filter(u_optimal_raw);
        thrust = u_filtered_dynamic(3);
        body_rates << u_filtered_dynamic(0), u_filtered_dynamic(1), u_filtered_dynamic(2);
    }

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 100,
        "MPC u_raw=[%.3f,%.3f,%.3f,%.3f] cmd=[%.3f,%.3f,%.3f] T=%.3f pn=%d boot=%d",
        u_optimal_raw(0), u_optimal_raw(1), u_optimal_raw(2), u_optimal_raw(3),
        body_rates(0) * ctrl_coeff_, body_rates(1) * ctrl_coeff_, body_rates(2) * ctrl_coeff_,
        thrust, pn_active_, timer_boot_count_);

    auto control_msg = std::make_unique<quadrotor_msgs::msg::ControlCommand>();
    control_msg->header.stamp = this->get_clock()->now();
    control_msg->armed = true;
    control_msg->collective_thrust = thrust;
    if (rate_control_enabled_) {
        control_msg->control_mode = quadrotor_msgs::msg::ControlCommand::BODY_RATES;
        control_msg->bodyrates.x = body_rates(0) * ctrl_coeff_;
        control_msg->bodyrates.y = body_rates(1) * ctrl_coeff_;
        control_msg->bodyrates.z = body_rates(2) * ctrl_coeff_;
    } else {
        control_msg->control_mode = quadrotor_msgs::msg::ControlCommand::ATTITUDE;
        control_msg->expected_execution_time.sec = static_cast<int>(dt_);
        control_msg->expected_execution_time.nanosec = static_cast<int>((dt_ - static_cast<int>(dt_)) * 1e9);
        Eigen::Vector4d quat = get_quaternion(curr_drone_state);
        control_msg->orientation.w = quat(0);
        control_msg->orientation.x = quat(1);
        control_msg->orientation.y = quat(2);
        control_msg->orientation.z = quat(3);
        control_msg->bodyrates.x = body_rates(0);
        control_msg->bodyrates.y = body_rates(1);
        control_msg->bodyrates.z = body_rates(2);
    }
    control_pub_->publish(std::move(control_msg));
}

// ============================================================================
// Image-Based PN Guidance (3D TPN with priority tilt budget)
// ============================================================================

VectorU InterceptorNodeBase::calculate_pn_controls_image_based(
    const Eigen::VectorXd& drone_state,
    const Eigen::Vector3d& pos_est, const Eigen::Vector3d& vel_est)
{
    VectorU u_pn;
    u_pn << 0.0, 0.0, 0.0, G_;

    if (pn_active_ == 2) {
        return pn_last_u_pn_;
    }

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

    double azimuth = los_state(0);
    double elevation = los_state(1);
    double azimuth_rate = los_state(2);
    double elevation_rate = los_state(3);

    if (std::isnan(azimuth_rate) || std::isnan(elevation_rate)) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 100,
            "PN: Invalid LOS rates (NaN), returning hover");
        return u_pn;
    }

    // Miss detection
    double off_boresight_rad = std::sqrt(azimuth * azimuth + elevation * elevation);
    double off_boresight_deg = off_boresight_rad * 180.0 / M_PI;
    if (off_boresight_deg > 30.0 && pn_active_ == 1) {
        pn_active_ = 2;
        pn_last_u_pn_.head<3>() *= 0.3;
        RCLCPP_WARN(this->get_logger(),
            "PN: Miss detected! Off-boresight=%.1f° > 30°, switching to terminal coast",
            off_boresight_deg);
        return pn_last_u_pn_;
    }

    // 3D TPN computation
    Eigen::Vector3d drone_pos = get_position(drone_state);
    Eigen::Vector3d drone_vel = get_velocity_world(drone_state);

    Eigen::Vector3d rel_pos = pos_est - drone_pos;
    Eigen::Vector3d rel_vel = vel_est - drone_vel;

    double dist_to_target = rel_pos.norm();
    if (dist_to_target < 0.1) {
        RCLCPP_WARN(this->get_logger(), "PN: Target too close (%.2fm)", dist_to_target);
        return pn_last_u_pn_;
    }

    Eigen::Vector3d los_unit = rel_pos / dist_to_target;

    double closing_speed = -rel_pos.dot(rel_vel) / dist_to_target;
    if (closing_speed < 1.0) {
        closing_speed = 3.0;
    }

    double tti = dist_to_target / closing_speed;

    // Terminal phase
    const double terminal_dist_threshold = ekf_->getEstimatedWidth() * 2.5;
    const double terminal_tti_threshold = 0.1;
    if ((dist_to_target < terminal_dist_threshold || tti < terminal_tti_threshold) && pn_active_ == 1) {
        pn_active_ = 2;
        pn_last_u_pn_.head<3>() *= 0.5;
        RCLCPP_INFO(this->get_logger(),
            "PN: Entering terminal phase (dist=%.2fm, TTI=%.2fs), coasting with zero angular rates",
            dist_to_target, tti);
        return pn_last_u_pn_;
    }

    // N adaptation
    double base_N = pn_nav_constant_;
    double adaptive_N = base_N;
    if (closing_speed > 15.0) {
        adaptive_N = base_N * (15.0 / closing_speed);
    }
    adaptive_N = std::max(2.0, adaptive_N);

    // 3D omega_los from EKF 3D state
    double dist_sq = dist_to_target * dist_to_target;
    Eigen::Vector3d omega_los = rel_pos.cross(rel_vel) / dist_sq;

    // EMA filter
    Eigen::VectorXd omega_raw(3);
    omega_raw << omega_los.x(), omega_los.y(), omega_los.z();
    Eigen::VectorXd omega_filtered = pn_omega_los_filter_->filter(omega_raw);
    omega_los = Eigen::Vector3d(omega_filtered(0), omega_filtered(1), omega_filtered(2));
    pn_last_omega_los_ = omega_los;

    // PN acceleration
    double capped_closing_speed = std::min(closing_speed, 15.0);
    Eigen::Vector3d a_pn = adaptive_N * capped_closing_speed * omega_los.cross(los_unit);

    // Augmented PN: target acceleration compensation
    if (pn_target_vel_valid_ && !vel_est.hasNaN()) {
        Eigen::Vector3d a_target_raw = (vel_est - pn_last_target_vel_) / dt_;
        const double max_target_accel = 15.0;
        if (a_target_raw.norm() > max_target_accel) {
            a_target_raw = a_target_raw.normalized() * max_target_accel;
        }
        const double tau_accel = 0.3;
        const double alpha_accel = dt_ / (tau_accel + dt_);
        pn_target_accel_filtered_ = (1.0 - alpha_accel) * pn_target_accel_filtered_ + alpha_accel * a_target_raw;
        pn_apn_warmup_counter_++;

        const int apn_warmup_frames = static_cast<int>(0.3 / dt_);
        if (pn_apn_warmup_counter_ >= apn_warmup_frames) {
            Eigen::Vector3d a_target_perp = pn_target_accel_filtered_ -
                pn_target_accel_filtered_.dot(los_unit) * los_unit;
            a_pn += (adaptive_N / 3.0) * a_target_perp;
        }
    }
    if (!vel_est.hasNaN()) {
        pn_last_target_vel_ = vel_est;
        pn_target_vel_valid_ = true;
    }

    // Forward acceleration with drag feedforward
    double desired_closing_speed = std::max(pn_vc_initial_, V_DESIRED_);
    double closing_speed_error = desired_closing_speed - closing_speed;
    double drag_ff = pn_drag_ff_coeff_ * closing_speed;
    double ax_forward = std::clamp(1.5 * closing_speed_error + drag_ff, -5.0, 20.0);

    Eigen::Vector3d a_forward = ax_forward * los_unit;
    Eigen::Vector3d a_world = a_pn + a_forward;

    // Gravity floor
    const double gravity_floor = -(G_ - ControlLimits::THRUST_MIN);
    a_world.z() = std::max(a_world.z(), gravity_floor);

    // Priority-based tilt budget allocation (binary search)
    double max_tilt_tan = std::tan(att_limit_ * M_PI / 180.0);

    double a_xy = std::sqrt(a_world.x() * a_world.x() + a_world.y() * a_world.y());
    double a_z_total = std::max(0.1, a_world.z() + G_);
    double max_a_xy_by_tilt = a_z_total * max_tilt_tan;
    double actual_max_xy = std::min(pn_max_accel_xy_, max_a_xy_by_tilt);
    bool needs_scaling = (a_xy > actual_max_xy) || (std::abs(a_world.z()) > pn_max_accel_z_);

    if (needs_scaling) {
        auto check_feasible = [&](double s) -> bool {
            Eigen::Vector3d a_test = a_forward + s * a_pn;
            a_test.z() = std::max(a_test.z(), gravity_floor);
            double txy = std::sqrt(a_test.x() * a_test.x() + a_test.y() * a_test.y());
            double tzt = std::max(0.1, a_test.z() + G_);
            double mxy = std::min(pn_max_accel_xy_, tzt * max_tilt_tan);
            return (txy <= mxy) && (std::abs(a_test.z()) <= pn_max_accel_z_);
        };

        double pn_scale = 0.0;
        if (check_feasible(0.0)) {
            double s_lo = 0.0, s_hi = 1.0;
            for (int iter = 0; iter < 12; ++iter) {
                double s_mid = (s_lo + s_hi) * 0.5;
                if (check_feasible(s_mid)) s_lo = s_mid; else s_hi = s_mid;
            }
            pn_scale = s_lo;
            a_world = a_forward + pn_scale * a_pn;
            a_world.z() = std::max(a_world.z(), gravity_floor);
        } else {
            double fwd_xy = std::sqrt(a_forward.x() * a_forward.x() + a_forward.y() * a_forward.y());
            double fwd_zt = std::max(0.1, a_forward.z() + G_);
            double fwd_max = std::min(pn_max_accel_xy_, fwd_zt * max_tilt_tan);
            double sf = (fwd_xy > fwd_max && fwd_xy > 0.1) ? (fwd_max / fwd_xy) : 1.0;
            a_world = Eigen::Vector3d(a_forward.x() * sf, a_forward.y() * sf, a_forward.z());
            a_world.z() = std::clamp(a_world.z(), -pn_max_accel_z_, pn_max_accel_z_);
            a_world.z() = std::max(a_world.z(), gravity_floor);
        }

        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "PN correction scaled to %.0f%% for tilt/accel limits (a_xy_orig=%.1f, max_xy=%.1f)",
            pn_scale * 100.0, a_xy, actual_max_xy);
    }

    // Accel filter
    a_xy = std::sqrt(a_world.x() * a_world.x() + a_world.y() * a_world.y());
    Eigen::VectorXd accel_raw(2);
    accel_raw << a_xy, a_world.z();
    Eigen::VectorXd accel_filtered = pn_accel_filter_->filter(accel_raw);

    if (a_xy > 0.1) {
        double ratio = accel_filtered(0) / a_xy;
        a_world.x() *= ratio;
        a_world.y() *= ratio;
    }
    a_world.z() = accel_filtered(1);

    // Thrust and attitude computation
    Eigen::Vector3d a_total = a_world + Eigen::Vector3d(0.0, 0.0, G_);

    double eff = std::clamp(thrust_efficiency_, 0.5, 1.0);
    double thrust = a_total.norm() / eff;
    thrust = std::clamp(thrust, ControlLimits::THRUST_MIN, ControlLimits::THRUST_MAX);

    Eigen::Vector3d z_body_desired = a_total.normalized();
    if (z_body_desired.hasNaN() || z_body_desired.norm() < 0.5) {
        z_body_desired = Eigen::Vector3d(0.0, 0.0, 1.0);
    }

    // Angular rate computation
    Eigen::Vector4d quat_curr = get_quaternion(drone_state);
    Eigen::Quaterniond q_curr(quat_curr(0), quat_curr(1), quat_curr(2), quat_curr(3));
    Eigen::Matrix3d R_B_W = q_curr.toRotationMatrix();
    Eigen::Vector3d z_body_current = R_B_W.col(2);

    Eigen::Vector3d rotation_axis = z_body_current.cross(z_body_desired);
    double sin_angle = rotation_axis.norm();
    double cos_angle = z_body_current.dot(z_body_desired);

    const double tau_yaw = 0.63;
    const double attitude_gain = 1.0 / (pn_tau_attitude_ * ctrl_coeff_);
    const double yaw_gain = 1.0 / (tau_yaw * ctrl_coeff_);

    Eigen::Vector3d omega_world = Eigen::Vector3d::Zero();
    if (sin_angle > 1e-6) {
        rotation_axis.normalize();
        double angle = std::atan2(sin_angle, cos_angle);
        omega_world = rotation_axis * angle * attitude_gain;
    }

    Eigen::Vector3d omega_body = R_B_W.transpose() * omega_world;

    // Yaw control
    Eigen::Vector3d los_horizontal = los_unit;
    los_horizontal.z() = 0.0;
    if (los_horizontal.norm() > 0.1) {
        los_horizontal.normalize();
        Eigen::Vector3d x_body_current = R_B_W.col(0);
        Eigen::Vector3d x_body_horizontal = x_body_current;
        x_body_horizontal.z() = 0.0;
        if (x_body_horizontal.norm() > 0.1) {
            x_body_horizontal.normalize();
            double yaw_error = x_body_horizontal.cross(los_horizontal).z();
            double yaw_rate_cmd = yaw_gain * yaw_error;
            yaw_rate_cmd = std::clamp(yaw_rate_cmd, -ControlLimits::YAW_RATE_MAX, ControlLimits::YAW_RATE_MAX);
            omega_body(2) = yaw_rate_cmd;
        }
    }

    omega_body(0) = std::clamp(omega_body(0), -ControlLimits::ROLL_PITCH_RATE_MAX, ControlLimits::ROLL_PITCH_RATE_MAX);
    omega_body(1) = std::clamp(omega_body(1), -ControlLimits::ROLL_PITCH_RATE_MAX, ControlLimits::ROLL_PITCH_RATE_MAX);

    u_pn << omega_body(0), omega_body(1), omega_body(2), thrust;
    pn_last_u_pn_ = u_pn;

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 100,
        "PN: az=%.2f° el=%.2f° | az_rate=%.2f°/s el_rate=%.2f°/s | Vc=%.2fm/s dist=%.2fm TTI=%.2fs",
        azimuth * 180.0 / M_PI, elevation * 180.0 / M_PI,
        azimuth_rate * 180.0 / M_PI, elevation_rate * 180.0 / M_PI,
        closing_speed, dist_to_target, tti);

    double omega_los_deg = omega_los.norm() * 180.0 / M_PI;
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 100,
        "PN: omega_los=%.2f°/s | a_pn=[%.1f,%.1f,%.1f] a_fwd=%.1f | omega=[%.2f,%.2f,%.2f] | T=%.1f",
        omega_los_deg, a_pn.x(), a_pn.y(), a_pn.z(), ax_forward,
        omega_body(0), omega_body(1), omega_body(2), thrust);

    return u_pn;
}

// ============================================================================
// Delay Compensation
// ============================================================================

Eigen::Vector3d InterceptorNodeBase::get_history_body_rates(const Eigen::Vector3d& current_body_rates, double delay_time)
{
    std::shared_lock<std::shared_mutex> lock(mtx_control_history_);
    if (control_history_.empty()) {
        return current_body_rates;
    }
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

Eigen::VectorXd InterceptorNodeBase::predict_delayed_state(const Eigen::VectorXd& current_state, double delay_time)
{
    Eigen::VectorXd predicted_state = current_state;

    {
        std::shared_lock<std::shared_mutex> lock(mtx_control_history_);
        if (control_history_.empty() || delay_time < 1e-3) {
            return predicted_state;
        }
    }

    double current_time = this->get_clock()->now().seconds();
    double prediction_start_time = current_time - delay_time;

    Eigen::Vector3d average_body_rates = Eigen::Vector3d::Zero();
    int valid_samples = 0;

    {
        std::shared_lock<std::shared_mutex> lg(mtx_control_history_);
        auto it = std::lower_bound(
            control_history_.begin(), control_history_.end(),
            prediction_start_time,
            [](const auto& entry, double t) { return entry.first < t; }
        );
        for (; it != control_history_.end(); ++it) {
            average_body_rates += it->second;
            valid_samples++;
        }
    }

    if (valid_samples > 0) {
        average_body_rates /= valid_samples;
    } else {
        average_body_rates = get_omega_body(current_state);
    }

    return predict_state_forward(current_state, delay_time, average_body_rates);
}

void InterceptorNodeBase::update_control_history(double current_time, const Eigen::Vector3d& body_rates)
{
    std::unique_lock<std::shared_mutex> lock(mtx_control_history_);
    control_history_.emplace_back(current_time, body_rates);
    double cutoff_time = current_time - max_history_time_;
    while (!control_history_.empty() && control_history_.front().first < cutoff_time) {
        control_history_.pop_front();
    }
}

// ============================================================================
// State Validation and Safety
// ============================================================================

bool InterceptorNodeBase::validate_state(const Eigen::VectorXd& state, const char* context) const
{
    if (state.hasNaN()) {
        RCLCPP_ERROR(this->get_logger(), "NaN detected in state (%s)", context);
        return false;
    }
    if (!validate_orientation(state, context)) {
        return false;
    }
    Eigen::Vector3d pos = get_position(state);
    if (pos.norm() > 10000.0) {
        RCLCPP_ERROR(this->get_logger(), "Position out of range (%s)", context);
        return false;
    }
    Eigen::Vector3d vel = get_velocity_world(state);
    if (vel.norm() > 100.0) {
        RCLCPP_WARN(this->get_logger(), "Velocity unusually high: %.2f m/s (%s)", vel.norm(), context);
    }
    return true;
}

bool InterceptorNodeBase::validate_control(const VectorU& u) const
{
    if (u.hasNaN()) {
        RCLCPP_ERROR(this->get_logger(), "NaN detected in control output");
        return false;
    }
    if (u(3) < ControlLimits::THRUST_MIN * 0.5 || u(3) > ControlLimits::THRUST_MAX * 1.5) {
        RCLCPP_WARN(this->get_logger(), "Thrust out of expected range: %.2f m/s^2", u(3));
        return false;
    }
    if (std::abs(u(0)) > ControlLimits::ROLL_PITCH_RATE_MAX * 1.5 ||
        std::abs(u(1)) > ControlLimits::ROLL_PITCH_RATE_MAX * 1.5 ||
        std::abs(u(2)) > ControlLimits::YAW_RATE_MAX * 1.5) {
        RCLCPP_WARN(this->get_logger(), "Body rates out of expected range: [%.2f, %.2f, %.2f]",
            u(0), u(1), u(2));
        return false;
    }
    return true;
}

VectorU InterceptorNodeBase::apply_control_limits(const VectorU& u_raw) const
{
    VectorU u_limited = u_raw;
    u_limited(0) = std::clamp(u_limited(0), -ControlLimits::ROLL_PITCH_RATE_MAX, ControlLimits::ROLL_PITCH_RATE_MAX);
    u_limited(1) = std::clamp(u_limited(1), -ControlLimits::ROLL_PITCH_RATE_MAX, ControlLimits::ROLL_PITCH_RATE_MAX);
    u_limited(2) = std::clamp(u_limited(2), -ControlLimits::YAW_RATE_MAX, ControlLimits::YAW_RATE_MAX);
    u_limited(3) = std::clamp(u_limited(3), ControlLimits::THRUST_MIN, ControlLimits::THRUST_MAX);
    return u_limited;
}

bool InterceptorNodeBase::check_attitude_safety(const Eigen::VectorXd& state)
{
    Eigen::Vector4d quat = get_quaternion(state);
    double qx = quat(1), qy = quat(2);
    double cos_tilt = 1 - 2 * (qx * qx + qy * qy);
    double tilt_angle = std::acos(std::clamp(cos_tilt, -1.0, 1.0)) * 180.0 / M_PI;

    if (tilt_angle > att_limit_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "Tilt angle %.1f deg exceeds limit %.1f deg",
            tilt_angle, att_limit_);
        return false;
    }
    return true;
}

void InterceptorNodeBase::check_ekf_health()
{
    std::shared_lock<std::shared_mutex> lock(mtx_ekf_);
    if (!ekf_ || !ekf_->is_initialized()) return;
    if (++ekf_health_check_counter_ < 50) return;
    ekf_health_check_counter_ = 0;
    if (!ekf_->is_healthy()) {
        RCLCPP_WARN(this->get_logger(), "EKF state contains NaN, resetting...");
        lock.unlock();
        std::unique_lock<std::shared_mutex> write_lock(mtx_ekf_);
        ekf_->reset();
    }
}

// ============================================================================
// Solver Recovery and State Machine
// ============================================================================

SolverRecoveryAction InterceptorNodeBase::handle_solver_failure(int status)
{
    consecutive_solver_failures_++;
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
        "ACADOS solver failed with status %d (consecutive: %d)",
        status, consecutive_solver_failures_);

    if (consecutive_solver_failures_ <= MAX_SOLVER_FAILURES_WARN) {
        return SolverRecoveryAction::CONTINUE;
    } else if (consecutive_solver_failures_ <= MAX_SOLVER_FAILURES_FAILSAFE) {
        RCLCPP_WARN(this->get_logger(),
            "Solver failures reached %d, resetting warm start",
            consecutive_solver_failures_);
        return SolverRecoveryAction::RESET_WARM;
    } else if (consecutive_solver_failures_ <= MAX_SOLVER_FAILURES_ABORT) {
        RCLCPP_ERROR(this->get_logger(),
            "Solver failures reached %d, entering failsafe mode",
            consecutive_solver_failures_);
        status_publisher_->publish(
            mavros_msgs::msg::StatusText::WARNING,
            1, "Solver issues, failsafe mode", true);
        return SolverRecoveryAction::FAILSAFE;
    } else {
        return SolverRecoveryAction::ABORT;
    }
}

void InterceptorNodeBase::reset_solver_warm_start(const Eigen::VectorXd& current_state)
{
    RCLCPP_INFO(this->get_logger(), "Resetting solver warm start...");
    for (int i = 0; i <= N_HORIZON_; ++i) {
        x_warm_start_[i] = current_state;
        normalize_warm_start_orientation(x_warm_start_[i]);
    }
    for (int i = 0; i < N_HORIZON_; ++i) {
        u_warm_start_[i] << 0, 0, 0, G_;
    }
    consecutive_solver_failures_ = consecutive_solver_failures_ / 2;
}

void InterceptorNodeBase::transition_to_state(MissionState new_state, const char* reason)
{
    MissionState old_state = mission_state_.get();
    if (old_state == new_state) return;
    mission_state_.set(new_state);
    RCLCPP_INFO(this->get_logger(), "Mission state: %s -> %s (%s)",
        MissionStateMachine::state_name(old_state),
        MissionStateMachine::state_name(new_state),
        reason);
}
