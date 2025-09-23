#include <iostream>
#include <vector>
#include <deque>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "vision_msgs/msg/detection2_d.hpp"

#include "quadrotor_msgs/msg/control_command.hpp"

// Eigen for linear algebra
#include <Eigen/Dense>

#include "utils.h"

// ACADOS
extern "C" {
#include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_drone_interceptor_mpc.h"
}

using namespace std::chrono_literals;

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

// Helper to convert global angle velocities to body rates
Eigen::Vector3d get_body_rates(double phi, double theta, const Eigen::Vector3d& w) {
    // phi, theta as Euler angle Roll and Pitch in ZYX order
    // w as [dphi, dtheta, dpsi] (global angle velocities)
    
    // Check for gimbal lock condition (theta near ±π/2)
    double abs_theta = std::abs(theta);
    if (abs_theta > M_PI/2 - 0.1) { // Within 0.1 rad (5.7°) of gimbal lock
        // Use simplified transformation to avoid singularity
        return w; // Fallback: assume global rates ≈ body rates near gimbal lock
    }
    
    // Correct transformation matrix for ZYX Euler angles
    double c_phi = std::cos(phi);
    double s_phi = std::sin(phi);
    double c_theta = std::cos(theta);
    double s_theta = std::sin(theta);
    
    // Additional safety check for numerical stability
    if (std::abs(c_theta) < 0.1) {
        return w; // Fallback to avoid division by near-zero
    }
    
    Eigen::Matrix3d T_matrix;
    T_matrix << 1, 0, -s_theta,
                0, c_phi, c_theta * s_phi,
                0, -s_phi, c_theta * c_phi;
    
    Eigen::Vector3d body_rates = T_matrix * w;
    
    // Additional safety check on output
    if (body_rates.cwiseAbs().maxCoeff() > 20.0) {
        // Scale down extreme values
        double scale = 20.0 / body_rates.cwiseAbs().maxCoeff();
        body_rates *= scale;
    }
    
    return body_rates;
}

// Helper to predict future attitude for control execution delay compensation
std::pair<double, double> predict_future_attitude(const Vector9d& drone_state, const Eigen::Vector3d& current_body_rates, double dt_predict, double att_limit) {
    // Current attitude
    double phi = drone_state(6);   // roll
    double theta = drone_state(7); // pitch
    double psi = drone_state(8);   // yaw
    
    // Get current body rates (actual rates from previous control cycle)
    double p = current_body_rates(0);  // roll rate
    double q = current_body_rates(1);  // pitch rate
    double r = current_body_rates(2);  // yaw rate
    
    // Convert current body rates to global angle velocities for attitude integration
    double c_phi = std::cos(phi);
    double s_phi = std::sin(phi);
    double c_theta = std::cos(theta);
    double s_theta = std::sin(theta);
    double t_theta = std::tan(theta);
    
    // Avoid singularity near gimbal lock
    if (std::abs(c_theta) < 0.1) {
        // Use simplified prediction near gimbal lock
        double phi_future = phi + p * dt_predict;
        double theta_future = theta + q * dt_predict;
        phi_future = std::clamp(phi_future, -att_limit, att_limit);
        theta_future = std::clamp(theta_future, -att_limit, att_limit);
        return std::make_pair(phi_future, theta_future);
    }
    
    // Transform body rates to global angle rates
    double dphi = p + s_phi * t_theta * q + c_phi * t_theta * r;
    double dtheta = c_phi * q - s_phi * r;
    
    // Predict future attitude using current body rates
    double phi_future = phi + dphi * dt_predict;
    double theta_future = theta + dtheta * dt_predict;
    
    // Limit predicted angles to prevent divergence
    phi_future = std::clamp(phi_future, -att_limit, att_limit);
    theta_future = std::clamp(theta_future, -att_limit, att_limit);
    
    return std::make_pair(phi_future, theta_future);
}

// Helper to check attitude safety and apply protective scaling if necessary
Eigen::Vector3d apply_attitude_safety_limit(const Vector9d& drone_state, const Eigen::Vector3d& body_rates, double att_limit, double safety_margin = 0.1) {
    double phi = drone_state(6);   // current roll
    double theta = drone_state(7); // current pitch
    
    // First apply hard limits on body rates to prevent extreme values
    Eigen::Vector3d safe_body_rates = body_rates;
    double max_body_rate = 8.0; // 8 rad/s maximum body rate (conservative)
    safe_body_rates(0) = std::clamp(safe_body_rates(0), -max_body_rate, max_body_rate);
    safe_body_rates(1) = std::clamp(safe_body_rates(1), -max_body_rate, max_body_rate);
    safe_body_rates(2) = std::clamp(safe_body_rates(2), -max_body_rate, max_body_rate);
    
    // Check if current attitude is near limits
    double phi_abs = std::abs(phi);
    double theta_abs = std::abs(theta);
    double limit_with_margin = att_limit - safety_margin;
    
    // Apply protective scaling for roll - but ensure scale_factor is always positive
    if (phi_abs > limit_with_margin && limit_with_margin > 0) {
        double remaining_margin = att_limit - phi_abs;
        double scale_factor = std::clamp(remaining_margin / safety_margin, 0.0, 1.0);
        if ((phi > 0 && safe_body_rates(0) > 0) || (phi < 0 && safe_body_rates(0) < 0)) {
            // Body rate would increase attitude angle further - scale it down
            safe_body_rates(0) *= scale_factor;
        }
    }
    
    // Apply protective scaling for pitch
    if (theta_abs > limit_with_margin && limit_with_margin > 0) {
        double remaining_margin = att_limit - theta_abs;
        double scale_factor = std::clamp(remaining_margin / safety_margin, 0.0, 1.0);
        if ((theta > 0 && safe_body_rates(1) > 0) || (theta < 0 && safe_body_rates(1) < 0)) {
            // Body rate would increase attitude angle further - scale it down
            safe_body_rates(1) *= scale_factor;
        }
    }
    
    return safe_body_rates;
}


// ============================================================================
// Interceptor ROS2 Node
// ============================================================================
class InterceptorNode : public rclcpp::Node {
public:
    InterceptorNode() : Node("interceptor_node") {
        declare_parameters();
        load_parameters();

        acados_capsule_ = drone_interceptor_mpc_acados_create_capsule();
        int status = drone_interceptor_mpc_acados_create(acados_capsule_);
        if (status) {
            RCLCPP_ERROR(this->get_logger(), "acados_create() returned status %d. Exiting.", status);
            throw std::runtime_error("Failed to create ACADOS solver");
        }
        nlp_config_ = drone_interceptor_mpc_acados_get_nlp_config(acados_capsule_);
        nlp_dims_ = drone_interceptor_mpc_acados_get_nlp_dims(acados_capsule_);
        nlp_in_ = drone_interceptor_mpc_acados_get_nlp_in(acados_capsule_);
        nlp_out_ = drone_interceptor_mpc_acados_get_nlp_out(acados_capsule_);
        
        ekf_ = std::make_unique<TargetEKF>(dt_, cam_focal_length_, target_physical_size_est_,
                                            bbox_pos_std_dev_, bbox_size_std_dev_);
        output_filter_ = std::make_unique<EMAFilter>(static_cast<int>(N_HORIZON_ / TF_ * 0.2), 4);
        
        x_warm_start_.resize(N_HORIZON_ + 1, Vector9d::Zero());
        u_warm_start_.resize(N_HORIZON_, Eigen::Vector4d::Zero());
        for (int i = 0; i <= N_HORIZON_; ++i) x_warm_start_[i] = drone_state_;
        for (int i = 0; i < N_HORIZON_; ++i) u_warm_start_[i] << 0, 0, 0, G_;

        // Initialize previous body rates to zero
        previous_body_rates_ = Eigen::Vector3d::Zero();

        control_pub_ = this->create_publisher<quadrotor_msgs::msg::ControlCommand>("/fpv/control_command", rclcpp::SensorDataQoS());
        if (use_simulation_) {
            RCLCPP_INFO(this->get_logger(), "Running in SIMULATION mode.");
            timer_ = this->create_wall_timer(std::chrono::duration<double>(dt_), std::bind(&InterceptorNode::simulation_loop, this));
        } else {
            RCLCPP_INFO(this->get_logger(), "Running in REAL mode.");
            odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                "/fpv/odom", rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(), std::bind(&InterceptorNode::odom_callback, this, std::placeholders::_1));
            bbox_sub_ = this->create_subscription<vision_msgs::msg::Detection2D>(
                "/fpv/target_bbox", rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(), 
                std::bind(&InterceptorNode::bbox_callback, this, std::placeholders::_1));
            // For testing without real drone and camera
            timer_ = this->create_wall_timer(std::chrono::duration<double>(dt_), std::bind(&InterceptorNode::real_loop, this));
        }
    }
    
    ~InterceptorNode() {
        drone_interceptor_mpc_acados_free(acados_capsule_);
        drone_interceptor_mpc_acados_free_capsule(acados_capsule_);
    }

private:
    void declare_parameters() {
        this->declare_parameter<bool>("use_simulation", true);
        // this->declare_parameter<double>("physics.mass", 1.5);
        this->declare_parameter<double>("physics.g", 9.81);
        // this->declare_parameter<double>("physics.t_max", 30.0);
        // this->declare_parameter<double>("physics.t_min", 1.0);
        this->declare_parameter<double>("physics.drag_coeff", 0.1);
        this->declare_parameter<double>("limits.roll_pitch_rate_max", 4.0 * M_PI);
        this->declare_parameter<double>("limits.yaw_rate_max", 2.0 * M_PI);
        this->declare_parameter<double>("limits.att_limit", M_PI / 4.0);
        this->declare_parameter<int>("mpc.n_horizon", 50);
        this->declare_parameter<double>("mpc.tf", 1.0);
        this->declare_parameter<double>("mpc.v_desired", 25.0);
        this->declare_parameter<double>("sim.time", 15.0);
        this->declare_parameter<double>("sim.interception_dist", 0.5);
        this->declare_parameter<std::vector<double>>("initial_conditions.drone", {0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>("initial_conditions.target_pos", {100.0, 20.0, 15.0});
        this->declare_parameter<std::vector<double>>("initial_conditions.target_vel", {-10.0, 10.0, 2.5});
        this->declare_parameter<double>("perception.delay", 0.05);
        this->declare_parameter<double>("perception.bbox_pos_std_dev", 2.0);
        this->declare_parameter<double>("perception.bbox_size_std_dev", 5.0);
        this->declare_parameter<double>("control.execution_delay", 0.1);
        this->declare_parameter<double>("target.physical_size_truth", 0.8);
        this->declare_parameter<double>("target.physical_size_est", 1.0);
        this->declare_parameter<double>("camera.fov_deg", 40.0);
        this->declare_parameter<int>("camera.width_px", 0);
        this->declare_parameter<double>("camera.focal_length_mm", 12.0);
        this->declare_parameter<double>("camera.pixel_size_um", 3.45);
        this->declare_parameter<double>("cost.w_los", 100.0);
        this->declare_parameter<double>("cost.w_align", 0.5);
        this->declare_parameter<double>("cost.w_closing_speed", 1.0);
        this->declare_parameter<double>("cost.w_u_rates", 1.0);
        this->declare_parameter<double>("cost.w_u_thrust", 0.1);
    }
    
    void load_parameters() {
        this->get_parameter("use_simulation", use_simulation_);
        this->get_parameter("physics.g", G_);
        this->get_parameter("mpc.n_horizon", N_HORIZON_);
        this->get_parameter("mpc.tf", TF_);
        this->get_parameter("mpc.v_desired", V_DESIRED_);
        this->get_parameter("physics.drag_coeff", DRAG_COEFF_);
        this->get_parameter("perception.bbox_pos_std_dev", bbox_pos_std_dev_);
        this->get_parameter("perception.bbox_size_std_dev", bbox_size_std_dev_);
        this->get_parameter("target.physical_size_est", target_physical_size_est_);
        this->get_parameter("control.execution_delay", control_execution_delay_);
        this->get_parameter("limits.att_limit", att_limit_);
        dt_ = TF_ / N_HORIZON_;
        double cam_focal_mm, cam_pixel_um;
        int cam_width_px;
        double fov_deg; 
        this->get_parameter("camera.fov_deg", fov_deg);
        this->get_parameter("camera.width_px", cam_width_px);
        this->get_parameter("camera.focal_length_mm", cam_focal_mm);
        this->get_parameter("camera.pixel_size_um", cam_pixel_um);
        if (cam_width_px > 0) {
            cam_focal_length_ = cam_width_px * 0.5 / tan(fov_deg * M_PI / 180.0 / 2.0);
        }
        else
        {
            cam_focal_length_ = cam_focal_mm * 1000.0 / cam_pixel_um;
        }
        std::vector<double> drone_initial_state_vec;
        this->get_parameter("initial_conditions.drone", drone_initial_state_vec);
        drone_state_ = Eigen::Map<Vector9d>(drone_initial_state_vec.data());
        if (use_simulation_) {
            std::vector<double> target_pos_vec, target_vel_vec;
            this->get_parameter("initial_conditions.target_pos", target_pos_vec);
            this->get_parameter("initial_conditions.target_vel", target_vel_vec);
            target_pos_true_ = Eigen::Map<Eigen::Vector3d>(target_pos_vec.data());
            target_vel_true_ = Eigen::Map<Eigen::Vector3d>(target_vel_vec.data());
            this->get_parameter("sim.interception_dist", interception_dist_);
            double fov_deg; this->get_parameter("camera.fov_deg", fov_deg);
            max_angle_rad_ = fov_deg * M_PI / 180.0 / 2.0;
            this->get_parameter("target.physical_size_truth", target_physical_size_truth_);
            rng_ = std::mt19937(rd_());
            pos_dist_ = std::normal_distribution<double>(0, bbox_pos_std_dev_);
            size_dist_ = std::normal_distribution<double>(0, bbox_size_std_dev_);
        }
        std::vector<double> cost_weights_vec(5);
        this->get_parameter("cost.w_los", cost_weights_vec[0]);
        this->get_parameter("cost.w_align", cost_weights_vec[1]);
        this->get_parameter("cost.w_closing_speed", cost_weights_vec[2]);
        this->get_parameter("cost.w_u_rates", cost_weights_vec[3]);
        this->get_parameter("cost.w_u_thrust", cost_weights_vec[4]);
        for (int i=0; i < 5; ++i) w_cost_(i) = std::sqrt(cost_weights_vec[i]);
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        drone_state_.head<3>() << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
        
        // Get quaternion and convert to rotation matrix
        Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                             msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        
        // Check quaternion validity
        if (std::abs(q.norm() - 1.0) > 0.1) {
            RCLCPP_WARN(this->get_logger(), "Invalid quaternion received! norm = %.3f", q.norm());
            return; // Skip this update if quaternion is invalid
        }
        
        // Transform body frame velocities to world frame velocities
        Eigen::Vector3d vel_body(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);
        Eigen::Vector3d vel_world = q.toRotationMatrix() * vel_body;
        drone_state_.segment<3>(3) = vel_world;
        
        Eigen::Vector3d euler_angles = to_euler_angles(q);
        drone_state_.tail<3>() = euler_angles;
        
        // Check for extreme attitude values
        if (std::abs(euler_angles.x()) > M_PI/2 || std::abs(euler_angles.y()) > M_PI/2) {
            RCLCPP_WARN(this->get_logger(), "Extreme attitude detected! Roll: %.2f°, Pitch: %.2f°, Yaw: %.2f°", 
                        euler_angles.x() * 180.0/M_PI, euler_angles.y() * 180.0/M_PI, euler_angles.z() * 180.0/M_PI);
        }
    }
    
    void bbox_callback(const vision_msgs::msg::Detection2D::SharedPtr msg) 
    { 
        // run_control_cycle(*msg, true);
        current_bbox_ = *msg;
        is_visible_ = true;
        printf("Received bbox: center=(%.2f, %.2f), size=(%.2f, %.2f)\n", 
              current_bbox_.bbox.center.position.x, current_bbox_.bbox.center.position.y,
              current_bbox_.bbox.size_x, current_bbox_.bbox.size_y);
        if (!timer_) {
            timer_ = this->create_wall_timer(std::chrono::duration<double>(dt_), std::bind(&InterceptorNode::real_loop, this));
        }
    }

    void real_loop() {
        // run_control_cycle(current_bbox_, is_visible_);
        // is_visible_ = false; // reset visibility for next cycle
        run_control_cycle_test();
        Eigen::Vector3d pos_est {0.0, 0.0, 20.0};
        double distance = (pos_est - drone_state_.head<3>()).norm();
        if (distance < 0.5) {
            RCLCPP_INFO(this->get_logger(), "[SUCCESS] Target intercepted at distance %.2f m. Stopping simulation.", distance);
            timer_->cancel();
            quadrotor_msgs::msg::ControlCommand::UniquePtr control_msg = 
            std::make_unique<quadrotor_msgs::msg::ControlCommand>();
            control_msg->header.stamp = this->get_clock()->now();
            control_msg->control_mode = quadrotor_msgs::msg::ControlCommand::ATTITUDE;
            control_msg->armed = true;
            control_msg->orientation.w = 1.0;
            control_msg->orientation.x = 0.0;
            control_msg->orientation.y = 0.0;
            control_msg->orientation.z = 0.0;
            control_msg->collective_thrust = G_;
            control_pub_->publish(std::move(control_msg));
        }
    }
    
    void simulation_loop() {
        // auto t_start = std::chrono::high_resolution_clock::now();
        is_visible_ = project_and_get_measurement(drone_state_, target_pos_true_, current_bbox_);
        run_control_cycle(current_bbox_, is_visible_);
        target_pos_true_ += target_vel_true_ * dt_;
        double distance = (target_pos_true_ - drone_state_.head<3>()).norm();
        if (distance < interception_dist_) {
            RCLCPP_INFO(this->get_logger(), "[SUCCESS] Target intercepted at distance %.2f m. Stopping simulation.", distance);
            timer_->cancel();
        }
        // auto t_end = std::chrono::high_resolution_clock::now();
        // RCLCPP_INFO(this->get_logger(), "simulation_loop time: %ld us", std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());

    }

    bool project_and_get_measurement(const Vector9d& drone_state, const Eigen::Vector3d& target_pos, vision_msgs::msg::Detection2D& bbox_out) {
        Eigen::Vector3d drone_pos = drone_state.head<3>();
        double psi = drone_state(8);
        Eigen::Vector3d los_W = target_pos - drone_pos;
        double cpsi = std::cos(psi), spsi = std::sin(psi);
        Eigen::Matrix3d R_W_C; R_W_C << cpsi, spsi, 0, -spsi, cpsi, 0, 0, 0, 1;
        Eigen::Vector3d los_C = R_W_C * los_W;
        double dist_cam_x = los_C.x();
        if (dist_cam_x <= 0.1) return false;
        double angle_to_center = std::atan2(std::sqrt(los_C.y()*los_C.y() + los_C.z()*los_C.z()), dist_cam_x);
        if (angle_to_center > max_angle_rad_) return false;
        double pixel_size = cam_focal_length_ * target_physical_size_truth_ / dist_cam_x;
        bbox_out.bbox.center.position.x = cam_focal_length_ * los_C.y() / dist_cam_x + pos_dist_(rng_);
        bbox_out.bbox.center.position.y = cam_focal_length_ * los_C.z() / dist_cam_x + pos_dist_(rng_);
        bbox_out.bbox.size_x = std::max(1.0, pixel_size + size_dist_(rng_));
        bbox_out.bbox.size_y = bbox_out.bbox.size_x;
        return true;
    }

    void run_control_cycle(const vision_msgs::msg::Detection2D& meas_bbox, bool meas_visible) {
        auto t_start = std::chrono::high_resolution_clock::now();
        if (ekf_->is_initialized()) ekf_->predict(dt_);
        
        if (meas_visible) {
            target_lost_counter_ = 0;
            Eigen::Vector3d raw_pos_est = ekf_->noisy_bbox_to_pos(meas_bbox, drone_state_.head<3>(), drone_state_(8));
            if (!ekf_->is_initialized()) ekf_->initialize(raw_pos_est);
            else ekf_->update(meas_bbox, drone_state_.head<3>(), drone_state_(8));
        } else {
            target_lost_counter_++;
            if (ekf_->is_initialized()) RCLCPP_WARN(this->get_logger(), "Target lost! Blind flight count: %d/%d", target_lost_counter_, N_HORIZON_);
        }
        
        if (target_lost_counter_ > N_HORIZON_) {
            RCLCPP_ERROR(this->get_logger(), "Target lost for too long. Aborting.");
            if (timer_) timer_->cancel();
            return;
        }

        if (!ekf_->is_initialized()) {
            RCLCPP_WARN(this->get_logger(), "EKF not initialized. Skipping planning step.");
            return;
        }
        
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", drone_state_.data());
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", drone_state_.data());

        const auto& ekf_state = ekf_->get_state();
        Eigen::Vector3d pos_est = ekf_state.head<3>();
        Eigen::Vector3d vel_est = ekf_state.tail<3>();

        Vector13d p_vec;
        for (int k = 0; k <= N_HORIZON_; ++k) {
            Eigen::Vector3d predicted_target_pos = pos_est + (k * dt_) * vel_est;
            p_vec << predicted_target_pos, vel_est, DRAG_COEFF_, V_DESIRED_, w_cost_;
            drone_interceptor_mpc_acados_update_params(acados_capsule_, k, p_vec.data(), p_vec.size());
        }

        for (int i = 0; i < N_HORIZON_; ++i) {
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "x", x_warm_start_[i].data());
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "u", u_warm_start_[i].data());
        }
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, N_HORIZON_, "x", x_warm_start_[N_HORIZON_].data());

        // auto t_start = std::chrono::high_resolution_clock::now();
        int status = drone_interceptor_mpc_acados_solve(acados_capsule_);
        auto t_end = std::chrono::high_resolution_clock::now();
        
        if (status != ACADOS_SUCCESS) RCLCPP_WARN(this->get_logger(), "ACADOS solver returned status %d.", status);
        RCLCPP_INFO(this->get_logger(), "ACADOS solve time: %ld us", 
            std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());

        Eigen::Vector4d u_optimal_raw;
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", u_optimal_raw.data());
        
        if (use_simulation_) ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", drone_state_.data());

        for (int i = 0; i < N_HORIZON_ - 1; ++i) {
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i + 1, "x", x_warm_start_[i].data());
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i + 1, "u", u_warm_start_[i].data());
        }
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, N_HORIZON_, "x", x_warm_start_[N_HORIZON_ - 1].data());
        x_warm_start_[N_HORIZON_] = x_warm_start_[N_HORIZON_ - 1];
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, N_HORIZON_ - 1, "u", u_warm_start_[N_HORIZON_ - 1].data());

        // Apply output filtering to raw optimal control
        Eigen::VectorXd u_filtered_dynamic = output_filter_->filter(u_optimal_raw);
        
        // Apply control execution delay compensation
        // Predict where the drone attitude will be when the new body rates take effect
        auto [phi_predicted, theta_predicted] = predict_future_attitude(drone_state_, previous_body_rates_, control_execution_delay_, att_limit_);
        double thrust_accel = u_filtered_dynamic(3);
        
        // Convert global angle velocities to body rates using predicted future attitude
        Eigen::Vector3d global_ang_vel(u_filtered_dynamic(0), u_filtered_dynamic(1), u_filtered_dynamic(2));
        Eigen::Vector3d body_rates_raw = get_body_rates(phi_predicted, theta_predicted, global_ang_vel);
        
        // Check for extreme values before safety limiting
        if (body_rates_raw.cwiseAbs().maxCoeff() > 10.0) {
            RCLCPP_WARN(this->get_logger(), "Extreme body rates detected before safety limit! [%.2f, %.2f, %.2f]", 
                        body_rates_raw(0), body_rates_raw(1), body_rates_raw(2));
        }
        
        // Apply attitude safety limit to prevent dangerous maneuvers
        Eigen::Vector3d body_rates = apply_attitude_safety_limit(drone_state_, body_rates_raw, att_limit_);
        
        // Final safety check - hard limit on all body rates
        double max_safe_rate = 6.0; // 6 rad/s absolute maximum
        body_rates(0) = std::clamp(body_rates(0), -max_safe_rate, max_safe_rate);
        body_rates(1) = std::clamp(body_rates(1), -max_safe_rate, max_safe_rate);
        body_rates(2) = std::clamp(body_rates(2), -max_safe_rate, max_safe_rate);
        
        // Log if safety limiting was applied
        if ((body_rates - body_rates_raw).cwiseAbs().maxCoeff() > 0.1) {
            RCLCPP_INFO(this->get_logger(), "Safety limiting applied: raw=[%.2f,%.2f,%.2f] -> safe=[%.2f,%.2f,%.2f]", 
                        body_rates_raw(0), body_rates_raw(1), body_rates_raw(2),
                        body_rates(0), body_rates(1), body_rates(2));
        }
        
        // Create control message with body rates
        quadrotor_msgs::msg::ControlCommand::UniquePtr control_msg = 
            std::make_unique<quadrotor_msgs::msg::ControlCommand>();
        control_msg->header.stamp = this->get_clock()->now();
        control_msg->control_mode = quadrotor_msgs::msg::ControlCommand::BODY_RATES;
        control_msg->armed = true;
        control_msg->bodyrates.x = body_rates(0);  // body p rate
        control_msg->bodyrates.y = body_rates(1);  // body q rate
        control_msg->bodyrates.z = body_rates(2);  // body r rate
        control_msg->collective_thrust = thrust_accel;
        control_pub_->publish(std::move(control_msg));
        
        // Store current body rates for next cycle's delay compensation
        previous_body_rates_ = body_rates;
    }

    void run_control_cycle_test() 
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", drone_state_.data());
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", drone_state_.data());

        Eigen::Vector3d pos_est {0.0, 0.0, 20.0};
        Eigen::Vector3d vel_est {0.0, 0.0, 0.0};

        Vector13d p_vec;
        for (int k = 0; k <= N_HORIZON_; ++k) {
            Eigen::Vector3d predicted_target_pos = pos_est + (k * dt_) * vel_est;
            p_vec << predicted_target_pos, vel_est, DRAG_COEFF_, V_DESIRED_, w_cost_;
            drone_interceptor_mpc_acados_update_params(acados_capsule_, k, p_vec.data(), p_vec.size());
        }

        for (int i = 0; i < N_HORIZON_; ++i) {
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "x", x_warm_start_[i].data());
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "u", u_warm_start_[i].data());
        }
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, N_HORIZON_, "x", x_warm_start_[N_HORIZON_].data());

        // auto t_start = std::chrono::high_resolution_clock::now();
        int status = drone_interceptor_mpc_acados_solve(acados_capsule_);
        auto t_end = std::chrono::high_resolution_clock::now();
        
        if (status != ACADOS_SUCCESS) RCLCPP_WARN(this->get_logger(), "ACADOS solver returned status %d.", status);
        RCLCPP_INFO(this->get_logger(), "ACADOS solve time: %ld us", std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());

        Eigen::Vector4d u_optimal_raw;
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", u_optimal_raw.data());
        
        if (use_simulation_) ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", drone_state_.data());

        for (int i = 0; i < N_HORIZON_ - 1; ++i) {
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i + 1, "x", x_warm_start_[i].data());
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i + 1, "u", u_warm_start_[i].data());
        }
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, N_HORIZON_, "x", x_warm_start_[N_HORIZON_ - 1].data());
        x_warm_start_[N_HORIZON_] = x_warm_start_[N_HORIZON_ - 1];
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, N_HORIZON_ - 1, "u", u_warm_start_[N_HORIZON_ - 1].data());

        // Apply output filtering to raw optimal control
        Eigen::VectorXd u_filtered_dynamic = output_filter_->filter(u_optimal_raw);
        
        // Apply control execution delay compensation  
        auto [phi_predicted, theta_predicted] = predict_future_attitude(drone_state_, previous_body_rates_, control_execution_delay_, att_limit_);
        double thrust_accel = u_filtered_dynamic(3);
        
        // Convert global angle velocities to body rates using predicted future attitude
        Eigen::Vector3d global_ang_vel(u_filtered_dynamic(0), u_filtered_dynamic(1), u_filtered_dynamic(2));
        Eigen::Vector3d body_rates = get_body_rates(phi_predicted, theta_predicted, global_ang_vel);
        
        // Apply attitude safety limit to prevent dangerous maneuvers
        Eigen::Vector3d safe_body_rates = apply_attitude_safety_limit(drone_state_, body_rates, att_limit_);
        
        printf("Control outputs - global_ang_vel: [%.2f, %.2f, %.2f], body_rates: [%.2f, %.2f, %.2f], safe_body_rates: [%.2f, %.2f, %.2f]\n",
               global_ang_vel(0), global_ang_vel(1), global_ang_vel(2),
               body_rates(0), body_rates(1), body_rates(2),
               safe_body_rates(0), safe_body_rates(1), safe_body_rates(2));
        
        // Create control message with body rates
        quadrotor_msgs::msg::ControlCommand::UniquePtr control_msg = 
            std::make_unique<quadrotor_msgs::msg::ControlCommand>();
        control_msg->header.stamp = this->get_clock()->now();
        control_msg->control_mode = quadrotor_msgs::msg::ControlCommand::BODY_RATES;
        control_msg->armed = true;
        control_msg->bodyrates.x = safe_body_rates(0) * 0.5;  // body p rate
        control_msg->bodyrates.y = safe_body_rates(1) * 0.5;  // body q rate
        control_msg->bodyrates.z = safe_body_rates(2) * 0.5;  // body r rate
        // control_msg->bodyrates.x = global_ang_vel(0);  // body p rate
        // control_msg->bodyrates.y = global_ang_vel(1);  // body q rate
        // control_msg->bodyrates.z = global_ang_vel(2);  // body r rate
        control_msg->collective_thrust = thrust_accel;
        control_pub_->publish(std::move(control_msg));
        
        // Store current body rates for next cycle's delay compensation
        previous_body_rates_ = safe_body_rates;
    }

    // Member Variables
    bool use_simulation_;
    double G_, DRAG_COEFF_, V_DESIRED_;
    double bbox_pos_std_dev_, bbox_size_std_dev_;
    double target_physical_size_est_, target_physical_size_truth_;
    double cam_focal_length_, max_angle_rad_;
    double control_execution_delay_;  // Control execution delay compensation
    double att_limit_;  // Attitude limit for safety checking
    Eigen::Vector3d previous_body_rates_;  // Store previous body rates for delay compensation
    int N_HORIZON_;
    double TF_, dt_;
    Vector5d w_cost_;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<quadrotor_msgs::msg::ControlCommand>::SharedPtr control_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr bbox_sub_;

    Vector9d drone_state_;
    Eigen::Vector3d target_pos_true_, target_vel_true_; // For sim
    double interception_dist_;
    int target_lost_counter_ = 0;

    std::unique_ptr<TargetEKF> ekf_;
    std::unique_ptr<EMAFilter> output_filter_;
    
    drone_interceptor_mpc_solver_capsule* acados_capsule_;
    ocp_nlp_config* nlp_config_;
    ocp_nlp_dims* nlp_dims_;
    ocp_nlp_in* nlp_in_;
    ocp_nlp_out* nlp_out_;
    
    std::vector<Vector9d> x_warm_start_;
    std::vector<Eigen::Vector4d> u_warm_start_;
    
    std::random_device rd_;
    std::mt19937 rng_;
    std::normal_distribution<double> pos_dist_;
    std::normal_distribution<double> size_dist_;

    vision_msgs::msg::Detection2D current_bbox_;
    bool is_visible_ = false;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InterceptorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

