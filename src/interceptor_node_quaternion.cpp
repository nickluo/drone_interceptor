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

// ACADOS - 使用四元数版本的求解器
extern "C" {
#include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_drone_interceptor_mpc_quat.h"
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

// ============================================================================
// Interceptor ROS2 Node - 四元数版本
// ============================================================================
class InterceptorNodeQuaternion : public rclcpp::Node {
public:
    InterceptorNodeQuaternion() : Node("interceptor_node_quaternion") {
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
        // output_filter_ = std::make_unique<EMAFilter>(1, 4);
        
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
            // For testing without real drone and camera
            timer_ = this->create_wall_timer(std::chrono::duration<double>(dt_), 
                std::bind(&InterceptorNodeQuaternion::real_loop, this),
                timer_callback_group_);
        }
    }
    
    ~InterceptorNodeQuaternion() {
        drone_interceptor_mpc_quat_acados_free(acados_capsule_);
        drone_interceptor_mpc_quat_acados_free_capsule(acados_capsule_);
    }

private:
    void declare_parameters() {
        this->declare_parameter<bool>("use_simulation", true);

        this->declare_parameter<double>("physics.g", 9.81);
        this->declare_parameter<double>("physics.drag_coeff", 0.1);
        this->declare_parameter<double>("limits.att_limit", M_PI / 4.0);
        this->declare_parameter<int>("mpc.n_horizon", 50);
        this->declare_parameter<double>("mpc.tf", 1.0);
        this->declare_parameter<double>("mpc.v_desired", 25.0);
        
        // Simulation Only
        this->declare_parameter<double>("sim.interception_dist", 1.0);
        this->declare_parameter<std::vector<double>>("sim.initial_conditions.drone", {0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>("sim.initial_conditions.target_pos", {100.0, 20.0, 15.0});
        this->declare_parameter<std::vector<double>>("sim.initial_conditions.target_vel", {-10.0, 10.0, 2.5});

        this->declare_parameter<double>("perception.delay", 0.05);
        this->declare_parameter<double>("perception.bbox_pos_std_dev", 2.0);
        this->declare_parameter<double>("perception.bbox_size_std_dev", 5.0);
        this->declare_parameter<double>("control.execution_delay", 0.05);
        this->declare_parameter<double>("control.rate_tc", 0.05);
        // this->declare_parameter<double>("control.body_rate_scale", 0.5);
        this->declare_parameter<double>("control.max_history_time_factor", 2.0);

        this->declare_parameter<double>("target.physical_size_truth", 0.8);
        this->declare_parameter<double>("target.physical_size_est", 1.0);

        this->declare_parameter<double>("camera.fov_deg", 40.0);
        this->declare_parameter<int>("camera.width_px", 0);
        this->declare_parameter<double>("camera.focal_length_mm", 12.0);
        this->declare_parameter<double>("camera.pixel_size_um", 3.45);

        this->declare_parameter<double>("cost.w_los", 100.0);
        this->declare_parameter<double>("cost.w_align", 0.5);
        this->declare_parameter<double>("cost.w_closing_speed", 1.0);
        this->declare_parameter<double>("cost.w_rates", 1.0);
        this->declare_parameter<double>("cost.w_tilt", 1.0);
        this->declare_parameter<double>("cost.w_u_alpha", 50.0);
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
        // this->get_parameter("control.body_rate_scale", body_rate_scale_);
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
            cam_focal_length_ = cam_width_px * 0.5 / tan(fov_deg * M_PI / 180.0 / 2.0);
        } else {
            cam_focal_length_ = cam_focal_mm * 1000.0 / cam_pixel_um;
        }

        if (use_simulation_) {
            // 初始化四元数状态 [p(3), v(3), q(4)]
            std::vector<double> drone_initial_state_vec;
            this->get_parameter("sim.initial_conditions.drone", drone_initial_state_vec);
            if (drone_initial_state_vec.size() == 10) {
                // 已经是四元数格式
                drone_state_ = Eigen::Map<Vector13d>(drone_initial_state_vec.data());
            } else if (drone_initial_state_vec.size() == 9) {
                // 转换欧拉角到四元数
                Eigen::Vector3d pos = Eigen::Map<Eigen::Vector3d>(&drone_initial_state_vec[0]);
                Eigen::Vector3d vel = Eigen::Map<Eigen::Vector3d>(&drone_initial_state_vec[3]);
                Eigen::Vector3d euler = Eigen::Map<Eigen::Vector3d>(&drone_initial_state_vec[6]);
                Eigen::Vector4d quat = euler_to_quaternion(euler(0), euler(1), euler(2));
                Eigen::Vector3d omega = Eigen::Map<Eigen::Vector3d>(&drone_initial_state_vec[9]);
                drone_state_ << pos, vel, quat, omega;
            } else {
                RCLCPP_ERROR(this->get_logger(), 
                    "Invalid drone initial state size: %zu", drone_initial_state_vec.size());
                // 使用默认值
                drone_state_ << 0, 0, 10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
            }
            odom_received_ = true; // In simulation, we assume we know the state
            
            max_angle_rad_ = fov_deg * M_PI / 180.0 / 2.0;
            this->get_parameter("target.physical_size_truth", target_physical_size_truth_);
            rng_ = std::mt19937(rd_());
            pos_dist_ = std::normal_distribution<double>(0, bbox_pos_std_dev_);
            size_dist_ = std::normal_distribution<double>(0, bbox_size_std_dev_);

        }

        std::vector<double> target_pos_vec, target_vel_vec;
        this->get_parameter("sim.initial_conditions.target_pos", target_pos_vec);
        this->get_parameter("sim.initial_conditions.target_vel", target_vel_vec);
        target_pos_true_ = Eigen::Map<Eigen::Vector3d>(target_pos_vec.data());
        target_vel_true_ = Eigen::Map<Eigen::Vector3d>(target_vel_vec.data());
        this->get_parameter("sim.interception_dist", interception_dist_);

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

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) 
    {
        Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                             msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        Eigen::Vector4d quat_normalized(q.w(), q.x(), q.y(), q.z());
        // 转换机体系速度到世界系速度
        Eigen::Vector3d vel_body(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);
        Eigen::Vector3d vel_world = q.toRotationMatrix() * vel_body;
        {
            std::lock_guard<std::mutex> lock(mtx_drone_state_);
            // 提取位置
            drone_state_.head<3>() << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
            // 提取四元数
            drone_state_.segment<4>(6) = quat_normalized;
            // 提取速度
            drone_state_.segment<3>(3) = vel_world;
            // 提取角速度
            drone_state_.tail<3>() << msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z;
            update_control_history(rclcpp::Time(msg->header.stamp).seconds(), drone_state_.tail<3>());
        }
        
        // 检查极端姿态值
        double qx = quat_normalized(1), qy = quat_normalized(2);
        double cos_tilt_angle = 1 - 2*(qx*qx + qy*qy);
        double tilt_angle = std::acos(std::clamp(cos_tilt_angle, -1.0, 1.0));
        if (tilt_angle > att_limit_) { // 警告
            // RCLCPP_WARN(this->get_logger(), "Large tilt angle detected! Tilt: %.2f°", 
            //             tilt_angle * 180.0/M_PI);
            printf("Large tilt angle detected! Tilt: %.2f°\n", tilt_angle * 180.0/M_PI);
        }
        if (!odom_received_)
        {
            for (int i = 0; i <= N_HORIZON_; ++i)
                x_warm_start_[i] = drone_state_;
            for (int i = 0; i < N_HORIZON_; ++i) 
                u_warm_start_[i] << msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z, G_;
            odom_received_ = true;
        }
    }
    
    void bbox_callback(const vision_msgs::msg::Detection2D::SharedPtr msg) 
    { 
        current_bbox_ = *msg;
        is_visible_ = true;
        // printf("Received bbox: center=(%.2f, %.2f), size=(%.2f, %.2f)\n", 
        //       current_bbox_.bbox.center.position.x, current_bbox_.bbox.center.position.y,
        //       current_bbox_.bbox.size_x, current_bbox_.bbox.size_y);
        if (!timer_) {
            timer_ = this->create_wall_timer(std::chrono::duration<double>(dt_), 
                std::bind(&InterceptorNodeQuaternion::real_loop, this),
                timer_callback_group_);
        }
    }

    void failsafe(bool hover) 
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

    void real_loop() {
        if (!odom_received_) {
            // RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for odometry...");
            printf("Waiting for odometry...\n");
            return;
        }
        run_control_cycle_test();
        target_pos_true_ += target_vel_true_ * dt_; // 仅用于测试 
        Eigen::Vector3d pos_est = target_pos_true_;
        double distance = (pos_est - drone_state_.head<3>()).norm();
        printf("Distance to target: %.2f m\n", distance);
        if (distance < interception_dist_) {
            // RCLCPP_INFO(this->get_logger(), "[SUCCESS] Target intercepted at distance %.2f m. Stopping simulation.", distance);
            printf("[SUCCESS] Target intercepted at distance %.2f m. Stopping simulation. \n", distance);
            timer_->cancel();
            
            // 悬停控制
            failsafe(true);
        }
    }
    
    void simulation_loop() {
        // 从四元数状态获取偏航角进行投影
        is_visible_ = project_and_get_measurement(drone_state_, target_pos_true_, current_bbox_);
        run_control_cycle(current_bbox_, is_visible_);
        target_pos_true_ += target_vel_true_ * dt_;
        double distance = (target_pos_true_ - drone_state_.head<3>()).norm();
        if (distance < interception_dist_) {
            RCLCPP_INFO(this->get_logger(), "[SUCCESS] Target intercepted at distance %.2f m. Stopping simulation.", distance);
            timer_->cancel();
        }
    }

    bool project_and_get_measurement(const Vector13d& drone_state, const Eigen::Vector3d& target_pos, vision_msgs::msg::Detection2D& bbox_out) {
        Eigen::Vector3d drone_pos = drone_state.head<3>();
        double psi = get_yaw_from_quaternion(drone_state.segment<4>(6));
        
        Eigen::Vector3d los_W = target_pos - drone_pos;
        double cpsi = std::cos(psi), spsi = std::sin(psi);
        Eigen::Matrix3d R_W_C; 
        R_W_C << cpsi, spsi, 0, -spsi, cpsi, 0, 0, 0, 1;
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

    void run_control_cycle(const vision_msgs::msg::Detection2D& meas_bbox, bool meas_visible) 
    {    
        if (ekf_->is_initialized()) ekf_->predict(dt_);
        
        if (meas_visible) {
            target_lost_counter_ = 0;
            double psi = get_yaw_from_quaternion(drone_state_.segment<4>(6));
            Eigen::Vector3d raw_pos_est = ekf_->noisy_bbox_to_pos(meas_bbox, drone_state_.head<3>(), psi);
            if (!ekf_->is_initialized()) ekf_->initialize(raw_pos_est);
            else ekf_->update(meas_bbox, drone_state_.head<3>(), psi);
        } else {
            target_lost_counter_++;
            if (ekf_->is_initialized()) 
                RCLCPP_WARN(this->get_logger(), 
                    "Target lost! Blind flight count: %d/%d", target_lost_counter_, N_HORIZON_);
        }
        
        if (target_lost_counter_ > N_HORIZON_) {
            RCLCPP_ERROR(this->get_logger(), 
                "Target lost for too long. Aborting.");
            if (timer_) timer_->cancel();
            return;
        }

        if (!ekf_->is_initialized()) {
            // RCLCPP_WARN(this->get_logger(), "EKF not initialized. Skipping planning step.");
            return;
        }
        
        // 使用延迟补偿的预测状态设置初始条件约束
        Vector13d predicted_state = predict_delayed_state(drone_state_, control_execution_delay_);
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", predicted_state.data());
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", predicted_state.data());

        const auto& ekf_state = ekf_->get_state();
        Eigen::Vector3d pos_est = ekf_state.head<3>();
        Eigen::Vector3d vel_est = ekf_state.tail<3>();

        // 更新预测轨迹的参数
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

        // 求解MPC
        auto t_start = std::chrono::high_resolution_clock::now();
        int status = drone_interceptor_mpc_quat_acados_solve(acados_capsule_);
        auto t_end = std::chrono::high_resolution_clock::now();
        
        if (status != ACADOS_SUCCESS)
            RCLCPP_WARN(this->get_logger(), "ACADOS solver returned status %d.", status);
        // RCLCPP_INFO(this->get_logger(), "ACADOS solve time: %ld us", std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());
        printf("ACADOS solve time: %ld us\n", 
            std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());

        Eigen::Vector4d u_optimal_raw;
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", u_optimal_raw.data());
        
        // 在仿真模式下更新状态
        if (use_simulation_) ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", drone_state_.data());

        // 更新热启动
        for (int i = 0; i < N_HORIZON_ - 1; ++i) {
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i + 1, "x", x_warm_start_[i].data());
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i + 1, "u", u_warm_start_[i].data());
        }
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, N_HORIZON_, "x", x_warm_start_[N_HORIZON_ - 1].data());
        x_warm_start_[N_HORIZON_] = x_warm_start_[N_HORIZON_ - 1];
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, N_HORIZON_ - 1, "u", u_warm_start_[N_HORIZON_ - 1].data());

        // 输出滤波
        Eigen::VectorXd u_filtered_dynamic = output_filter_->filter(u_optimal_raw);
        
        // 提取控制量：机体角速度和推力加速度
        double thrust_accel = u_filtered_dynamic(3);
        
        Eigen::Vector3d body_rates(u_filtered_dynamic(0), u_filtered_dynamic(1), u_filtered_dynamic(2));
        
        
        // 更新控制历史
        // update_control_history(body_rates);
        
        // 创建控制消息
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
    }

    void run_control_cycle_test() 
    {
        // 使用延迟补偿的预测状态设置初始条件约束
        Vector13d curr_drone_state;
        {
            std::lock_guard<std::mutex> lock(mtx_drone_state_);
            curr_drone_state = drone_state_;
        }
        Vector13d predicted_state = predict_delayed_state(curr_drone_state, control_execution_delay_);
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", predicted_state.data());
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", predicted_state.data());

        // 测试目标
        Eigen::Vector3d pos_est = target_pos_true_;
        Eigen::Vector3d vel_est = target_vel_true_;

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
        // RCLCPP_INFO(this->get_logger(), "ACADOS solve time: %ld us", std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());
        printf("ACADOS solve time: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());
        
        if (status != ACADOS_SUCCESS)
        {
            // RCLCPP_WARN(this->get_logger(), "ACADOS solver returned status %d.", status);
            printf("ACADOS solver returned status %d.\n", status);
            // timer_->cancel();
            
            // 悬停控制
            failsafe(false);
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
        
        // 提取控制量
        double thrust_accel = u_filtered_dynamic(3);
        Eigen::Vector3d body_rates(u_filtered_dynamic(0), u_filtered_dynamic(1), u_filtered_dynamic(2));
        
        // 转换世界系速度到机体系速度 for debug
        // Eigen::Quaterniond q_inv(x_warm_start_[0].data()[6], -x_warm_start_[0].data()[7], -x_warm_start_[0].data()[8], -x_warm_start_[0].data()[9]);
        // Eigen::Vector3d vel_world(x_warm_start_[0].data()[3], x_warm_start_[0].data()[4], x_warm_start_[0].data()[5]);
        // Eigen::Vector3d vel_body = q_inv.toRotationMatrix() * vel_world;

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

    Eigen::Vector3d get_history_body_rates(const Eigen::Vector3d &current_body_rates, double delay_time) 
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
    Vector13d predict_delayed_state(const Vector13d& current_state, double delay_time) 
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
    
    // 四元数积分函数
    Eigen::Vector4d integrate_quaternion(const Eigen::Vector4d& quat, const Eigen::Vector3d& body_rates, double dt) {
        // 使用四元数微分方程进行积分
        double p = body_rates(0), q = body_rates(1), r = body_rates(2);
        // double qw = quat(0), qx = quat(1), qy = quat(2), qz = quat(3);
        
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
    
    // 更新控制历史
    void update_control_history(double current_time, const Eigen::Vector3d& body_rates) {
        // double current_time = this->get_clock()->now().seconds();
        control_history_.emplace_back(current_time, body_rates);
        
        // 清理过旧的历史数据
        double cutoff_time = current_time - max_history_time_;
        while (!control_history_.empty() && control_history_.front().first < cutoff_time) {
            control_history_.pop_front();
        }
    }

    // Member Variables - 四元数版本
    bool use_simulation_;
    double G_, DRAG_COEFF_, V_DESIRED_, RATE_TC_;
    double bbox_pos_std_dev_, bbox_size_std_dev_;
    double target_physical_size_est_, target_physical_size_truth_;
    double cam_focal_length_, max_angle_rad_;
    double control_execution_delay_;
    // double body_rate_scale_;
    double att_limit_;

    // 延迟补偿相关变量
    std::deque<std::pair<double, Eigen::Vector3d>> control_history_;  // 时间戳和角速度指令历史
    double max_history_time_;  // 最大历史保存时间
    
    int N_HORIZON_;
    double TF_, dt_;
    Vector7d w_cost_;

    rclcpp::CallbackGroup::SharedPtr sub_callback_group_;
    rclcpp::CallbackGroup::SharedPtr timer_callback_group_;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<quadrotor_msgs::msg::ControlCommand>::SharedPtr control_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr bbox_sub_;

    std::mutex mtx_drone_state_;
    Vector13d drone_state_;  // 四元数状态: [p(3), v(3), q(4), w(3)]
    Eigen::Vector3d target_pos_true_, target_vel_true_; // For sim
    double interception_dist_;
    int target_lost_counter_ = 0;

    std::unique_ptr<TargetEKF> ekf_;
    std::unique_ptr<EMAFilter> output_filter_;
    
    // 四元数版本的ACADOS接口
    drone_interceptor_mpc_quat_solver_capsule* acados_capsule_;
    ocp_nlp_config* nlp_config_;
    ocp_nlp_dims* nlp_dims_;
    ocp_nlp_in* nlp_in_;
    ocp_nlp_out* nlp_out_;
    
    std::vector<Vector13d> x_warm_start_;  // 四元数状态热启动
    std::vector<Eigen::Vector4d> u_warm_start_;
    
    std::random_device rd_;
    std::mt19937 rng_;
    std::normal_distribution<double> pos_dist_;
    std::normal_distribution<double> size_dist_;

    vision_msgs::msg::Detection2D current_bbox_;
    std::atomic_bool is_visible_  {false};
    std::atomic_bool odom_received_ {false};
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InterceptorNodeQuaternion>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
