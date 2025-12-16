#pragma once

#include <vector>
#include <deque>
#include <memory>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <std_msgs/msg/bool.hpp>

#include "quadrotor_msgs/msg/control_command.hpp"
#include "utils.h"

using VectorX = Eigen::Matrix<double, 16, 1>;
using VectorU = Eigen::Matrix<double, 4, 1>;
using VectorP = Eigen::Matrix<double, 18, 1>;
using VectorWcost = Eigen::Matrix<double, 7, 1>;

// ACADOS - 使用四元数版本的求解器
extern "C" {
    #include "acados/utils/print.h"
    #include "acados_c/ocp_nlp_interface.h"
    #include "acados_c/external_function_interface.h"
    #include "acados_solver_drone_interceptor_mpc_quat.h"
}

class InterceptorNodeQuaternion : public rclcpp::Node 
{
public:
    InterceptorNodeQuaternion();
    ~InterceptorNodeQuaternion();

private:
    void declare_parameters() {
        this->declare_parameter<bool>("use_simulation", true);

        this->declare_parameter<double>("physics.g", 9.81);
        this->declare_parameter<double>("physics.drag_coeff", 0.1);
        this->declare_parameter<double>("limits.att_limit", M_PI / 4.0);
        this->declare_parameter<int>("mpc.ctrl_rate", 50);
        this->declare_parameter<double>("mpc.tf", 1.0);
        this->declare_parameter<double>("mpc.v_desired", 25.0);
        this->declare_parameter<double>("mpc.tracking_distance", 0.0);
        
        // Simulation Only
        this->declare_parameter<double>("sim.interception_dist", 1.0);
        this->declare_parameter<std::vector<double>>("sim.initial_conditions.target_pos", {100.0, 20.0, 15.0});
        this->declare_parameter<std::vector<double>>("sim.initial_conditions.target_vel", {-10.0, 10.0, 2.5});

        this->declare_parameter<double>("perception.bbox_pos_std_dev", 2.0);
        this->declare_parameter<double>("perception.bbox_size_std_dev", 5.0);
        this->declare_parameter<double>("perception.target_physical_size_est", 1.0);

        this->declare_parameter<double>("control.execution_delay", 0.05);
        this->declare_parameter<double>("control.rate_tc", 0.05);
        this->declare_parameter<double>("control.max_history_time_factor", 2.0);

        this->declare_parameter<int>("camera.width_px", 640);

        // For fov based calculation
        this->declare_parameter<double>("camera.fov_deg", -1.0);

        // For focal length based calculation
        this->declare_parameter<int>("camera.native_width_px", 640);
        this->declare_parameter<double>("camera.focal_length_mm", 12.0);
        this->declare_parameter<double>("camera.pixel_size_um", 3.45);

        this->declare_parameter<double>("gimbal.rp_coeff", 1.2);
        this->declare_parameter<double>("gimbal.y_coeff", 1.5);

        this->declare_parameter<double>("cost.w_los", 100.0);
        this->declare_parameter<double>("cost.w_align", 0.5);
        this->declare_parameter<double>("cost.w_closing_speed", 1.0);
        this->declare_parameter<double>("cost.w_rates", 1.0);
        this->declare_parameter<double>("cost.w_tilt", 1.0);
        this->declare_parameter<double>("cost.w_u_alpha", 50.0);
        this->declare_parameter<double>("cost.w_u_thrust", 0.1);
    }
    void load_parameters();

    void mission_complete_callback(const std_msgs::msg::Bool::SharedPtr msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void bbox_callback(const vision_msgs::msg::Detection2D::SharedPtr msg);
    void gimbal_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

    void failsafe(bool hover);
    void real_loop();
    void simulation_loop();
    void run_control_cycle(const Eigen::Vector3d &pos_est, const Eigen::Vector3d &vel_est);

    void update_control_history(double current_time, const Eigen::Vector3d& body_rates);
    Eigen::Vector3d get_history_body_rates(const Eigen::Vector3d &current_body_rates, double delay_time);
    VectorX predict_delayed_state(const VectorX& current_state, double delay_time);

    // Member Variables
    bool use_simulation_;
    double G_, DRAG_COEFF_, V_DESIRED_, RATE_TC_, TRACKING_DISTANCE_, TRACKING_GAIN_;
    double bbox_pos_std_dev_, bbox_size_std_dev_;
    double target_physical_size_est_;
    double cam_focal_length_, max_angle_rad_, cam_width_px_;
    double control_execution_delay_;
    double att_limit_;

    // 延迟补偿相关变量
    std::deque<std::pair<double, Eigen::Vector3d>> control_history_;  // 时间戳和角速度指令历史
    double max_history_time_;  // 最大历史保存时间
    
    int N_HORIZON_;
    double dt_;
    VectorWcost w_cost_;

    double gimbal_rp_coeff_, gimbal_y_coeff_;

    rclcpp::CallbackGroup::SharedPtr sub_callback_group_;
    rclcpp::CallbackGroup::SharedPtr timer_callback_group_;

    rclcpp::TimerBase::SharedPtr timer_;
    int timer_boot_count_ = 0;

    rclcpp::Publisher<quadrotor_msgs::msg::ControlCommand>::SharedPtr control_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr mission_complete_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr bbox_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr gimbal_sub_;

    VectorX drone_state_;
    Eigen::Vector3d target_pos_true_, target_vel_true_; // For sim
    double interception_dist_;
    int target_lost_counter_ = 0;
    int solver_failure_counter_ = 0;

    std::unique_ptr<TargetEKF> ekf_;
    std::unique_ptr<EMAFilter> output_filter_;
    
    // 四元数版本的ACADOS接口
    drone_interceptor_mpc_quat_solver_capsule* acados_capsule_;
    ocp_nlp_config* nlp_config_;
    ocp_nlp_dims* nlp_dims_;
    ocp_nlp_in* nlp_in_;
    ocp_nlp_out* nlp_out_;
    
    std::vector<VectorX> x_warm_start_;  // 四元数状态热启动
    std::vector<VectorU> u_warm_start_;

    std::mutex mtx_drone_state_;
    std::mutex mtx_ekf_;
    std::mutex mtx_bbox_;
    std::mutex mtx_gimbal_;

    Eigen::Quaterniond current_gimbal_orientation_;

    vision_msgs::msg::Detection2D current_bbox_;
    
    bool is_visible_ = false;
    bool is_completed_ = false;
    std::atomic_bool odom_received_ {false};
    std::atomic_bool gimbal_received_ {false};
};
