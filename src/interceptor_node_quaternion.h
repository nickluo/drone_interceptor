#pragma once

#include <vector>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <queue>
#include <tuple>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/header.hpp>
#include <mavros_msgs/msg/status_text.hpp>

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

class StatusPublisher
{
public:
    StatusPublisher(rclcpp::Node &node, int hz=1);
    ~StatusPublisher();
    bool publish(uint8_t severity, int count,const std::string& text, bool force=false);
private:
    rclcpp::Publisher<mavros_msgs::msg::StatusText>::SharedPtr status_pub_;
    rclcpp::TimerBase::SharedPtr status_timer_;
    std::mutex status_mtx_;
    std::queue<std::tuple<uint8_t, int, std::string>> status_queue_;
    int status_counter_ = -1;
};


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
        this->declare_parameter<double>("limits.att_warning_deg", 45.0);
        this->declare_parameter<int>("mpc.ctrl_rate", 50);
        this->declare_parameter<double>("mpc.tf", 1.0);
        this->declare_parameter<double>("mpc.v_desired", 25.0);
        this->declare_parameter<double>("mpc.tracking_distance", 0.0);

        this->declare_parameter<bool>("pn.enabled", true);
        this->declare_parameter<double>("pn.nav_constant", 4.0);
        this->declare_parameter<double>("pn.switch_dist", 15.0);
        this->declare_parameter<double>("pn.fov_switch_deg", 15.0);
        
        // Simulation Only
        this->declare_parameter<double>("sim.interception_dist", 1.0);
        this->declare_parameter<std::vector<double>>("sim.initial_conditions.target_pos", {100.0, 20.0, 15.0});
        this->declare_parameter<std::vector<double>>("sim.initial_conditions.target_vel", {-10.0, 10.0, 2.5});

        this->declare_parameter<double>("perception.bbox_pos_std_dev", 2.0);
        this->declare_parameter<double>("perception.bbox_size_std_dev", 5.0);
        this->declare_parameter<double>("perception.target_physical_size_est", 1.0);
        this->declare_parameter<double>("perception.perspective_compensation", 1.0);  // 透视补偿因子
        this->declare_parameter<double>("perception.detection_latency", 0.08);  // 检测延迟估计

        this->declare_parameter<double>("control.execution_delay", 0.05);
        this->declare_parameter<double>("control.rate_tc", 0.05);
        // this->declare_parameter<double>("control.ctrl_scale", 0.8);  // Scale factor for auto-calculated ctrl_coeff
        this->declare_parameter<double>("control.max_history_time_factor", 2.0);
        this->declare_parameter<bool>("control.rate_control_enabled", false);

        this->declare_parameter<int>("camera.width_px", 640);
        this->declare_parameter<int>("camera.height_px", 640);  // Camera height in pixels

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

    void mission_launch_callback(const std_msgs::msg::Header::SharedPtr msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void bbox_callback(const vision_msgs::msg::Detection2D::SharedPtr msg);
    void gimbal_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

    void failsafe(bool hover);
    void real_loop();
    void simulation_loop();
    void run_control_cycle(const Eigen::Vector3d &pos_est, const Eigen::Vector3d &vel_est);
    void stop();

    void update_control_history(double current_time, const Eigen::Vector3d& body_rates);
    Eigen::Vector3d get_history_body_rates(const Eigen::Vector3d &current_body_rates, double delay_time);
    VectorX predict_delayed_state(const VectorX& current_state, double delay_time);
    
    // 状态验证和安全检查函数
    bool validate_state(const VectorX& state, const char* context) const;
    bool validate_control(const VectorU& u) const;
    VectorU apply_control_limits(const VectorU& u_raw) const;
    bool check_attitude_safety(const VectorX& state);
    void check_ekf_health();
    
    // 求解器故障恢复
    SolverRecoveryAction handle_solver_failure(int status);
    void reset_solver_warm_start(const VectorX& current_state);
    void transition_to_state(MissionState new_state, const char* reason);

    // Image-based PN Guidance (直接使用检测框计算LOS)
    VectorU calculate_pn_controls_image_based(const VectorX& drone_state);

    // Member Variables
    bool use_simulation_;
    double G_, DRAG_COEFF_, V_DESIRED_, RATE_TC_, TRACKING_DISTANCE_, TRACKING_GAIN_;
    
    // PN Parameters
    bool pn_enabled_;
    double pn_nav_constant_;
    double pn_switch_dist_;
    double pn_fov_switch_deg_;
    bool pn_active_ {false}; // Latched state for PN
    double pn_vc_initial_ {0.0}; // 切换到PN时body系x轴速度
    
    double bbox_pos_std_dev_, bbox_size_std_dev_;
    double target_physical_size_est_;
    double perspective_compensation_;  // 透视补偿因子
    double detection_latency_;  // 检测延迟估计
    double cam_focal_length_, max_angle_rad_, cam_width_px_, cam_height_px_;
    double control_execution_delay_;
    double ctrl_coeff_;
    double att_limit_;
    bool rate_control_enabled_;

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
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr mission_state_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr mission_launch_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr bbox_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr gimbal_sub_;
    
    VectorX drone_state_;
    Eigen::Vector3d target_pos_true_, target_vel_true_; // For sim
    double interception_dist_;
    int target_lost_counter_ = 0;
    int solver_failure_counter_ = 0;
    
    // 性能监控变量
    double avg_solve_time_us_ = 0.0;
    int solve_count_ = 0;
    int ekf_health_check_counter_ = 0;

    std::unique_ptr<DroneEKF> ekf_;
    std::unique_ptr<TargetEKF> ekf_legacy_;
    std::unique_ptr<EMAFilter> output_filter_;
    std::unique_ptr<StatusPublisher> status_publisher_;
    
    // 四元数版本的ACADOS接口
    drone_interceptor_mpc_quat_solver_capsule* acados_capsule_;
    ocp_nlp_config* nlp_config_;
    ocp_nlp_dims* nlp_dims_;
    ocp_nlp_in* nlp_in_;
    ocp_nlp_out* nlp_out_;
    
    std::vector<VectorX> x_warm_start_;  // 四元数状态热启动
    std::vector<VectorU> u_warm_start_;

    // 使用shared_mutex减少读写竞争
    mutable std::shared_mutex mtx_drone_state_;  // 无人机状态，频繁读取
    mutable std::shared_mutex mtx_ekf_;          // EKF状态，频繁读取
    mutable std::shared_mutex mtx_gimbal_;       // 云台状态，频繁读取
    std::mutex mtx_bbox_;                        // bbox更新不频繁，保持mutex

    Eigen::Quaterniond current_gimbal_orientation_;

    vision_msgs::msg::Detection2D current_bbox_;
    
    bool is_visible_ = false;
    
    // 图像直接测量的LOS角度和角速率 (相机系)
    mutable std::shared_mutex mtx_los_;
    Eigen::Vector2d los_angles_ {0.0, 0.0};       // [azimuth, elevation] in camera frame (rad)
    Eigen::Vector2d los_angles_prev_ {0.0, 0.0};  // 上一帧LOS角度
    Eigen::Vector2d los_rates_ {0.0, 0.0};        // LOS角速率 (rad/s)
    double los_measurement_time_ {0.0};           // 上次测量时间戳
    bool los_initialized_ {false};                // LOS是否已初始化
    
    // 线程安全状态机
    MissionStateMachine mission_state_;
    
    // 传感器接收状态
    std::atomic_bool odom_received_ {false};
    std::atomic_bool gimbal_received_ {false};
    
    // 求解器连续失败计数（用于故障恢复策略）
    int consecutive_solver_failures_ = 0;
    static constexpr int MAX_SOLVER_FAILURES_WARN = 5;
    static constexpr int MAX_SOLVER_FAILURES_FAILSAFE = 10;
    static constexpr int MAX_SOLVER_FAILURES_ABORT = 20;
};
