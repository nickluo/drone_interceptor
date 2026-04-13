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

// ACADOS generic interface (solver-specific headers in subclasses)
extern "C" {
    #include "acados/utils/print.h"
    #include "acados_c/ocp_nlp_interface.h"
    #include "acados_c/external_function_interface.h"
}

// Shared control vector type (4D: roll_rate, pitch_rate, yaw_rate, thrust)
using VectorU = Eigen::Matrix<double, 4, 1>;

// ============================================================================
// Control limits (identical for both solver variants)
// ============================================================================
namespace ControlLimits {
    constexpr double ROLL_PITCH_RATE_MAX = 2.0 * M_PI;  // rad/s
    constexpr double YAW_RATE_MAX = 1.0 * M_PI;         // rad/s
    constexpr double THRUST_MIN = 1.0;                   // m/s^2
    constexpr double THRUST_MAX = 20.0;                  // m/s^2
}

// ============================================================================
// StatusPublisher (unified, replaces StatusPublisher + StatusPublisherDQ)
// ============================================================================
class StatusPublisher
{
public:
    StatusPublisher(rclcpp::Node &node, int hz = 1);
    ~StatusPublisher();
    bool publish(uint8_t severity, int count, const std::string& text, bool force = false);
private:
    rclcpp::Publisher<mavros_msgs::msg::StatusText>::SharedPtr status_pub_;
    rclcpp::TimerBase::SharedPtr status_timer_;
    std::mutex status_mtx_;
    std::queue<std::tuple<uint8_t, int, std::string>> status_queue_;
    int status_counter_ = -1;
};

// ============================================================================
// InterceptorNodeBase — abstract base for quaternion and DQ variants
//
// Design pattern: Template Method
// Base class owns the algorithm skeleton (control loop, PN guidance, callbacks).
// Subclasses provide state-representation and solver-specific hooks via
// pure virtual methods.
//
// State storage uses Eigen::VectorXd to avoid template complexity.
// At 50 Hz the dynamic allocation overhead is negligible.
// ============================================================================
class InterceptorNodeBase : public rclcpp::Node
{
public:
    virtual ~InterceptorNodeBase();

protected:
    // Protected constructor — only subclasses instantiate
    explicit InterceptorNodeBase(const std::string& node_name);

    // Two-phase init: subclass constructor calls this after its own vtable is set up
    void initialize();

    // ========================================================================
    // Pure virtual interface — subclasses must implement
    // ========================================================================

    // --- State accessors (representation-specific extraction) ---
    virtual Eigen::Vector3d get_position(const Eigen::VectorXd& state) const = 0;
    virtual Eigen::Vector3d get_velocity_world(const Eigen::VectorXd& state) const = 0;
    virtual Eigen::Vector4d get_quaternion(const Eigen::VectorXd& state) const = 0;
    virtual Eigen::Vector3d get_omega_body(const Eigen::VectorXd& state) const = 0;

    // --- State lifecycle ---
    virtual int state_dim() const = 0;                   // 16 for quat, 17 for DQ
    virtual Eigen::VectorXd make_identity_state() const = 0;
    virtual void normalize_warm_start_orientation(Eigen::VectorXd& state) = 0;
    virtual void encode_odom(Eigen::VectorXd& state,
                             const Eigen::Vector3d& pos,
                             const Eigen::Quaterniond& q,
                             const Eigen::Vector3d& vel_body,
                             const Eigen::Vector3d& omega) = 0;

    // --- Solver interface ---
    virtual bool init_solver() = 0;       // Creates capsule, sets nlp_* pointers
    virtual void destroy_solver() = 0;
    virtual int  solve_step() = 0;        // Calls solver, returns ACADOS status
    virtual void update_solver_params(int k, double* data, int size) = 0;

    // --- MPC parameter packing (layout differs between variants) ---
    virtual void pack_and_set_mpc_params(const Eigen::Vector3d& pos_est,
                                         const Eigen::Vector3d& vel_est,
                                         const Eigen::Vector3d& drone_pos) = 0;

    // --- Delay compensation (state-specific integration) ---
    virtual Eigen::VectorXd predict_state_forward(const Eigen::VectorXd& state,
                                                   double delay,
                                                   const Eigen::Vector3d& avg_rates) = 0;

    // --- Orientation validation (quat norm vs DQ norm) ---
    virtual bool validate_orientation(const Eigen::VectorXd& state, const char* ctx) const = 0;

    // --- Subclass parameter and EKF setup ---
    virtual void declare_specific_parameters() = 0;
    virtual void load_specific_parameters() = 0;
    virtual void setup_ekf() = 0;

    // ========================================================================
    // Shared implementation (called by base, available to subclasses)
    // ========================================================================

    void declare_common_parameters();
    void load_common_parameters();

    // ROS callbacks
    void mission_launch_callback(const std_msgs::msg::Header::SharedPtr msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void bbox_callback(const vision_msgs::msg::Detection2D::SharedPtr msg);
    void gimbal_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

    void failsafe(bool hover);
    void real_loop();
    void simulation_loop();
    void run_control_cycle(const Eigen::Vector3d& pos_est, const Eigen::Vector3d& vel_est);
    void stop();

    // Delay compensation
    void update_control_history(double current_time, const Eigen::Vector3d& body_rates);
    Eigen::Vector3d get_history_body_rates(const Eigen::Vector3d& current_body_rates, double delay_time);
    Eigen::VectorXd predict_delayed_state(const Eigen::VectorXd& current_state, double delay_time);

    // Validation and safety
    bool validate_state(const Eigen::VectorXd& state, const char* context) const;
    bool validate_control(const VectorU& u) const;
    VectorU apply_control_limits(const VectorU& u_raw) const;
    bool check_attitude_safety(const Eigen::VectorXd& state);
    void check_ekf_health();

    // Solver recovery
    SolverRecoveryAction handle_solver_failure(int status);
    void reset_solver_warm_start(const Eigen::VectorXd& current_state);
    void transition_to_state(MissionState new_state, const char* reason);

    // PN Guidance (full 3D TPN — shared implementation using virtual state accessors)
    VectorU calculate_pn_controls_image_based(const Eigen::VectorXd& drone_state,
                                               const Eigen::Vector3d& pos_est,
                                               const Eigen::Vector3d& vel_est);

    // ========================================================================
    // Shared member variables
    // ========================================================================

    // Physics and MPC
    bool use_simulation_;
    double G_, DRAG_COEFF_, V_DESIRED_, RATE_TC_, TRACKING_DISTANCE_, TRACKING_GAIN_;
    int N_HORIZON_;
    double dt_;

    // PN Parameters
    bool pn_enabled_;
    double pn_nav_constant_;
    double pn_switch_dist_;
    double pn_fov_switch_deg_;
    double pn_max_accel_xy_;
    double pn_max_accel_z_;
    double pn_drag_ff_coeff_;
    double pn_tau_attitude_;

    // PN runtime state
    uint8_t pn_active_ = 0;
    double pn_vc_initial_ {0.0};
    VectorU pn_last_u_pn_ {0.0, 0.0, 0.0, 9.81};
    Eigen::Vector3d pn_last_omega_los_ {0.0, 0.0, 0.0};

    // Augmented PN
    Eigen::Vector3d pn_last_target_vel_ {0.0, 0.0, 0.0};
    bool pn_target_vel_valid_ {false};
    Eigen::Vector3d pn_target_accel_filtered_ {0.0, 0.0, 0.0};
    int pn_apn_warmup_counter_ {0};

    // PN Filters
    std::unique_ptr<EMAFilter> pn_accel_filter_;
    std::unique_ptr<EMAFilter> pn_omega_los_filter_;

    // PN collision detection (gyroscope)
    Eigen::Vector3d pn_last_cmd_rates_ {0.0, 0.0, 0.0};
    bool pn_collision_detected_ {false};

    // Thrust efficiency (MPC-phase learning, currently disabled)
    double thrust_efficiency_ {1.0};
    Eigen::Vector3d last_accel_world_ {0.0, 0.0, 0.0};
    double last_thrust_cmd_ {0.0};

    // Perception
    double bbox_pos_std_dev_, bbox_size_std_dev_;
    double target_physical_size_est_;
    double detection_latency_;
    double control_execution_delay_;
    double ctrl_coeff_;
    double att_limit_;
    bool rate_control_enabled_;
    CameraIntrinsics cam_intrinsics_;
    double detection_ratio_ = 0.0;

    // EKF process noise for W (set by subclass, used in bbox_callback)
    double ekf_std_q_W_ = 0.002;

    // Delay compensation
    std::deque<std::pair<double, Eigen::Vector3d>> control_history_;
    double max_history_time_;
    mutable std::shared_mutex mtx_control_history_;

    // Gimbal
    double gimbal_rp_coeff_, gimbal_y_coeff_;

    // ROS interfaces
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

    // State
    Eigen::VectorXd drone_state_;
    Eigen::Vector3d target_pos_true_, target_vel_true_;
    double interception_dist_;
    int target_lost_counter_ = 0;
    int solver_failure_counter_ = 0;

    // Performance monitoring
    double avg_solve_time_us_ = 0.0;
    int solve_count_ = 0;
    int ekf_health_check_counter_ = 0;

    // EKF and filters
    std::unique_ptr<DroneEKF> ekf_;
    std::unique_ptr<EMAFilter> output_filter_;
    std::unique_ptr<StatusPublisher> status_publisher_;

    // ACADOS generic interface (populated by subclass init_solver)
    ocp_nlp_config* nlp_config_ = nullptr;
    ocp_nlp_dims*   nlp_dims_   = nullptr;
    ocp_nlp_in*     nlp_in_     = nullptr;
    ocp_nlp_out*    nlp_out_    = nullptr;

    // Warm start (VectorXd for x, VectorU for u)
    std::vector<Eigen::VectorXd> x_warm_start_;
    std::vector<VectorU> u_warm_start_;

    // Synchronization
    mutable std::shared_mutex mtx_drone_state_;
    mutable std::shared_mutex mtx_ekf_;
    mutable std::shared_mutex mtx_gimbal_;
    std::mutex mtx_bbox_;

    Eigen::Quaterniond current_gimbal_orientation_;
    vision_msgs::msg::Detection2D current_bbox_;
    bool is_visible_ = false;

    DroneEKF::Bbox last_bbox_;
    bool last_bbox_valid_ = false;
    int bbox_jump_counter_ = 0;

    MissionStateMachine mission_state_;

    std::atomic_bool odom_received_ {false};
    std::atomic_bool gimbal_received_ {false};

    int consecutive_solver_failures_ = 0;
    static constexpr int MAX_SOLVER_FAILURES_WARN = 5;
    static constexpr int MAX_SOLVER_FAILURES_FAILSAFE = 10;
    static constexpr int MAX_SOLVER_FAILURES_ABORT = 20;
};
