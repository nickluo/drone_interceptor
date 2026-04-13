#include <chrono>
#include <cmath>
#include <Eigen/Dense>

#include "interceptor_node_quaternion.h"

// ============================================================================
// Constructor
// ============================================================================

InterceptorNodeQuaternion::InterceptorNodeQuaternion()
    : InterceptorNodeBase("interceptor_node_quaternion")
{
    initialize();
}

InterceptorNodeQuaternion::~InterceptorNodeQuaternion()
{
    destroy_solver();
}

// ============================================================================
// State lifecycle
// ============================================================================

Eigen::VectorXd InterceptorNodeQuaternion::make_identity_state() const
{
    Eigen::VectorXd s = Eigen::VectorXd::Zero(16);
    s(6) = 1.0;  // quaternion w
    return s;
}

void InterceptorNodeQuaternion::normalize_warm_start_orientation(Eigen::VectorXd& s)
{
    s.segment<4>(6) = normalize_quaternion(Eigen::Vector4d(s.segment<4>(6)));
}

void InterceptorNodeQuaternion::encode_odom(Eigen::VectorXd& s,
                                             const Eigen::Vector3d& pos,
                                             const Eigen::Quaterniond& q,
                                             const Eigen::Vector3d& vel_body,
                                             const Eigen::Vector3d& omega)
{
    Eigen::Vector3d vel_world = q * vel_body;
    s.head<3>() = pos;
    s.segment<3>(3) = vel_world;
    s.segment<4>(6) << q.w(), q.x(), q.y(), q.z();
    s.segment<3>(10) = omega;
}

// ============================================================================
// Solver interface
// ============================================================================

bool InterceptorNodeQuaternion::init_solver()
{
    acados_capsule_ = drone_interceptor_mpc_quat_acados_create_capsule();
    int status = drone_interceptor_mpc_quat_acados_create(acados_capsule_);
    if (status) {
        RCLCPP_ERROR(this->get_logger(), "acados_create() returned status %d.", status);
        return false;
    }
    nlp_config_ = drone_interceptor_mpc_quat_acados_get_nlp_config(acados_capsule_);
    nlp_dims_   = drone_interceptor_mpc_quat_acados_get_nlp_dims(acados_capsule_);
    nlp_in_     = drone_interceptor_mpc_quat_acados_get_nlp_in(acados_capsule_);
    nlp_out_    = drone_interceptor_mpc_quat_acados_get_nlp_out(acados_capsule_);
    return true;
}

void InterceptorNodeQuaternion::destroy_solver()
{
    if (acados_capsule_) {
        drone_interceptor_mpc_quat_acados_free(acados_capsule_);
        drone_interceptor_mpc_quat_acados_free_capsule(acados_capsule_);
        acados_capsule_ = nullptr;
    }
}

int InterceptorNodeQuaternion::solve_step()
{
    return drone_interceptor_mpc_quat_acados_solve(acados_capsule_);
}

void InterceptorNodeQuaternion::update_solver_params(int k, double* data, int size)
{
    drone_interceptor_mpc_quat_acados_update_params(acados_capsule_, k, data, size);
}

void InterceptorNodeQuaternion::pack_and_set_mpc_params(
    const Eigen::Vector3d& pos_est, const Eigen::Vector3d& vel_est,
    const Eigen::Vector3d& /*drone_pos*/)
{
    VectorPQuat p_vec;
    for (int k = 0; k <= N_HORIZON_; ++k) {
        Eigen::Vector3d predicted_target_pos = pos_est + (k * dt_) * vel_est;
        double tracking_coeff = 1.0;
        if (TRACKING_DISTANCE_ > 0.001) {
            Eigen::VectorXd x_k = Eigen::VectorXd::Zero(state_dim());
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "x", x_k.data());
            auto x_k_pos = x_k.head<3>();
            double dist_to_target = (predicted_target_pos - x_k_pos).norm();
            double dist_error = (dist_to_target - TRACKING_DISTANCE_) / TRACKING_DISTANCE_;
            tracking_coeff = 2.0 / (1.0 + std::exp(-TRACKING_GAIN_ * dist_error)) - 1.0;
        }
        p_vec << predicted_target_pos, vel_est, DRAG_COEFF_,
                 tracking_coeff * V_DESIRED_, RATE_TC_, w_cost_,
                 gimbal_rp_coeff_, gimbal_y_coeff_;
        drone_interceptor_mpc_quat_acados_update_params(acados_capsule_, k, p_vec.data(), p_vec.size());
    }
}

// ============================================================================
// Delay compensation
// ============================================================================

Eigen::VectorXd InterceptorNodeQuaternion::predict_state_forward(
    const Eigen::VectorXd& state, double delay, const Eigen::Vector3d& avg_rates)
{
    Eigen::VectorXd predicted = state;

    Eigen::Vector4d current_quat = state.segment<4>(6);
    Eigen::Vector4d predicted_quat = integrate_quaternion(current_quat, avg_rates, delay);
    predicted.segment<4>(6) = normalize_quaternion(predicted_quat);

    Eigen::Vector3d current_vel = state.segment<3>(3);
    predicted.head<3>() += current_vel * delay;

    return predicted;
}

// ============================================================================
// Validation
// ============================================================================

bool InterceptorNodeQuaternion::validate_orientation(const Eigen::VectorXd& s, const char* ctx) const
{
    Eigen::Vector4d q = s.segment<4>(6);
    double quat_norm = q.norm();
    if (std::abs(quat_norm - 1.0) > 0.1) {
        RCLCPP_ERROR(this->get_logger(), "Quaternion norm: %.6f (expected 1.0) (%s)", quat_norm, ctx);
        return false;
    }
    return true;
}

// ============================================================================
// Parameters and EKF
// ============================================================================

void InterceptorNodeQuaternion::declare_specific_parameters()
{
    this->declare_parameter<double>("cost.w_los", 100.0);
    this->declare_parameter<double>("cost.w_align", 0.5);
    this->declare_parameter<double>("cost.w_closing_speed", 1.0);
    this->declare_parameter<double>("cost.w_rates", 1.0);
    this->declare_parameter<double>("cost.w_tilt", 1.0);
    this->declare_parameter<double>("cost.w_u_alpha", 50.0);
    this->declare_parameter<double>("cost.w_u_thrust", 0.1);
    this->declare_parameter<double>("cost.w_zem", 5.0);
}

void InterceptorNodeQuaternion::load_specific_parameters()
{
    std::vector<double> cw(8);
    this->get_parameter("cost.w_los", cw[0]);
    this->get_parameter("cost.w_align", cw[1]);
    this->get_parameter("cost.w_closing_speed", cw[2]);
    this->get_parameter("cost.w_rates", cw[3]);
    this->get_parameter("cost.w_tilt", cw[4]);
    this->get_parameter("cost.w_u_alpha", cw[5]);
    this->get_parameter("cost.w_u_thrust", cw[6]);
    this->get_parameter("cost.w_zem", cw[7]);

    for (int i = 0; i < 8; ++i) w_cost_(i) = std::sqrt(cw[i]);

    if (TRACKING_DISTANCE_ > 0.001) {
        auto coeff = std::sqrt(10.0);
        auto los_coeff = std::sqrt(TRACKING_DISTANCE_ * 0.1);
        if (los_coeff > coeff) los_coeff = coeff;
        w_cost_(0) /= los_coeff;
        w_cost_(2) *= coeff;
    }

    ekf_std_q_W_ = 0.002;
}

void InterceptorNodeQuaternion::setup_ekf()
{
    Eigen::Matrix<double, 9, 1> process_noise;
    process_noise << bbox_pos_std_dev_ * 0.1,
                     bbox_pos_std_dev_ * 0.5,
                     bbox_pos_std_dev_ * 0.25,
                     bbox_size_std_dev_ * 0.1,
                     bbox_size_std_dev_ * 0.5,
                     bbox_size_std_dev_ * 0.25,
                     0.002,
                     1.0,
                     0.0;

    Eigen::Vector3d measurement_noise;
    measurement_noise << bbox_pos_std_dev_, bbox_size_std_dev_, bbox_size_std_dev_ * 0.1;

    const int predicted_imu_freq = 100;
    ekf_ = std::make_unique<DroneEKF>(cam_intrinsics_, process_noise, measurement_noise,
                                       predicted_imu_freq, detection_latency_,
                                       target_physical_size_est_);
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InterceptorNodeQuaternion>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
