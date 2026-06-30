#include <chrono>
#include <cmath>
#include <Eigen/Dense>

#include "interceptor_node_dq.h"

// ============================================================================
// DQ math helpers (scalar-first [w,x,y,z] Hamilton product)
// ============================================================================

Eigen::Vector4d InterceptorNodeDQ::quat_product(const Eigen::Vector4d& p, const Eigen::Vector4d& q)
{
    return Eigen::Vector4d(
        p(0)*q(0) - p(1)*q(1) - p(2)*q(2) - p(3)*q(3),
        p(0)*q(1) + p(1)*q(0) + p(2)*q(3) - p(3)*q(2),
        p(0)*q(2) - p(1)*q(3) + p(2)*q(0) + p(3)*q(1),
        p(0)*q(3) + p(1)*q(2) - p(2)*q(1) + p(3)*q(0));
}

Eigen::Vector4d InterceptorNodeDQ::quat_conjugate(const Eigen::Vector4d& q)
{
    return Eigen::Vector4d(q(0), -q(1), -q(2), -q(3));
}

Eigen::Vector3d InterceptorNodeDQ::rotate_vector(const Eigen::Vector4d& q, const Eigen::Vector3d& v)
{
    // v' = q * [0,v] * q*
    Eigen::Vector4d v_quat(0.0, v(0), v(1), v(2));
    Eigen::Vector4d result = quat_product(quat_product(q, v_quat), quat_conjugate(q));
    return result.tail<3>();
}

Eigen::Vector3d InterceptorNodeDQ::dq_get_translation(const Eigen::VectorXd& dq8)
{
    // t = 2 * d * q*   where dq = [q(4), d(4)]
    Eigen::Vector4d q = dq8.head<4>();
    Eigen::Vector4d d = dq8.segment<4>(4);
    Eigen::Vector4d t_quat = 2.0 * quat_product(d, quat_conjugate(q));
    return t_quat.tail<3>();
}

void InterceptorNodeDQ::normalize_dq(Eigen::VectorXd& dq8)
{
    // Normalize real part
    Eigen::Vector4d q = dq8.head<4>();
    double norm_q = q.norm();
    if (norm_q < 1e-8) {
        dq8.head<4>() << 1.0, 0.0, 0.0, 0.0;
        dq8.segment<4>(4).setZero();
        return;
    }
    q /= norm_q;
    Eigen::Vector4d d = dq8.segment<4>(4) / norm_q;
    // Enforce orthogonality: q . d = 0
    double qd_dot = q.dot(d);
    d -= qd_dot * q;
    dq8.head<4>() = q;
    dq8.segment<4>(4) = d;
}

// ============================================================================
// Constructor
// ============================================================================

InterceptorNodeDQ::InterceptorNodeDQ()
    : InterceptorNodeBase("interceptor_node_dq")
{
    initialize();
}

InterceptorNodeDQ::~InterceptorNodeDQ()
{
    destroy_solver();
}

// ============================================================================
// State lifecycle
// ============================================================================

Eigen::VectorXd InterceptorNodeDQ::make_identity_state() const
{
    Eigen::VectorXd s = Eigen::VectorXd::Zero(DRONE_INTERCEPTOR_DQ_NX);
    s(0) = 1.0;  // quaternion w in DQ real part
    return s;
}

void InterceptorNodeDQ::normalize_warm_start_orientation(Eigen::VectorXd& s)
{
    // Normalize the real part of the dual quaternion and enforce orthogonality of the dual part.
    Eigen::VectorXd dq8 = s.head<8>();
    normalize_dq(dq8);
    s.head<8>() = dq8;
}

Eigen::Vector3d InterceptorNodeDQ::get_position(const Eigen::VectorXd& s) const
{
    return dq_get_translation(s.head<8>());
}

Eigen::Vector3d InterceptorNodeDQ::get_velocity_world(const Eigen::VectorXd& s) const
{
    // DQ state stores body-frame velocity at indices 11-13
    Eigen::Vector4d q = s.head<4>();
    Eigen::Vector3d v_body = s.segment<3>(11);
    return rotate_vector(q, v_body);
}

void InterceptorNodeDQ::encode_odom(Eigen::VectorXd& s,
                                     const Eigen::Vector3d& pos,
                                     const Eigen::Quaterniond& q_eigen,
                                     const Eigen::Vector3d& vel_body,
                                     const Eigen::Vector3d& omega)
{
    // Convert pose to dual quaternion: d = 0.5 * [0,t] * q
    Eigen::Vector4d q(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
    Eigen::Vector4d t_quat(0.0, pos(0), pos(1), pos(2));
    Eigen::Vector4d d = 0.5 * quat_product(t_quat, q);

    s.head<4>() = q;
    s.segment<4>(4) = d;
    s.segment<3>(8) = omega;
    s.segment<3>(11) = vel_body;  // DQ uses body-frame velocity directly

    if (camera_fixed_) {
        gimbal_received_ = true;
        // Equivalent gimbal calculation: R_gimbal = R_C_B.transpose() * R_B_W.transpose() * R_yaw_W
        double psi = get_yaw_from_quaternion(q);
        Eigen::Matrix3d R_yaw_W = Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        Eigen::Matrix3d R_B_W = q_eigen.toRotationMatrix();
        Eigen::Matrix3d R_C_B = Eigen::AngleAxisd(camera_pitch_deg_ * M_PI / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Matrix3d R_gimbal = R_C_B.transpose() * R_B_W.transpose() * R_yaw_W;

        // Extract roll, pitch, yaw from R_gimbal (Y-X-Z intrinsic sequence in drone_model_dq.py)
        double roll = std::asin(std::clamp(R_gimbal(1, 2), -1.0, 1.0));
        double yaw = std::atan2(-R_gimbal(1, 0), R_gimbal(1, 1));
        double pitch = std::atan2(-R_gimbal(0, 2), R_gimbal(2, 2));

        s.segment<3>(14) << roll, pitch, yaw;

        // Update current_gimbal_orientation_ to full camera-to-world orientation
        {
            std::unique_lock<std::shared_mutex> lock(mtx_gimbal_);
            current_gimbal_orientation_ = q_eigen * Eigen::Quaterniond(R_C_B);
        }
    }
}

// ============================================================================
// Solver interface
// ============================================================================

bool InterceptorNodeDQ::init_solver()
{
    acados_capsule_ = drone_interceptor_dq_acados_create_capsule();
    int status = drone_interceptor_dq_acados_create(acados_capsule_);
    if (status) {
        RCLCPP_ERROR(this->get_logger(), "acados_create() returned status %d.", status);
        return false;
    }
    nlp_config_ = drone_interceptor_dq_acados_get_nlp_config(acados_capsule_);
    nlp_dims_   = drone_interceptor_dq_acados_get_nlp_dims(acados_capsule_);
    nlp_in_     = drone_interceptor_dq_acados_get_nlp_in(acados_capsule_);
    nlp_out_    = drone_interceptor_dq_acados_get_nlp_out(acados_capsule_);
    return true;
}

void InterceptorNodeDQ::destroy_solver()
{
    if (acados_capsule_) {
        drone_interceptor_dq_acados_free(acados_capsule_);
        drone_interceptor_dq_acados_free_capsule(acados_capsule_);
        acados_capsule_ = nullptr;
    }
}

int InterceptorNodeDQ::solve_step()
{
    return drone_interceptor_dq_acados_solve(acados_capsule_);
}

void InterceptorNodeDQ::update_solver_params(int k, double* data, int size)
{
    drone_interceptor_dq_acados_update_params(acados_capsule_, k, data, size);
}

void InterceptorNodeDQ::pack_and_set_mpc_params(
    const Eigen::Vector3d& pos_est, const Eigen::Vector3d& vel_est,
    const Eigen::Vector3d& /*drone_pos*/)
{
    // DQ parameter layout (19D):
    //   [target_pos(3), target_vel(3), v_desired, drag_coeff, W_cost(9), gimbal_rp_coeff, gimbal_y_coeff]
    VectorPDQ p_vec;
    double rp_coeff = camera_fixed_ ? 0.0 : gimbal_rp_coeff_;
    double y_coeff = camera_fixed_ ? 0.0 : gimbal_y_coeff_;
    for (int k = 0; k <= N_HORIZON_; ++k) {
        Eigen::Vector3d predicted_target_pos = pos_est + (k * dt_) * vel_est;
        double tracking_coeff = 1.0;
        if (TRACKING_DISTANCE_ > 0.001) {
            Eigen::VectorXd x_k = Eigen::VectorXd::Zero(state_dim());
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "x", x_k.data());
            Eigen::Vector3d x_k_pos = dq_get_translation(x_k.head<8>());
            double dist_to_target = (predicted_target_pos - x_k_pos).norm();
            double dist_error = (dist_to_target - TRACKING_DISTANCE_) / TRACKING_DISTANCE_;
            tracking_coeff = 2.0 / (1.0 + std::exp(-TRACKING_GAIN_ * dist_error)) - 1.0;
        }
        p_vec << predicted_target_pos, vel_est,
                 tracking_coeff * V_DESIRED_, DRAG_COEFF_,
                 w_cost_,
                 rp_coeff, y_coeff;
        drone_interceptor_dq_acados_update_params(acados_capsule_, k, p_vec.data(), p_vec.size());
    }
}

// ============================================================================
// Delay compensation
// ============================================================================

Eigen::VectorXd InterceptorNodeDQ::predict_state_forward(
    const Eigen::VectorXd& state, double delay, const Eigen::Vector3d& avg_rates)
{
    Eigen::VectorXd predicted = state;

    // Integrate quaternion (real part of DQ) forward
    Eigen::Vector4d current_quat = state.head<4>();
    Eigen::Vector4d predicted_quat = integrate_quaternion(current_quat, avg_rates, delay);
    predicted_quat = normalize_quaternion(predicted_quat);
    predicted.head<4>() = predicted_quat;

    // Extrapolate position and reconstruct dual part
    Eigen::Vector3d current_pos = dq_get_translation(state.head<8>());
    Eigen::Vector3d current_vel_world = get_velocity_world(state);
    Eigen::Vector3d predicted_pos = current_pos + current_vel_world * delay;

    // Reconstruct dual part: d = 0.5 * [0,t] * q
    Eigen::Vector4d t_quat(0.0, predicted_pos(0), predicted_pos(1), predicted_pos(2));
    predicted.segment<4>(4) = 0.5 * quat_product(t_quat, predicted_quat);

    return predicted;
}

// ============================================================================
// Validation
// ============================================================================

bool InterceptorNodeDQ::validate_orientation(const Eigen::VectorXd& s, const char* ctx) const
{
    Eigen::Vector4d q = s.head<4>();
    double quat_norm = q.norm();
    if (std::abs(quat_norm - 1.0) > 0.1) {
        RCLCPP_ERROR(this->get_logger(), "DQ real quat norm: %.6f (expected 1.0) (%s)", quat_norm, ctx);
        return false;
    }
    return true;
}

// ============================================================================
// Parameters and EKF
// ============================================================================

void InterceptorNodeDQ::declare_specific_parameters()
{
    // 9D cost weights matching drone_model_dq.py parameter order
    this->declare_parameter<double>("cost.w_los", 100.0);
    this->declare_parameter<double>("cost.w_zem", 10.0);
    this->declare_parameter<double>("cost.w_closing", 0.3);
    this->declare_parameter<double>("cost.w_align", 5.0);
    this->declare_parameter<double>("cost.w_control_F", 0.001);
    this->declare_parameter<double>("cost.w_control_tau", 1.0);
    this->declare_parameter<double>("cost.w_tilt", 0.01);
    this->declare_parameter<double>("cost.w_omega", 0.001);
    this->declare_parameter<double>("cost.w_fov", 10.0);
}

void InterceptorNodeDQ::load_specific_parameters()
{
    // DQ NONLINEAR_LS model embeds sqrt(w) in residuals, so pass raw weights.
    // Cost = ||sqrt(w) * error||^2 = w * error^2
    std::vector<double> cw(9);
    this->get_parameter("cost.w_los", cw[0]);
    this->get_parameter("cost.w_zem", cw[1]);
    this->get_parameter("cost.w_closing", cw[2]);
    this->get_parameter("cost.w_align", cw[3]);
    this->get_parameter("cost.w_control_F", cw[4]);
    this->get_parameter("cost.w_control_tau", cw[5]);
    this->get_parameter("cost.w_tilt", cw[6]);
    this->get_parameter("cost.w_omega", cw[7]);
    this->get_parameter("cost.w_fov", cw[8]);

    for (int i = 0; i < 9; ++i) w_cost_(i) = cw[i];

    if (TRACKING_DISTANCE_ > 0.001) {
        auto coeff = std::sqrt(10.0);
        auto los_coeff = std::sqrt(TRACKING_DISTANCE_ * 0.1);
        if (los_coeff > coeff) los_coeff = coeff;
        w_cost_(0) /= los_coeff;   // w_los
        w_cost_(2) *= coeff;        // w_closing
    }

    ekf_std_q_W_ = 0.002;
}

void InterceptorNodeDQ::setup_ekf()
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
    auto node = std::make_shared<InterceptorNodeDQ>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
