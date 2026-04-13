#pragma once

#include "interceptor_node_base.h"

// ACADOS — dual quaternion solver
extern "C" {
    #include "acados_solver_drone_interceptor_dq.h"
}

// DQ cost weights: 9D [w_los, w_zem, w_closing, w_align, w_control_F, w_control_tau, w_tilt, w_omega, w_fov]
using VectorWcostDQ = Eigen::Matrix<double, 9, 1>;
// DQ parameter vector: 19D [target_pos(3), target_vel(3), v_desired, drag_coeff, W(9), gimbal_coeff(2)]
using VectorPDQ     = Eigen::Matrix<double, DRONE_INTERCEPTOR_DQ_NP, 1>;

class InterceptorNodeDQ : public InterceptorNodeBase
{
public:
    InterceptorNodeDQ();
    ~InterceptorNodeDQ() override;

protected:
    // State accessors — DQ layout: [dq(8) omega_body(3) v_body(3) gimbal(3)]
    Eigen::Vector3d get_position(const Eigen::VectorXd& s) const override;
    Eigen::Vector3d get_velocity_world(const Eigen::VectorXd& s) const override;
    Eigen::Vector4d get_quaternion(const Eigen::VectorXd& s) const override { return s.segment<4>(0); }
    Eigen::Vector3d get_omega_body(const Eigen::VectorXd& s) const override { return s.segment<3>(8); }

    int  state_dim() const override { return DRONE_INTERCEPTOR_DQ_NX; }  // 17
    Eigen::VectorXd make_identity_state() const override;
    void normalize_warm_start_orientation(Eigen::VectorXd& s) override;
    void encode_odom(Eigen::VectorXd& s, const Eigen::Vector3d& pos,
                     const Eigen::Quaterniond& q, const Eigen::Vector3d& vel_body,
                     const Eigen::Vector3d& omega) override;

    bool init_solver() override;
    void destroy_solver() override;
    int  solve_step() override;
    void update_solver_params(int k, double* data, int size) override;
    void pack_and_set_mpc_params(const Eigen::Vector3d& pos_est,
                                 const Eigen::Vector3d& vel_est,
                                 const Eigen::Vector3d& drone_pos) override;

    Eigen::VectorXd predict_state_forward(const Eigen::VectorXd& state, double delay,
                                           const Eigen::Vector3d& avg_rates) override;
    bool validate_orientation(const Eigen::VectorXd& s, const char* ctx) const override;

    void declare_specific_parameters() override;
    void load_specific_parameters() override;
    void setup_ekf() override;

private:
    // DQ math helpers (scalar-first [w,x,y,z])
    static Eigen::Vector4d quat_product(const Eigen::Vector4d& p, const Eigen::Vector4d& q);
    static Eigen::Vector4d quat_conjugate(const Eigen::Vector4d& q);
    static Eigen::Vector3d rotate_vector(const Eigen::Vector4d& q, const Eigen::Vector3d& v);
    static Eigen::Vector3d dq_get_translation(const Eigen::VectorXd& dq8);
    static void normalize_dq(Eigen::VectorXd& dq8);

    drone_interceptor_dq_solver_capsule* acados_capsule_ = nullptr;
    VectorWcostDQ w_cost_;
};
