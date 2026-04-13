#pragma once

#include "interceptor_node_base.h"

// ACADOS — quaternion solver
extern "C" {
    #include "acados_solver_drone_interceptor_mpc_quat.h"
}

using VectorWcostQuat = Eigen::Matrix<double, 8, 1>;
using VectorPQuat     = Eigen::Matrix<double, 19, 1>;

class InterceptorNodeQuaternion : public InterceptorNodeBase
{
public:
    InterceptorNodeQuaternion();
    ~InterceptorNodeQuaternion() override;

protected:
    // State accessors — quaternion layout: [pos(3) vel_world(3) quat(4) omega(3) gimbal(3)]
    Eigen::Vector3d get_position(const Eigen::VectorXd& s) const override { return s.head<3>(); }
    Eigen::Vector3d get_velocity_world(const Eigen::VectorXd& s) const override { return s.segment<3>(3); }
    Eigen::Vector4d get_quaternion(const Eigen::VectorXd& s) const override { return s.segment<4>(6); }
    Eigen::Vector3d get_omega_body(const Eigen::VectorXd& s) const override { return s.segment<3>(10); }

    int  state_dim() const override { return 16; }
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
    drone_interceptor_mpc_quat_solver_capsule* acados_capsule_ = nullptr;
    VectorWcostQuat w_cost_;
};
