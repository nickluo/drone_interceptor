#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/header.hpp>
#include <quadrotor_msgs/msg/control_command.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define THROTTLE_INTERVAL_MS    100

namespace {

constexpr int kStateDim = 10;
constexpr int kMeasDim = 3;
constexpr int kModeCount = 3;

static double clamp(double value, double low, double high) {
    return std::max(low, std::min(high, value));
}

static Eigen::Vector3d pixel_to_los_nav(double u, double v, double focal_length,
                                    const Eigen::Matrix3d& R_b2n, 
                                    const Eigen::Matrix3d& R_b2c) {
    Eigen::Vector3d ray_cam(u / focal_length, v / focal_length, 1.0);
    ray_cam.normalize();
    const Eigen::Vector3d ray_body = R_b2c.transpose() * ray_cam;
    return (R_b2n * ray_body).normalized();
}

}  // namespace


class UnscentedKalmanFilter10D {
private:
    static Eigen::MatrixXd robust_cholesky(const Eigen::MatrixXd& matrix) {
        Eigen::MatrixXd sym = (matrix + matrix.transpose()) * 0.5;
        sym += Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols()) * 1e-8;

        Eigen::LLT<Eigen::MatrixXd> llt(sym);
        if (llt.info() == Eigen::Success) {
            return llt.matrixL();
        }

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(sym);
        Eigen::VectorXd values = eig.eigenvalues();
        for (int i = 0; i < values.size(); ++i) {
            values(i) = std::max(values(i), 1e-8);
        }
        return eig.eigenvectors() * values.cwiseSqrt().asDiagonal();
    }
    static Eigen::Vector3d h_meas_pixel(const Eigen::VectorXd& x, 
                                 const Eigen::Matrix3d& R_n2c, 
                                 double focal_length) {
        // Relative-state form: x[0:3] IS the relative position (target - ownship) in world frame G.
        // No uav_pos needed — the measurement is purely relative; R_n2c = R_b2c @ R_b2n^T carries the
        // ownship ATTITUDE (from IMU orientation), which is all the projection requires.
        const Eigen::Vector3d rel_p = x.segment<3>(0);
        Eigen::Vector3d p_cam = R_n2c * rel_p;
        p_cam(2) = std::max(p_cam(2), 0.5);
        const double u = focal_length * p_cam(0) / p_cam(2);
        const double v = focal_length * p_cam(1) / p_cam(2);
        // Clamp w_est to physical range to prevent negative-width sigma points from
        // creating asymmetric measurement predictions that bias the UKF update.
        const double w_est = clamp(x(9), 0.3, 5.0);
        const double s = focal_length * w_est / p_cam(2);
        return Eigen::Vector3d(u, v, s);
    }
    // Relative-state motion models. State x = [r(3), v_rel(3), a_t(3), W(1)] in a gravity-aligned
    // world frame G (origin arbitrary). r/v_rel are TARGET-OWNSHIP (relative); a_t is the target's
    // OWN acceleration; `a_own` is the ownship's world-frame acceleration over dt, supplied as an
    // exogenous input from IMU preintegration (dv_own/dt, gravity-corrected). Ownship pos/vel are
    // NOT state — relative quantities are drift-free because each bbox update closes the loop on r.
    // Exact mirror of the verified Python prototype (sim_test_relative_imm_ukf->py).
    static Eigen::VectorXd f_singer(const Eigen::VectorXd& x, double dt, const Eigen::Vector3d& a_own) {
        Eigen::VectorXd x_new = x;
        const double decay = std::exp(-dt / 1.5);
        const Eigen::Vector3d a_rel = x.segment<3>(6) - a_own;      // relative accel = a_target - a_ownship
        x_new.segment<3>(6) = x.segment<3>(6) * decay;              // target own accel decays (Singer)
        x_new.segment<3>(3) = x.segment<3>(3) + a_rel * dt;         // v_rel
        x_new.segment<3>(0) = x.segment<3>(0) + x.segment<3>(3) * dt + 0.5 * a_rel * dt * dt;   // r
        x_new(9) = x(9);
        return x_new;
    }

    static Eigen::VectorXd f_ct(const Eigen::VectorXd& x, double dt, const Eigen::Vector3d& a_own) {
        Eigen::VectorXd x_new = x;
        const Eigen::Vector3d p = x.segment<3>(0);
        const Eigen::Vector3d v = x.segment<3>(3);      // v_rel
        const Eigen::Vector3d a_t = x.segment<3>(6);    // target own accel
        const Eigen::Vector3d a_rel = a_t - a_own;
        const double v_xy2 = std::max(v(0) * v(0) + v(1) * v(1), 1e-4);
        // approx: v_rel stands in for target vel in the turn-rate estimate
        const double omega = clamp((v(0) * a_t(1) - v(1) * a_t(0)) / v_xy2, -2.0, 2.0);

        if (std::abs(omega) < 1e-3) {
            x_new.segment<3>(0) = p + v * dt + 0.5 * a_rel * dt * dt;
            x_new.segment<3>(3) = v + a_rel * dt;
        } else {
            const double sin_wt = std::sin(omega * dt);
            const double cos_wt = std::cos(omega * dt);
            x_new(0) = p(0) + (v(0) * sin_wt - v(1) * (1.0 - cos_wt)) / omega;
            x_new(1) = p(1) + (v(1) * sin_wt + v(0) * (1.0 - cos_wt)) / omega;
            x_new(2) = p(2) + v(2) * dt + 0.5 * a_rel(2) * dt * dt;
            x_new(3) = v(0) * cos_wt - v(1) * sin_wt;
            x_new(4) = v(0) * sin_wt + v(1) * cos_wt;
            x_new(5) = v(2) + a_rel(2) * dt;
            x_new(6) = a_t(0) * cos_wt - a_t(1) * sin_wt;   // target own accel rotates with the turn
            x_new(7) = a_t(0) * sin_wt + a_t(1) * cos_wt;
            x_new(8) = a_t(2);
        }
        x_new(9) = x(9);
        return x_new;
    }

    static Eigen::VectorXd f_rotor(const Eigen::VectorXd& x, double dt, const Eigen::Vector3d& a_own,
                                   double g_drag_coeff) {
        Eigen::VectorXd x_new = x;
        const Eigen::Vector3d v = x.segment<3>(3);      // v_rel
        const Eigen::Vector3d a_t = x.segment<3>(6);    // target own accel
        const Eigen::Vector3d a_rel = a_t - a_own;
        // NOTE: the original absolute-state f_rotor modeled target aerodynamic drag as
        // a_real = a_thrust - k_drag*|v_target|*v_target. In relative coordinates v_target is
        // UNOBSERVABLE (it would need ownship vel). Substituting v_rel is wrong: when the target is
        // static but the ownship is moving, |v_rel| is large and the drag term spuriously brakes the
        // Target aerodynamic drag (restored): a_drag = -k*|v|*v. In relative coordinates v_rel is an
        // approximation of v_target (exact for a static ownship; ownship motion is captured separately
        // via a_own). With drag_coeff=0 the model reverts to pure accel-relaxation (tau=2s).
        const Eigen::Vector3d a_drag = -g_drag_coeff * v.norm() * v;
        const Eigen::Vector3d a_rel_total = a_rel + a_drag;
        x_new.segment<3>(0) = x.segment<3>(0) + v * dt + 0.5 * a_rel_total * dt * dt;
        x_new.segment<3>(3) = v + a_rel_total * dt;
        x_new.segment<3>(6) = a_t * std::exp(-dt / 2.0);
        x_new(9) = x(9);
        return x_new;
    }

    Eigen::VectorXd apply_model(const Eigen::VectorXd& x, double dt, const Eigen::Vector3d& a_own) {
        if (mode == 0) return f_singer(x, dt, a_own);
        if (mode == 1) return f_ct(x, dt, a_own);
        return f_rotor(x, dt, a_own, drag_coeff);
    }
public:
    Eigen::VectorXd x = Eigen::VectorXd::Zero(kStateDim);
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(kStateDim, kStateDim);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(kStateDim, kStateDim);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    int mode = 0;
    double drag_coeff = 0.0;
    std::vector<Eigen::VectorXd> sigmas_f;
    bool predicted = false;

    UnscentedKalmanFilter10D() {
        sigmas_f.assign(2 * kStateDim + 1, Eigen::VectorXd::Zero(kStateDim));
        R.diagonal() << 0.005 * 0.005, 0.005 * 0.005, 0.015 * 0.015;
    }

    void predict(double dt, const Eigen::Vector3d& a_own) {
        const int n = kStateDim;
        const double Wi = 0.5 / n;

        const Eigen::MatrixXd L = robust_cholesky(n * P);
        std::vector<Eigen::VectorXd> sigmas(2 * n + 1, Eigen::VectorXd::Zero(n));
        sigmas[0] = x;
        for (int i = 0; i < n; ++i) {
            sigmas[i + 1] = x + L.col(i);
            sigmas[i + 1 + n] = x - L.col(i);
        }

        sigmas_f.resize(2 * n + 1, Eigen::VectorXd::Zero(n));
        // a_own is a deterministic exogenous input — same value applied to every sigma point.
        for (int i = 0; i <= 2 * n; ++i) {
            sigmas_f[i] = apply_model(sigmas[i], dt, a_own);
        }

        Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(n);
        for (int i = 1; i <= 2 * n; ++i) {
            x_pred += Wi * sigmas_f[i];
        }

        Eigen::MatrixXd P_pred = Eigen::MatrixXd::Zero(n, n);
        for (int i = 1; i <= 2 * n; ++i) {
            const Eigen::VectorXd diff = sigmas_f[i] - x_pred;
            P_pred += Wi * diff * diff.transpose();
        }
        P_pred += Q;

        x = x_pred;
        P = (P_pred + P_pred.transpose()) * 0.5;
        predicted = true;
    }

    double update(const Eigen::Vector3d& z_pixel, const Eigen::Matrix3d& R_n2c, double focal_length) {
        const int n = kStateDim;
        const double Wi = 0.5 / n;

        if (!predicted) {
            const Eigen::MatrixXd L = robust_cholesky(n * P);
            sigmas_f.resize(2 * n + 1, Eigen::VectorXd::Zero(n));
            sigmas_f[0] = x;
            for (int i = 0; i < n; ++i) {
                sigmas_f[i + 1] = x + L.col(i);
                sigmas_f[i + 1 + n] = x - L.col(i);
            }
        }
        predicted = false;

        std::vector<Eigen::Vector3d> sigmas_h(2 * n + 1, Eigen::Vector3d::Zero());
        for (int i = 0; i <= 2 * n; ++i) {
            sigmas_h[i] = h_meas_pixel(sigmas_f[i], R_n2c, focal_length);
        }

        Eigen::Vector3d z_pred = Eigen::Vector3d::Zero();
        for (int i = 1; i <= 2 * n; ++i) {
            z_pred += Wi * sigmas_h[i];
        }

        Eigen::Matrix3d P_zz = Eigen::Matrix3d::Zero();
        Eigen::MatrixXd P_xz = Eigen::MatrixXd::Zero(n, kMeasDim);
        for (int i = 1; i <= 2 * n; ++i) {
            const Eigen::Vector3d dz = sigmas_h[i] - z_pred;
            const Eigen::VectorXd dx = sigmas_f[i] - x;
            P_zz += Wi * dz * dz.transpose();
            P_xz += Wi * dx * dz.transpose();
        }
        P_zz += R;
        P_zz = (P_zz + P_zz.transpose()) * 0.5;

        Eigen::LDLT<Eigen::Matrix3d> ldlt(P_zz);
        if (ldlt.info() != Eigen::Success) {
            return 1e-30;
        }

        const Eigen::MatrixXd K = P_xz * ldlt.solve(Eigen::Matrix3d::Identity());
        const Eigen::Vector3d y = z_pixel - z_pred;
        if (!K.allFinite() || !y.allFinite()) {
            return 1e-30;
        }

        x = x + K * y;
        P = P - K * P_zz * K.transpose();
        P = (P + P.transpose()) * 0.5;

        const double det = std::max(std::abs(P_zz.determinant()), 1e-30);
        const double exponent = -0.5 * y.dot(ldlt.solve(y));
        double likelihood = std::exp(clamp(exponent, -80.0, 80.0)) / std::sqrt(std::pow(2.0 * M_PI, 3) * det);
        if (!std::isfinite(likelihood) || likelihood < 1e-30) {
            likelihood = 1e-30;
        }
        return likelihood;
    }
};

class RobustIMMEstimator10D {
private:
    static Eigen::MatrixXd make_Q(double q_pos, double q_vel, double q_acc, double q_w) {
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(kStateDim, kStateDim);
        for (int i = 0; i < 3; ++i) {
            Q(i, i) = q_pos;
            Q(i + 3, i + 3) = q_vel;
            Q(i + 6, i + 6) = q_acc;
        }
        Q(9, 9) = q_w;
        return Q;
    }
    double drag_coeff = 0.0;
public:
    std::vector<UnscentedKalmanFilter10D> filters;
    Eigen::Vector3d mu;
    Eigen::Matrix3d M;
    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    Eigen::Vector3d cbar;
    Eigen::Matrix3d omega;

    explicit RobustIMMEstimator10D(double drag_coeff_val) 
        : drag_coeff(drag_coeff_val) {
        filters.resize(kModeCount);
        for (int i = 0; i < kModeCount; ++i) {
            filters[i].mode = i;
            filters[i].drag_coeff = drag_coeff;
        }
        filters[0].Q = make_Q(1e-3, 0.1, 6.0, 0.05);
        filters[1].Q = make_Q(1e-3, 0.1, 2.0, 0.05);
        filters[2].Q = make_Q(1e-3, 0.1, 5.0, 0.05);
        mu << 0.2, 0.4, 0.4;
        M << 0.85, 0.05, 0.10,
             0.05, 0.90, 0.05,
             0.10, 0.05, 0.85;
        x = Eigen::VectorXd::Zero(kStateDim);
        P = Eigen::MatrixXd::Identity(kStateDim, kStateDim);
        compute_mixing_probabilities();
    }

    void compute_mixing_probabilities() {
        cbar = M.transpose() * mu;
        for (int j = 0; j < kModeCount; ++j) {
            cbar(j) = std::max(cbar(j), 1e-30);
        }
        for (int i = 0; i < kModeCount; ++i) {
            for (int j = 0; j < kModeCount; ++j) {
                omega(i, j) = (M(i, j) * mu(i)) / cbar(j);
            }
        }
    }

    void predict(double dt, const Eigen::Vector3d& a_own) {
        std::vector<Eigen::VectorXd> xs(kModeCount, Eigen::VectorXd::Zero(kStateDim));
        std::vector<Eigen::MatrixXd> Ps(kModeCount, Eigen::MatrixXd::Zero(kStateDim, kStateDim));

        for (int j = 0; j < kModeCount; ++j) {
            for (int i = 0; i < kModeCount; ++i) {
                xs[j] += omega(i, j) * filters[i].x;
            }
            for (int i = 0; i < kModeCount; ++i) {
                const Eigen::VectorXd diff = filters[i].x - xs[j];
                Ps[j] += omega(i, j) * (filters[i].P + diff * diff.transpose());
            }
        }

        for (int j = 0; j < kModeCount; ++j) {
            filters[j].x = xs[j];
            filters[j].P = (Ps[j] + Ps[j].transpose()) * 0.5;
            filters[j].predict(dt, a_own);
        }
        compute_mixing_probabilities();
        compute_state_estimate();
    }

    void update(const Eigen::Vector3d& z_pixel, const Eigen::Matrix3d& R_n2c, double focal_length) {
        Eigen::Vector3d likelihoods;
        for (int j = 0; j < kModeCount; ++j) {
            likelihoods(j) = filters[j].update(z_pixel, R_n2c, focal_length);
        }
        mu = cbar.cwiseProduct(likelihoods);
        const double mu_sum = mu.sum();
        if (!std::isfinite(mu_sum) || mu_sum < 1e-30) {
            mu = Eigen::Vector3d::Constant(1.0 / kModeCount);
        } else {
            mu /= mu_sum;
        }
        compute_mixing_probabilities();
        compute_state_estimate();
    }

    void compute_state_estimate() {
        x.setZero();
        for (int j = 0; j < kModeCount; ++j) {
            x += mu(j) * filters[j].x;
        }
        P.setZero();
        for (int j = 0; j < kModeCount; ++j) {
            const Eigen::VectorXd diff = filters[j].x - x;
            P += mu(j) * (filters[j].P + diff * diff.transpose());
        }
        P = (P + P.transpose()) * 0.5;
    }
};

class RelativeIMMUKFEstimator {
private:
    static Eigen::MatrixXd make_Q(double q_pos, double q_vel, double q_acc, double q_w) {
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(kStateDim, kStateDim);
        for (int i = 0; i < 3; ++i) {
            Q(i, i) = q_pos;
            Q(i + 3, i + 3) = q_vel;
            Q(i + 6, i + 6) = q_acc;
        }
        Q(9, 9) = q_w;
        return Q;
    }
public:
    double dt_cam;
    double focal_length;
    double w_guess;
    Eigen::Matrix3d R_b2c;
    // Range below which the monocular width/scale is frozen. Mid-course W is well observable
    // (~1.5); freezing it before the terminal phase (default 15 m) stops a bad close-range
    // detection (motion blur / partial bbox) from corrupting scale (observed W 1.5 -> 2.8 after a
    // dropout+reacquire -> phantom target 11 m below -> dive). Old value ~5 m was too late.
    double width_freeze_range_ = 15.0;
    // Size-plausibility gate is CLOSE-RANGE ONLY: it exists to reject close-range mis-detections
    // (a garbage bbox that corrupts the monocular scale). Applied at long range it starves the
    // relative estimator: r_est lags the true range during a closing -> s_pred too small -> good
    // measurements rejected -> r never corrects (deadlock).
    double size_gate_range_ = 12.0;
    double size_gate_lo_ = 0.6;
    double size_gate_hi_ = 1.8;

    std::unique_ptr<RobustIMMEstimator10D> imm_ukf;
    bool initialized = false;
    double frozen_width = 0.0;
    bool has_frozen_width = false;

    double last_cap_time = 0.0;
    bool has_last_cap_time = false;
    double w_init_time = -1.0;   // first measurement time — for Q(9,9) time annealing

    // Soft-recovery: first measurement after a bbox dropout uses inflated measurement noise so the
    // large innovation does not slam the state (especially vel) to an extreme value.
    bool first_meas_after_dropout = false;
    // Cached LOS unit vector (world) from the last detection, for the divergence reset direction.
    Eigen::Vector3d last_los_nav = Eigen::Vector3d(1.0, 0.0, 0.0);
    bool has_last_los_nav = false;

    RelativeIMMUKFEstimator(double dt_cam_val, double focal_length_val, double W_guess_val,
                            const Eigen::Matrix3d& R_b2c_val, double drag_coeff_val)
        : dt_cam(dt_cam_val), focal_length(focal_length_val), w_guess(W_guess_val), R_b2c(R_b2c_val)
    {
        imm_ukf = std::make_unique<RobustIMMEstimator10D>(drag_coeff_val);
    }

    void update_process_noise(double r_norm, double w_est, double bbox_s, double obs_duration) {
        (void)w_est;
        const double q_scale = 1.0;
        const double r_freeze = width_freeze_range_;
        const double snr_factor = bbox_s > 0.0 ? clamp(bbox_s / 40.0, 0.1, 1.0) : 1.0;
        const double q_w_scale = std::pow(clamp((r_norm - r_freeze) / 20.0, 0.0, 1.0), 2.0);
        // Time-based annealing: W is a physical constant, so as observations accumulate, Q_w decays
        // exponentially → W_est "locks in" its converged value and resists late-time noise. tau=4s.
        constexpr double w_converge_tau = 6.0;
        const double anneal = std::exp(-std::max(0.0, obs_duration) / w_converge_tau);
        const double q_w_dynamic = 0.01 * anneal * q_w_scale * snr_factor;
        // Relative-state: v_rel is driven by the a_own predict (high-quality, IMU preintegration).
        // The monocular measurement constrains v_rel only weakly (depth is noisy), so a large Q_vel
        // makes the update TRUST the lagged measurement and pull v_rel back toward it (under-playing
        // the closing speed). Keep Q_vel SMALL so K(vel) stays small and the a_own predict dominates.
        const double q_vel = 0.02 * q_scale;
        imm_ukf->filters[0].Q = make_Q(0.05 * q_scale, q_vel, 1.0 * q_scale, q_w_dynamic * q_scale);
        imm_ukf->filters[1].Q = make_Q(0.05 * q_scale, q_vel, 0.5 * q_scale, q_w_dynamic * q_scale);
        imm_ukf->filters[2].Q = make_Q(0.05 * q_scale, q_vel, 1.0 * q_scale, q_w_dynamic * q_scale);
    }

    void sanitize_filters() {
        // Relative-state form: r_norm is just |x[0:3]| (no uav_pos subtraction). On divergence,
        // reset along the last observed LOS direction at a nominal range.
        bool should_freeze = false;
        for (auto& filter : imm_ukf->filters) {
            if (filter.x.segment<3>(0).norm() < width_freeze_range_) {
                should_freeze = true;
                break;
            }
        }
        if (!should_freeze) {
            has_frozen_width = false;
        } else if (!has_frozen_width) {
            // Clamp at capture time: an un-clamped x(9) (e.g. 0.10 from a bad UKF update) would
            // otherwise be frozen permanently and corrupt every subsequent measurement prediction.
            frozen_width = clamp(imm_ukf->x(9), 0.3, 5.0);
            has_frozen_width = true;
        }

        const Eigen::Vector3d los_dir = has_last_los_nav ? last_los_nav : Eigen::Vector3d(1.0, 0.0, 0.0);
        for (auto& filter : imm_ukf->filters) {
            bool diverged = !filter.x.allFinite() || !filter.P.allFinite();
            if (!diverged) {
                for (int i = 0; i < kStateDim; ++i) {
                    if (std::abs(filter.x(i)) > 1e6) {
                        diverged = true;
                        break;
                    }
                }
            }
            if (diverged) {
                filter.x.setZero();
                filter.x.segment<3>(0) = 10.0 * los_dir;   // relative pos along last LOS
                filter.x(9) = w_guess;
                filter.P = Eigen::MatrixXd::Zero(kStateDim, kStateDim);
                filter.P.diagonal() << 100.0, 100.0, 100.0, 25.0, 25.0, 25.0, 10.0, 10.0, 10.0, 1.0;
                continue;
            }

            const Eigen::Vector3d rel_p = filter.x.segment<3>(0);
            const double r_norm = rel_p.norm();
            if (r_norm > 200.0) {
                filter.x.segment<3>(0) = rel_p / std::max(r_norm, 1e-6) * 120.0;
            }
            for (int i = 3; i < 6; ++i) filter.x(i) = clamp(filter.x(i), -30.0, 30.0);
            for (int i = 6; i < 9; ++i) filter.x(i) = clamp(filter.x(i), -15.0, 15.0);
            filter.x(9) = clamp(filter.x(9), 0.3, 5.0);

            if (has_frozen_width) {
                filter.x(9) = frozen_width;
                filter.P.row(9).setZero();
                filter.P.col(9).setZero();
                filter.P(9, 9) = 1e-8;
            }

            filter.P = (filter.P + filter.P.transpose()) * 0.5;
            // Per-dimension P floor: position (0-2) must stay large enough that measurements can
            // still correct it (a P(pos)~0 makes K(pos)~0 and forces all residual into vel/acc,
            // which then jumps wildly). vel (3-5) floor moderate; acc/width keep small floor.
            for (int i = 0; i < 3; ++i) {
                filter.P(i, i) = clamp(filter.P(i, i), 2.0, 500.0);
            }
            for (int i = 3; i < 6; ++i) {
                filter.P(i, i) = clamp(filter.P(i, i), 0.5, 500.0);
            }
            // acc (6-8): cap P(acc) at 10.0. A large P(acc) makes the acc channel absorb all
            // measurement residual, injecting noise into acc -> vel -> pos.
            for (int i = 6; i < 9; ++i) {
                filter.P(i, i) = clamp(filter.P(i, i), 1e-4, 10.0);
            }
            filter.P(9, 9) = clamp(filter.P(9, 9), 1e-4, 500.0);
        }
        imm_ukf->compute_state_estimate();
        if (has_frozen_width) {
            imm_ukf->x(9) = frozen_width;
            imm_ukf->P.row(9).setZero();
            imm_ukf->P.col(9).setZero();
            imm_ukf->P(9, 9) = 1e-8;
        } else {
            imm_ukf->x(9) = clamp(imm_ukf->x(9), 0.3, 5.0);
        }
    }

    // Relative-state estimator. Inputs that replaced ownship pos/vel:
    //   R_b2n_cur — ownship ATTITUDE (from IMU orientation), used for cam projection + LOS ray.
    //   dv_own    — ownship velocity change over [last_cam, now] in world frame G, from IMU
    //               preintegration (dv_own = ∫(R_b2n·(f_imu − bias) − g)dτ). Pure relative
    //               quantity, recomputed each window — NOT a drifting integrated velocity.
    // No uav_pos/uav_vel anywhere; r in the state IS the relative position.
    std::pair<bool, Eigen::VectorXd> process_camera(bool has_bbox, double bbox_u, double bbox_v, double bbox_s,
                                                    double t_now, const Eigen::Matrix3d& R_b2n_cur,
                                                    const Eigen::Vector3d& dv_own) {
        if (!has_bbox) {
            if (!initialized) {
                return {false, Eigen::VectorXd::Constant(kStateDim, std::numeric_limits<double>::quiet_NaN())};
            }
            const double w_prev = imm_ukf->x(9);
            const double r_norm = imm_ukf->x.segment<3>(0).norm();
            const Eigen::Vector3d a_own = dv_own / std::max(dt_cam, 1e-3);
            update_process_noise(r_norm, w_prev, 0.0, t_now - w_init_time);
            // ALWAYS predict r forward using ownship motion (a_own). The absolute-state estimator
            // could skip the predict for a stationary target; the RELATIVE position r = target - own
            // moves by the ownship's motion EVERY frame. Skipping the predict freezes r while the
            // ownship keeps flying -> unbounded drift the moment the target leaves the FOV.
            imm_ukf->predict(dt_cam, a_own);
            // Coast: assume the target stops ACCELERATING (a_t->0, constant-velocity target) but
            // DO NOT touch v_rel. In relative coordinates v_rel = v_target - v_own; zeroing or
            // decaying it would erase the ownship's ongoing motion from r's predict. Keep v_rel;
            // the a_own-driven predict carries the ownship motion, the retained v_rel the target's.
            for (auto& filter : imm_ukf->filters) {
                filter.x.segment<3>(6).setZero();
                filter.x(9) = w_prev;
                filter.P(9, 9) = std::max(0.001, filter.P(9, 9));
            }
            imm_ukf->compute_state_estimate();
            imm_ukf->x(9) = w_prev;
            sanitize_filters();
            // Mark dropout so the next bbox update uses inflated measurement noise (soft recovery).
            first_meas_after_dropout = true;
            return {true, imm_ukf->x};
        }

        const Eigen::Vector3d los_nav = pixel_to_los_nav(bbox_u, bbox_v, focal_length, R_b2n_cur, R_b2c);
        last_los_nav = los_nav;
        has_last_los_nav = true;
        const Eigen::Vector3d z_pixel(bbox_u, bbox_v, bbox_s);
        const Eigen::Matrix3d R_n2c = R_b2c * R_b2n_cur.transpose();
        Eigen::Matrix3d R_k = Eigen::Matrix3d::Zero();
        R_k(0,0) = 25.0;
        R_k(1,1) = 25.0;
        const double size_noise_std = std::max(5.0, 300.0 / std::max(1.0, bbox_s));
        R_k(2,2) = size_noise_std * size_noise_std;
        // Soft recovery: inflate measurement noise on the first bbox after a dropout so the large
        // innovation (from dead-reckoning drift during the dropout) does not slam vel to an extreme.
        if (first_meas_after_dropout) {
            R_k *= 5.0;
        }

        if (initialized) {
            const double dt_meas = has_last_cap_time ? clamp(t_now - last_cap_time, 0.001, 0.1) : dt_cam;
            const Eigen::Vector3d a_own = dv_own / std::max(dt_meas, 1e-3);
            update_process_noise(imm_ukf->x.segment<3>(0).norm(), imm_ukf->x(9), bbox_s, t_now - w_init_time);
            imm_ukf->predict(dt_meas, a_own);

            // Size-plausibility gate (close range only, see size_gate_range_): reject detections
            // whose bbox size is wildly inconsistent with predicted range/width -> predict-only,
            // mark soft-recovery. Stops a garbage close-range size from corrupting scale/position.
            const double r_pred = imm_ukf->x.segment<3>(0).norm();
            const double s_pred = focal_length * std::max(0.1, imm_ukf->x(9)) / std::max(0.5, r_pred);
            const double size_ratio = bbox_s / std::max(1.0, s_pred);
            if (r_pred < size_gate_range_ &&
                (size_ratio < size_gate_lo_ || size_ratio > size_gate_hi_)) {
                imm_ukf->compute_state_estimate();
                first_meas_after_dropout = true;
                last_cap_time = t_now;
                has_last_cap_time = true;
                return {true, imm_ukf->x};
            }

            for (auto& filter : imm_ukf->filters) filter.R = R_k;
            const double w_before = imm_ukf->x(9);
            imm_ukf->update(z_pixel, R_n2c, focal_length);
            // W_est rate limiter: cap |ΔW| per update to prevent the positive-feedback cascade
            // where one noisy bbox (at a dither extreme) crashes W, which shifts range/dz, which
            // reinforces the wrong W on the next update. ±0.05/frame ≈ ±1.5/s at 30Hz — still
            // converges to the true W within ~1-2s for W_guess errors up to ~1m.
            {
                constexpr double w_max_delta = 0.05;
                const double w_after = clamp(imm_ukf->x(9), w_before - w_max_delta, w_before + w_max_delta);
                imm_ukf->x(9) = w_after;
                for (auto& filter : imm_ukf->filters) {
                    filter.x(9) = clamp(filter.x(9), w_before - w_max_delta, w_before + w_max_delta);
                }
            }

            last_cap_time = t_now;
            has_last_cap_time = true;
            first_meas_after_dropout = false;
            sanitize_filters();
            // (No fused v clamp to +-15: that was an absolute-state target-speed limit. In relative
            // coordinates v_rel = v_target - v_ownship easily exceeds 15 m/s during a high-speed
            // closing; clamping caps the closing-speed estimate and starves the controller.
            // sanitize_filters already bounds v_rel to +-30.)
            return {true, imm_ukf->x};
        } else {
            const double W_init = clamp(w_guess, 0.3, 5.0);
            const double r0 = clamp(focal_length * W_init / std::max(1.0, bbox_s), 10.0, 300.0);
            Eigen::VectorXd x0 = Eigen::VectorXd::Zero(kStateDim);
            x0.segment<3>(0) = r0 * los_nav;            // relative pos = range * LOS (no uav_pos)
            x0(9) = W_init;

            Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(kStateDim, kStateDim);
            P0.diagonal() << 100.0, 100.0, 100.0, 25.0, 25.0, 25.0, 10.0, 10.0, 10.0, 0.09;
            for (auto& filter : imm_ukf->filters) {
                filter.x = x0;
                filter.P = P0;
                filter.R = R_k;
                filter.predicted = false;
            }
            imm_ukf->x = x0;
            imm_ukf->P = P0;
            initialized = true;
            w_init_time = t_now;
            last_cap_time = t_now;
            has_last_cap_time = true;

            sanitize_filters();
            return {true, imm_ukf->x};
        }
    }
};


class AdvancedHybridController {
public:
    double N;
    double K_att;
    double K_yaw;
    double v_des;
    double max_tilt_rad;
    double g = 9.81;             // gravity (set from physics.g config via set_g)
    double fov_half_v_ = M_PI;
    double fov_half_h_ = M_PI;
    double fov_keep_margin_ = 0.85;
    double axial_jerk_max_ = 1e9;
    bool enable_fov_clamp_ = false;
    double mount_pitch_ = 0.0;
    double last_axial_accel_ = 0.0;
    double last_t_ = -1.0;
    bool has_last_ = false;
    // Vertical-accel slew-limit state (Fix C): bounds |da_z/dt|.
    double last_a_z_ = 0.0;
    double last_az_t_ = -1.0;
    bool has_last_az_ = false;
    double a_z_slew_ = 8.0;   // m/s^3, rate limit on the vertical accel command
    double lateral_decay_floor_ = 0.4;  // Fix D: a_lat_mag floor at terminal_brake_range (close-range weave damp)
    double lat_decay_start_ = 25.0;     // Fix F: range at which the close-range lateral attenuation begins (m)
    double max_body_rate_ = 3.0;
    double terminal_brake_range_ = 5.0;
    double terminal_brake_floor_ = 1.0;
    // Decoupled gains (config-driven via set_guidance_tuning).
    double K_fwd_ = 2.0;        // axial: a_fwd = K_fwd * (v_des - v_fwd_body)
    double K_az_ = 2.0;         // vertical-speed tracking gain: a_z = K_az*(v_z_desired - v_z)
    double a_z_limit_ = 8.0;    // vertical accel clamp (m/s^2) — airframe thrust/accel limit
    double terminal_bbox_size_ = 80.0;  // bbox px at terminal_brake_range (computed from focal/W)
    // Semi-strapdown single-axis pitch gimbal: keeps the camera boresight near-horizontal
    // (world elev = -gimbal_bias_) regardless of body pitch, so the vertical FOV is DECOUPLED
    // from theta_cmd. When true, the fixed-mount pitch FOV clamp is skipped and vertical
    // keep-in-frame is done through ALTITUDE (a_z) instead (see compute_control).
    bool has_gimbal_ = false;
    double gimbal_bias_ = 0.0;   // rad, down-gaze bias: boresight elev = -gimbal_bias_
    // Vertical cascade (cascaded velocity-limited altitude track): v_climb_/v_descent_ = max
    // commanded climb/sink rate [m/s] (the cap); K_dz_up_/dn_ = dz->rate gain [1/s] (closure
    // tau = 1/K). Asymmetric: climb more responsive than descent (avoid below-FOV over-dive).
    double v_climb_   = 8.0;
    double v_descent_ = 8.0;
    double K_dz_up_   = 2.0;
    double K_dz_dn_   = 1.0;
    // Speed-scheduled PN + attitude reference slew governor (v_des=20 oscillation fixes).
    double v_pn_ref_ = 1e9;       // cap on v_fwd in PN lateral accel -> bounds effective N*v at high speed
    double max_att_rate_ = 1e9;   // max |d(phi_cmd)/dt|, |d(theta_cmd)/dt| [rad/s] (slew governor)
    double last_phi_cmd_ = 0.0;   // governor state (dedicated; axial block reuses last_t_/has_last_)
    double last_theta_cmd_ = 0.0;
    double last_att_t_ = -1.0;
    bool   has_last_att_ = false;
    // Attitude-reference EMA low-pass (suppresses camera-rate content / roll limit-cycle).
    double att_ref_tc_ = 0.15;    // EMA time constant on phi_cmd/theta_cmd [s]
    double last_phi_cmd_f_ = 0.0;   // EMA-filtered reference state
    double last_theta_cmd_f_ = 0.0;
    // Terminal authority (maneuvering-target tracking): relax roll clamp + schedule a slower approach.
    double max_tilt_terminal_rad_ = 45.0 * M_PI / 180.0;  // roll clamp inside terminal_tilt_range_
    double terminal_tilt_range_ = 15.0;  // m, below this range phi_cmd may use max_tilt_terminal_rad_
    double terminal_decel_range_ = 30.0; // m, start ramping v_des -> v_terminal_ here
    double v_terminal_ = 12.0;           // m/s, desired speed at the terminal phase (more bank agility)
    // Active monocular-observability dither: a small lateral sinusoidal acceleration that gives
    // the ownship a horizontal parallax baseline. With x != 0 the pinhole u=fx*x/z becomes
    // sensitive to z (depth), so the depth/width degeneracy (size=fx*W/z) is broken and the
    // along-LOS v_rel — otherwise only weakly observable via the noisy size channel — becomes
    // observable through the bearing channel. The IMU preintegration captures the dither motion
    // as a_own, so the estimator predict carries it; the update resolves depth from the parallax.
    double dither_amp_ = 0.8;          // m/s^2, lateral dither amplitude
    double dither_freq_ = 0.5;         // Hz, lateral dither frequency
    double dither_min_range_ = 10.0;   // m, suppress dither inside this range (don't disturb terminal)
    // Geometric APN: recover the target lateral accel from TRUE-PN geometry + IMU (NOT the
    // estimator a_t state — unobservable with a single bbox): a_t_perp = range*ydot +
    // 2*lambda_dot*v_range + a_own_perp, where ydot = d(lambda_dot)/dt. Leads a turning target.
    double K_apn_ = 0.8;
    double K_apn_min_at_ = 0.5;        // m/s^2, gate on |a_t_perp_geom|
    double ydot_tc_ = 0.8;             // s, EMA time constant on d(lambda_dot)/dt
    double atp_tc_ = 0.6;              // s, EMA time constant on a_t_perp_geom
    double prev_los_rate_ = 0.0;
    double prev_lr_t_ = -1.0;
    bool   has_prev_lr_ = false;
    double ydot_ema_ = 0.0;
    double atp_ema_ = 0.0;
    double aown_perp_ema_ = 0.0;
    double last_los_dt_ = 0.02;   // s, last valid LOS-sample dt (for atp EMA alpha); updated each camera frame

    AdvancedHybridController(double N_val,
                             double K_att_val, double K_yaw_val, double v_des_val, double max_tilt_deg,
                             double fov_half_v, double fov_half_h, double fov_keep_margin,
                             double axial_jerk_max, bool enable_fov_clamp,
                             double mount_pitch_rad,
                             double max_body_rate, double terminal_brake_range, double terminal_brake_floor,
                             double terminal_bbox_size,
                             double v_pn_ref, double max_att_rate, double att_ref_tc,
                             double max_tilt_terminal_deg, double terminal_tilt_range,
                             double terminal_decel_range, double v_terminal)
        : N(N_val), K_att(K_att_val),
          K_yaw(K_yaw_val), v_des(v_des_val), max_tilt_rad(max_tilt_deg * M_PI / 180.0),
          fov_half_v_(fov_half_v), fov_half_h_(fov_half_h), fov_keep_margin_(fov_keep_margin),
          axial_jerk_max_(axial_jerk_max), enable_fov_clamp_(enable_fov_clamp),
          mount_pitch_(mount_pitch_rad),
          max_body_rate_(max_body_rate), terminal_brake_range_(terminal_brake_range),
          terminal_brake_floor_(terminal_brake_floor), terminal_bbox_size_(terminal_bbox_size),
          v_pn_ref_(v_pn_ref), max_att_rate_(max_att_rate), att_ref_tc_(att_ref_tc),
          max_tilt_terminal_rad_(max_tilt_terminal_deg * M_PI / 180.0),
          terminal_tilt_range_(terminal_tilt_range), terminal_decel_range_(terminal_decel_range),
          v_terminal_(v_terminal) {}

    void reset_dynamic_state() {
        last_axial_accel_ = 0.0;
        last_t_ = -1.0;
        has_last_ = false;
        last_phi_cmd_ = 0.0;
        last_theta_cmd_ = 0.0;
        last_att_t_ = -1.0;
        has_last_att_ = false;
        last_phi_cmd_f_ = 0.0;
        last_theta_cmd_f_ = 0.0;
        last_a_z_ = 0.0;
        last_az_t_ = -1.0;
        has_last_az_ = false;
        prev_los_rate_ = 0.0;
        prev_lr_t_ = -1.0;
        has_prev_lr_ = false;
        ydot_ema_ = 0.0;
        atp_ema_ = 0.0;
        aown_perp_ema_ = 0.0;
        last_los_dt_ = 0.02;
    }
    void set_a_z_slew(double v) { a_z_slew_ = (v > 0.0) ? v : 1e9; }  // <=0 disables the slew limit
    void set_lateral_decay_floor(double v) { lateral_decay_floor_ = clamp(v, 0.0, 1.0); }
    void set_lat_decay_start(double v) { lat_decay_start_ = std::max(v, terminal_brake_range_ + 1.0); }
    void set_g(double g_val) { g = g_val; }
    void set_gimbal(bool has_gimbal, double gimbal_bias_rad) {
        has_gimbal_ = has_gimbal;
        gimbal_bias_ = gimbal_bias_rad;
    }
    // Vertical-cascade tuning (cascaded velocity-limited altitude track). Independent so the
    // operator can set the climb/sink RATE (m/s, the cap — calibrate to the airframe's real
    // capability so the commanded rate is achievable) and the RESPONSIVENESS (K_dz 1/s, closure
    // tau = 1/K) separately for climb vs descent.
    void set_vertical_climb(double v_climb, double v_descent, double K_dz_up, double K_dz_dn) {
        v_climb_   = std::max(0.5, v_climb);
        v_descent_ = std::max(0.5, v_descent);
        K_dz_up_   = std::max(0.1, K_dz_up);
        K_dz_dn_   = std::max(0.1, K_dz_dn);
    }
    // Guidance tuning (operator-facing): axial gain, vertical tracking gain + accel limit, and the
    // lateral observability dither. All independent config params.
    void set_guidance_tuning(double K_fwd, double K_az, double a_z_limit,
                             double dither_amp, double dither_freq, double dither_min_range) {
        K_fwd_           = std::max(0.1, K_fwd);
        K_az_            = std::max(0.1, K_az);
        a_z_limit_       = std::max(1.0, a_z_limit);
        dither_amp_      = std::max(0.0, dither_amp);
        dither_freq_     = std::max(0.01, dither_freq);
        dither_min_range_= std::max(1.0, dither_min_range);
    }

    // Decoupled control (odometry-free, exact mirror of the Python prototype). Inputs:
    //   r_est, v_est : RELATIVE target pos/vel in world frame G (estimator state directly)
    //   los_az       : target azimuth in body heading frame (rad, 0 = dead ahead)
    //   los_el       : target elevation in world frame (rad, 0 = level, negative = below)
    //   v_fwd_body   : closing speed proxy = -v_rel·r̂ (positive = approaching); NOT odom
    //   bbox_size    : target bbox size in px (observable proximity cue)
    //   v_z          : ownship vertical speed proxy = -v_rel(2) (target-level approximation)
    //   target_speed : |v_rel| clamped — relative-speed proxy for the brake floor
    //   los_rate     : EMA-smoothed measured bearing rate (lambda_dot)
    //   a_own_world  : ownship world accel over the last camera window (IMU preintegration)
    //   t            : mission-relative time (s)
    std::vector<double> compute_control(double roll, double pitch, double yaw,
                                        const Eigen::Vector3d& r_est, const Eigen::Vector3d& v_est,
                                        double los_az, double los_el, double t,
                                        double v_fwd_body = 0.0, double bbox_size = 0.0,
                                        double v_z = 0.0, double target_speed = 0.0,
                                        double los_rate = 0.0,
                                        const Eigen::Vector3d& a_own_world = Eigen::Vector3d::Zero(),
                                        double boresight_el_meas = std::numeric_limits<double>::quiet_NaN()) {
        (void)v_est;
        const double r_norm = r_est.norm();

        // ---- (A) Bearing-only lateral guidance -------------------------------------------
        // LOS rate from the measured bearing ONLY (clean, no 1/r blowup, no v_est noise).
        double lambda_dot = los_rate;
        // Deadband: a steady bearing (<0.02 rad/s) yields ZERO lateral lead. Stops residual bearing
        // noise around zero from sign-flipping a_lat.
        if (std::abs(lambda_dot) < 0.02) {
            lambda_dot = 0.0;
        }
        const double omega_los_mag = std::min(std::abs(lambda_dot), 1.0);
        double N_eff = N;
        if (r_norm > 15.0) {
            N_eff = N - (N - 2.0) * clamp((r_norm - 15.0) / 35.0, 0.0, 1.0);
        }
        // Speed-scheduled PN: use v_des (OPEN-LOOP) for the v in a_lat = N*v*lambda_dot, NOT the
        // measured v_closing. v_closing is under-observable along-LOS until the estimator anchors
        // scale; using it starves PN lead in the first seconds and lets a crossing target escape
        // the FOV. v_des is the intended ownship speed — the correct PN proxy — available from t=0.
        const double v_pn = std::min(v_des, v_pn_ref_);
        double a_lat_mag = N_eff * v_pn * omega_los_mag;
        // Close-range lateral attenuation (Fix D/F): gently scale a_lat_mag toward
        // lateral_decay_floor_ over [terminal_brake_range, lat_decay_start_] to damp the
        // close-approach PN weave. NOT a hard zero — the floor keeps PN lead for turning targets.
        const double lat_f = clamp((r_norm - terminal_brake_range_) /
                                   std::max(lat_decay_start_ - terminal_brake_range_, 1.0), 0.0, 1.0);
        a_lat_mag *= lateral_decay_floor_ + (1.0 - lateral_decay_floor_) * lat_f;
        // Lateral direction in world horizontal plane: perpendicular to LOS projected to xy.
        Eigen::Vector2d e_los_xy(r_est(0), r_est(1));
        if (e_los_xy.norm() < 1e-3) e_los_xy << 1.0, 0.0;
        e_los_xy.normalize();
        Eigen::Vector2d e_perp_xy(-e_los_xy(1), e_los_xy(0));  // 90 deg CCW
        const double turn_sign = (lambda_dot >= 0.0) ? 1.0 : -1.0;
        Eigen::Vector2d a_lat_world = turn_sign * a_lat_mag * e_perp_xy;
        // Active observability dither (mid-course only): lateral sinusoid for monocular scale/depth.
        if (dither_amp_ > 0.0 && r_norm > dither_min_range_) {
            const double a_dither = dither_amp_ * std::sin(2.0 * M_PI * dither_freq_ * t);
            a_lat_world += a_dither * e_perp_xy;
        }
        // Geometric APN: classic PN lags a turning target. Recover the target lateral accel from
        // TRUE-PN geometry + IMU: a_t_perp = range*ydot + 2*lambda_dot*v_range + a_own_perp.
        // EMA a_own_perp with the SAME tc as ydot so the ownship-dither component cancels for a
        // static/CV target; EMA a_t_perp_geom so pixel-noise spikes average toward 0.
        if (has_prev_lr_ && t > prev_lr_t_) {
            const double dt_lr = t - prev_lr_t_;
            if (dt_lr > 1e-3 && dt_lr < 1.0) {
                const double ydot_raw = (los_rate - prev_los_rate_) / dt_lr;
                const double a_ydot = clamp(dt_lr / ydot_tc_, 0.0, 1.0);
                ydot_ema_ += a_ydot * (ydot_raw - ydot_ema_);
                const double a_ao = clamp(dt_lr / ydot_tc_, 0.0, 1.0);
                const double a_own_perp = a_own_world.head<2>().dot(e_perp_xy);
                aown_perp_ema_ += a_ao * (a_own_perp - aown_perp_ema_);
                last_los_dt_ = dt_lr;
            }
        }
        prev_los_rate_ = los_rate;
        prev_lr_t_ = t;
        has_prev_lr_ = true;
        const double a_t_perp_geom = r_norm * ydot_ema_ + 2.0 * los_rate * v_fwd_body + aown_perp_ema_;
        {
            const double a_atp = clamp(last_los_dt_ / atp_tc_, 0.0, 1.0);
            atp_ema_ += a_atp * (a_t_perp_geom - atp_ema_);
        }
        if (std::abs(a_t_perp_geom) > K_apn_min_at_) {
            a_lat_world += K_apn_ * a_t_perp_geom * e_perp_xy;
        }

        // ---- (A) Axial guidance: fixed forward thrust, range-independent ------------------
        double v_des_eff = v_des;
        // Range-based terminal decel schedule: arrive at the terminal phase ~v_terminal (slower),
        // where the same bank gives ~2x the turn rate. Feedforward on the monocular range estimate.
        if (r_norm < terminal_decel_range_) {
            const double v_term_eff = std::min(std::max(v_terminal_, target_speed + 1.5), v_des);
            const double frac = clamp((terminal_decel_range_ - r_norm) /
                                      std::max(terminal_decel_range_ - terminal_brake_range_, 1.0), 0.0, 1.0);
            v_des_eff = v_des + (v_term_eff - v_des) * frac;   // ramp v_des -> v_term_eff (<= v_des)
        }
        // Turn-radius-aware speed cap: limit the closing speed so the minimum turn radius
        // (v^2 / (g*tan(max_tilt))) stays inside the current range. Without this, v_des=20 yields a
        // ~40 m min turn radius at 45 deg bank and the interceptor PHYSICALLY cannot bend onto a
        // target inside that -> overshoot/fly-by (AirSim v_des=20 ~60% vs v_des=15 ~80%; the Python
        // sim's ideal rate loop hides it). Bites only at close range (at r=40 m the cap is ~20 m/s).
        // The brake floor below still keeps enough closing to gain on a moving target.
        v_des_eff = std::min(v_des_eff, std::sqrt(g * std::tan(max_tilt_rad) * r_norm));
        const double bbox_engage = terminal_bbox_size_ * 0.2;   // engage at ~5x brake_range
        if (bbox_size > bbox_engage) {
            const double prox = clamp((bbox_size - bbox_engage) /
                                      std::max(1.0, terminal_bbox_size_ - bbox_engage), 0.0, 1.0);
            // Speed-aware brake floor: never brake below target speed + margin when the target
            // is actually moving; a near-stationary target keeps the configured floor.
            const double moving_floor = (target_speed > 1.0) ? (target_speed + 1.5) : terminal_brake_floor_;
            const double floor_eff = std::min(std::max(terminal_brake_floor_, moving_floor), v_des);
            v_des_eff = v_des_eff + (floor_eff - v_des_eff) * prox;
        }
        // v_fwd_body (= v_closing from v_rel) is under-observable along-LOS until scale converges;
        // a low v_closing inflates a_fwd -> ownship over-accelerates ON-AXIS, starving lateral PN
        // lead. Scene-adaptive 3-level floor: turning target (|atp_ema| large) -> 0.88 (don't blast
        // through a circle); laterally-moving target (v_perp > 1) -> 0.82; static/slow -> 0.8.
        // Braking (v_closing above the floor) passes through unchanged.
        const double v_perp = std::abs(r_norm * los_rate);
        const double v_floor_frac = (std::abs(atp_ema_) > 0.5) ? 0.88 : ((v_perp > 1.0) ? 0.82 : 0.8);
        const double v_fwd_for_afwd = std::max(v_fwd_body, v_floor_frac * v_des_eff);
        double a_fwd = K_fwd_ * (v_des_eff - v_fwd_for_afwd);
        if (has_last_ && t > last_t_) {
            const double dt = t - last_t_;
            if (dt > 0.0 && dt < 1.0) {
                a_fwd = clamp(a_fwd, last_axial_accel_ - axial_jerk_max_ * dt,
                              last_axial_accel_ + axial_jerk_max_ * dt);
            }
        }
        last_axial_accel_ = a_fwd;
        last_t_ = t;
        has_last_ = true;
        Eigen::Vector2d a_fwd_world = a_fwd * e_los_xy;

        // ---- (B) Decoupled vertical: cascaded velocity-limited altitude track ------------
        const double dz_target = r_est(2);
        // Command a vertical SPEED toward the target altitude (capped), then accelerate to track it.
        // (1) the cap prevents runaway dive/overshoot; (2) cap saturation makes a biased monocular
        // dz (wrong target width) NOT inflate the descent — robust to wrong_w. Damping rides on the
        // IMU own vertical velocity. v_climb/v_descent (m/s, max commanded climb/sink rate) and
        // K_dz_up/dn (1/s, responsiveness; closure tau = 1/K) are INDEPENDENT config params
        // (set_vertical_climb). CLIMB uses a higher gain than DESCENT so an upper-FOV target does
        // not asymptotically stall ~1 m short, while a below-FOV target still brakes without
        // over-dive. Set v_climb/v_descent to the airframe's real climb/sink capability so the
        // commanded rate is achievable (else lag -> stall).
        // K_az (vertical-speed tracking gain) and a_z_limit (vertical accel clamp) are config-driven
        // members (set_guidance_tuning); see the controller params in the yaml.
        double v_z_desired;
        if (dz_target > 0.0) {
            v_z_desired = clamp(dz_target * K_dz_up_, 0.0, v_climb_);
        } else {
            v_z_desired = clamp(dz_target * K_dz_dn_, -v_descent_, 0.0);
        }
        double a_z = clamp(K_az_ * (v_z_desired - v_z), -a_z_limit_, a_z_limit_);
        // Slew limit |da_z/dt|: the z-estimate can jump O(10 m) in one cycle; ramp a_z instead of
        // stepping it (which would pitch the body over).
        if (has_last_az_ && t > last_az_t_) {
            const double dt_az = t - last_az_t_;
            if (dt_az > 0.0 && dt_az < 1.0) {
                a_z = clamp(a_z, last_a_z_ - a_z_slew_ * dt_az,
                            last_a_z_ + a_z_slew_ * dt_az);
            }
        }
        last_a_z_ = a_z;
        last_az_t_ = t;
        has_last_az_ = true;

        // ---- (B) Total horizontal acceleration (lateral + forward) ----------------------
        Eigen::Vector2d a_h_world = a_lat_world + a_fwd_world;
        // Rotate world-horizontal accel into body heading frame for thrust allocation.
        const double cy = std::cos(yaw);
        const double sy = std::sin(yaw);
        const double fx_h = cy * a_h_world(0) + sy * a_h_world(1);
        const double fy_h = -sy * a_h_world(0) + cy * a_h_world(1);

        // ---- (C) Thrust-vector cascade ---------------------------------------------------
        // Standard quadrotor control: the desired body-frame acceleration vector
        //   t_body = [fx_h (fwd), fy_h (right), a_z + g (up)]
        // is realized by tilting the thrust axis toward it. Pitch/roll commands come
        // directly from the horizontal/vertical force ratio, so forward accel (fx_h > 0)
        // commands nose-down (positive theta in this frame's convention, where pitch>0
        // tilts body-z toward +x = forward) and produces real forward thrust.
        // This fixes the original bug where theta_cmd only tracked los_el, leaving a_fwd
        // with no path to the pitch axis (UAV could not self-propel forward).
        // Body frame is FLU: X=fwd, Y=left, Z=up. fy_h>0 = need rightward accel => roll
        // right (negative phi). fx_h>0 = need forward accel => nose down (positive theta).
        double theta_cmd = clamp( std::atan2(fx_h, a_z + g), -max_tilt_rad, max_tilt_rad);
        // Roll clamp is relaxed inside the terminal phase: a maneuvering target needs more bank
        // authority at close range (where the bbox is large and coast handles brief dropout) than
        // the conservative mid-course max_tilt permits. Pitch/FOV keep the conservative limit.
        const double max_tilt_roll = (r_norm < terminal_tilt_range_) ? max_tilt_terminal_rad_ : max_tilt_rad;
        double phi_cmd   = clamp(-std::atan2(fy_h, a_z + g), -max_tilt_roll, max_tilt_roll);

        const bool target_behind = std::abs(los_az) > (M_PI / 2.0);
        if (enable_fov_clamp_ && has_gimbal_ && !target_behind) {
            // Semi-strapdown gimbal: the boresight is decoupled from body pitch (gimbal cancels
            // it; elev ~= -gimbal_bias_), so a theta clamp cannot keep the target in the vertical
            // FOV — ALTITUDE can. If the target approaches the bottom edge (los_el well below the
            // boresight), command a descent so the frame re-centers; symmetric for the top edge.
            // Overrides the PD a_z only near the edge (outer 30% of the half-FOV). Mirrors the
            // verified Python implementation (sim_test_relative_imm_ukf->py).
            const double fov_keep = fov_half_v_ * fov_keep_margin_;
            // Boresight world elevation: prefer the ACTUAL camera orientation measured by the gimbal
            // (q_C_W) — this tracks the true boresight INCLUDING servo lag, so the vertical-FOV clamp
            // reacts to where the camera really points during pitch transients (closer to real HW).
            // Fall back to the constant -gimbal_bias_ assumption when no measurement is available.
            const double boresight_el = std::isfinite(boresight_el_meas)
                ? boresight_el_meas : -gimbal_bias_;
            const double el_err = los_el - boresight_el;   // negative = target below boresight
            const double edge = 0.7 * fov_keep;
            if (el_err < -edge) {
                a_z = std::min(a_z, -8.0 * clamp((-el_err - edge) /
                                                 std::max(fov_keep - edge, 1e-3), 0.0, 1.0));
            } else if (el_err > edge) {
                a_z = std::max(a_z, 8.0 * clamp((el_err - edge) /
                                                std::max(fov_keep - edge, 1e-3), 0.0, 1.0));
            }
        } else if (enable_fov_clamp_ && !target_behind && r_norm < 15.0) {
            const double fov_hard = fov_half_v_;
            // theta_hi (nose-down): kept conservative (fov_hard - los_el - mount_pitch). The
            // geometrically correct form (mount_pitch + fov_hard - los_el) allows large nose-down
            // and was tested -> aggressive dive on low targets -> image smear -> overshoot.
            const double theta_hi = std::min(max_tilt_rad, fov_hard - los_el - mount_pitch_);
            // theta_lo (nose-up): CORRECT form to keep the target inside the vertical FOV. boresight
            // elev = mount_pitch - theta, so boresight <= los_el + fov_keep  =>  theta >= mount_pitch
            // - fov_keep - los_el. The old (-fov_hard - los_el - mount_pitch) sign-flipped mount_pitch
            // and allowed ~25 deg nose-up; with the 15 deg up-mount the boresight rose to ~40 deg and
            // dropped a near-level target past the bottom edge at 8.7 m (3 s terminal dropout, 1.93 m
            // miss). Under rate attenuation (AirSim/HW mode) this ran away completely (UAV pitched up
            // and climbed to z=318). fov_keep (margin) leaves headroom so attitude errors don't re-lose it.
            const double fov_keep_v = fov_half_v_ * fov_keep_margin_;
            const double theta_lo = std::max(-max_tilt_rad, mount_pitch_ - fov_keep_v - los_el);
            theta_cmd = clamp(theta_cmd, std::min(theta_lo, theta_hi), std::max(theta_lo, theta_hi));
        }

        // ---- Attitude reference EMA low-pass ------------------------------------------------
        // Suppress camera-rate content / roll limit-cycle before the P-law. An EMA (vs the slew
        // governor below) is the right tool for a steady oscillation: it attenuates high-freq phi_cmd
        // swings while preserving the slow PN turn bias. Distinct from the body-rate decay coeff
        // ctrl_coeff_; this filters the attitude REFERENCE, not the rate output.
        if (has_last_att_ && att_ref_tc_ > 1e-6) {
            const double dt = t - last_att_t_;
            if (dt > 0.0 && dt < 1.0) {
                const double a_ref = clamp(dt / att_ref_tc_, 0.0, 1.0);
                phi_cmd   = last_phi_cmd_f_   + a_ref * (phi_cmd   - last_phi_cmd_f_);
                theta_cmd = last_theta_cmd_f_ + a_ref * (theta_cmd - last_theta_cmd_f_);
            }
        }
        last_phi_cmd_f_   = phi_cmd;
        last_theta_cmd_f_ = theta_cmd;

        // ---- Attitude reference slew governor ----------------------------------------------
        // Bound |d(phi_cmd)/dt| and |d(theta_cmd)/dt| so the desired attitude cannot flip +-max_tilt
        // in a single cycle. That flip is what bang-bangs omega_x/omega_y into +-max_body_rate
        // saturation (via the P-law below) and drives the roll limit-cycle at high v_des. Mirrors the
        // axial_jerk_max_ slew limit above; uses dedicated state because the axial block already
        // overwrote last_t_/has_last_ for this cycle. Does NOT change steady-state lateral authority.
        if (has_last_att_) {
            const double dt = t - last_att_t_;
            if (dt > 0.0 && dt < 1.0) {
                const double max_dphi = max_att_rate_ * dt;
                phi_cmd   = clamp(phi_cmd,   last_phi_cmd_   - max_dphi, last_phi_cmd_   + max_dphi);
                theta_cmd = clamp(theta_cmd, last_theta_cmd_ - max_dphi, last_theta_cmd_ + max_dphi);
            }
        }
        last_phi_cmd_   = phi_cmd;
        last_theta_cmd_ = theta_cmd;
        last_att_t_     = t;
        has_last_att_   = true;

        // ---- (B) Thrust: magnitude to realize the vertical component + tilt overhead --------
        // Throttle from the ACTUAL attitude, not the commanded tilt: during a hard brake the
        // attitude lags theta_cmd, and (a_z+g)/cos(commanded tilt) then over-thrusts vertically ->
        // brake-climb -> target exits the bottom of the (gimbal-horizontal) frame at 5-7 m.
        // T*cos(actual tilt) = a_z + g  =>  vertical accel tracks a_z regardless of the attitude
        // lag. Floor the divisor slightly beyond max_tilt so a transient over-bank cannot blow
        // the throttle up. Mirrors the verified Python implementation.
        const double cos_tilt_actual = std::max(std::cos(roll) * std::cos(pitch),
                                                std::cos(max_tilt_rad + 0.2));
        const double throttle = clamp((a_z + g) / cos_tilt_actual, 2.0, 25.0);

        // ---- Attitude rate commands (P on attitude error) -------------------------------
        double omega_x = clamp(K_att * (phi_cmd - roll), -max_body_rate_, max_body_rate_);
        double omega_y = clamp(K_att * (theta_cmd - pitch), -max_body_rate_, max_body_rate_);
        double omega_z = clamp(K_yaw * los_az, -max_body_rate_, max_body_rate_);
        return {throttle, omega_x, omega_y, omega_z};
    }
};

class InterceptorNodeRelativeIMMUKF : public rclcpp::Node {
public:
    InterceptorNodeRelativeIMMUKF() : Node("interceptor_node_relative_imm_ukf") {
        declare_parameters();

        auto sub_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        timer_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = sub_group;

        load_parameters();

        focal_length_ = (cam_width_ / 2.0) / std::tan((cam_fov_deg_ * M_PI / 180.0) / 2.0);

        estimator_ = std::make_unique<RelativeIMMUKFEstimator>(dt_cam_, focal_length_, w_guess_, R_b2c_, g_drag_coeff_);

        mission_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/fpv/mission_launch", 10, std::bind(&InterceptorNodeRelativeIMMUKF::mission_callback, this, std::placeholders::_1));
        tracker_trigger_sub_ = this->create_subscription<std_msgs::msg::Header>(
            "/fpv/tracker_trigger", 10, std::bind(&InterceptorNodeRelativeIMMUKF::mission_launch_callback, this, std::placeholders::_1));

        // odom_sub_ = nullptr;   // ODOMETRY-FREE: the relative-state estimator + IMU strapdown
        // IMU: SOLE ego-motion source. Orientation -> R_b2n (attitude for cam projection +
        // control); linear_acceleration -> preintegrated dv_own between camera frames (exogenous
        // a_own for the relativized process model); angular_velocity -> collision detection.
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/mavros/imu/data", rclcpp::SensorDataQoS(), std::bind(&InterceptorNodeRelativeIMMUKF::imu_callback, this, std::placeholders::_1), sub_options);

        bbox_sub_ = this->create_subscription<vision_msgs::msg::Detection2D>(
            "/fpv/target_bbox", 10, std::bind(&InterceptorNodeRelativeIMMUKF::bbox_callback, this, std::placeholders::_1), sub_options);

        // Gimbal orientation, IDENTICAL convention to InterceptorNodeBase / the hardware gimbal
        // driver / AirSim: a sensor_msgs/Imu whose orientation field is the camera-to-world
        // quaternion q_C_W (camera attitude in world/ENU). angular_velocity and orientation_covariance
        // are ignored by this node (it has its own IMM-UKF + body-IMU strapdown); they are populated
        // by the publisher for contract consistency. Only consumed when camera.use_gimbal is true.
        gimbal_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/fpv/gimbal", rclcpp::SensorDataQoS(),
            std::bind(&InterceptorNodeRelativeIMMUKF::gimbal_callback, this, std::placeholders::_1), sub_options);

        control_pub_ = this->create_publisher<quadrotor_msgs::msg::ControlCommand>("/fpv/control_command", 10);
        mission_state_pub_ = this->create_publisher<std_msgs::msg::Bool>("/fpv/mission_state", 10);

        mission_state_.set(MissionState::IDLE);
        RCLCPP_INFO(this->get_logger(), "Relative IMM-UKF node ready (ODOMETRY-FREE, IMU strapdown). IMU=/mavros/imu/data BBox=/fpv/target_bbox Launch=/fpv/tracker_trigger Control=/fpv/control_command");
    }

private:
    void declare_parameters() {
        this->declare_parameter<int>("camera.width_px", 640);
        this->declare_parameter<int>("camera.height_px", 480);
        this->declare_parameter<double>("camera.fov_deg", 60.0);
        // Semi-strapdown single-axis pitch gimbal: gimbal_bias_deg is the UNIFIED camera down-gaze
        // angle (positive = down). Used for both the gimbal (world-stabilized boresight) and the
        // fixed mount (body-relative). The gimbal rebuilds R_b2c from q_C_W each frame; the fixed
        // mount uses the initial value.
        this->declare_parameter<bool>("camera.use_gimbal", false);
        this->declare_parameter<double>("camera.gimbal_bias_deg", 5.0);
        this->declare_parameter<double>("physics.g", 9.81);
        this->declare_parameter<double>("physics.drag_coeff", 0.1625);
        this->declare_parameter<double>("controller.N", 4.5);
        this->declare_parameter<double>("controller.K_att", 9.0);
        this->declare_parameter<double>("controller.K_yaw", 5.0);
        this->declare_parameter<double>("controller.v_des", 15.0);
        this->declare_parameter<double>("physics.max_tilt_deg", 35.0);
        this->declare_parameter<double>("controller.fov_keep_margin", 0.85);
        this->declare_parameter<double>("controller.axial_jerk_max", 15.0);
        this->declare_parameter<double>("physics.max_body_rate", 3.0);
        this->declare_parameter<double>("controller.terminal_brake_range", 5.0);
        this->declare_parameter<double>("controller.terminal_brake_floor", 1.0);
        this->declare_parameter<double>("controller.terminal_range", 10.0);
        this->declare_parameter<double>("controller.terminal_N", 3.0);
        this->declare_parameter<double>("controller.v_pn_ref", 12.0);
        this->declare_parameter<double>("controller.max_att_rate", 3.5);   // Fix G: clamp |d(phi/theta)_cmd/dt| (was 5)
        this->declare_parameter<double>("controller.att_ref_tc", 0.15);
        this->declare_parameter<double>("physics.max_tilt_terminal_deg", 45.0);
        this->declare_parameter<double>("controller.terminal_tilt_range", 15.0);
        this->declare_parameter<double>("controller.terminal_decel_range", 30.0);
        this->declare_parameter<double>("controller.v_terminal", 12.0);
        this->declare_parameter<double>("controller.a_z_slew", 20.0);   // Fix C/E: |da_z/dt| limit (m/s^3)
        this->declare_parameter<bool>("controller.enable_fov_clamp", true);
        // Guidance tuning (set_guidance_tuning) — operator-facing knobs promoted from hardcoded.
        this->declare_parameter<double>("controller.K_fwd", 2.0);       // axial accel gain
        this->declare_parameter<double>("controller.K_az", 2.0);        // vertical-speed tracking gain
        this->declare_parameter<double>("physics.a_z_limit", 8.0);  // vertical accel clamp (m/s^2)
        this->declare_parameter<double>("controller.dither_amp", 0.8);  // m/s^2, lateral observability dither
        this->declare_parameter<double>("controller.dither_freq", 0.5); // Hz
        this->declare_parameter<double>("controller.dither_min_range", 10.0); // m, suppress dither inside
        this->declare_parameter<double>("controller.lateral_decay_floor", 0.4);  // Fix D: close-range a_lat floor
        this->declare_parameter<double>("controller.lateral_decay_start", 25.0); // Fix F: range lateral attenuation begins (m)
        // Vertical cascade (cascaded velocity-limited altitude track). Set v_climb/v_descent to the
        // airframe's real climb/sink capability (m/s) so the commanded rate is achievable; K_dz_up/dn
        // (1/s, closure tau=1/K) tune responsiveness independently for climb vs descent.
        this->declare_parameter<double>("physics.v_climb", 8.0);     // m/s, max commanded climb rate (cap)
        this->declare_parameter<double>("physics.v_descent", 8.0);   // m/s, max commanded sink rate (cap)
        this->declare_parameter<double>("controller.K_dz_up", 2.0);     // 1/s, climb responsiveness (upper target)
        this->declare_parameter<double>("controller.K_dz_dn", 1.0);     // 1/s, descent responsiveness (lower target; < K_up to avoid over-dive)
        this->declare_parameter<double>("estimation.w_guess", 1.5);
        this->declare_parameter<double>("estimation.vel_smooth_tc", 0.25);  // Fix B: target-vel EMA (s, 0=off)
        // Accelerometer bias in body frame, subtracted before the strapdown preintegration.
        // The Python-harness sim publishes a bias-free IMU (zeros); set calibrated values on real HW.
        this->declare_parameter<std::vector<double>>("imu_bias", {0.0, 0.0, 0.0});
        this->declare_parameter<double>("control.rate", 100.0);
        this->declare_parameter<double>("control.rate_tc", 0.02);
        this->declare_parameter<bool>("control.rate_control_attenuation", true);
        this->declare_parameter<bool>("use_simulation", false);
    }

    void load_parameters() {
        cam_width_ = this->get_parameter("camera.width_px").as_int();
        cam_height_ = this->get_parameter("camera.height_px").as_int();
        cam_fov_deg_ = this->get_parameter("camera.fov_deg").as_double();
        const bool use_simulation = this->get_parameter("use_simulation").as_bool();
        use_simulation_ = use_simulation;   // mirrors InterceptorNodeBase; gates gyro collision detect
        // FLU(body)→RDF(cam) standard extrinsic — a frame convention, not a tuning param.
        R_b2c_base_ << 0.0, -1.0, 0.0,
                       0.0,  0.0, -1.0,
                       1.0,  0.0,  0.0;
        g_ = this->get_parameter("physics.g").as_double();
        g_drag_coeff_ = this->get_parameter("physics.drag_coeff").as_double();
        use_gimbal_ = this->get_parameter("camera.use_gimbal").as_bool();
        gimbal_bias_rad_ = this->get_parameter("camera.gimbal_bias_deg").as_double() * M_PI / 180.0;
        // gimbal_bias_deg (positive = down-gaze) is the unified camera tilt for BOTH the gimbal
        // (world-stabilized) and the fixed mount (body-relative). The gimbal rebuilds R_b2c_ from
        // q_C_W each frame; the fixed mount keeps this initial value.
        R_b2c_ = gimbal_R_b2c(gimbal_bias_rad_);

        w_guess_ = this->get_parameter("estimation.w_guess").as_double();
        control_rate_ = this->get_parameter("control.rate").as_double();
        const double rate_tc = this->get_parameter("control.rate_tc").as_double();
        const bool rate_attenuation = this->get_parameter("control.rate_control_attenuation").as_bool();
        // First-order low-pass coefficient on body-rate output (mirrors InterceptorNodeBase).
        // EMA form: out_new = out_old + ctrl_coeff_ * (raw - out_old). Attenuates high-frequency
        // cmd jumps from estimator/guidance noise before they reach the attitude controller.
        if (!use_simulation && rate_attenuation && rate_tc > 1e-6 && control_rate_ > 1e-6) {
            ctrl_coeff_ = 1.0 - std::exp(-1.0 / (control_rate_ * rate_tc));
        } else {
            ctrl_coeff_ = 1.0;
        }
        double N = this->get_parameter("controller.N").as_double();
        double K_att = this->get_parameter("controller.K_att").as_double();
        double K_yaw = this->get_parameter("controller.K_yaw").as_double();
        double v_des = this->get_parameter("controller.v_des").as_double();
        double max_tilt = this->get_parameter("physics.max_tilt_deg").as_double();
        double fov_keep = this->get_parameter("controller.fov_keep_margin").as_double();
        double jerk = this->get_parameter("controller.axial_jerk_max").as_double();
        double max_body_rate = this->get_parameter("physics.max_body_rate").as_double();
        double terminal_brake_range = this->get_parameter("controller.terminal_brake_range").as_double();
        double terminal_brake_floor = this->get_parameter("controller.terminal_brake_floor").as_double();
        double v_pn_ref = this->get_parameter("controller.v_pn_ref").as_double();
        double max_att_rate = this->get_parameter("controller.max_att_rate").as_double();
        double att_ref_tc = this->get_parameter("controller.att_ref_tc").as_double();
        double max_tilt_terminal_deg = this->get_parameter("physics.max_tilt_terminal_deg").as_double();
        double terminal_tilt_range = this->get_parameter("controller.terminal_tilt_range").as_double();
        double terminal_decel_range = this->get_parameter("controller.terminal_decel_range").as_double();
        double v_terminal = this->get_parameter("controller.v_terminal").as_double();
        double fov_half_v = (cam_height_ / 2.0) / std::max(1.0, focal_length_);
        double fov_half_h = (cam_width_ / 2.0) / std::max(1.0, focal_length_);
        // Compute terminal_bbox_size from camera intrinsics so the brake triggers at the correct
        // range regardless of focal length. bbox_size = focal * W / range, so at terminal_brake_range:
        // terminal_bbox_size = focal * w_guess / terminal_brake_range. For AirSim (focal=831, W=1.5,
        // range=5): 249px. For Python sim (focal=554, W=1.5, range=5): 166px. Previously hardcoded
        // 80px which triggered full brake at ~15m in AirSim (bbox=83px at R=15), stalling the UAV.
        const double terminal_bbox_size = focal_length_ * w_guess_ / std::max(0.1, terminal_brake_range);

        const bool enable_fov_clamp = this->get_parameter("controller.enable_fov_clamp").as_bool();
        controller_ = std::make_unique<AdvancedHybridController>(
            N, K_att, K_yaw, v_des, max_tilt,
            fov_half_v, fov_half_h, fov_keep, jerk, enable_fov_clamp, gimbal_bias_rad_,
            max_body_rate, terminal_brake_range, terminal_brake_floor, terminal_bbox_size,
            v_pn_ref, max_att_rate, att_ref_tc,
            max_tilt_terminal_deg, terminal_tilt_range, terminal_decel_range, v_terminal);
        controller_->set_a_z_slew(this->get_parameter("controller.a_z_slew").as_double());
        controller_->set_lateral_decay_floor(this->get_parameter("controller.lateral_decay_floor").as_double());
        controller_->set_lat_decay_start(this->get_parameter("controller.lateral_decay_start").as_double());
        controller_->set_gimbal(use_gimbal_, gimbal_bias_rad_);
        controller_->set_g(g_);
        controller_->set_vertical_climb(
            this->get_parameter("physics.v_climb").as_double(),
            this->get_parameter("physics.v_descent").as_double(),
            this->get_parameter("controller.K_dz_up").as_double(),
            this->get_parameter("controller.K_dz_dn").as_double());
        controller_->set_guidance_tuning(
            this->get_parameter("controller.K_fwd").as_double(),
            this->get_parameter("controller.K_az").as_double(),
            this->get_parameter("physics.a_z_limit").as_double(),
            this->get_parameter("controller.dither_amp").as_double(),
            this->get_parameter("controller.dither_freq").as_double(),
            this->get_parameter("controller.dither_min_range").as_double());
        vel_smooth_tc_ = this->get_parameter("estimation.vel_smooth_tc").as_double();
        const std::vector<double> bias = this->get_parameter("imu_bias").as_double_array();
        if (bias.size() == 3) {
            imu_bias_ = Eigen::Vector3d(bias[0], bias[1], bias[2]);
        }

    }

    void reset_estimator() {
        estimator_ = std::make_unique<RelativeIMMUKFEstimator>(dt_cam_, focal_length_, w_guess_, R_b2c_, g_drag_coeff_);
    }

    void reset_control_cache() {
        std::lock_guard<std::mutex> cache_lock(control_cache_mtx_);
        control_cache_.valid = false;
    }

    void start_control_timer() {
        if (control_timer_) return;
        const auto period = std::chrono::duration<double>(1.0 / control_rate_);
        control_timer_ = this->create_wall_timer(
            period, std::bind(&InterceptorNodeRelativeIMMUKF::control_timer_callback, this), timer_callback_group_);
    }

    void bbox_callback(const vision_msgs::msg::Detection2D::SharedPtr msg) {
        if (!mission_state_.is_active()) {
            return;
        }

        std::lock_guard<std::mutex> lock(estimator_mtx_);

        if (last_imu_time_ < 0.0 || !yaw_align_init_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Waiting for IMU to initialize...");
            return;
        }

        const double t_now = rclcpp::Time(msg->header.stamp).seconds();

        // Adaptive dt_cam_: measure the actual inter-bbox interval from timestamps. Initial value
        // is the 30Hz guess (0.033); after the first few frames dt_cam_ tracks the real camera rate.
        // Used by the estimator's dropout predict and the control cache's a_own = dv_own/dt_cam_.
        if (last_bbox_cam_t_ > 0.0 && t_now > last_bbox_cam_t_) {
            const double dt_measured = t_now - last_bbox_cam_t_;
            if (dt_measured > 1e-3 && dt_measured < 0.2) {   // sanity: 5 Hz .. 1000 Hz
                dt_cam_ = dt_measured;
                if (estimator_) estimator_->dt_cam = dt_measured;
            }
        }
        last_bbox_cam_t_ = t_now;

        const bool has_bbox = !msg->header.frame_id.empty();
        const double u = msg->bbox.center.position.x - cam_width_ / 2.0;
        const double v = msg->bbox.center.position.y - cam_height_ / 2.0;
        const double s = std::max(msg->bbox.size_x, msg->bbox.size_y);

        Eigen::Matrix3d R_b2n;
        Eigen::Vector3d dv_own;
        Eigen::Quaterniond gimbal_q_now = Eigen::Quaterniond::Identity();
        bool gimbal_rcvd = false;
        {
            std::lock_guard<std::mutex> ol(odom_mtx_);
            R_b2n = latest_R_b2n_;
            gimbal_q_now = latest_gimbal_q_;
            gimbal_rcvd = gimbal_received_;
            // Drain the IMU-preintegration window for this camera frame: dv_own is the ownship
            // velocity change since the last camera frame (world frame G, gravity-corrected).
            dv_own = dv_own_accum_;
            dv_own_accum_.setZero();
            last_dv_own_ = dv_own;
        }

        // Semi-strapdown gimbal: rebuild R_b2c from the ACTUAL (lagged) camera-to-world
        // orientation q_C_W so the measurement model and the LOS reconstruction match the physical
        // camera orientation at capture time. The orientation is measured in real time, so lag is
        // fully accounted for.
        //
        // NOTE: in gimbal mode R_b2c_ is a SYNTHETIC matrix, not a true body->camera rotation. Its
        // only purpose is to make the estimator's R_n2c = R_b2c*R_b2n^T and pixel_to_los_nav's
        // R_b2n*R_b2c^T*ray reproduce the camera projection from q_C_W: with R_b2c_ = q_C_W^T * R_b2n,
        // R_n2c = q_C_W^T and LOS = q_C_W*ray_cam. The same local R_b2n read here is passed to
        // process_camera below (so the two cancel exactly). Until the first /fpv/gimbal msg arrives,
        // fall back to the bias-derived body->camera mount (R_n2c then includes attitude via R_b2n^T).
        if (use_gimbal_) {
            if (gimbal_rcvd) {
                R_b2c_ = gimbal_q_now.normalized().toRotationMatrix().transpose() * R_b2n;
            } else {
                R_b2c_ = gimbal_R_b2c(gimbal_bias_rad_);
            }
            estimator_->R_b2c = R_b2c_;
        }

        const bool was_initialized = estimator_->initialized;
        const auto result = estimator_->process_camera(has_bbox, u, v, s, t_now, R_b2n, dv_own);

        if (!was_initialized && estimator_->initialized) {
            transition_to_state(MissionState::TRACKING, "Target initialized");
        }
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS,
            "bbox=%d size=%.1f init=%d->%d", has_bbox, s, was_initialized, estimator_->initialized);

        if (!estimator_->initialized) return;

        // Relative-state estimator: x[0:3] IS the relative position, x[3:6] the relative velocity.
        Eigen::Vector3d r_est = result.second.segment<3>(0);
        Eigen::Vector3d rel_vel_now = result.second.segment<3>(3);
        // Output smoothing of the relative velocity (Fix B): EMA the raw UKF v_rel so single-frame
        // swings do not reach guidance. vel_smooth_tc_<=0 disables it.
        if (vel_smooth_tc_ > 1e-6) {
            if (has_target_vel_smoothed_ && t_now > last_vel_smooth_t_ &&
                (t_now - last_vel_smooth_t_) < 1.0) {
                const double a = clamp((t_now - last_vel_smooth_t_) / vel_smooth_tc_, 0.0, 1.0);
                target_vel_smoothed_ += a * (rel_vel_now - target_vel_smoothed_);
            } else {
                target_vel_smoothed_ = rel_vel_now;
            }
            last_vel_smooth_t_ = t_now;
            has_target_vel_smoothed_ = true;
            rel_vel_now = target_vel_smoothed_;
        }
        const Eigen::Vector3d v_est = rel_vel_now;   // relative velocity (no ownship vel available)

        double los_az_world = std::atan2(r_est(1), r_est(0));
        double los_el = std::asin(clamp(r_est(2) / std::max(r_est.norm(), 0.1), -1.0, 1.0));

        if (has_bbox) {
            // bbox is real-time: capture orientation == current orientation.
            const Eigen::Vector3d los_nav = pixel_to_los_nav(u, v, focal_length_, R_b2n, R_b2c_);
            los_az_world = std::atan2(los_nav(1), los_nav(0));
            los_el = std::asin(clamp(los_nav(2), -1.0, 1.0));
        }

        const double yaw = std::atan2(R_b2n(1, 0), R_b2n(0, 0));
        double los_az = los_az_world - yaw;
        while (los_az > M_PI) los_az -= 2.0 * M_PI;
        while (los_az < -M_PI) los_az += 2.0 * M_PI;

        // Smoothed inertial LOS bearing rate (lambda_dot) for PN lateral lead. Equals omega_los(2)
        // algebraically, but derived from the observed bearing so it does not inherit v_est's
        // close-range noise nor blow up as 1/r. EMA over ~3 camera frames suppresses detector
        // jitter. Updated at the camera/detection rate; held between updates.
        if (has_prev_los_) {
            double daz = los_az_world - prev_los_az_world_;
            while (daz > M_PI) daz -= 2.0 * M_PI;
            while (daz < -M_PI) daz += 2.0 * M_PI;
            const double dlos_t = t_now - prev_los_t_;
            if (dlos_t > 1e-3 && dlos_t < 1.0) {
                const double raw_rate = daz / dlos_t;
                const double alpha_los = 0.25;
                los_rate_ema_ = (1.0 - alpha_los) * los_rate_ema_ + alpha_los * raw_rate;
            }
        }
        prev_los_az_world_ = los_az_world;
        prev_los_t_ = t_now;
        has_prev_los_ = true;

        ControlCacheData cache;
        cache.valid = true;
        cache.has_bbox = has_bbox;
        cache.r_est = r_est;
        cache.v_est = v_est;
        cache.los_az_world = los_az_world;
        cache.los_el = los_el;
        cache.los_rate = los_rate_ema_;
        cache.source_time = t_now;
        cache.update_time = this->get_clock()->now().seconds();
        cache.bbox_size = s;
        // Relative-speed proxy (ownship absolute speed unobservable without odom): drives the
        // speed-aware terminal-brake floor. Mirror of the Python prototype's target_speed.
        cache.target_speed = std::min(rel_vel_now.norm(), 15.0);
        // Ownship world accel over the last camera window (for the controller's geometric APN).
        cache.a_own = last_dv_own_ / std::max(dt_cam_, 1e-3);

        {
            std::lock_guard<std::mutex> cache_lock(control_cache_mtx_);
            control_cache_ = cache;
        }

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS,
            "rel=[%.2f %.2f %.2f] vel=[%.2f %.2f %.2f] w_est=%.2f r_meas=%.1f r_est=%.1f",
            r_est(0), r_est(1), r_est(2), v_est(0), v_est(1), v_est(2),
            estimator_->imm_ukf->x(9),
            (has_bbox && s > 0.1) ? (focal_length_ * estimator_->w_guess / s) : 0.0,
            r_est.norm());

        // IMM diagnostic: mode probabilities, per-filter vel, and key P diagonals. Helps locate
        // which motion model is driving the vel jumps when the estimate oscillates.
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS*4,
            "IMM mu=[%.2f %.2f %.2f] v0=[%.1f %.1f %.1f] v1=[%.1f %.1f %.1f] v2=[%.1f %.1f %.1f] Pd=[%.1f %.1f %.1f %.1f]",
            estimator_->imm_ukf->mu(0), estimator_->imm_ukf->mu(1), estimator_->imm_ukf->mu(2),
            estimator_->imm_ukf->filters[0].x(3), estimator_->imm_ukf->filters[0].x(4), estimator_->imm_ukf->filters[0].x(5),
            estimator_->imm_ukf->filters[1].x(3), estimator_->imm_ukf->filters[1].x(4), estimator_->imm_ukf->filters[1].x(5),
            estimator_->imm_ukf->filters[2].x(3), estimator_->imm_ukf->filters[2].x(4), estimator_->imm_ukf->filters[2].x(5),
            estimator_->imm_ukf->P(0,0), estimator_->imm_ukf->P(3,3), estimator_->imm_ukf->P(6,6), estimator_->imm_ukf->P(9,9));
    }

    void control_timer_callback() {
        if (!mission_state_.is_active() || !controller_) return;

        ControlCacheData cache;
        {
            std::lock_guard<std::mutex> lock(control_cache_mtx_);
            cache = control_cache_;
        }
        if (!cache.valid) return;

        const rclcpp::Time now = this->get_clock()->now();
        const double stale_limit = std::max(2.5, 2.0 / std::max(1.0, control_rate_));
        if (now.seconds() - cache.update_time > stale_limit) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Skipping stale control cache: age=%.3f s", now.seconds() - cache.update_time);
            return;
        }

        Eigen::Matrix3d R_b2n;
        Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
        Eigen::Quaterniond gimbal_q_now = Eigen::Quaterniond::Identity();
        bool gimbal_rcvd = false;
        Eigen::Vector3d own_vel_now = Eigen::Vector3d::Zero();
        double t_now = 0.0;
        double sim_start_time = -1.0;
        {
            std::lock_guard<std::mutex> odom_lock(odom_mtx_);
            R_b2n = latest_R_b2n_;
            gyro = latest_gyro_;
            gimbal_q_now = latest_gimbal_q_;
            gimbal_rcvd = gimbal_received_;
            own_vel_now = own_vel_;
            t_now = (last_imu_time_ > 0.0) ? last_imu_time_ : now.seconds();
            // Anchor mission-relative time on the first control tick: the controller's jerk/slew/
            // EMA states and the observability DITHER (sin(2*pi*f*t)) all need a running t.
            if (sim_start_time_ < 0.0) {
                sim_start_time_ = t_now;
            }
            sim_start_time = sim_start_time_;
        }
        // Actual camera boresight world elevation from the measured q_C_W (camera->world): the
        // camera forward axis is +Z (RDF), so its world z-component is sin(boresight_elev). Fed to
        // the controller's vertical-FOV clamp so it tracks the TRUE boresight (incl. servo lag).
        // NaN when the gimbal is disabled or no /fpv/gimbal msg has arrived yet (clamp then falls
        // back to the -gimbal_bias_ assumption).
        double boresight_el_meas = std::numeric_limits<double>::quiet_NaN();
        if (use_gimbal_ && gimbal_rcvd) {
            const Eigen::Matrix3d R_c2n = gimbal_q_now.normalized().toRotationMatrix();
            boresight_el_meas = std::asin(clamp(R_c2n(2, 2), -1.0, 1.0));
        }

        const double roll = std::atan2(R_b2n(2, 1), R_b2n(2, 2));
        const double pitch = std::atan2(-R_b2n(2, 0), std::sqrt(R_b2n(2, 1) * R_b2n(2, 1) + R_b2n(2, 2) * R_b2n(2, 2)));
        const double yaw = std::atan2(R_b2n(1, 0), R_b2n(0, 0));
        // Closing speed along LOS from relative quantities only (no ownship vel):
        // v_closing = -d|r|/dt = -v_rel·r̂ (positive = approaching). Replaces odom v_fwd_body.
        const double v_closing = cache.valid
            ? -cache.v_est.dot(cache.r_est) / std::max(cache.r_est.norm(), 1e-3)
            : 0.0;
        double los_az = cache.los_az_world - yaw;
        los_az = std::atan2(std::sin(los_az), std::cos(los_az));
        const double t_rel = (sim_start_time >= 0.0) ? t_now - sim_start_time : 0.0;

        // Body-rate feedback via finite difference of R_b2n (for attitude-loop diagnostics).
        // omega_body ≈ R_b2n^T * dR/dt  →  extract roll/pitch/yaw rates from Euler angle diffs.
        double rate_roll = 0.0, rate_pitch = 0.0, rate_yaw = 0.0;
        if (has_prev_attitude_ && t_now > prev_attitude_t_ && (t_now - prev_attitude_t_) < 0.2) {
            const double dt = t_now - prev_attitude_t_;
            const double prev_roll = std::atan2(prev_R_b2n_(2, 1), prev_R_b2n_(2, 2));
            const double prev_pitch = std::atan2(-prev_R_b2n_(2, 0), std::sqrt(prev_R_b2n_(2, 1) * prev_R_b2n_(2, 1) + prev_R_b2n_(2, 2) * prev_R_b2n_(2, 2)));
            const double prev_yaw = std::atan2(prev_R_b2n_(1, 0), prev_R_b2n_(0, 0));
            rate_roll = (roll - prev_roll) / dt;
            rate_pitch = (pitch - prev_pitch) / dt;
            double dyaw = yaw - prev_yaw;
            while (dyaw > M_PI) dyaw -= 2.0 * M_PI;
            while (dyaw < -M_PI) dyaw += 2.0 * M_PI;
            rate_yaw = dyaw / dt;
        }
        prev_R_b2n_ = R_b2n;
        prev_attitude_t_ = t_now;
        has_prev_attitude_ = true;

        // Collision detection (gyro spike) — AirSim / real HW only (mirror InterceptorNodeBase).
        // The ROS2 Python sim (use_simulation_=true) has ground-truth range and judges the intercept
        // externally (ros2_sim_node.py), so the gyro check is skipped there. On contact the body rate
        // spikes well beyond what was commanded -> declare COMPLETED (intercept success) and stop.
        if (!use_simulation_ && !collision_detected_) {
            const Eigen::Vector3d rate_error = gyro - last_cmd_;
            const double rate_error_norm = rate_error.norm();
            if (rate_error_norm > 4.0) {
                collision_detected_ = true;
                RCLCPP_WARN(this->get_logger(),
                    "[COLLISION] Gyro spike! |err|=%.1f rad/s gyro=[%.2f,%.2f,%.2f] "
                    "cmd=[%.2f,%.2f,%.2f]",
                    rate_error_norm,
                    gyro.x(), gyro.y(), gyro.z(),
                    last_cmd_.x(), last_cmd_.y(), last_cmd_.z());
                transition_to_state(MissionState::COMPLETED, "Collision detected via gyro");
                stop();
                return;
            }
        }

        // Capture lockout: latch true once the UAV has intercepted, then BRAKE-TO-HOLD for the rest
        // of the mission. Resuming normal guidance after a close approach — bbox flickered back or
        // dead-reckon chased a reacquired target — flew the UAV 70-250 m off after an otherwise-clean
        // intercept at v_des=20. Two reliable triggers (the close approach happens during a bbox
        // dropout, when the range ESTIMATE is biased and useless, so do NOT key off r_est alone):
        //   (1) dead-reckoned range < capture_range (only fires if the estimate happens to be good), OR
        //   (2) target now BEHIND the UAV (|los_az|>90°, observable) after being close at the last
        //       detection (last_good_range < coast_range) — i.e. the UAV flew past the target.
        constexpr double capture_range = 2.0;
        const bool overshot = std::abs(los_az) > (M_PI / 2.0);
        if (cache.valid) {
            min_range_seen_ = std::min(min_range_seen_, cache.r_est.norm());
        }
        // Reopen: range bottomed out and has since grown past it — the UAV flew through the target.
        // Robust under a noisy close-range estimate (high-noise scenario) where los_az never cleanly
        // flips past 90 deg. Only arms after a real close approach (min_range_seen < 15 m).
        const bool range_reopened = has_frozen_cmd_ && min_range_seen_ < 15.0 &&
            cache.valid && cache.r_est.norm() > min_range_seen_ + 8.0;
        if (!captured_ && cache.valid &&
            (cache.r_est.norm() < capture_range ||
             (overshot && has_frozen_cmd_ && last_good_range_ < 12.0) ||
             range_reopened)) {
            captured_ = true;
            // Seed the brake-scoped ownship velocity: the estimator's v_closing flips sign once
            // the dead-reckoned r_est passes behind (fly-through), so braking on it runs AWAY
            // (positive feedback). Instead, take the ownship velocity at capture ≈ last good
            // closing speed along the current heading, then track it by IMU strapdown integration
            // (imu_callback) — the brake then always opposes the REAL ownship velocity.
            {
                std::lock_guard<std::mutex> ol(odom_mtx_);
                const double v_seed = std::max(v_closing, 0.0);
                v_own_brake_ = Eigen::Vector3d(v_seed * std::cos(yaw), v_seed * std::sin(yaw), 0.0);
                brake_vel_seeded_ = true;
            }
            RCLCPP_INFO(this->get_logger(),
                "CAPTURE: holding (brake-to-hold) to end of mission (r_est=%.1fm los_az=%.0fdeg v_seed=%.1f)",
                cache.r_est.norm(), los_az * 180.0 / M_PI, std::max(v_closing, 0.0));
        }
        if (captured_) {
            // Brake-to-hold on the strapdown brake velocity (NOT the estimator v_closing — its
            // sign is unreliable once the target is behind): nose-up pitch cancels the forward
            // component, roll cancels the lateral component, tilt-compensated hover thrust holds
            // altitude. Both angles -> 0 as the velocity -> 0, settling into a level hover.
            const double g_hover = g_;
            constexpr double max_brake_tilt = 0.436;   // 25 deg
            Eigen::Vector3d v_brake;
            {
                std::lock_guard<std::mutex> ol(odom_mtx_);
                v_brake = v_own_brake_;
            }
            const double v_fwd_brake = std::cos(yaw) * v_brake(0) + std::sin(yaw) * v_brake(1);
            const double v_lat_brake = -std::sin(yaw) * v_brake(0) + std::cos(yaw) * v_brake(1);
            const double theta_brake = clamp(-0.12 * v_fwd_brake, -max_brake_tilt, max_brake_tilt);
            const double phi_brake = clamp(0.12 * v_lat_brake, -max_brake_tilt, max_brake_tilt);
            const double tilt = std::sqrt(roll * roll + pitch * pitch);
            auto cmd = std::make_unique<quadrotor_msgs::msg::ControlCommand>();
            cmd->header.stamp = now;
            cmd->armed = true;
            cmd->control_mode = quadrotor_msgs::msg::ControlCommand::BODY_RATES;
            cmd->collective_thrust = g_hover / std::max(std::cos(tilt), 0.3);
            cmd->bodyrates.x = clamp(5.0 * (phi_brake - roll), -3.0, 3.0) * ctrl_coeff_;
            cmd->bodyrates.y = clamp(5.0 * (theta_brake - pitch), -3.0, 3.0) * ctrl_coeff_;
            cmd->bodyrates.z = 0.0;
            last_cmd_ = Eigen::Vector3d(cmd->bodyrates.x, cmd->bodyrates.y, cmd->bodyrates.z);
            control_pub_->publish(std::move(cmd));
            return;
        }

        // Terminal dropout handling. If the bbox drops inside the terminal phase, the dead-reckoned
        // r_est (especially r_est(2), the vertical) is unreliable and chasing it is what dived the UAV
        // at a phantom target (AirSim: estimate diverged 11 m below after a close-range reacquire) and,
        // at v_des=20, flew it from 9 m out to 39 m past the target. Two phases:
        //  (1) COAST (first coast_max_dur): freeze the last good cmd and fly straight through the last
        //      known line — it carries the correct thrust for the current (banked) attitude, so no sink,
        //      and a close intercept can complete / the detector can reacquire.
        //  (2) LEVEL-AND-HOLD (past coast): actively level the body + hold altitude. Zero body rate on a
        //      BANKED body sinks (thrust tilts off-vertical) — that was the v_des=20 dive. Drive
        //      roll/pitch->0 with a tilt-compensated hover thrust; forward velocity coasts the UAV
        //      straight (level) toward the last bearing instead of curving off chasing a dead-reckon
        //      phantom. Never fall through to dead-reckon guidance at close range.
        const double coast_range = 12.0;
        const double coast_max_dur = 1.5;
        const double dropout_age = now.seconds() - last_bbox_time_;
        const bool terminal_dropout = !cache.has_bbox && has_frozen_cmd_ &&
            (last_good_range_ < coast_range);
        if (terminal_dropout) {
            auto cmd = std::make_unique<quadrotor_msgs::msg::ControlCommand>();
            cmd->header.stamp = now;
            cmd->armed = true;
            cmd->control_mode = quadrotor_msgs::msg::ControlCommand::BODY_RATES;
            if (dropout_age < coast_max_dur) {
                // COAST: reproduce the exact last-published attitude/thrust (straight-line flight).
                cmd->collective_thrust = frozen_thrust_;
                cmd->bodyrates.x = frozen_rate_x_;
                cmd->bodyrates.y = frozen_rate_y_;
                cmd->bodyrates.z = frozen_rate_z_;
            } else {
                // LEVEL-AND-HOLD: leveling rates + tilt-compensated hover thrust.
                const double g_hover = g_;
                const double tilt = std::sqrt(roll * roll + pitch * pitch);
                cmd->collective_thrust = g_hover / std::max(std::cos(tilt), 0.3);
                cmd->bodyrates.x = clamp(-3.0 * roll, -3.0, 3.0) * ctrl_coeff_;
                cmd->bodyrates.y = clamp(-3.0 * pitch, -3.0, 3.0) * ctrl_coeff_;
                cmd->bodyrates.z = 0.0;
            }
            last_cmd_ = Eigen::Vector3d(cmd->bodyrates.x, cmd->bodyrates.y, cmd->bodyrates.z);
            control_pub_->publish(std::move(cmd));
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS/2,
                "%s (terminal dropout): age=%.2fs last_good_range=%.1fm",
                (dropout_age < coast_max_dur) ? "COAST" : "LEVELHOLD", dropout_age, last_good_range_);
            return;
        }

        // Vertical damping from the IMU-integrated ownship vertical velocity (robust; the
        // monocular v_est(2) used before is poorly observable and left the vertical PD to over-dive
        // on below-FOV targets — own_vel_(2) is the actual own climb/sink rate).
        const double v_z = own_vel_now(2);
        const auto ctrl = controller_->compute_control(roll, pitch, yaw, cache.r_est, cache.v_est,
                                                       los_az, cache.los_el, t_rel, v_closing,
                                                       cache.bbox_size, v_z, cache.target_speed,
                                                       cache.los_rate, cache.a_own, boresight_el_meas);
        if (!std::isfinite(ctrl[0]) || !std::isfinite(ctrl[1]) || !std::isfinite(ctrl[2]) || !std::isfinite(ctrl[3])) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS*4,
                "Skipping non-finite control command");
            return;
        }

        auto cmd = std::make_unique<quadrotor_msgs::msg::ControlCommand>();
        cmd->header.stamp = now;
        cmd->armed = true;
        cmd->control_mode = quadrotor_msgs::msg::ControlCommand::BODY_RATES;
        cmd->collective_thrust = ctrl[0];
        // EMA low-pass on body rates: out_new = out_old + ctrl_coeff_*(raw - out_old).
        // Attenuates high-frequency cmd jumps (from estimator/guidance noise) before the attitude
        // controller, suppressing pitch/roll/yaw oscillation. Mirrors InterceptorNodeBase logic.
        // Rate scaling: ctrl_coeff_ scales the raw body-rate command. This acts as a
        // fixed gain reduction (~0.39 with rate_tc=0.02) that damps the attitude loop.
        // Correct for AirSim/real hardware (inner attitude loop adds gain). For the Python
        // sim (bodyrates execute directly) this under-powers the UAV, but retuning K_att
        // for EMA full authority is a separate task. Keep gain form as the stable baseline.
        cmd->bodyrates.x = ctrl_coeff_ * ctrl[1];
        cmd->bodyrates.y = ctrl_coeff_ * ctrl[2];
        cmd->bodyrates.z = ctrl_coeff_ * ctrl[3];
        last_cmd_ = Eigen::Vector3d(cmd->bodyrates.x, cmd->bodyrates.y, cmd->bodyrates.z);
        // Capture the last good (detected) command for terminal coast-on-dropout. Stored AFTER the
        // EMA low-pass so coast reproduces the exact last-published attitude/thrust (straight-line
        // flight) rather than a raw guidance jump.
        if (cache.has_bbox) {
            last_bbox_time_ = cache.update_time;
            last_good_range_ = cache.r_est.norm();
            frozen_thrust_ = cmd->collective_thrust;
            frozen_rate_x_ = cmd->bodyrates.x;
            frozen_rate_y_ = cmd->bodyrates.y;
            frozen_rate_z_ = cmd->bodyrates.z;
            has_frozen_cmd_ = true;
        }
        control_pub_->publish(std::move(cmd));

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS,
            "ctrl rel=[%.2f %.2f %.2f] vel=[%.2f %.2f %.2f] cmd=[%.2f %.2f %.2f %.2f] pitch=%.2f los_el=%.2f",
            cache.r_est(0), cache.r_est(1), cache.r_est(2), cache.v_est(0), cache.v_est(1), cache.v_est(2),
            ctrl[0], ctrl[1], ctrl[2], ctrl[3], pitch, cache.los_el);

        // Attitude-loop diagnostic: actual Euler angles + measured body rates (finite-differenced)
        // vs commanded body rates. Reveals whether oscillation is from cmd jumps (estimator/guidance)
        // or from P-gain too high (cmd smooth but rate overshoots).
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS,
            "att rpy=[%.3f %.3f %.3f] rate=[%.2f %.2f %.2f] cmd_rate=[%.2f %.2f %.2f]",
            roll, pitch, yaw, rate_roll, rate_pitch, rate_yaw, ctrl[1], ctrl[2], ctrl[3]);
    }

    void mission_launch_callback(const std_msgs::msg::Header::SharedPtr msg) {
        if (msg->frame_id == "OFF" && mission_state_.is_active()) {
            RCLCPP_INFO(this->get_logger(), "Mission abort received! Stopping control...");
            transition_to_state(MissionState::ABORTED, "User abort command");
            stop();
            return;
        }

        if (!mission_state_.is_terminated()) {
            return;
        }

        if (msg->frame_id == "MISSION") {
            RCLCPP_INFO(this->get_logger(), "Mission launch received! Starting control...");
            std::unique_lock<std::mutex> lock(estimator_mtx_);
            reset_estimator();
            reset_control_cache();
            if (controller_) controller_->reset_dynamic_state();
            {
                std::lock_guard<std::mutex> ol(odom_mtx_);
                dv_own_accum_.setZero();
                last_dv_own_.setZero();
                last_imu_time_ = -1.0;
                sim_start_time_ = -1.0;
                gimbal_received_ = false;
                latest_gimbal_q_ = Eigen::Quaterniond::Identity();
                latest_gyro_ = Eigen::Vector3d::Zero();
                latest_R_b2n_ = Eigen::Matrix3d::Identity();
                last_bbox_cam_t_ = -1.0;
                dt_cam_ = 0.033;
            }
            last_cmd_ = Eigen::Vector3d::Zero();
            has_target_vel_smoothed_ = false;
            has_prev_attitude_ = false;
            has_prev_los_ = false;
            los_rate_ema_ = 0.0;
            prev_los_t_ = -1.0;
            has_frozen_cmd_ = false;
            last_good_range_ = 1e9;
            last_bbox_time_ = 0.0;
            captured_ = false;
            min_range_seen_ = 1e9;
            collision_detected_ = false;
            yaw_align_init_ = false;
            {
                std::lock_guard<std::mutex> ol(odom_mtx_);
                v_own_brake_.setZero();
                own_vel_.setZero();
                brake_vel_seeded_ = false;
            }
            transition_to_state(MissionState::INITIALIZING, "Mission launch command");
            publish_mission_state(true);
            start_control_timer();
            RCLCPP_INFO(this->get_logger(), "Mission started: odometry tracking active");
        }
    }

    void mission_callback(const std_msgs::msg::Bool::SharedPtr msg) {
        auto header = std::make_shared<std_msgs::msg::Header>();
        header->stamp = this->get_clock()->now();
        header->frame_id = msg->data ? "MISSION" : "OFF";
        mission_launch_callback(header);
    }

    // IMU: sole ego-motion input (ODOMETRY-FREE). Orientation -> attitude R_b2n; angular velocity
    // -> collision detection; linear acceleration -> strapdown preintegration of the ownship
    // velocity change dv_own = ∫(R_b2n·(f_imu − bias) − g)dτ over the current camera window.
    // The accumulator is drained by bbox_callback each camera frame; it is a pure RELATIVE
    // quantity per window, so there is no long-term velocity drift.
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        const double t = rclcpp::Time(msg->header.stamp).seconds();
        std::lock_guard<std::mutex> odom_lock(odom_mtx_);
        latest_gyro_(0) = msg->angular_velocity.x;
        latest_gyro_(1) = msg->angular_velocity.y;
        latest_gyro_(2) = msg->angular_velocity.z;
        const Eigen::Quaterniond q(msg->orientation.w, msg->orientation.x,
                                   msg->orientation.y, msg->orientation.z);
        if (q.norm() > 1e-6) {
            latest_R_b2n_ = q.normalized().toRotationMatrix();
        }
        if (last_imu_time_ > 0.0 && t > last_imu_time_) {
            const double dt = std::min(t - last_imu_time_, 0.05);   // guard against stamp gaps
            const Eigen::Vector3d f_body(msg->linear_acceleration.x - imu_bias_(0),
                                         msg->linear_acceleration.y - imu_bias_(1),
                                         msg->linear_acceleration.z - imu_bias_(2));
            const Eigen::Vector3d a_world = latest_R_b2n_ * f_body - Eigen::Vector3d(0.0, 0.0, g_);
            dv_own_accum_ += a_world * dt;
            own_vel_ += a_world * dt;   // always-on ownship vel (world) for vertical damping
            // Brake-scoped ownship velocity (capture hold only): seeded from the last good
            // v_closing at capture, then propagated by pure strapdown integration. Short-horizon
            // (seconds), so accel-noise drift is negligible.
            if (brake_vel_seeded_) {
                v_own_brake_ += a_world * dt;
            }
        }
        last_imu_time_ = t;
        yaw_align_init_ = true;
    }

    // /fpv/gimbal (Imu): the orientation field is q_C_W in the FLU camera frame (x=fwd, y=left,
    // z=up) — the SAME convention as InterceptorNodeBase/DroneEKF and airsim_tracker, so the
    // base/quaternion/DQ nodes consume /fpv/gimbal unchanged. This node's estimator uses the RDF
    // optical frame internally (x=right, y=down, z=fwd), so convert FLU->RDF at intake:
    //   q_C_W_rdf = q_C_W_flu * R(R_b2c_base_^T),  since (FLU->ENU)*(RDF->FLU) = RDF->ENU.
    // R_b2c_base_ is the FLU(body)->RDF(cam) extrinsic. angular_velocity / orientation_covariance
    // are NOT consumed here — this node runs its own IMM-UKF + body-IMU strapdown.
    void gimbal_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        const Eigen::Quaterniond q_flu(msg->orientation.w, msg->orientation.x,
                                       msg->orientation.y, msg->orientation.z);
        if (q_flu.norm() > 1e-6) {
            const Eigen::Quaterniond q_rdf = q_flu.normalized() *
                                             Eigen::Quaterniond(R_b2c_base_.transpose());
            std::lock_guard<std::mutex> odom_lock(odom_mtx_);
            latest_gimbal_q_ = q_rdf.normalized();
            gimbal_received_ = true;
        }
    }

    // Build R_b2c from the given gimbal cam-x pitch angle (positive = down-gaze).
    Eigen::Matrix3d gimbal_R_b2c(double angle) const {
        Eigen::Matrix3d R_g;
        R_g << 1.0, 0.0, 0.0,
               0.0, std::cos(angle), -std::sin(angle),
               0.0, std::sin(angle), std::cos(angle);
        return R_g * R_b2c_base_;
    }

    void transition_to_state(MissionState new_state, const char* reason) {
        const MissionState old_state = mission_state_.get();
        if (old_state == new_state) return;
        mission_state_.set(new_state);
        RCLCPP_INFO(this->get_logger(), "Mission state: %s -> %s (%s)",
                    MissionStateMachine::state_name(old_state),
                    MissionStateMachine::state_name(new_state), reason);
    }

    void publish_mission_state(bool active) {
        auto state_msg = std::make_unique<std_msgs::msg::Bool>();
        state_msg->data = active;
        mission_state_pub_->publish(std::move(state_msg));
    }

    void stop() {
        if (control_timer_) {
            control_timer_->cancel();
            control_timer_ = nullptr;
        }
        reset_control_cache();
        publish_mission_state(false);
        const double g_hover = g_;
        auto cmd = std::make_unique<quadrotor_msgs::msg::ControlCommand>();
        cmd->header.stamp = this->get_clock()->now();
        cmd->armed = true;
        cmd->control_mode = quadrotor_msgs::msg::ControlCommand::ATTITUDE;
        cmd->collective_thrust = g_hover;
        control_pub_->publish(std::move(cmd));
    }

    int cam_width_ = 640;
    int cam_height_ = 480;
    double cam_fov_deg_ = 60.0;
    double focal_length_ = 554.256;
    Eigen::Matrix3d R_b2c_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_b2c_base_ = Eigen::Matrix3d::Identity();
    // Semi-strapdown gimbal state (see parameter declaration for the convention).
    bool use_gimbal_ = false;
    double gimbal_bias_rad_ = 0.0;
    // Camera-to-world quaternion q_C_W from /fpv/gimbal (Imu). Guarded by odom_mtx_.
    Eigen::Quaterniond latest_gimbal_q_ = Eigen::Quaterniond::Identity();
    bool gimbal_received_ = false;       // true once the first valid /fpv/gimbal msg arrives

    double dt_cam_ = 0.033;          // s, camera-frame dt (adaptive; updated from bbox timestamps)
    double last_bbox_cam_t_ = -1.0;  // last bbox timestamp (for dt_cam_ measurement)
    double w_guess_ = 1.5;
    double control_rate_ = 100.0;

    std::unique_ptr<RelativeIMMUKFEstimator> estimator_;
    std::mutex estimator_mtx_;

    std::unique_ptr<AdvancedHybridController> controller_;

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr mission_sub_;
    rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr tracker_trigger_sub_;
    // rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr bbox_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr gimbal_sub_;
    rclcpp::Publisher<quadrotor_msgs::msg::ControlCommand>::SharedPtr control_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr mission_state_pub_;

    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::CallbackGroup::SharedPtr timer_callback_group_;

    std::mutex control_cache_mtx_;
    struct ControlCacheData {
        bool valid = false;
        bool has_bbox = false;
        Eigen::Vector3d r_est = Eigen::Vector3d::Zero();
        Eigen::Vector3d v_est = Eigen::Vector3d::Zero();
        double los_az_world = 0.0;
        double los_el = 0.0;
        double los_rate = 0.0;
        double source_time = 0.0;
        double update_time = 0.0;
        double timestamp = 0.0;
        Eigen::Vector3d target_pos = Eigen::Vector3d::Zero();
        Eigen::Vector3d target_vel = Eigen::Vector3d::Zero();
        double w_est = 0.0;
        double r_meas = 0.0;
        double los_az = 0.0;
        Eigen::Matrix3d R_b2n = Eigen::Matrix3d::Identity();
        double bbox_size = 0.0;
        double target_speed = 0.0;
        // Ownship world accel over the last camera window (IMU preintegration; geometric APN input).
        Eigen::Vector3d a_own = Eigen::Vector3d::Zero();
    };
    ControlCacheData control_cache_;

    MissionStateMachine mission_state_;

    std::mutex odom_mtx_;
    // ODOMETRY-FREE ego state: attitude + preintegrated velocity change from the IMU only.
    Eigen::Matrix3d latest_R_b2n_ = Eigen::Matrix3d::Identity();   // from IMU orientation
    Eigen::Vector3d imu_bias_ = Eigen::Vector3d::Zero();           // accel bias (body frame)
    Eigen::Vector3d dv_own_accum_ = Eigen::Vector3d::Zero();       // ∫(R(f-b)-g)dt since last cam frame
    Eigen::Vector3d last_dv_own_ = Eigen::Vector3d::Zero();        // last drained camera window
    // Output smoothing of the estimator's RELATIVE velocity (Fix B): EMA before it drives guidance.
    Eigen::Vector3d target_vel_smoothed_ = Eigen::Vector3d::Zero();
    bool has_target_vel_smoothed_ = false;
    double last_vel_smooth_t_ = -1.0;
    double vel_smooth_tc_ = 0.25;   // s, EMA time constant on the relative-velocity estimate (0 = off)

    double last_imu_time_ = -1.0;
    double sim_start_time_ = -1.0;
    bool yaw_align_init_ = false;

    // Attitude-loop diagnostics: previous R_b2n for body-rate finite differencing.
    Eigen::Matrix3d prev_R_b2n_ = Eigen::Matrix3d::Identity();
    double prev_attitude_t_ = -1.0;
    bool has_prev_attitude_ = false;

    // PN lateral LOS bearing-rate EMA state (lambda_dot from the measured bearing).
    double prev_los_az_world_ = 0.0;
    double prev_los_t_ = -1.0;
    bool has_prev_los_ = false;
    double los_rate_ema_ = 0.0;

    // Terminal coast-on-dropout state: last good (detected) command + when/where it was captured.
    double last_bbox_time_ = 0.0;
    double last_good_range_ = 1e9;
    bool has_frozen_cmd_ = false;
    // Capture lockout: latched true once the UAV reaches capture_range of the target. After capture
    // the interceptor HOLDS (level-and-hold) for the rest of the mission — resuming normal guidance
    // on a reacquired-and-now-distant target is what flies the UAV off after a clean close approach.
    bool captured_ = false;
    // Brake-scoped ownship velocity (capture hold): seeded at capture from the last good closing
    // speed, then strapdown-integrated in imu_callback. Guarded by odom_mtx_.
    Eigen::Vector3d v_own_brake_ = Eigen::Vector3d::Zero();
    bool brake_vel_seeded_ = false;
    // Always-on ownship velocity (world frame) from IMU strapdown. Robust damping signal for the
    // vertical guidance: the monocular relative v_est(2) it used before is poorly observable, so the
    // vertical PD over-dived on below-FOV targets. Drifts long-term from accel bias, but the
    // high-frequency component (what damping needs) is accurate; reset on mission launch.
    Eigen::Vector3d own_vel_ = Eigen::Vector3d::Zero();
    // Collision detection (AirSim/real HW only, mirror InterceptorNodeBase): a gyro spike vs the
    // last commanded body rate signals physical contact -> intercept success -> COMPLETED. Gated on
    // !use_simulation_: the ROS2 Python sim has ground-truth range and judges intercept externally.
    bool collision_detected_ = false;
    bool use_simulation_ = false;
    Eigen::Vector3d latest_gyro_ = Eigen::Vector3d::Zero();   // body angular velocity from odom twist
    // Min estimated range seen this mission (reopen detection): once the range bottoms out and
    // reopens, the UAV flew past the target. More robust than los_az under a noisy close-range
    // estimate (high-noise scenario), where the bearing never cleanly flips past 90 deg.
    double min_range_seen_ = 1e9;
    double frozen_thrust_ = 0.0;
    double frozen_rate_x_ = 0.0;
    double frozen_rate_y_ = 0.0;
    double frozen_rate_z_ = 0.0;

    double ctrl_coeff_ = 1.0;
    // Target aerodynamic drag coefficient (set from physics.drag_coeff by the node). Used in f_rotor.
    double g_ = 9.81;              // gravity (from physics.g config)
    double g_drag_coeff_ = 0.0;
    Eigen::Vector3d last_cmd_ = Eigen::Vector3d::Zero();
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InterceptorNodeRelativeIMMUKF>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
