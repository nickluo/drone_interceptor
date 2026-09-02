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
                                   double /*g_drag_coeff*/) {
        Eigen::VectorXd x_new = x;
        const Eigen::Vector3d v = x.segment<3>(3);      // v_rel
        const Eigen::Vector3d a_t = x.segment<3>(6);    // target own accel
        const Eigen::Vector3d a_rel = a_t - a_own;
        // RELATIVE-COORDINATE DRAG IS DISABLED. The absolute-state f_rotor modeled the target's
        // aerodynamic drag as a_drag = -k*|v_target|*v_target, but v_target is UNOBSERVABLE in this
        // strapdown/relative variant (it would need the ownship velocity). Substituting v_rel is
        // wrong: when the target is near-static but the ownship is closing fast, |v_rel| is large and
        // the term spuriously brakes the RELATIVE state — at |v_rel|~15 and drag_coeff=0.1625 that is
        // ~36 m/s^2 of phantom deceleration injected into the v_rel predict every cycle, which the
        // radar update then yanks back, chattering v_closing and braking the mid-course flight.
        // g_drag_coeff is intentionally ignored (kept in the signature only because it is wired
        // through the ctor/apply_model, and the sim reuses physics.drag_coeff for the OWNship drag);
        // the model is pure accel-relaxation with tau=2s. Do NOT re-enable without first expressing
        // the term as a function of an observable target-velocity estimate.
        x_new.segment<3>(0) = x.segment<3>(0) + v * dt + 0.5 * a_rel * dt * dt;
        x_new.segment<3>(3) = v + a_rel * dt;
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

    // Radar-relay measurement: a direct observation of the RELATIVE position/velocity
    // z = [r(3), v_rel(3)] = H x with H = [I6 | 0]. Linear in the state, so a plain KF update is
    // exact — no sigma points needed (and it stays valid at ranges where the pixel model is
    // unusable because the target is outside the camera FOV).
    double update_posvel(const Vector6d& z, const Matrix6d& R6) {
        predicted = false;
        const Vector6d y = z - x.head<6>();
        Matrix6d S = P.topLeftCorner<6, 6>() + R6;
        S = (S + S.transpose()) * 0.5;

        Eigen::LDLT<Matrix6d> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            return 1e-30;
        }
        // P * H^T = the first 6 columns of P.
        const Eigen::Matrix<double, kStateDim, 6> K =
            P.leftCols<6>() * ldlt.solve(Matrix6d::Identity());
        if (!K.allFinite() || !y.allFinite()) {
            return 1e-30;
        }

        x = x + K * y;
        P = P - K * S * K.transpose();
        P = (P + P.transpose()) * 0.5;

        const double det = std::max(std::abs(S.determinant()), 1e-30);
        const double exponent = -0.5 * y.dot(ldlt.solve(y));
        double likelihood = std::exp(clamp(exponent, -80.0, 80.0)) / std::sqrt(std::pow(2.0 * M_PI, 6) * det);
        if (!std::isfinite(likelihood) || likelihood < 1e-30) {
            likelihood = 1e-30;
        }
        return likelihood;
    }

    // Velocity-only update (radar vel aid during terminal vision): 3D KF update on x[3:6] only.
    // The gain spans all 10 rows, so the accurate radar v_rel also corrects pos/acc/width through
    // the cross-covariance — a small, decaying, self-consistent nudge (the 30 Hz monocular bearing
    // update dominates position, so the radar channel cannot run away with range). Mirrors
    // update_posvel restricted to the velocity block.
    double update_vel(const Eigen::Vector3d& z, const Eigen::Matrix3d& R) {
        predicted = false;
        const Eigen::Vector3d y = z - x.segment<3>(3);
        Eigen::Matrix3d S = P.block<3, 3>(3, 3) + R;
        S = (S + S.transpose()) * 0.5;
        Eigen::LDLT<Eigen::Matrix3d> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            return 1e-30;
        }
        // P * H^T = columns 3..5 of P (the velocity block's cross-covariance with the full state).
        const Eigen::Matrix<double, kStateDim, 3> K =
            P.block<kStateDim, 3>(0, 3) * ldlt.solve(Eigen::Matrix3d::Identity());
        if (!K.allFinite() || !y.allFinite()) {
            return 1e-30;
        }
        x = x + K * y;
        P = P - K * S * K.transpose();
        P = (P + P.transpose()) * 0.5;
        const double det = std::max(std::abs(S.determinant()), 1e-30);
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
        mix_likelihoods(likelihoods);
    }

    // Radar-relay update: direct [r, v_rel] observation applied to every mode filter.
    void update_posvel(const Vector6d& z, const Matrix6d& R6) {
        Eigen::Vector3d likelihoods;
        for (int j = 0; j < kModeCount; ++j) {
            likelihoods(j) = filters[j].update_posvel(z, R6);
        }
        mix_likelihoods(likelihoods);
    }
    // Radar vel-only aid (terminal vision): v_rel observation applied to every mode filter.
    void update_vel(const Eigen::Vector3d& z, const Eigen::Matrix3d& R) {
        Eigen::Vector3d likelihoods;
        for (int j = 0; j < kModeCount; ++j) {
            likelihoods(j) = filters[j].update_vel(z, R);
        }
        mix_likelihoods(likelihoods);
    }

    void mix_likelihoods(const Eigen::Vector3d& likelihoods) {
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
    // Sanitizer range ceiling. Monocular-only operation never sees a credible target past ~200 m,
    // but a radar relay hands over tracks at several hundred metres — the node raises this when
    // radar is enabled so the sanitizer does not fold a valid long-range track back to 120 m.
    double max_range_ = 200.0;
    // Set once the width has been fixed from a radar-range + bbox-size handover. sanitize_filters
    // then keeps the freeze regardless of range (the normal freeze releases above
    // width_freeze_range_, which would let a converged handover width drift again).
    bool width_externally_locked_ = false;

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

    // Single time base shared by every measurement source (camera bbox, camera dropout coast, radar
    // relay). The predict dt is always "time since the previous estimator step", so mixing a 30 Hz
    // vision stream with a 10 Hz radar stream never double-propagates: whichever source runs next
    // advances the filters by exactly the elapsed interval, and the IMU dv_own accumulator drained
    // by that same source covers exactly the same window.
    double advance_time(double t_now) {
        const double dt = has_last_cap_time ? clamp(t_now - last_cap_time, 0.001, 0.3) : dt_cam;
        last_cap_time = t_now;
        has_last_cap_time = true;
        return dt;
    }

    // Fix the target width from an external, trusted source (radar range x bbox size at handover).
    // Terminal monocular range r = focal*W/s is only as good as W, so pinning it at handover is
    // what makes the vision terminal phase metrically consistent with the radar mid-course track.
    void lock_width(double W) {
        const double w = clamp(W, 0.3, 5.0);
        w_guess = w;
        frozen_width = w;
        has_frozen_width = true;
        width_externally_locked_ = true;
        for (auto& filter : imm_ukf->filters) {
            filter.x(9) = w;
            filter.P.row(9).setZero();
            filter.P.col(9).setZero();
            filter.P(9, 9) = 1e-8;
        }
        imm_ukf->compute_state_estimate();
        imm_ukf->x(9) = w;
    }

    // Radar-relay measurement of the RELATIVE state: z = [r, v_rel] in world frame G. Used for
    // mid-course guidance before the target enters the camera FOV, and as the metric anchor for the
    // vision handover. dv_own is the IMU-preintegrated ownship velocity change over the same window
    // as the predict (drained by the caller), exactly as in process_camera.
    std::pair<bool, Eigen::VectorXd> process_radar(const Eigen::Vector3d& r_meas,
                                                   const Eigen::Vector3d& v_meas,
                                                   double t_now, const Eigen::Vector3d& dv_own,
                                                   double pos_std, double vel_std) {
        Matrix6d R6 = Matrix6d::Zero();
        for (int i = 0; i < 3; ++i) {
            R6(i, i) = pos_std * pos_std;
            R6(i + 3, i + 3) = vel_std * vel_std;
        }

        if (!initialized) {
            Eigen::VectorXd x0 = Eigen::VectorXd::Zero(kStateDim);
            x0.segment<3>(0) = r_meas;
            x0.segment<3>(3) = v_meas;
            x0(9) = clamp(w_guess, 0.3, 5.0);

            Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(kStateDim, kStateDim);
            const double pv = pos_std * pos_std;
            const double vv = vel_std * vel_std;
            P0.diagonal() << pv, pv, pv, vv, vv, vv, 10.0, 10.0, 10.0, 0.25;
            for (auto& filter : imm_ukf->filters) {
                filter.x = x0;
                filter.P = P0;
                filter.predicted = false;
            }
            imm_ukf->x = x0;
            imm_ukf->P = P0;
            initialized = true;
            w_init_time = t_now;
            last_cap_time = t_now;
            has_last_cap_time = true;
            if (r_meas.norm() > 1e-3) {
                last_los_nav = r_meas.normalized();
                has_last_los_nav = true;
            }
            sanitize_filters();
            return {true, imm_ukf->x};
        }

        const double dt = advance_time(t_now);
        const Eigen::Vector3d a_own = dv_own / std::max(dt, 1e-3);
        update_process_noise(imm_ukf->x.segment<3>(0).norm(), imm_ukf->x(9), 0.0, t_now - w_init_time);
        imm_ukf->predict(dt, a_own);

        Vector6d z;
        z.head<3>() = r_meas;
        z.tail<3>() = v_meas;
        imm_ukf->update_posvel(z, R6);

        if (r_meas.norm() > 1e-3) {
            last_los_nav = r_meas.normalized();
            has_last_los_nav = true;
        }
        sanitize_filters();
        return {true, imm_ukf->x};
    }

    // Velocity-only radar update (terminal-vision aid): predict + a 3D update on v_rel only. Keeps
    // the along-LOS closing speed observable without injecting the metro-level radar POSITION
    // (which would corrupt the monocular range). Estimator is always initialized when this is
    // called (only runs once vision is locked). vel_std sets how gently the aid pulls (large =>
    // trust monocular more); steady-state P(vel) ~= vel_std^2.
    void process_radar_vel(const Eigen::Vector3d& v_meas, double t_now,
                           const Eigen::Vector3d& dv_own, double vel_std) {
        const double dt = advance_time(t_now);
        const Eigen::Vector3d a_own = dv_own / std::max(dt, 1e-3);
        update_process_noise(imm_ukf->x.segment<3>(0).norm(), imm_ukf->x(9), 0.0, t_now - w_init_time);
        imm_ukf->predict(dt, a_own);
        const Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * (vel_std * vel_std);
        imm_ukf->update_vel(v_meas, R);
        sanitize_filters();
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
        bool should_freeze = width_externally_locked_;
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
            if (r_norm > max_range_) {
                filter.x.segment<3>(0) = rel_p / std::max(r_norm, 1e-6) * (0.6 * max_range_);
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
            const double dt_coast = advance_time(t_now);
            const Eigen::Vector3d a_own = dv_own / std::max(dt_coast, 1e-3);
            update_process_noise(r_norm, w_prev, 0.0, t_now - w_init_time);
            // ALWAYS predict r forward using ownship motion (a_own). The absolute-state estimator
            // could skip the predict for a stationary target; the RELATIVE position r = target - own
            // moves by the ownship's motion EVERY frame. Skipping the predict freezes r while the
            // ownship keeps flying -> unbounded drift the moment the target leaves the FOV.
            imm_ukf->predict(dt_coast, a_own);
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
            const double dt_meas = advance_time(t_now);
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
    // Decoupled gains (config-driven via set_guidance_tuning).
    double K_fwd_ = 2.0;        // axial: a_fwd = K_fwd * (v_des - v_fwd_body)
    double K_az_ = 2.0;         // vertical-speed tracking gain: a_z = K_az*(v_z_desired - v_z)
    double a_z_limit_ = 8.0;    // vertical accel clamp (m/s^2) — airframe thrust/accel limit
    // Turn-radius cap scaling factor. The cap v ≤ factor·sqrt(g·tan(tilt)·r) is the single adaptive
    // terminal-speed reducer (keeps min turn radius inside range). 1.0 = physics limit; <1 brakes sooner.
    double turn_radius_cap_factor_ = 0.8;
    // Axial deadband: coast (a_fwd=0) within ±this of v_des_eff — kills radar-noise-driven pitch oscillation.
    double axial_deadband_ = 1.5;
    // Motion-dependent LOS gate widening: bearing_tol += motion_gate_factor * v_perp * delay / r.
    double motion_gate_factor_ = 3.0;
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
    // Terminal authority (maneuvering-target tracking): relax the roll clamp inside terminal_tilt_range_.
    // max_tilt_terminal_rad_ is ALSO the bank used by the turn-radius speed cap in that band, so raising
    // it both permits tighter turns AND raises the close-range speed cap (win-win for terminal maneuverability).
    double max_tilt_terminal_rad_ = 45.0 * M_PI / 180.0;  // roll clamp + cap tilt inside terminal_tilt_range_
    double terminal_tilt_range_ = 15.0;  // m, below this range phi_cmd and the speed cap use max_tilt_terminal_rad_
    // Active monocular-observability dither: a small lateral sinusoidal acceleration that gives
    // the ownship a horizontal parallax baseline. With x != 0 the pinhole u=fx*x/z becomes
    // sensitive to z (depth), so the depth/width degeneracy (size=fx*W/z) is broken and the
    // along-LOS v_rel — otherwise only weakly observable via the noisy size channel — becomes
    // observable through the bearing channel. The IMU preintegration captures the dither motion
    // as a_own, so the estimator predict carries it; the update resolves depth from the parallax.
    double dither_amp_ = 0.8;          // m/s^2, lateral dither amplitude
    double dither_freq_ = 0.5;         // Hz, lateral dither frequency
    double dither_min_range_ = 10.0;   // m, suppress dither inside this range (don't disturb terminal)
    // Axial-only range EMA (radar smoothness): isolates the turn-radius cap + terminal-decel ramp
    // from 10 Hz radar single-sample range noise. r_norm elsewhere (PN, dz, APN) stays unsmoothed.
    // Lag is conservative during closing (over-estimates range -> cap allows slightly more speed,
    // decel engages slightly later), so it biases against the over-braking we are fixing.
    double r_norm_smooth_tc_ = 0.0;    // s, 0 = off (passthrough)
    double r_norm_ema_ = 0.0;
    double last_rn_ema_t_ = -1.0;
    bool   has_rn_ema_ = false;
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
                             double max_body_rate, double terminal_brake_range,
                             double v_pn_ref, double max_att_rate, double att_ref_tc,
                             double max_tilt_terminal_deg, double terminal_tilt_range)
        : N(N_val), K_att(K_att_val),
          K_yaw(K_yaw_val), v_des(v_des_val), max_tilt_rad(max_tilt_deg * M_PI / 180.0),
          fov_half_v_(fov_half_v), fov_half_h_(fov_half_h), fov_keep_margin_(fov_keep_margin),
          axial_jerk_max_(axial_jerk_max), enable_fov_clamp_(enable_fov_clamp),
          mount_pitch_(mount_pitch_rad),
          max_body_rate_(max_body_rate), terminal_brake_range_(terminal_brake_range),
          v_pn_ref_(v_pn_ref), max_att_rate_(max_att_rate), att_ref_tc_(att_ref_tc),
          max_tilt_terminal_rad_(max_tilt_terminal_deg * M_PI / 180.0),
          terminal_tilt_range_(terminal_tilt_range) {}

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
        r_norm_ema_ = 0.0;
        last_rn_ema_t_ = -1.0;
        has_rn_ema_ = false;
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
    // Axial-only range EMA time constant (radar smoothness). <=0 disables (passthrough r_norm).
    void set_range_smooth_tc(double tc) { r_norm_smooth_tc_ = std::max(0.0, tc); }
    // Turn-radius cap factor: clamped to [0.3, 1.0] (1.0 = physics limit; <1 brakes sooner). The cap is
    // the single adaptive terminal-speed reducer (see compute_control). Out-of-range is WARNed at the
    // call site so a stale yaml "3.0 = disabled" is not silently swallowed.
    void set_turn_radius_cap_factor(double v) { turn_radius_cap_factor_ = clamp(v, 0.3, 1.0); }
    void set_axial_deadband(double v) { axial_deadband_ = std::max(0.0, v); }
    void set_motion_gate_factor(double v) { motion_gate_factor_ = std::max(0.0, v); }

    // Decoupled control (odometry-free, exact mirror of the Python prototype). Inputs:
    //   r_est, v_est : RELATIVE target pos/vel in world frame G (estimator state directly)
    //   los_az       : target azimuth in body heading frame (rad, 0 = dead ahead)
    //   los_el       : target elevation in world frame (rad, 0 = level, negative = below)
    //   v_fwd_body   : closing speed proxy = -v_rel·r̂ (positive = approaching); NOT odom
    //   bbox_size    : target bbox size in px (observable proximity cue)
    //   v_z          : ownship vertical speed proxy = -v_rel(2) (target-level approximation)
    //   target_speed : transverse (escape) speed = |r_xy×v_xy|/|r_xy| — drives the speed-aware
    //                  terminal-brake floor (must NOT brake below a crossing target's lateral speed).
    //                  NOT the full |v_rel|: that is the closing speed and would defeat the brake.
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
                                        double boresight_el_meas = std::numeric_limits<double>::quiet_NaN(),
                                        bool radar_owns_guidance = false) {
        (void)v_est;
        const double r_norm = r_est.norm();
        // Axial-only range EMA (radar smoothness): r_norm carries +-~4 m of 10 Hz radar noise; fed
        // 1:1 into the turn-radius cap (sqrt(g*tan*r)) it swings v_des_eff and brakes every cycle.
        // Smooth it ONLY for the axial chain (cap + terminal decel). Lag is conservative on closing.
        double r_norm_axial = r_norm;
        if (r_norm_smooth_tc_ > 1e-6) {
            if (has_rn_ema_ && t > last_rn_ema_t_ && (t - last_rn_ema_t_) < 1.0) {
                const double dt_rn = t - last_rn_ema_t_;
                const double a_rn = clamp(dt_rn / r_norm_smooth_tc_, 0.0, 1.0);
                r_norm_ema_ += a_rn * (r_norm - r_norm_ema_);
                r_norm_axial = r_norm_ema_;
            } else {
                r_norm_ema_ = r_norm;
                has_rn_ema_ = true;
            }
        }
        last_rn_ema_t_ = t;

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
        // Suppressed while the radar owns guidance: the radar gives absolute scale directly, so the
        // monocular-observability job the dither does is redundant during radar-only flight, and the
        // 0.3 Hz lateral sinusoid only adds an airspeed oscillation. Re-engages at/after vision lock.
        if (dither_amp_ > 0.0 && r_norm > dither_min_range_ && !radar_owns_guidance) {
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

        // ---- (A) Axial guidance: flat v_des + single adaptive turn-radius cap ----
        // No deliberate terminal brake: v_des_eff starts at the cruise ceiling v_des and is reduced
        // ONLY by the turn-radius cap below (keeps min turn radius inside the current range -> the
        // interceptor can still bend onto a dodging target). sqrt(r) is smooth; the only derivative
        // kink is at the cap/v_des crossover (far from terminal) and is absorbed by axial_jerk_max.
        double v_des_eff = v_des;
        // Speed-aware floor: never below a maneuvering target's escape speed + margin (it must be chased).
        // Static/slow target -> 3.0 m/s stall guard (the cap, not this floor, governs close-range speed).
        {
            const double moving_floor = (target_speed > 1.0) ? (target_speed + 1.5) : 3.0;
            v_des_eff = std::max(v_des_eff, std::min(moving_floor, v_des));
        }
        // Turn-radius maneuverability cap: v ≤ factor·sqrt(g·tan(tilt)·r). Uses the range-appropriate
        // bank (terminal tilt inside terminal_tilt_range) so the raised terminal bank translates into
        // sustained close-range speed AND tighter turns. This is the binding terminal-speed governor.
        const double tilt_for_cap = (r_norm_axial < terminal_tilt_range_) ? max_tilt_terminal_rad_ : max_tilt_rad;
        v_des_eff = std::min(v_des_eff, turn_radius_cap_factor_ * std::sqrt(g * std::tan(tilt_for_cap) * r_norm_axial));
        // v_fwd_body (= v_closing from v_rel) is under-observable along-LOS until scale converges;
        // a low v_closing inflates a_fwd -> ownship over-accelerates ON-AXIS, starving lateral PN
        // lead. Scene-adaptive 3-level floor: turning target (|atp_ema| large) -> 0.88 (don't blast
        // through a circle); laterally-moving target (v_perp > 1) -> 0.82; static/slow -> 0.8.
        // Braking (v_closing above the floor) passes through unchanged.
        const double v_perp = std::abs(r_norm * los_rate);
        const double v_floor_frac = (std::abs(atp_ema_) > 0.5) ? 0.88 : ((v_perp > 1.0) ? 0.82 : 0.8);
        const double v_fwd_for_afwd = std::max(v_fwd_body, v_floor_frac * v_des_eff);
        // Axial deadband: coast (a_fwd=0) within ±axial_deadband_ of v_des_eff. Kills the pitch
        // oscillation from radar v_closing noise (±1 m/s) during blind flight — the interceptor
        // cruises at v_des±deadband (all plenty for a kinetic-energy intercept) instead of chasing
        // every noise sample. Outside the band K_fwd applies, shifted by the deadband (no step).
        const double v_error = v_des_eff - v_fwd_for_afwd;
        double a_fwd;
        if (std::abs(v_error) <= axial_deadband_) {
            a_fwd = 0.0;
        } else {
            a_fwd = K_fwd_ * (v_error - std::copysign(axial_deadband_, v_error));
        }
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

        if (radar_enable_) {
            estimator_->max_range_ = radar_max_range_;
            radar_target_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                radar_topic_target_, 10,
                [this](const nav_msgs::msg::Odometry::SharedPtr msg) { radar_callback(msg, true); },
                sub_options);
            radar_own_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                radar_topic_own_, 10,
                [this](const nav_msgs::msg::Odometry::SharedPtr msg) { radar_callback(msg, false); },
                sub_options);
            RCLCPP_INFO(this->get_logger(),
                "Radar relay ENABLED: target=%s interceptor=%s pos_std=%.1fm vel_std=%.1fm/s handover=%d frames",
                radar_topic_target_.c_str(), radar_topic_own_.c_str(),
                radar_pos_std_, radar_vel_std_, handover_frames_);
        }

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
        this->declare_parameter<double>("controller.v_pn_ref", 12.0);
        this->declare_parameter<double>("controller.max_att_rate", 3.5);   // Fix G: clamp |d(phi/theta)_cmd/dt| (was 5)
        this->declare_parameter<double>("controller.att_ref_tc", 0.15);
        this->declare_parameter<double>("physics.max_tilt_terminal_deg", 45.0);
        this->declare_parameter<double>("controller.terminal_tilt_range", 15.0);
        this->declare_parameter<double>("controller.a_z_slew", 20.0);   // Fix C/E: |da_z/dt| limit (m/s^3)
        this->declare_parameter<bool>("controller.enable_fov_clamp", true);
        // Guidance tuning (set_guidance_tuning) — operator-facing knobs promoted from hardcoded.
        this->declare_parameter<double>("controller.K_fwd", 2.0);       // axial accel gain
        this->declare_parameter<double>("controller.K_az", 2.0);        // vertical-speed tracking gain
        this->declare_parameter<double>("physics.a_z_limit", 8.0);  // vertical accel clamp (m/s^2)
        this->declare_parameter<double>("controller.dither_amp", 0.8);  // m/s^2, lateral observability dither
        this->declare_parameter<double>("controller.dither_freq", 0.5); // Hz
        this->declare_parameter<double>("controller.dither_min_range", 10.0); // m, suppress dither inside
        this->declare_parameter<double>("controller.range_smooth_tc", 0.3); // s, EMA on the range feeding the turn-radius cap (0=off; radar smoothness)
        this->declare_parameter<double>("controller.turn_radius_cap_factor", 0.8); // turn-radius maneuverability cap factor (clamped [0.3,1.0]; 1.0=physics limit). The binding terminal-speed reducer — see compute_control.
        this->declare_parameter<double>("controller.axial_deadband", 1.5); // m/s, coast within ±this of v_des — kills radar-noise-driven pitch oscillation during blind flight
        this->declare_parameter<double>("radar.motion_gate_factor", 3.0); // widens the bbox-radar LOS gate for moving targets: bearing_tol += this * v_perp * delay / r
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
        // ---- Radar relay (mid-course guidance before the target enters the camera FOV) --------
        // An external radar tracks BOTH the target and the interceptor and reports each as a
        // world-ENU nav_msgs/Odometry at ~10 Hz with metre-level noise and 50-100 ms latency. The
        // node differences the pair into a relative measurement, delay-compensates it, and feeds it
        // to the same IMM-UKF as a direct [r, v_rel] observation.
        this->declare_parameter<bool>("radar.enable", false);
        this->declare_parameter<std::string>("radar.topic_target", "/fpv/radar/target");
        this->declare_parameter<std::string>("radar.topic_interceptor", "/fpv/radar/interceptor");
        this->declare_parameter<double>("radar.pos_std", 3.0);        // m, 1-sigma per axis
        this->declare_parameter<double>("radar.vel_std", 1.0);        // m/s, 1-sigma per axis
        this->declare_parameter<double>("radar.pair_window", 0.15);   // s, max target/own stamp skew
        this->declare_parameter<double>("radar.timeout", 1.0);        // s, track considered stale
        this->declare_parameter<double>("radar.max_delay_comp", 0.3); // s, cap on latency extrapolation
        this->declare_parameter<double>("radar.max_range", 800.0);    // m, sanitizer ceiling with radar
        // Handover: consecutive vision frames required to declare a visual lock, and the window in
        // which the radar range + bbox size fixes the target width for terminal guidance.
        this->declare_parameter<int>("radar.handover_frames", 10);
        this->declare_parameter<double>("radar.handover_max_range", 200.0);
        this->declare_parameter<double>("radar.width_min", 0.3);
        this->declare_parameter<double>("radar.width_max", 5.0);
        this->declare_parameter<bool>("radar.disable_after_lock", true);  // SUPERSEDED for the vision branch by vel_aid_after_lock (kept so an old yaml still parses; the pos+vel path still keys off it if vel aiding is off)
        this->declare_parameter<bool>("radar.vel_aid_after_lock", false); // OFF by default: terminal vision must be PURE vision (the radar node must not influence the endgame). Opt-in experiment: fuse radar VELOCITY (not position) while vision is locked to anchor v_closing — but in AirSim the lagged/10 Hz radar vel fought the 30 Hz vision update and HURT, so it is off.
        this->declare_parameter<double>("radar.vel_aid_std", 2.5);       // m/s, 1-sigma of the vel-only aid (large = trust the monocular solution more; steady-state P(vel) ~= vel_aid_std^2)
        this->declare_parameter<double>("radar.vision_lost_timeout", 1.5);
        // vision_hold is how long a bbox counts as "current": past it the radar takes the filter
        // and guidance back immediately instead of dead-reckoning.
        this->declare_parameter<double>("radar.vision_hold", 0.35);
        // The radar reports the interceptor's OWN world velocity — the only drift-free correction
        // available for the IMU strapdown own_vel_ that damps the vertical loop.
        this->declare_parameter<bool>("radar.own_vel_aiding", true);
        this->declare_parameter<double>("radar.own_vel_tc", 5.0);
        // Yaw rotation from the radar world frame into the IMU-referenced estimator world frame.
        // Nonzero only when the FCU heading reference is offset from the radar ENU x-axis; the
        // handover bearing check measures exactly this angle and logs it.
        this->declare_parameter<double>("radar.yaw_offset_deg", 0.0);
        this->declare_parameter<double>("radar.handover_max_bearing_err", 0.20);
        // False-positive rejection. The detector fires on clouds/structures at long range and ONE
        // fused phantom pixel measurement is enough to drag the monocular width — and with it the
        // whole range scale — off. Every bbox is checked against the radar track before it reaches
        // the filter, not just before the handover.
        this->declare_parameter<bool>("radar.bbox_gate", true);
        this->declare_parameter<double>("radar.bbox_range_ratio", 2.0);
        this->declare_parameter<double>("radar.lock_min_duration", 0.7);
        this->declare_parameter<double>("radar.width_window", 1.5); // s, rolling window for the width median (survives transient misses)
        this->declare_parameter<double>("radar.width_min_bbox", 25.0); // px, only anchor the width from a bbox this large (a tiny long-range bbox is over-bound by the tracker -> inflated W). Shifts the lock closer where the bbox is accurate.
        this->declare_parameter<double>("radar.width_scale", 1.0);   // multiplicative correction for tracker over-bounding: locked_W *= scale. Calibrate as true_W / locked_W (e.g. 0.66 if it locks 2.26 for a 1.5 m target).
        // Re-intercept: on a terminal vision loss (a miss, NOT a clean hit), re-engage the radar
        // relay for another mid-course + terminal pass instead of brake-to-hold forever. Off by default.
        this->declare_parameter<bool>("radar.reintercept_enable", false);   // master switch
        this->declare_parameter<int>("radar.reintercept_max", 2);            // max re-intercept attempts (total terminal passes = 1 + this)
        this->declare_parameter<double>("radar.reintercept_kill_range", 1.0);// m; if the closest approach was <= this, treat as a clean hit (don't re-attempt). 0 = always re-attempt.
        this->declare_parameter<double>("radar.reintercept_min_range", 15.0);// m; re-engage only once radar shows the target back beyond this standoff (clean re-approach). 0 = immediate.
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
        // physics.drag_coeff models the OWNship aero drag in the sim; the RELATIVE-state f_rotor
        // ignores it (see f_rotor comment). Logged once so an operator setting it is not surprised.
        if (g_drag_coeff_ > 1e-3) {
            RCLCPP_INFO(this->get_logger(),
                "physics.drag_coeff=%.4f is IGNORED by the relative-state f_rotor "
                "(kept for the sim's ownship drag); rotor mode runs as pure accel-relaxation tau=2s.",
                g_drag_coeff_);
        }
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
        double v_pn_ref = this->get_parameter("controller.v_pn_ref").as_double();
        double max_att_rate = this->get_parameter("controller.max_att_rate").as_double();
        double att_ref_tc = this->get_parameter("controller.att_ref_tc").as_double();
        double max_tilt_terminal_deg = this->get_parameter("physics.max_tilt_terminal_deg").as_double();
        double terminal_tilt_range = this->get_parameter("controller.terminal_tilt_range").as_double();
        double fov_half_v = (cam_height_ / 2.0) / std::max(1.0, focal_length_);
        double fov_half_h = (cam_width_ / 2.0) / std::max(1.0, focal_length_);

        const bool enable_fov_clamp = this->get_parameter("controller.enable_fov_clamp").as_bool();
        controller_ = std::make_unique<AdvancedHybridController>(
            N, K_att, K_yaw, v_des, max_tilt,
            fov_half_v, fov_half_h, fov_keep, jerk, enable_fov_clamp, gimbal_bias_rad_,
            max_body_rate, terminal_brake_range,
            v_pn_ref, max_att_rate, att_ref_tc,
            max_tilt_terminal_deg, terminal_tilt_range);
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
        controller_->set_range_smooth_tc(this->get_parameter("controller.range_smooth_tc").as_double());
        {
            const double trcf = this->get_parameter("controller.turn_radius_cap_factor").as_double();
            if (trcf < 0.3 - 1e-9 || trcf > 1.0 + 1e-9) {
                RCLCPP_WARN(this->get_logger(),
                    "controller.turn_radius_cap_factor=%.3f is outside [0.3, 1.0] -> clamped. "
                    "The cap is the binding terminal-speed reducer; >1.0 would disable it.", trcf);
            }
            controller_->set_turn_radius_cap_factor(trcf);
        }
        controller_->set_axial_deadband(this->get_parameter("controller.axial_deadband").as_double());
        controller_->set_motion_gate_factor(this->get_parameter("radar.motion_gate_factor").as_double());
        vel_smooth_tc_ = this->get_parameter("estimation.vel_smooth_tc").as_double();
        const std::vector<double> bias = this->get_parameter("imu_bias").as_double_array();
        if (bias.size() == 3) {
            imu_bias_ = Eigen::Vector3d(bias[0], bias[1], bias[2]);
        }

        radar_enable_ = this->get_parameter("radar.enable").as_bool();
        printf("radar_enable: %d\n", radar_enable_);
        radar_topic_target_ = this->get_parameter("radar.topic_target").as_string();
        radar_topic_own_ = this->get_parameter("radar.topic_interceptor").as_string();
        radar_pos_std_ = std::max(0.1, this->get_parameter("radar.pos_std").as_double());
        radar_vel_std_ = std::max(0.05, this->get_parameter("radar.vel_std").as_double());
        radar_pair_window_ = this->get_parameter("radar.pair_window").as_double();
        radar_timeout_ = this->get_parameter("radar.timeout").as_double();
        radar_max_delay_comp_ = this->get_parameter("radar.max_delay_comp").as_double();
        radar_max_range_ = this->get_parameter("radar.max_range").as_double();
        handover_frames_ = std::max(1, static_cast<int>(this->get_parameter("radar.handover_frames").as_int()));
        handover_max_range_ = this->get_parameter("radar.handover_max_range").as_double();
        width_min_ = this->get_parameter("radar.width_min").as_double();
        width_max_ = this->get_parameter("radar.width_max").as_double();
        radar_disable_after_lock_ = this->get_parameter("radar.disable_after_lock").as_bool();
        radar_vel_aid_after_lock_ = this->get_parameter("radar.vel_aid_after_lock").as_bool();
        radar_vel_aid_std_ = std::max(0.5, this->get_parameter("radar.vel_aid_std").as_double());
        vision_lost_timeout_ = this->get_parameter("radar.vision_lost_timeout").as_double();
        radar_vision_hold_ = this->get_parameter("radar.vision_hold").as_double();
        radar_own_vel_aiding_ = this->get_parameter("radar.own_vel_aiding").as_bool();
        radar_own_vel_tc_ = std::max(0.05, this->get_parameter("radar.own_vel_tc").as_double());
        handover_max_bearing_err_ = this->get_parameter("radar.handover_max_bearing_err").as_double();
        radar_bbox_gate_ = this->get_parameter("radar.bbox_gate").as_bool();
        radar_bbox_range_ratio_ = std::max(1.1, this->get_parameter("radar.bbox_range_ratio").as_double());
        lock_min_duration_ = this->get_parameter("radar.lock_min_duration").as_double();
        width_window_s_ = std::max(0.1, this->get_parameter("radar.width_window").as_double());
        width_min_bbox_ = std::max(1.0, this->get_parameter("radar.width_min_bbox").as_double());
        width_scale_ = this->get_parameter("radar.width_scale").as_double();
        reintercept_enable_ = this->get_parameter("radar.reintercept_enable").as_bool();
        reintercept_max_ = std::max(0, static_cast<int>(this->get_parameter("radar.reintercept_max").as_int()));
        reintercept_kill_range_ = std::max(0.0, this->get_parameter("radar.reintercept_kill_range").as_double());
        reintercept_min_range_ = std::max(0.0, this->get_parameter("radar.reintercept_min_range").as_double());
        const double radar_yaw_off = this->get_parameter("radar.yaw_offset_deg").as_double() * M_PI / 180.0;
        radar_R_yaw_ = Eigen::AngleAxisd(radar_yaw_off, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        vision_locked_ = !radar_enable_;
    }

    void reset_estimator() {
        estimator_ = std::make_unique<RelativeIMMUKFEstimator>(dt_cam_, focal_length_, w_guess_, R_b2c_, g_drag_coeff_);
        if (radar_enable_) estimator_->max_range_ = radar_max_range_;
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

    // Output smoothing of the estimator's RELATIVE velocity (Fix B): EMA the raw v_rel so a
    // single-frame swing never reaches guidance. Shared by the vision and radar paths so a
    // mid-course -> terminal handover does not step the velocity fed to the controller.
    Eigen::Vector3d smooth_rel_vel(const Eigen::Vector3d& v_raw, double t_now) {
        if (vel_smooth_tc_ <= 1e-6) return v_raw;
        if (has_target_vel_smoothed_ && t_now > last_vel_smooth_t_ &&
            (t_now - last_vel_smooth_t_) < 1.0) {
            const double a = clamp((t_now - last_vel_smooth_t_) / vel_smooth_tc_, 0.0, 1.0);
            target_vel_smoothed_ += a * (v_raw - target_vel_smoothed_);
        } else {
            target_vel_smoothed_ = v_raw;
        }
        last_vel_smooth_t_ = t_now;
        has_target_vel_smoothed_ = true;
        return target_vel_smoothed_;
    }

    // Is vision currently delivering a consistent track? True when a radar-validated bbox has been
    // seen within vision_lost_timeout_. Used to decide guidance-cache ownership: vision owns the
    // cache as soon as it's delivering (even before the full handover completes), radar is the
    // fallback only when vision is absent/inconsistent for a definitive-loss period.
    bool vision_delivering(double now) const {
        return last_vision_time_ > 0.0 && (now - last_vision_time_) < vision_lost_timeout_;
    }

    // ---- Radar relay -------------------------------------------------------------------------
    // Each report carries ONE object (target or interceptor) as a world-ENU nav_msgs/Odometry,
    // stamped with the MEASUREMENT epoch: the radar publisher holds every sample for the sensor
    // latency before sending, so header.stamp is genuinely in the past and the lag is recoverable.
    // A relative measurement needs both objects, so samples are paired by stamp.
    void radar_callback(const nav_msgs::msg::Odometry::SharedPtr msg, bool is_target) {
        RadarSample s;
        s.stamp = rclcpp::Time(msg->header.stamp).seconds();
        s.pos = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y,
                                msg->pose.pose.position.z);
        s.vel = Eigen::Vector3d(msg->twist.twist.linear.x, msg->twist.twist.linear.y,
                                msg->twist.twist.linear.z);
        if (!s.pos.allFinite() || !s.vel.allFinite()) return;
        s.valid = true;
        {
            std::lock_guard<std::mutex> lock(radar_mtx_);
            if (is_target) { radar_target_ = s; ++radar_msg_target_; }
            else { radar_own_ = s; ++radar_msg_own_; }
        }
        fuse_radar();
    }

    void fuse_radar() {
        if (!mission_state_.is_active()) return;

        RadarSample tgt, own;
        {
            std::lock_guard<std::mutex> lock(radar_mtx_);
            tgt = radar_target_;
            own = radar_own_;
        }
        if (!tgt.valid || !own.valid) return;
        if (std::abs(tgt.stamp - own.stamp) > radar_pair_window_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Radar pair rejected: target/interceptor stamps %.3fs apart (radar.pair_window=%.3f)",
                std::abs(tgt.stamp - own.stamp), radar_pair_window_);
            return;
        }

        const double t_meas = std::min(tgt.stamp, own.stamp);
        const double now = this->get_clock()->now().seconds();

        std::lock_guard<std::mutex> lock(estimator_mtx_);
        if (t_meas <= radar_last_fused_t_ + 1e-6) return;   // this pair was already consumed
        radar_last_fused_t_ = t_meas;

        // Latency compensation: extrapolate the relative position from the measurement epoch to
        // `now` along the measured relative velocity. Both objects are sampled at the same instant,
        // so the ownship leg of the extrapolation is carried by v_meas as well.
        // radar_R_yaw_ rotates the radar world frame onto the IMU-referenced estimator frame; the
        // guidance mixes the two (los_az = atan2(r) - imu_yaw), so a yaw offset here flies the
        // interceptor at a constant angle off the target.
        const double lag = clamp(now - t_meas, 0.0, radar_max_delay_comp_);
        // Online yaw correction: rotate the radar measurement by the EMA of the camera-radar az diff
        // (bearing_err_ema_). This auto-calibrates the systematic frame misalignment that
        // radar.yaw_offset_deg is meant to correct — no manual calibration needed. Applied here
        // (before the UKF) so a wrong correction makes the gate TIGHTER (conservative, self-correcting)
        // rather than looser (divergent, as a gate-side correction would be).
        Eigen::Matrix3d R_online_yaw = Eigen::Matrix3d::Identity();
        if (has_bearing_err_ && std::abs(bearing_err_ema_) < 0.5) {  // sanity cap: < 28.6°
            R_online_yaw = Eigen::AngleAxisd(bearing_err_ema_, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        }
        const Eigen::Vector3d v_meas = R_online_yaw * radar_R_yaw_ * (tgt.vel - own.vel);
        const Eigen::Vector3d r_meas = R_online_yaw * radar_R_yaw_ * (tgt.pos - own.pos) + v_meas * lag;

        radar_rel_pos_ = r_meas;
        radar_rel_time_ = now;
        radar_has_rel_ = true;

        // Ownship velocity aiding. own_vel_ is otherwise an unbounded IMU integral running since the
        // MISSION trigger, and it damps the vertical loop — a long wait for the FCU to accept
        // external control is enough to poison it. This is a slow DRIFT TRIM, not a replacement:
        // the vertical loop needs own_vel_'s high-frequency content, and blending in a 10 Hz,
        // lag-delayed radar velocity at a short time constant adds enough phase lag to make that
        // loop oscillate (throttle bang-bangs between its clamps and takes the whole attitude loop
        // with it). Keep own_vel_tc long compared with the vertical-loop time constant.
        if (radar_own_vel_aiding_) {
            const Eigen::Vector3d v_own_radar = radar_R_yaw_ * own.vel;
            const double dt_aid = (radar_own_vel_t_ > 0.0)
                ? clamp(now - radar_own_vel_t_, 0.001, 1.0) : radar_own_vel_tc_;
            radar_own_vel_t_ = now;
            std::lock_guard<std::mutex> ol(odom_mtx_);
            if (!own_vel_aided_) {
                own_vel_ = v_own_radar;   // strapdown has no absolute anchor: hard-seed the first fix
                own_vel_aided_ = true;
            } else {
                own_vel_ += clamp(dt_aid / radar_own_vel_tc_, 0.0, 1.0) * (v_own_radar - own_vel_);
            }
        }

        // Drain the IMU preintegration window for this radar epoch (shared by both branches).
        Eigen::Vector3d dv_own = Eigen::Vector3d::Zero();
        {
            std::lock_guard<std::mutex> ol(odom_mtx_);
            if (last_imu_time_ < 0.0 || !yaw_align_init_) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "Radar track received but IMU not ready yet");
                return;
            }
            dv_own = dv_own_accum_;
            dv_own_accum_.setZero();
            last_dv_own_ = dv_own;
        }

        // Terminal vision is PURE vision: once vision is locked the radar node must not influence
        // the estimator/guidance at all (the radar is for MID-COURSE only). So while vision_locked_,
        // fully disengage — no radar fusion (the radar's lagged 10 Hz vel fought the 30 Hz vision
        // update and HURT in AirSim, and its position's metro noise corrupts the monocular range).
        // The radar re-engages full pos+vel only once vision is DEFINITIVELY lost (vision_locked_ ->
        // false via vision_lost_timeout), restarting mid-course relay. vel_aid_after_lock is an
        // opt-in experiment (off by default): fuse radar VELOCITY only to anchor v_closing.
        if (vision_locked_) {
            if (radar_vel_aid_after_lock_) {
                estimator_->process_radar_vel(v_meas, now, dv_own, radar_vel_aid_std_);
            }
            return;   // vision owns the guidance cache while locked
        }

        const bool was_initialized = estimator_->initialized;
        // Window the IMU accumulator actually covers — the interval since the previous estimator
        // step, whichever source produced it. Needed for a_own below: dividing by the camera dt
        // would treble it whenever radar is the only measurement source.
        const double dt_step = estimator_->has_last_cap_time
            ? clamp(now - estimator_->last_cap_time, 0.001, 0.3) : dt_cam_;
        const auto result = estimator_->process_radar(r_meas, v_meas, now, dv_own,
                                                      radar_pos_std_, radar_vel_std_);
        if (!result.first) return;
        if (!was_initialized) {
            transition_to_state(MissionState::BLIND_FLIGHT, "Radar relay acquired target");
            RCLCPP_INFO(this->get_logger(),
                "Radar handover start: r=[%.1f %.1f %.1f] |r|=%.1fm |v_rel|=%.1fm/s lag=%.0fms",
                r_meas(0), r_meas(1), r_meas(2), r_meas.norm(), v_meas.norm(), lag * 1000.0);
        }

        // Radar owns the guidance cache only when vision is NOT delivering a consistent track
        // (mid-course before visual acquisition, or after a definitive vision loss). When vision is
        // delivering the vision path owns the cache — the radar's only role then is the one-time
        // width anchor (block in update_vision_handover), not guidance.
        if (!vision_delivering(now)) {
            fill_cache_from_radar(result.second, now, dt_step);
        }

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS * 5,
            "[RADAR] meas=[%.1f %.1f %.1f] est=[%.1f %.1f %.1f] |r|=%.1f lag=%.0fms",
            r_meas(0), r_meas(1), r_meas(2),
            result.second(0), result.second(1), result.second(2),
            result.second.segment<3>(0).norm(), lag * 1000.0);
    }

    // Mid-course guidance cache from the radar-driven estimate. Same contract as the vision path
    // (bbox_size = 0 so the terminal brake stays disarmed), but the LOS rate is taken ANALYTICALLY
    // from the filtered relative state instead of differencing the bearing: at 10 Hz with
    // metre-level position noise, bearing differencing would inject ~0.3 rad/s straight into the
    // PN lateral command.
    void fill_cache_from_radar(const Eigen::VectorXd& state, double t_now, double dt_step) {
        const Eigen::Vector3d r_est = state.segment<3>(0);
        const Eigen::Vector3d v_est = smooth_rel_vel(state.segment<3>(3), t_now);

        const double los_az_world = std::atan2(r_est(1), r_est(0));
        const double los_el = std::asin(clamp(r_est(2) / std::max(r_est.norm(), 0.1), -1.0, 1.0));
        const double r_xy2 = std::max(r_est(0) * r_est(0) + r_est(1) * r_est(1), 1.0);
        const double raw_rate = (r_est(0) * v_est(1) - r_est(1) * v_est(0)) / r_xy2;
        constexpr double alpha_radar_los = 0.3;
        los_rate_ema_ = has_prev_los_
            ? (1.0 - alpha_radar_los) * los_rate_ema_ + alpha_radar_los * raw_rate
            : raw_rate;
        prev_los_az_world_ = los_az_world;
        prev_los_t_ = t_now;
        has_prev_los_ = true;

        ControlCacheData cache;
        cache.valid = true;
        cache.has_bbox = false;
        cache.r_est = r_est;
        cache.v_est = v_est;
        cache.los_az_world = los_az_world;
        cache.los_el = los_el;
        cache.los_rate = los_rate_ema_;
        cache.source_time = t_now;
        cache.update_time = this->get_clock()->now().seconds();
        cache.bbox_size = 0.0;
        // Transverse (escape) speed — see the vision path. (Not |v_rel|, which is the closing speed
        // and would defeat the brake floor during mid-course closure.)
        {
            const double r_xy = std::sqrt(r_est(0) * r_est(0) + r_est(1) * r_est(1));
            const double v_transverse = (r_xy > 1e-3)
                ? std::abs(r_est(0) * v_est(1) - r_est(1) * v_est(0)) / r_xy : 0.0;
            cache.target_speed = std::min(v_transverse, 15.0);
        }
        cache.a_own = last_dv_own_ / std::max(dt_step, 1e-3);
        {
            std::lock_guard<std::mutex> cache_lock(control_cache_mtx_);
            control_cache_ = cache;
        }
    }

    // Transition bookkeeping run on every bbox message while the radar relay is active.
    // Called with estimator_mtx_ held.
    //
    // Width handover: the monocular scale W is the one quantity a single camera cannot observe, and
    // the whole terminal range estimate r = focal*W/s rides on it. During the overlap window — the
    // target is inside the FOV AND the radar still has a valid range — W falls straight out of the
    // pinhole relation W = s * r_radar / focal. Averaging over handover_frames rejects per-frame
    // bbox noise; the result is then LOCKED so the terminal phase runs on a trusted scale instead
    // of the configured guess.
    // A detection that disagrees with the radar track is a false positive, and ONE fused phantom
    // pixel measurement is enough to walk the monocular width off (observed w_est 1.50 -> 0.66 with
    // the estimate slewing tens of metres). Angular check always; the range check only once the
    // width is trusted, because r = focal*W/s is meaningless before that.
    bool bbox_matches_radar(const Eigen::Vector3d& los_nav, double bbox_s, double* d_az_out) {
        if (!radar_bbox_gate_) return true;
        const bool radar_fresh = radar_has_rel_ &&
            (this->get_clock()->now().seconds() - radar_rel_time_) < radar_timeout_;
        const double r_radar = radar_rel_pos_.norm();
        if (!radar_fresh || r_radar < 1.0 || los_nav.norm() < 0.5) return true;

        const Eigen::Vector3d u_radar = radar_rel_pos_ / r_radar;
        const double az_cam = std::atan2(los_nav(1), los_nav(0));
        const double az_radar = std::atan2(u_radar(1), u_radar(0));
        if (d_az_out) {
            *d_az_out = std::atan2(std::sin(az_cam - az_radar), std::cos(az_cam - az_radar));
        }

        const double d_ang = std::acos(clamp(los_nav.normalized().dot(u_radar), -1.0, 1.0));
        // Range- and motion-dependent bearing tolerance. At close range with a fast-crossing target,
        // the residual radar delay introduces a bearing error ~ v_perp * lag / r. Scale by a safety
        // factor and add to the base tolerance. This also accommodates the radar yaw offset + noise
        // (the base is widened to cover the ~6° steady offset the [FRAME] log reports). Together they
        // break the reject->no-fusion->drift->reject feedback loop that grows the error to 35°.
        double v_perp_gate = 0.0;
        if (estimator_ && estimator_->initialized) {
            const auto& x = estimator_->imm_ukf->x;
            const double r_xy = std::sqrt(x(0)*x(0) + x(1)*x(1));
            v_perp_gate = (r_xy > 1e-3)
                ? std::abs(x(0)*x(4) - x(1)*x(3)) / r_xy : 0.0;   // |r_xy × v_xy| / |r_xy|
        }
        const double motion_tol = controller_->motion_gate_factor_ * v_perp_gate *
            radar_max_delay_comp_ / std::max(r_radar, 1.0);
        // pos_noise_tol DISABLED: the yaw auto-correction above removes the systematic 6° offset,
        // leaving only ~5-7° random noise — well within the 11.5° base gate. A range-dependent
        // pos-noise term was tried but it over-widened at close range and let false positives through.
        const double pos_noise_tol = 0.0;
        const double bearing_limit = handover_max_bearing_err_ + motion_tol + pos_noise_tol;
        if (d_ang > bearing_limit) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                "[BBOX-REJECT] LOS %.1fdeg off (limit %.1f+%.1f+%.1f=%.1f, r=%.0f v_perp=%.1f bbox=%.0fpx)",
                d_ang * 180.0 / M_PI, handover_max_bearing_err_ * 180.0 / M_PI,
                motion_tol * 180.0 / M_PI, pos_noise_tol * 180.0 / M_PI,
                bearing_limit * 180.0 / M_PI, r_radar, v_perp_gate, bbox_s);
            return false;
        }
        if (width_locked_ && bbox_s > 1.0) {
            const double r_vis = focal_length_ * estimator_->imm_ukf->x(9) / bbox_s;
            const double ratio = r_vis / r_radar;
            if (ratio > radar_bbox_range_ratio_ || ratio < 1.0 / radar_bbox_range_ratio_) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                    "[BBOX-REJECT] implied range %.0fm vs radar %.0fm, bbox=%.0fpx",
                    r_vis, r_radar, bbox_s);
                return false;
            }
        }
        return true;
    }

    // Called with only radar-validated frames. bbox_ok false means "no usable detection this frame"
    // — either the detector saw nothing or the detection failed the radar cross-check.
    void update_vision_handover(bool bbox_ok, double bbox_s, double t_now, double d_az) {
        if (!bbox_ok) {
            vision_frames_ = 0;   // the vision LOCK still requires an unbroken run (false-positive guard)
            vision_lock_t0_ = -1.0;
            // The width buffer is NOT cleared here: a single transient missed frame (radar stall,
            // one rejected bbox) no longer throws away the accumulated width samples. The rolling
            // window below ages them out instead, so the width lock survives brief gaps.
            if (vision_locked_ && (t_now - last_vision_time_) > vision_lost_timeout_) {
                vision_locked_ = false;
                RCLCPP_WARN(this->get_logger(),
                    "Vision gone %.1fs — radar relay owns guidance again",
                    t_now - last_vision_time_);
            }
            return;
        }
        last_vision_time_ = t_now;

        const bool radar_fresh = radar_has_rel_ &&
            (this->get_clock()->now().seconds() - radar_rel_time_) < radar_timeout_;

        // Track the SYSTEMATIC yaw offset (constant frame misalignment). Subtract the expected
        // motion-induced az diff (v_perp * delay / r) from d_az before updating the EMA — this
        // removes the target-motion component so the EMA converges to the constant offset only.
        // The EMA is then applied in fuse_radar as an online yaw correction (rotating the radar
        // measurement to match the camera frame), auto-calibrating radar.yaw_offset_deg.
        if (radar_fresh && std::isfinite(d_az)) {
            double d_az_sys = d_az;
            if (estimator_ && estimator_->initialized) {
                const auto& xs = estimator_->imm_ukf->x;
                const double rxy = std::sqrt(xs(0)*xs(0) + xs(1)*xs(1));
                if (rxy > 1e-3) {
                    const double vp = std::abs(xs(0)*xs(4) - xs(1)*xs(3)) / rxy;
                    const double motion_az = vp * radar_max_delay_comp_ / rxy;
                    d_az_sys = d_az - std::copysign(motion_az, d_az);
                }
            }
            bearing_err_ema_ = has_bearing_err_ ? (0.92 * bearing_err_ema_ + 0.08 * d_az_sys) : d_az_sys;
            has_bearing_err_ = true;
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "[FRAME] cam-radar az diff=%.1f deg (motion-corrected %.1f, yaw_ema %.1f)",
                d_az * 180.0 / M_PI, d_az_sys * 180.0 / M_PI, bearing_err_ema_ * 180.0 / M_PI);
        }
        if (!width_locked_ && radar_fresh && bbox_s >= width_min_bbox_) {
            // Filtered, radar-anchored range (the same quantity the controller trusts) — strictly
            // lower-variance than the raw single-sample radar_rel_pos_, so the locked width does not
            // inherit per-sample latency-extrapolation noise/bias. Runs before process_camera, i.e. on
            // the pre-camera-update (radar-predicted) blended state — exactly the independent anchor.
            const double r_filter = estimator_ ? estimator_->imm_ukf->x.head<3>().norm() : 0.0;
            if (r_filter > 1.0 && r_filter < handover_max_range_) {
                const double w_meas = clamp(bbox_s * r_filter / focal_length_, width_min_, width_max_);
                width_samples_.push_back({w_meas, t_now});
                // Rolling window: age out stale samples so a transient missed frame (which no longer
                // wipes the buffer) does not leave stale data, and the median tracks a moving range.
                width_samples_.erase(
                    std::remove_if(width_samples_.begin(), width_samples_.end(),
                                   [&](const WidthSample& s){ return (t_now - s.t) > width_window_s_; }),
                    width_samples_.end());
                if (estimator_) estimator_->w_guess = w_meas;   // helps a monocular-only re-init
                if (static_cast<int>(width_samples_.size()) >= handover_frames_) {
                    // MAD outlier rejection: a phantom bbox that passes the bearing gate but has a wild
                    // size is rejected instead of contaminating the median (observed w_est 1.5 -> 0.66).
                    std::vector<double> ws;
                    for (const auto& s : width_samples_) ws.push_back(s.w);
                    std::sort(ws.begin(), ws.end());
                    const double med = ws[ws.size() / 2];
                    double mad = 0.0;
                    for (double w : ws) mad += std::abs(w - med);
                    mad /= std::max((size_t)1, ws.size());
                    constexpr double kappa = 3.0;
                    std::vector<double> kept;
                    for (const auto& s : width_samples_) {
                        if (mad < 1e-6 || std::abs(s.w - med) <= kappa * 1.4826 * mad) kept.push_back(s.w);
                    }
                    if (!kept.empty()) std::sort(kept.begin(), kept.end());
                    const double w_med = kept.empty() ? med : kept[kept.size() / 2];
                    // Apply the detector over-bounding correction (width_scale) and clamp. A visual
                    // tracker systematically over-bounds small targets (observed W~2.26 vs ~1.5 at
                    // 186 m / 16 px); width_scale = true_W / locked_W calibrates that multiplicative
                    // bias. The width_min_bbox gate above already shifted the anchor closer (larger,
                    // more accurate bboxes) which shrinks any additive padding bias.
                    const double w_fused = clamp(width_scale_ * w_med, width_min_, width_max_);
                    if (estimator_) estimator_->lock_width(w_fused);
                    width_locked_ = true;
                    RCLCPP_INFO(this->get_logger(),
                        "[HANDOVER] Target width fixed from radar range: W=%.2fm "
                        "(median %.2f, scale %.3f, kept %zu/%zu, r_filter=%.1fm bbox=%.1fpx focal=%.0f)",
                        w_fused, w_med, width_scale_, kept.size(), ws.size(), r_filter, bbox_s, focal_length_);
                }
            } else {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Width fix skipped: filtered range %.0fm outside (1, %.0f]m",
                    r_filter, handover_max_range_);
            }
        }

        if (!vision_locked_) {
            if (vision_frames_ == 0) vision_lock_t0_ = t_now;
            ++vision_frames_;
            // Frame count AND elapsed time: a burst of consecutive false positives can clear a pure
            // frame count in a fraction of a second, which is how a phantom took the terminal phase.
            const bool run_ok = vision_frames_ >= handover_frames_ &&
                                (t_now - vision_lock_t0_) >= lock_min_duration_;
            // Handing the endgame to vision before the radar has fixed the width would run the
            // terminal range r = focal*W/s on the configured GUESS, and nothing would pin W
            // afterwards — it drifts with every update. Only skip this when there is no radar at all.
            const bool width_ready = width_locked_ || !radar_has_rel_;
            if (run_ok && width_ready) {
                vision_locked_ = true;
                had_terminal_attempt_ = true;   // a terminal phase began — gates re-intercept on a later vision loss
                RCLCPP_INFO(this->get_logger(),
                    "[HANDOVER] Visual lock: %d validated frames over %.1fs, W=%.2fm%s "
                    "— terminal guidance on vision, radar standing by",
                    vision_frames_, t_now - vision_lock_t0_,
                    estimator_ ? estimator_->imm_ukf->x(9) : 0.0,
                    width_locked_ ? " (radar-fixed)" : " (guess, no radar)");
                transition_to_state(MissionState::TRACKING, "Vision lock acquired");
            } else if (run_ok) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Visual run stable (%d frames); width not radar-fixed yet (vision guidance "
                    "already active on w_guess, W anchor pending)", vision_frames_);
            }
        }
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
        // Use the bbox WIDTH only. The pinhole size model s = focal*W/depth treats s as the
        // projection of a FIXED physical width W; the target's image height varies with viewing
        // angle and attitude (a banked/tilted drone), so max(size_x,size_y) can latch onto the
        // unstable height and inflate/corrupt the scale. size_x is the stable width dimension and
        // is what every consumer of s (the UKF size channel, the radar width anchor, the terminal
        // brake, the size gate) actually models. (The Python sim publishes size_x == size_y, so
        // this is a no-op there; it matters for AirSim's real tracker bbox.)
        const double s = msg->bbox.size_x;

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

        // Needs the rebuilt R_b2c_, so it runs here rather than at message intake.
        bool bbox_ok = has_bbox;
        if (radar_enable_) {
            double d_az = std::numeric_limits<double>::quiet_NaN();
            // Radar cross-check is a HANDOVER-time false-positive guard only. Once vision is locked
            // the target is terminal-authoritative: at close range the radar's metre-level noise
            // makes the camera-radar bearing disagree by 12-30 deg for the REAL target, so gating
            // then REJECTS good bboxes and forces a vision dropout right before intercept. Trust
            // vision unconditionally once locked; the estimator's close-range size gate backstops FPs.
            if (has_bbox && !vision_locked_) {
                const Eigen::Vector3d los_nav_meas =
                    pixel_to_los_nav(u, v, focal_length_, R_b2n, R_b2c_);
                bbox_ok = bbox_matches_radar(los_nav_meas, s, &d_az);
            }
            update_vision_handover(bbox_ok, s, t_now, d_az);
        }

        const bool was_initialized = estimator_->initialized;
        const auto result = estimator_->process_camera(bbox_ok, u, v, s, t_now, R_b2n, dv_own);

        if (!was_initialized && estimator_->initialized) {
            transition_to_state(MissionState::TRACKING, "Target initialized");
            had_terminal_attempt_ = true;   // camera-only path: vision owns guidance from frame 0
        }
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS,
            "bbox=%d%s size=%.1f init=%d->%d", has_bbox, (has_bbox && !bbox_ok) ? "(rejected)" : "",
            s, was_initialized, estimator_->initialized);

        if (!estimator_->initialized) return;

        // Vision owns the guidance cache as soon as a CONSISTENT bbox is in hand (the radar is for
        // mid-course when vision can't see the target, and for the one-time width anchor). Once
        // vision_locked_ the cache fills every frame (even dropouts, has_bbox=false -> COAST). Pre-
        // lock, fill only when a consistent bbox is available — the radar owns the cache otherwise.
        if (!vision_locked_ && !bbox_ok) return;

        // Relative-state estimator: x[0:3] IS the relative position, x[3:6] the relative velocity.
        Eigen::Vector3d r_est = result.second.segment<3>(0);
        const Eigen::Vector3d rel_vel_now = smooth_rel_vel(result.second.segment<3>(3), t_now);
        const Eigen::Vector3d v_est = rel_vel_now;   // relative velocity (no ownship vel available)

        double los_az_world = std::atan2(r_est(1), r_est(0));
        double los_el = std::asin(clamp(r_est(2) / std::max(r_est.norm(), 0.1), -1.0, 1.0));

        if (bbox_ok) {
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
        cache.has_bbox = bbox_ok;
        cache.r_est = r_est;
        cache.v_est = v_est;
        cache.los_az_world = los_az_world;
        cache.los_el = los_el;
        cache.los_rate = los_rate_ema_;
        cache.source_time = t_now;
        cache.update_time = this->get_clock()->now().seconds();
        cache.bbox_size = bbox_ok ? s : 0.0;
        // TRANSVERSE (escape) component of the relative velocity — the part that actually justifies
        // not braking (a target sliding sideways out of the FOV). The full |v_rel| is dominated by
        // the CLOSING speed (≈ own speed), so using it for the brake floor was circular: the faster
        // the closure, the higher the floor, the less it braked — the interceptor barrelled in at
        // v_des and collided. Use only the transverse component (r_xy × v_xy)/|r_xy|.
        {
            const double r_xy = std::sqrt(r_est(0) * r_est(0) + r_est(1) * r_est(1));
            const double v_transverse = (r_xy > 1e-3)
                ? std::abs(r_est(0) * rel_vel_now(1) - r_est(1) * rel_vel_now(0)) / r_xy : 0.0;
            cache.target_speed = std::min(v_transverse, 15.0);
        }
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

        // Silent radar is otherwise invisible: the node just behaves like the vision-only variant.
        if (radar_enable_ && !radar_has_rel_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Radar ENABLED but no usable track yet (%s: %lu msgs, %s: %lu msgs)",
                radar_topic_target_.c_str(), radar_msg_target_,
                radar_topic_own_.c_str(), radar_msg_own_);
        }

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

        // ---- Re-intercept (radar.reintercept_enable) ----------------------------------------
        // On a terminal vision-loss MISS (target gone > vision_lost_timeout AFTER a real terminal
        // phase) that was NOT a clean hit (closest approach > kill_range), if the radar has
        // re-acquired the target at a standoff and attempts remain: clear the capture/dropout
        // latches and resume radar mid-course guidance for another pass. The radar cache is already
        // being filled again (vision_locked_ flipped false at vision_lost_timeout, reopening fuse_radar
        // + fill_cache_from_radar). Placed BEFORE the capture/dropout blocks so the reset clears both,
        // then execution falls through to compute_control. Self-debouncing: had_terminal_attempt_ is
        // cleared here and only re-armed on the next vision lock.
        {
            const double prev_min = min_range_seen_;
            const bool terminal_miss = had_terminal_attempt_ && !vision_delivering(now.seconds()) &&
                                       min_range_seen_ > reintercept_kill_range_;
            const bool radar_reacquired = radar_enable_ && radar_has_rel_ &&
                                          radar_rel_pos_.norm() > reintercept_min_range_;
            if (reintercept_enable_ && terminal_miss && radar_reacquired &&
                reintercept_count_ < reintercept_max_) {
                ++reintercept_count_;
                captured_ = false;
                has_frozen_cmd_ = false;
                last_good_range_ = 1e9;
                min_range_seen_ = 1e9;
                min_vision_range_ = 1e9;
                last_bbox_time_ = 0.0;
                {
                    std::lock_guard<std::mutex> ol(odom_mtx_);
                    v_own_brake_.setZero();
                    brake_vel_seeded_ = false;
                }
                vision_locked_ = false;          // radar owns guidance again (force; likely already false)
                had_terminal_attempt_ = false;   // re-arms on the next vision lock
                transition_to_state(MissionState::BLIND_FLIGHT, "Re-intercept: radar relay re-engaged");
                RCLCPP_WARN(this->get_logger(),
                    "[REINTERCEPT] attempt %d/%d after terminal miss (min %.1fm > kill %.1fm); "
                    "radar standoff %.1fm — resuming mid-course relay",
                    reintercept_count_, reintercept_max_, prev_min, reintercept_kill_range_,
                    radar_rel_pos_.norm());
                // fall through to normal guidance (the radar-filled cache drives compute_control)
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
            // Vision-only running min: anchors the fly-through (range_reopened) detector. The
            // estimator/radar cache range is INCONSISTENT with vision during a radar re-engage — radar
            // drives min_range_seen_ down as the interceptor closes, but vision at handover still
            // measures the standoff range, so min_range_seen_ would false-trigger a reopen the instant
            // vision handsover. min_vision_range_ advances only on real vision frames, so a reopen
            // requires a genuine vision close approach first.
            if (cache.has_bbox) {
                min_vision_range_ = std::min(min_vision_range_, cache.r_est.norm());
            }
        }
        // Reopen: range bottomed out and has since grown past it — the UAV flew through the target.
        // Robust under a noisy close-range estimate (high-noise scenario) where los_az never cleanly
        // flips past 90 deg. Keyed on min_vision_range_ (vision-confirmed) so it only arms after a real
        // vision close approach — immune to the radar-cache inconsistency during radar re-engage.
        const bool range_reopened = has_frozen_cmd_ && min_vision_range_ < 15.0 &&
            cache.valid && cache.r_est.norm() > min_vision_range_ + 8.0;
        // Capture is a terminal-phase concept: only arm it once a confirmed vision terminal pass has
        // begun this engagement (had_terminal_attempt_). This suppresses a spurious range_reopened
        // capture during radar re-engage right after a re-intercept, where the noisy post-dropout
        // estimate dips min_range_seen_ < 15 then jumps (reads as a fly-through but is just the
        // re-approach). Legitimate captures (close approach / overshot / real fly-through) always
        // occur after vision engaged, so had_terminal_attempt_ is true for them.
        if (had_terminal_attempt_ && !captured_ && cache.valid &&
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
                                                       cache.los_rate, cache.a_own, boresight_el_meas,
                                                       (radar_enable_ && radar_has_rel_ && !vision_delivering(now.seconds())));
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
            vision_locked_ = !radar_enable_;
            vision_frames_ = 0;
            last_vision_time_ = -1.0;
            width_locked_ = false;
            width_samples_.clear();
            radar_last_fused_t_ = -1.0;
            radar_rel_time_ = -1.0;
            radar_has_rel_ = false;
            radar_rel_pos_.setZero();
            radar_own_vel_t_ = -1.0;
            own_vel_aided_ = false;
            bearing_err_ema_ = 0.0;
            has_bearing_err_ = false;
            vision_lock_t0_ = -1.0;
            radar_msg_target_ = 0;
            radar_msg_own_ = 0;
            {
                std::lock_guard<std::mutex> rl(radar_mtx_);
                radar_target_ = RadarSample();
                radar_own_ = RadarSample();
            }
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
            min_vision_range_ = 1e9;
            reintercept_count_ = 0;
            had_terminal_attempt_ = false;
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
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr radar_target_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr radar_own_sub_;
    rclcpp::Publisher<quadrotor_msgs::msg::ControlCommand>::SharedPtr control_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr mission_state_pub_;

    // ---- Radar relay state -------------------------------------------------------------------
    struct RadarSample {
        bool valid = false;
        double stamp = 0.0;                              // measurement epoch (NOT arrival time)
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();    // world ENU
        Eigen::Vector3d vel = Eigen::Vector3d::Zero();    // world ENU
    };
    std::mutex radar_mtx_;
    RadarSample radar_target_;
    RadarSample radar_own_;
    bool radar_enable_ = false;
    std::string radar_topic_target_ = "/fpv/radar/target";
    std::string radar_topic_own_ = "/fpv/radar/interceptor";
    double radar_pos_std_ = 3.0;
    double radar_vel_std_ = 1.0;
    double radar_pair_window_ = 0.15;
    double radar_timeout_ = 1.0;
    double radar_max_delay_comp_ = 0.3;
    double radar_max_range_ = 800.0;
    bool radar_disable_after_lock_ = true;
    bool radar_vel_aid_after_lock_ = false;   // OFF: pure terminal vision by default (radar vel aid is an opt-in experiment)
    double radar_vel_aid_std_ = 2.5;         // m/s, 1-sigma of the vel-only aid
    double radar_vision_hold_ = 0.35;
    bool radar_own_vel_aiding_ = true;
    double radar_own_vel_tc_ = 5.0;
    double radar_own_vel_t_ = -1.0;
    bool own_vel_aided_ = false;
    double handover_max_bearing_err_ = 0.20;
    bool radar_bbox_gate_ = true;
    double radar_bbox_range_ratio_ = 2.0;
    double lock_min_duration_ = 0.7;
    double vision_lock_t0_ = -1.0;
    uint64_t radar_msg_target_ = 0;
    uint64_t radar_msg_own_ = 0;
    Eigen::Matrix3d radar_R_yaw_ = Eigen::Matrix3d::Identity();
    // Camera-vs-radar LOS azimuth discrepancy. A steady nonzero value is the radar/IMU world-frame
    // yaw mismatch and should be dialled out with radar.yaw_offset_deg.
    double bearing_err_ema_ = 0.0;
    bool has_bearing_err_ = false;
    // Last fused relative position + when it was fused (guarded by estimator_mtx_). Kept even after
    // the visual lock stops radar updates, because it is the range reference for the width handover.
    double radar_last_fused_t_ = -1.0;
    Eigen::Vector3d radar_rel_pos_ = Eigen::Vector3d::Zero();
    double radar_rel_time_ = -1.0;
    bool radar_has_rel_ = false;
    // Radar -> vision handover (guarded by estimator_mtx_).
    int handover_frames_ = 5;
    double handover_max_range_ = 120.0;
    double width_min_ = 0.3;
    double width_max_ = 5.0;
    double vision_lost_timeout_ = 1.5;
    bool vision_locked_ = true;      // radar disabled => vision owns guidance from the first frame
    int vision_frames_ = 0;
    double last_vision_time_ = -1.0;
    bool width_locked_ = false;
    struct WidthSample { double w; double t; };
    std::vector<WidthSample> width_samples_;
    double width_window_s_ = 1.5;   // s, rolling window for the width-median buffer (survives transient misses)
    double width_min_bbox_ = 25.0;  // px, minimum bbox size to anchor the radar width (tiny far-range bboxes are over-bound)
    double width_scale_ = 1.0;      // multiplicative correction for tracker over-bounding, applied to the locked width

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
    // Vision-only running min of r_est (advances only on cache.has_bbox frames). Anchors
    // range_reopened so the fly-through detector isn't poisoned by the radar/estimator cache during
    // a radar re-engage. Reset on mission launch and on each re-intercept.
    double min_vision_range_ = 1e9;
    // Re-intercept (radar.reintercept_enable): on a terminal vision-loss miss, re-engage the radar
    // relay for another pass instead of holding forever. had_terminal_attempt_ gates it to a vision
    // loss that followed a real terminal phase; reintercept_count_ bounds the attempts.
    bool reintercept_enable_ = false;
    int  reintercept_max_ = 2;
    double reintercept_kill_range_ = 1.0;
    double reintercept_min_range_ = 15.0;
    int  reintercept_count_ = 0;
    bool had_terminal_attempt_ = false;
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
