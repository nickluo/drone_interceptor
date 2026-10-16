#pragma once

// Relative-state IMM-UKF target estimator stack.
//
// State (10D): x = [r(3), v_rel(3), a_t(3), W(1)] in a gravity-aligned world frame G.
//   r     — RELATIVE position (target - ownship); ownship pos/vel are NOT state.
//   v_rel — relative velocity (target - ownship).
//   a_t   — target OWN acceleration.
//   W     — target physical width (m), the monocular scale anchor.
// Ownship motion enters each predict as an exogenous, IMU-preintegrated acceleration
// a_own (dv_own/dt, gravity-corrected), so the relative quantities stay drift-free:
// every bbox update closes the loop on r.
//
// This is the C++ mirror of the verified Python prototype
// (prototype/sim_test_relative_imm_ukf.py).

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "utils.h"

// State/measurement dimensions shared by the IMM-UKF stack.
inline constexpr int kStateDim = 10;   // [r(3), v_rel(3), a_t(3), W(1)]
inline constexpr int kMeasDim = 3;     // pixel measurement [u, v, size]
inline constexpr int kModeCount = 3;   // IMM motion models: Singer / coordinated turn / rotor

// ============================================================================
// Single-mode 10D UKF. IMM runs one of these per motion model; every filter
// shares the state layout above. 2N+1 sigma points, robust Cholesky (LLT with
// eigendecomposition fallback) for the square-root of the covariance.
// ============================================================================
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

    static Eigen::VectorXd f_rotor(const Eigen::VectorXd& x, double dt, const Eigen::Vector3d& a_own) {
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
        // The model is therefore pure accel-relaxation with tau=2s. (config physics.drag_coeff
        // models the OWNship drag in the Python sim harness only; the estimator consumes no drag
        // coefficient.) Do NOT add a drag term without first expressing it as a function of an
        // observable target-velocity estimate.
        x_new.segment<3>(0) = x.segment<3>(0) + v * dt + 0.5 * a_rel * dt * dt;
        x_new.segment<3>(3) = v + a_rel * dt;
        x_new.segment<3>(6) = a_t * std::exp(-dt / 2.0);
        x_new(9) = x(9);
        return x_new;
    }

    Eigen::VectorXd apply_model(const Eigen::VectorXd& x, double dt, const Eigen::Vector3d& a_own) {
        if (mode == 0) return f_singer(x, dt, a_own);
        if (mode == 1) return f_ct(x, dt, a_own);
        return f_rotor(x, dt, a_own);
    }
public:
    Eigen::VectorXd x = Eigen::VectorXd::Zero(kStateDim);
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(kStateDim, kStateDim);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(kStateDim, kStateDim);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    int mode = 0;
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

// ============================================================================
// IMM coordinator over the kModeCount motion models: mixing probabilities,
// interacting (mixed-state) predict, per-model likelihood update, fused blend.
// ============================================================================
class RobustIMMEstimator10D {
public:
    // Block-diagonal process noise: [q_pos x3, q_vel x3, q_acc x3, q_w].
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

    std::vector<UnscentedKalmanFilter10D> filters;
    Eigen::Vector3d mu;
    Eigen::Matrix3d M;
    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    Eigen::Vector3d cbar;
    Eigen::Matrix3d omega;

    RobustIMMEstimator10D() {
        filters.resize(kModeCount);
        for (int i = 0; i < kModeCount; ++i) {
            filters[i].mode = i;
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

// ============================================================================
// Relative IMM-UKF estimator: measurement intake, single shared time base,
// process-noise scheduling, width freeze/lock, and divergence sanitization
// around the RobustIMMEstimator10D core.
// ============================================================================
class RelativeIMMUKFEstimator {
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
                            const Eigen::Matrix3d& R_b2c_val)
        : dt_cam(dt_cam_val), focal_length(focal_length_val), w_guess(W_guess_val), R_b2c(R_b2c_val)
    {
        imm_ukf = std::make_unique<RobustIMMEstimator10D>();
    }

    // Pixel measurement -> unit LOS ray in the world (nav) frame. Pinhole back-projection through
    // the camera extrinsic R_b2c, then rotated into the world frame by the ownship attitude R_b2n.
    static Eigen::Vector3d pixel_to_los_nav(double u, double v, double focal_length,
                                            const Eigen::Matrix3d& R_b2n,
                                            const Eigen::Matrix3d& R_b2c) {
        Eigen::Vector3d ray_cam(u / focal_length, v / focal_length, 1.0);
        ray_cam.normalize();
        const Eigen::Vector3d ray_body = R_b2c.transpose() * ray_cam;
        return (R_b2n * ray_body).normalized();
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
        imm_ukf->filters[0].Q = RobustIMMEstimator10D::make_Q(0.05 * q_scale, q_vel, 1.0 * q_scale, q_w_dynamic * q_scale);
        imm_ukf->filters[1].Q = RobustIMMEstimator10D::make_Q(0.05 * q_scale, q_vel, 0.5 * q_scale, q_w_dynamic * q_scale);
        imm_ukf->filters[2].Q = RobustIMMEstimator10D::make_Q(0.05 * q_scale, q_vel, 1.0 * q_scale, q_w_dynamic * q_scale);
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
