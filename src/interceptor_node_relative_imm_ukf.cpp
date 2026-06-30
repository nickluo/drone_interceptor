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

double clamp(double value, double low, double high) {
    return std::max(low, std::min(high, value));
}

Eigen::MatrixXd robust_cholesky(const Eigen::MatrixXd& matrix) {
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

Eigen::MatrixXd make_Q(double q_pos, double q_vel, double q_acc, double q_w) {
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(kStateDim, kStateDim);
    for (int i = 0; i < 3; ++i) {
        Q(i, i) = q_pos;
        Q(i + 3, i + 3) = q_vel;
        Q(i + 6, i + 6) = q_acc;
    }
    Q(9, 9) = q_w;
    return Q;
}

Eigen::VectorXd f_singer(const Eigen::VectorXd& x, double dt) {
    Eigen::VectorXd x_new = x;
    const double decay = std::exp(-dt / 1.5);
    x_new.segment<3>(6) = x.segment<3>(6) * decay;
    x_new.segment<3>(3) = x.segment<3>(3) + x.segment<3>(6) * dt;
    x_new.segment<3>(0) = x.segment<3>(0) + x.segment<3>(3) * dt + 0.5 * x.segment<3>(6) * dt * dt;
    x_new(9) = x(9);
    return x_new;
}

Eigen::VectorXd f_ct(const Eigen::VectorXd& x, double dt) {
    Eigen::VectorXd x_new = x;
    const Eigen::Vector3d p = x.segment<3>(0);
    const Eigen::Vector3d v = x.segment<3>(3);
    const Eigen::Vector3d a = x.segment<3>(6);
    const double v_xy2 = std::max(v(0) * v(0) + v(1) * v(1), 1e-4);
    const double omega = clamp((v(0) * a(1) - v(1) * a(0)) / v_xy2, -2.0, 2.0);

    if (std::abs(omega) < 1e-3) {
        x_new.segment<3>(0) = p + v * dt + 0.5 * a * dt * dt;
        x_new.segment<3>(3) = v + a * dt;
    } else {
        const double sin_wt = std::sin(omega * dt);
        const double cos_wt = std::cos(omega * dt);
        x_new(0) = p(0) + (v(0) * sin_wt - v(1) * (1.0 - cos_wt)) / omega;
        x_new(1) = p(1) + (v(1) * sin_wt + v(0) * (1.0 - cos_wt)) / omega;
        x_new(2) = p(2) + v(2) * dt + 0.5 * a(2) * dt * dt;
        x_new(3) = v(0) * cos_wt - v(1) * sin_wt;
        x_new(4) = v(0) * sin_wt + v(1) * cos_wt;
        x_new(5) = v(2) + a(2) * dt;
        x_new(6) = a(0) * cos_wt - a(1) * sin_wt;
        x_new(7) = a(0) * sin_wt + a(1) * cos_wt;
        x_new(8) = a(2);
    }
    x_new(9) = x(9);
    return x_new;
}

Eigen::VectorXd f_rotor(const Eigen::VectorXd& x, double dt) {
    Eigen::VectorXd x_new = x;
    const Eigen::Vector3d v = x.segment<3>(3);
    const Eigen::Vector3d a_thrust = x.segment<3>(6);
    const double k_drag = 0.05;
    const Eigen::Vector3d a_real = a_thrust - k_drag * v.norm() * v;
    x_new.segment<3>(0) = x.segment<3>(0) + v * dt + 0.5 * a_real * dt * dt;
    x_new.segment<3>(3) = v + a_real * dt;
    x_new.segment<3>(6) = a_thrust * std::exp(-dt / 2.0);
    x_new(9) = x(9);
    return x_new;
}

Eigen::VectorXd apply_model(const Eigen::VectorXd& x, double dt, int mode) {
    if (mode == 0) return f_singer(x, dt);
    if (mode == 1) return f_ct(x, dt);
    return f_rotor(x, dt);
}

Eigen::Vector3d h_meas_pixel(const Eigen::VectorXd& x, const Eigen::Vector3d& uav_pos, const Eigen::Matrix3d& R_n2c, double focal_length) {
    const Eigen::Vector3d rel_p = x.segment<3>(0) - uav_pos;
    Eigen::Vector3d p_cam = R_n2c * rel_p;
    p_cam(2) = std::max(p_cam(2), 0.5);
    const double u = focal_length * p_cam(0) / p_cam(2);
    const double v = focal_length * p_cam(1) / p_cam(2);
    // Clamp W_est to physical range to prevent negative-width sigma points from
    // creating asymmetric measurement predictions that bias the UKF update.
    const double W_est = clamp(x(9), 0.9, 2.5);
    const double s = focal_length * W_est / p_cam(2);
    return Eigen::Vector3d(u, v, s);
}

Eigen::Vector3d pixel_to_los_nav(double u, double v, double focal_length,
                                 const Eigen::Matrix3d& R_b2n, const Eigen::Matrix3d& R_b2c) {
    Eigen::Vector3d ray_cam(u / focal_length, v / focal_length, 1.0);
    ray_cam.normalize();
    const Eigen::Vector3d ray_body = R_b2c.transpose() * ray_cam;
    return (R_b2n * ray_body).normalized();
}

}  // namespace


class UnscentedKalmanFilter10D {
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

    void predict(double dt) {
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
        for (int i = 0; i <= 2 * n; ++i) {
            sigmas_f[i] = apply_model(sigmas[i], dt, mode);
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

    double update(const Eigen::Vector3d& z_pixel, const Eigen::Vector3d& uav_pos, const Eigen::Matrix3d& R_n2c, double focal_length) {
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
            sigmas_h[i] = h_meas_pixel(sigmas_f[i], uav_pos, R_n2c, focal_length);
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
public:
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

    void predict(double dt) {
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
            filters[j].predict(dt);
        }
        compute_mixing_probabilities();
        compute_state_estimate();
    }

    void update(const Eigen::Vector3d& z_pixel, const Eigen::Vector3d& uav_pos, const Eigen::Matrix3d& R_n2c, double focal_length) {
        Eigen::Vector3d likelihoods;
        for (int j = 0; j < kModeCount; ++j) {
            likelihoods(j) = filters[j].update(z_pixel, uav_pos, R_n2c, focal_length);
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
public:
    double dt_cam;
    double focal_length;
    double W_guess;
    Eigen::Matrix3d R_b2c;
    // Range below which the monocular width/scale is frozen. Mid-course W is well observable
    // (~1.5); freezing it before the terminal phase (default 15 m) stops a bad close-range
    // detection (motion blur / partial bbox) from corrupting scale (observed W 1.5 -> 2.8 after a
    // dropout+reacquire -> phantom target 11 m below -> dive). Old value ~5 m was too late.
    double width_freeze_range_ = 15.0;

    RobustIMMEstimator10D imm_ukf;
    bool initialized = false;
    double frozen_width = 0.0;
    bool has_frozen_width = false;

    double last_cap_time = 0.0;
    bool has_last_cap_time = false;

    // EMA-smoothed target velocity (world frame). PN guidance is sensitive to vel noise; the raw
    // UKF vel estimate jumps frame-to-frame and excites control jitter. Smooth before feeding the
    // controller. Initialized lazily on first valid estimate.
    Eigen::Vector3d v_est_smoothed = Eigen::Vector3d::Zero();
    bool has_v_smoothed = false;
    // Soft-recovery: first measurement after a bbox dropout uses inflated measurement noise so the
    // large innovation does not slam the state (especially vel) to an extreme value.
    bool first_meas_after_dropout = false;

    RelativeIMMUKFEstimator(double dt_cam_val, double focal_length_val, double W_guess_val,
                            const Eigen::Matrix3d& R_b2c_val)
        : dt_cam(dt_cam_val), focal_length(focal_length_val), W_guess(W_guess_val), R_b2c(R_b2c_val) {}

    void update_process_noise(double r_norm, double W_est, double bbox_s = 0.0) {
        const double q_scale = 1.0;
        const double r_freeze = width_freeze_range_;
        const double snr_factor = bbox_s > 0.0 ? clamp(bbox_s / 40.0, 0.1, 1.0) : 1.0;
        const double q_w_scale = std::pow(clamp((r_norm - r_freeze) / 20.0, 0.0, 1.0), 2.0);
        const double q_w_dynamic = 0.005 * q_w_scale * snr_factor;
        imm_ukf.filters[0].Q = make_Q(0.05 * q_scale, 0.05 * q_scale, 1.0 * q_scale, q_w_dynamic * q_scale);
        imm_ukf.filters[1].Q = make_Q(0.05 * q_scale, 0.05 * q_scale, 0.5 * q_scale, q_w_dynamic * q_scale);
        imm_ukf.filters[2].Q = make_Q(0.05 * q_scale, 0.05 * q_scale, 1.0 * q_scale, q_w_dynamic * q_scale);
    }

    void sanitize_filters(const Eigen::Vector3d& uav_pos) {
        bool should_freeze = false;
        for (auto& filter : imm_ukf.filters) {
            const double r_norm = (filter.x.segment<3>(0) - uav_pos).norm();
            const double r_freeze = width_freeze_range_;
            if (r_norm < r_freeze) {
                should_freeze = true;
                break;
            }
        }
        if (!should_freeze) {
            has_frozen_width = false;
        } else if (!has_frozen_width) {
            // Clamp at capture time: an un-clamped x(9) (e.g. 0.10 from a bad UKF update) would
            // otherwise be frozen permanently and corrupt every subsequent measurement prediction.
            frozen_width = clamp(imm_ukf.x(9), 1.0, 1.8);
            has_frozen_width = true;
        }

        for (auto& filter : imm_ukf.filters) {
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
                filter.x.segment<3>(0) = uav_pos + Eigen::Vector3d(10.0, 0.0, 0.0);
                filter.x(9) = W_guess;
                filter.P = Eigen::MatrixXd::Zero(kStateDim, kStateDim);
                filter.P.diagonal() << 100.0, 100.0, 100.0, 25.0, 25.0, 25.0, 10.0, 10.0, 10.0, 1.0;
                continue;
            }

            Eigen::Vector3d rel_p = filter.x.segment<3>(0) - uav_pos;
            const double r_norm = rel_p.norm();
            if (r_norm > 200.0) {
                filter.x.segment<3>(0) = uav_pos + rel_p / std::max(r_norm, 1e-6) * 120.0;
            }
            for (int i = 3; i < 6; ++i) filter.x(i) = clamp(filter.x(i), -30.0, 30.0);
            for (int i = 6; i < 9; ++i) filter.x(i) = clamp(filter.x(i), -15.0, 15.0);
            filter.x(9) = clamp(filter.x(9), 0.9, 2.5);

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
            // acc (6-8): cap P(acc) at 10.0. A large P(acc) (observed 80-114) makes the acc
            // channel absorb all measurement residual, injecting noise into acc -> vel -> pos and
            // destabilizing the estimate. width (9) keeps the existing wide range.
            for (int i = 6; i < 9; ++i) {
                filter.P(i, i) = clamp(filter.P(i, i), 1e-4, 10.0);
            }
            filter.P(9, 9) = clamp(filter.P(9, 9), 1e-4, 500.0);
        }
        imm_ukf.compute_state_estimate();
        if (has_frozen_width) {
            imm_ukf.x(9) = frozen_width;
            imm_ukf.P.row(9).setZero();
            imm_ukf.P.col(9).setZero();
            imm_ukf.P(9, 9) = 1e-8;
        } else {
            imm_ukf.x(9) = clamp(imm_ukf.x(9), 0.9, 2.5);
        }
    }

    std::pair<bool, Eigen::VectorXd> process_camera(bool has_bbox, double bbox_u, double bbox_v, double bbox_s,
                                                    double t_now, const Eigen::Vector3d& uav_pos_current,
                                                    const Eigen::Matrix3d& R_b2n_current) {
        if (!has_bbox) {
            if (!initialized) {
                return {false, Eigen::VectorXd::Constant(kStateDim, std::numeric_limits<double>::quiet_NaN())};
            }
            const double w_prev = imm_ukf.x(9);
            const double r_norm = (imm_ukf.x.segment<3>(0) - uav_pos_current).norm();
            if (r_norm < 25.0) {
                for (auto& filter : imm_ukf.filters) {
                    filter.x.segment<3>(3).setZero();
                    filter.x.segment<3>(6).setZero();
                    filter.x(9) = w_prev;
                }
                imm_ukf.compute_state_estimate();
                imm_ukf.x(9) = w_prev;
                return {true, imm_ukf.x};
            }
            update_process_noise(r_norm, w_prev);
            imm_ukf.predict(dt_cam);
            for (auto& filter : imm_ukf.filters) {
                filter.x.segment<3>(3) *= std::exp(-dt_cam / 3.0);
                filter.x.segment<3>(6).setZero();
                filter.x(9) = w_prev;
                filter.P(9, 9) = std::max(0.001, filter.P(9, 9));
            }
            imm_ukf.compute_state_estimate();
            imm_ukf.x(9) = w_prev;
            sanitize_filters(uav_pos_current);
            // Mark dropout so the next bbox update uses inflated measurement noise (soft recovery).
            first_meas_after_dropout = true;
            return {true, imm_ukf.x};
        }

        // bbox is real-time: capture pose == current pose, no history lookup needed.
        const Eigen::Vector3d& uav_pos_cap = uav_pos_current;
        const Eigen::Matrix3d& R_b2n_cap = R_b2n_current;

        const Eigen::Vector3d los_nav = pixel_to_los_nav(bbox_u, bbox_v, focal_length, R_b2n_cap, R_b2c);
        const double r_meas = focal_length * W_guess / std::max(1.0, bbox_s);
        const Eigen::Vector3d z_pixel(bbox_u, bbox_v, bbox_s);
        const Eigen::Matrix3d R_n2c = R_b2c * R_b2n_cap.transpose();
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
            const double dt_meas = has_last_cap_time ? clamp(t_now - last_cap_time, 0.001, dt_cam) : dt_cam;
            update_process_noise((imm_ukf.x.segment<3>(0) - uav_pos_cap).norm(), imm_ukf.x(9), bbox_s);
            imm_ukf.predict(dt_meas);

            // Size-plausibility gate: reject detections whose bbox size is wildly inconsistent
            // with the predicted range/width. A garbage size (e.g. a 32 px detection of a target
            // that should be ~138 px at this range) maps to a wildly wrong monocular range and
            // corrupts scale (W) and position -> the phantom-target dive seen in AirSim (W 1.5 ->
            // 2.8 after a dropout+reacquire). Reject -> predict-only (no update), treat the frame
            // as a dropout so the next GOOD measurement recovers with inflated noise.
            const double r_pred = (imm_ukf.x.segment<3>(0) - uav_pos_cap).norm();
            const double s_pred = focal_length * std::max(0.1, imm_ukf.x(9)) / std::max(0.5, r_pred);
            const double size_ratio = bbox_s / std::max(1.0, s_pred);
            const double size_gate_lo = 0.6;
            const double size_gate_hi = 1.8;
            if (size_ratio < size_gate_lo || size_ratio > size_gate_hi) {
                imm_ukf.compute_state_estimate();
                first_meas_after_dropout = true;
                last_cap_time = t_now;
                has_last_cap_time = true;
                return {true, imm_ukf.x};
            }

            for (auto& filter : imm_ukf.filters) filter.R = R_k;
            imm_ukf.update(z_pixel, uav_pos_cap, R_n2c, focal_length);

            last_cap_time = t_now;
            has_last_cap_time = true;
            first_meas_after_dropout = false;
            sanitize_filters(uav_pos_cap);
            // Clamp vel axes to reject single-frame UKF outliers without adding EMA lag (lag
            // destabilizes PN). ±15 m/s is well above any real target speed here.
            for (int i = 0; i < 3; ++i) {
                imm_ukf.x(3 + i) = clamp(imm_ukf.x(3 + i), -15.0, 15.0);
            }
            return {true, imm_ukf.x};
        } else {
            const double W_init = clamp(W_guess, 0.5, 4.0);
            const double r0 = clamp(focal_length * W_init / std::max(1.0, bbox_s), 10.0, 300.0);
            Eigen::VectorXd x0 = Eigen::VectorXd::Zero(kStateDim);
            x0.segment<3>(0) = uav_pos_cap + r0 * los_nav;
            x0.segment<3>(3).setZero();
            x0(9) = W_init;

            Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(kStateDim, kStateDim);
            P0.diagonal() << 100.0, 100.0, 100.0, 25.0, 25.0, 25.0, 10.0, 10.0, 10.0, 0.09;
            for (auto& filter : imm_ukf.filters) {
                filter.x = x0;
                filter.P = P0;
                filter.R = R_k;
                filter.predicted = false;
            }
            imm_ukf.x = x0;
            imm_ukf.P = P0;
            initialized = true;
            last_cap_time = t_now;
            has_last_cap_time = true;

            sanitize_filters(uav_pos_cap);
            return {true, imm_ukf.x};
        }
    }
};


class AdvancedHybridController {
public:
    double N;
    double K_v;
    double K_ibvs;
    double K_att;
    double K_yaw;
    double v_des;
    double max_tilt_rad;
    double g = 9.81;
    double fov_half_v_ = M_PI;
    double fov_half_h_ = M_PI;
    double fov_keep_margin_ = 0.85;
    double v_ramp_time_ = 0.0;
    double axial_jerk_max_ = 1e9;
    bool enable_fov_clamp_ = false;
    double mount_pitch_ = 0.0;
    double dither_gain_ = 1.5;
    double last_axial_accel_ = 0.0;
    double last_t_ = -1.0;
    bool has_last_ = false;
    double max_body_rate_ = 3.0;
    double terminal_brake_range_ = 5.0;
    double terminal_brake_floor_ = 1.0;
    // Decoupled gains.
    double K_pitch_ = 1.2;      // theta_cmd = K_pitch * los_el  (direct LOS pitch -> body pitch)
    double K_roll_ = 1.0;       // phi_cmd   = -K_roll * lateral_err (direct lateral -> roll)
    double K_z_ = 2.0;          // vertical proportional: a_z = K_z*dz_target - K_zd_*v_z
    double K_zd_ = 2.0;         // vertical damping on pursuer odom v_z (brakes climb rate before the
                                // zero-crossing; with K_z=2 -> zeta≈0.7. Was pure-P -> AirSim z porpoising)
    double K_fwd_ = 2.0;        // axial: a_fwd = K_fwd * (v_des - v_fwd_body)
    double terminal_bbox_size_ = 80.0;  // bbox px at terminal_brake_range (computed from focal/W)
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

    AdvancedHybridController(double N_val, double K_v_val, double K_ibvs_val,
                             double K_att_val, double K_yaw_val, double v_des_val, double max_tilt_deg,
                             double fov_half_v, double fov_half_h, double fov_keep_margin,
                             double v_ramp_time, double axial_jerk_max, bool enable_fov_clamp,
                             double mount_pitch_rad, double dither_gain,
                             double max_body_rate, double terminal_brake_range, double terminal_brake_floor,
                             double terminal_bbox_size, double K_z_val, double K_zd_val,
                             double v_pn_ref, double max_att_rate, double att_ref_tc,
                             double max_tilt_terminal_deg, double terminal_tilt_range,
                             double terminal_decel_range, double v_terminal)
        : N(N_val), K_v(K_v_val), K_ibvs(K_ibvs_val), K_att(K_att_val),
          K_yaw(K_yaw_val), v_des(v_des_val), max_tilt_rad(max_tilt_deg * M_PI / 180.0),
          fov_half_v_(fov_half_v), fov_half_h_(fov_half_h), fov_keep_margin_(fov_keep_margin),
          v_ramp_time_(v_ramp_time), axial_jerk_max_(axial_jerk_max), enable_fov_clamp_(enable_fov_clamp),
          mount_pitch_(mount_pitch_rad), dither_gain_(dither_gain),
          max_body_rate_(max_body_rate), terminal_brake_range_(terminal_brake_range),
          terminal_brake_floor_(terminal_brake_floor), terminal_bbox_size_(terminal_bbox_size),
          K_z_(K_z_val), K_zd_(K_zd_val),
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
    }

    // Decoupled control. Inputs:
    //   r_est, v_est : target rel pos/vel in NAV frame (v_est noisy, used only for LOS rate)
    //   los_az       : target azimuth in body heading frame (rad, 0 = dead ahead)
    //   los_el       : target elevation in world frame (rad, 0 = level, negative = below)
    //   v_fwd_body   : interceptor forward speed in body frame (m/s, from odom, observable)
    //   bbox_size    : target bbox size in px (observable proximity cue, monocular-range-independent)
    //   t            : mission-relative time (s)
    std::vector<double> compute_control(double roll, double pitch, double yaw,
                                        const Eigen::Vector3d& r_est, const Eigen::Vector3d& v_est,
                                        double los_az, double los_el, double t,
                                        double v_fwd_body = 0.0, double bbox_size = 0.0,
                                        double v_z = 0.0, double target_speed = 0.0,
                                        double los_rate = 0.0) {
        const double r_norm = r_est.norm();

        // ---- (A) Bearing-only lateral guidance -------------------------------------------
        // LOS angular rate (observable from bbox motion). v_est used only to form the cross
        // product; magnitude bounded so monocular vel noise cannot blow up PN.
        Eigen::Vector3d omega_los = Eigen::Vector3d::Zero();
        if (r_norm > 0.1) {
            omega_los = r_est.cross(v_est) / (r_norm * r_norm);
        }
        // LOS angular rate (lambda_dot) source. omega_los(2) = (r x v_est)_z / |r_xy|^2 is the
        // textbook PN LOS rate, but at close range v_est is noisy (observed +/-12 m/s jumps in
        // AirSim) and the 1/|r_xy| scaling blows it up -> violent banked turns. los_rate is the
        // EMA-smoothed derivative of the *measured* bearing (los_az_world), algebraically equal to
        // omega_los(2) when v_est is exact, but bounded and noise-robust. It is the primary source;
        // omega_los(2) is kept only as a turn-direction fallback while the bearing-rate EMA warms.
        double lambda_dot = los_rate;
        if (std::abs(lambda_dot) < 1e-4) {
            lambda_dot = omega_los(2);
        }
        const double omega_los_mag = std::min(std::abs(lambda_dot), 1.0);
        double N_eff = N;
        if (r_norm > 15.0) {
            N_eff = N - (N - 2.0) * clamp((r_norm - 15.0) / 35.0, 0.0, 1.0);
        }
        const double v_fwd_safe = clamp(v_fwd_body, 0.5, 20.0);
        // Speed-scheduled PN: cap the forward speed entering a_lat = N*v*lambda_dot. At v_des=20 the
        // raw N*v (~90) makes even 0.05 rad/s of LOS rate command ~4.5 m/s^2 lateral accel, slamming
        // phi_cmd to +-max_tilt and bang-banging omega_x into +-max_body_rate -> roll limit-cycle.
        // Capping v at v_pn_ref_ keeps the effective lateral gain near its tuned (lower-speed) value.
        const double v_pn = std::min(v_fwd_safe, v_pn_ref_);
        double a_lat_mag = N_eff * v_pn * omega_los_mag;
        // No near-range lateral decay. The old code zeroed lateral below 15 m to suppress the
        // v_est-noise (+/-12 m/s) banking, leaving pure-pursuit (yaw + axial) for the terminal
        // dash -> it trailed maneuvering targets (maneuvering-ct miss). Now that lambda_dot comes
        // from the clean, EMA-smoothed *bearing* derivative (not the 1/r v_est cross product),
        // there is no noise reason to attenuate, and full PN lead must survive to close range to
        // intercept a turning target. Dropping the decay takes maneuvering-ct from a 2.85 m miss
        // to a clean intercept, with no regression on nominal/crossing (lambda_dot ~= 0 for a
        // near-stationary target, so lateral is naturally idle there).
        // FOV keep-in-frame (horizontal): when target approaches the horizontal FOV edge,
        // reduce lateral accel so the UAV does not bank the target out of frame. The yaw
        // loop (boosted below) turns the nose toward the target instead.
        const double az_edge = fov_half_h_ * fov_keep_margin_;
        const double az_excess = clamp((std::abs(los_az) - az_edge) /
                                       std::max(fov_half_h_ - az_edge, 1e-6), 0.0, 1.0);
        a_lat_mag *= (1.0 - az_excess);
        // Lateral direction in world horizontal plane: perpendicular to LOS projected to xy.
        Eigen::Vector2d e_los_xy(r_est(0), r_est(1));
        if (e_los_xy.norm() < 1e-3) e_los_xy << 1.0, 0.0;
        e_los_xy.normalize();
        Eigen::Vector2d e_perp_xy(-e_los_xy(1), e_los_xy(0));  // 90 deg CCW
        const double turn_sign = (lambda_dot >= 0.0) ? 1.0 : -1.0;
        Eigen::Vector2d a_lat_world = turn_sign * a_lat_mag * e_perp_xy;

        // ---- (A) Axial guidance: fixed forward thrust, range-independent ------------------
        // Monocular range is unobservable, so do NOT close the loop on r_norm. Drive a constant
        // desired forward speed; proximity sensed via bbox_size growth.
        double v_des_eff = v_des;
        // Range-based terminal decel schedule: arrive at the terminal phase ~v_terminal (slower),
        // where the same bank gives ~2x the turn rate -> tracks maneuvering targets. This is a
        // feedforward schedule on the (now-stable, post W_est hardening) monocular range estimate,
        // NOT a closed range loop; a +/-15% range error only shifts the schedule slightly. The
        // speed-aware floor never lets it stall below target speed + margin against a fast target.
        if (r_norm < terminal_decel_range_) {
            const double v_term_eff = std::min(std::max(v_terminal_, target_speed + 1.5), v_des);
            const double frac = clamp((terminal_decel_range_ - r_norm) /
                                      std::max(terminal_decel_range_ - terminal_brake_range_, 1.0), 0.0, 1.0);
            v_des_eff = v_des + (v_term_eff - v_des) * frac;   // ramp v_des -> v_term_eff (<= v_des)
        }
        if (bbox_size > terminal_bbox_size_ * 0.2) {
            // Engage the proximity brake at ~5x terminal_brake_range (e.g. 25 m, not 10 m). At
            // v_des=20 with full rate authority (use_simulation:=true) the old 0.5 threshold (10 m)
            // engaged far too late to shed the closing speed — the UAV arrived at ~15 m/s, blew past
            // the target, lost the bbox and flew off chasing the dead-reckoned estimate. Ramp
            // v_des_eff -> floor over [bbox_engage, terminal_bbox_size_] = [~5R, R].
            const double bbox_engage = terminal_bbox_size_ * 0.2;
            const double prox = clamp((bbox_size - bbox_engage) /
                                      std::max(1.0, terminal_bbox_size_ - bbox_engage), 0.0, 1.0);
            // Speed-aware brake floor: against a moving target, braking to the configured floor
            // (1 m/s) stalls the interceptor below the target's speed so it levels off and falls
            // behind in the final approach (observed: crossing target at 3.6 m/s, UAV braked to
            // ~2 m/s and stuck at ~5 m). Never brake below target speed + margin when the target
            // is actually moving; a near-stationary target keeps the configured floor unchanged.
            const double moving_floor = (target_speed > 1.0) ? (target_speed + 1.5) : terminal_brake_floor_;
            const double floor_eff = std::min(std::max(terminal_brake_floor_, moving_floor), v_des);
            v_des_eff = v_des_eff + (floor_eff - v_des_eff) * prox;   // lower the scheduled v_des toward floor
        }
        double a_fwd = K_fwd_ * (v_des_eff - v_fwd_body);
        // Overshoot recovery: target behind (|los_az|>90°) = flew past. Constant v_des would
        // keep flying away (observed 14m->56m no turn-back). Brake + reverse closing direction
        // so body yaws back. Uses only los_az (observable), range-independent.
        bool target_behind = std::abs(los_az) > (M_PI / 2.0);
        if (target_behind) {
            v_des_eff = -v_des * 0.5;
            a_fwd = K_fwd_ * (v_des_eff - v_fwd_body);
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

        // ---- (B) Decoupled vertical: independent altitude track -------------------------
        // a_z = K_z * dz_target, with a soft vertical velocity limiter that begins reducing
        // vertical accel at |v_z|=4 m/s and zeroes it at |v_z|=8 m/s. This prevents:
        //  (1) Runaway climb (noise scenario: estimate divergence → dz_target=126m → sustained
        //      a_z=4.0 → 23 m/s climb → z=-220m).
        //  (2) Descent overshoot (target below: FOV clamp forces nose-down pitch → aggressive
        //      dive → v_z grows unchecked → UAV plunges past target before braking can engage).
        // The soft ramp (vs the old hard binary cutoff at 8 m/s) avoids the discontinuity that
        // caused limit-cycle oscillation near the threshold.
        const double dz_target = r_est(2);
        // PD on vertical: K_z on position error, K_zd damps pursuer odom v_z so the climb rate
        // built up while dz_target was large is braked BEFORE the zero-crossing. Without the
        // -K_zd*v_z term this is a pure spring (zeta=0) -> the +/-4 m z porpoising that caused
        // the AirSim miss (pursuer blasts through co-altitude at ~6 m/s). v_z is clean odom.
        double a_z = clamp(K_z_ * dz_target - K_zd_ * v_z, -4.0, 4.0);
        // Soft vertical velocity limiter: linearly decay a_z toward zero as |v_z| grows from
        // v_z_soft_start (4) to v_z_hard_zero (8). Only attenuates the component that would
        // increase |v_z| (same sign as v_z), allowing braking accel to pass through.
        const double v_z_soft_start = 5.0;
        const double v_z_hard_zero = 9.0;
        const double v_z_abs = std::abs(v_z);
        if (v_z_abs > v_z_soft_start) {
            const double decay = clamp(1.0 - (v_z_abs - v_z_soft_start) /
                                      (v_z_hard_zero - v_z_soft_start), 0.0, 1.0);
            // Only attenuate if a_z would increase |v_z| (same sign as v_z).
            if ((v_z > 0.0 && a_z > 0.0) || (v_z < 0.0 && a_z < 0.0)) {
                a_z *= decay;
            }
        }

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

        // FOV keep-in-frame (vertical): clamp pitch so target stays within vertical FOV.
        // Uses the original (less restrictive) sign convention: theta_hi = fov_hard - los_el - mount_pitch.
        // The "geometrically correct" sign (mount_pitch + fov_hard - los_el) was tested and caused
        // aggressive diving when target is below → image smear → bbox loss → overshoot. The original
        // sign is less restrictive but empirically keeps the target in frame during terminal phase
        // without forcing excessive nose-down pitch.
        //
        // Only active in terminal phase (r < 15 m): during mid-course the clamp limits pitch
        // to ~6° (mount_pitch=15° + narrow vertical FOV), capping forward accel at 1 m/s² and
        // making it impossible to catch moving targets. Accept brief FOV loss during approach;
        // the estimator dead-reckons and reacquires when the UAV levels out for terminal guidance.
        target_behind = std::abs(los_az) > (M_PI / 2.0);
        if (enable_fov_clamp_ && !target_behind && r_norm < 15.0) {
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
        // throttle = (a_z + g) / cos(tilt): the collective thrust along body-z needed so its
        // vertical projection matches (a_z + g). Horizontal projection comes from the tilt and
        // produces fx_h/fy_h once the attitude tracks theta_cmd/phi_cmd. Using |thrust_vec| here
        // would over-thrust while level (throttle ~ sqrt(fx_h^2+g^2) >> g) and cause pure climb
        // before the body can tilt; the /cos(tilt) form ramps throttle with tilt instead.
        const double a_h_mag = std::sqrt(fx_h * fx_h + fy_h * fy_h);
        double tilt = std::atan2(a_h_mag, a_z + g);
        if (tilt > max_tilt_rad) tilt = max_tilt_rad;
        const double throttle = clamp((a_z + g) / std::cos(tilt), 2.0, 25.0);

        // ---- Attitude rate commands (P on attitude error) -------------------------------
        // Yaw rate boost when target near horizontal FOV edge: turn nose toward target
        // faster to keep it in frame. az_excess (computed above) goes 0→1 as target
        // moves from keep-margin to FOV edge.
        const double yaw_boost = 1.0 + 2.0 * az_excess;
        double omega_x = clamp(K_att * (phi_cmd - roll), -max_body_rate_, max_body_rate_);
        double omega_y = clamp(K_att * (theta_cmd - pitch), -max_body_rate_, max_body_rate_);
        double omega_z = clamp(K_yaw * yaw_boost * los_az, -max_body_rate_, max_body_rate_);
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

        this->declare_parameter<int>("camera.width_px", 640);
        this->declare_parameter<int>("camera.height_px", 480);
        this->declare_parameter<double>("camera.fov_deg", 60.0);
        this->declare_parameter<double>("camera.mount_pitch_deg", 0.0);
        this->declare_parameter<std::vector<double>>("extrinsics.body_to_camera_rotation", 
            {0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0});
        this->declare_parameter<double>("controller.N", 4.5);
        this->declare_parameter<double>("controller.K_v", 2.2);
        this->declare_parameter<double>("controller.K_ibvs", 2.0);
        this->declare_parameter<double>("controller.K_att", 9.0);
        this->declare_parameter<double>("controller.K_yaw", 5.0);
        this->declare_parameter<double>("controller.K_z", 2.0);
        this->declare_parameter<double>("controller.K_zd", 2.0);
        this->declare_parameter<double>("controller.v_des", 15.0);
        this->declare_parameter<double>("controller.max_tilt_deg", 35.0);
        this->declare_parameter<double>("controller.fov_keep_margin", 0.85);
        this->declare_parameter<double>("controller.v_ramp_time", 2.0);
        this->declare_parameter<double>("controller.axial_jerk_max", 15.0);
        this->declare_parameter<double>("controller.max_body_rate", 3.0);
        this->declare_parameter<double>("controller.terminal_brake_range", 5.0);
        this->declare_parameter<double>("controller.terminal_brake_floor", 1.0);
        this->declare_parameter<double>("controller.terminal_range", 10.0);
        this->declare_parameter<double>("controller.terminal_N", 3.0);
        this->declare_parameter<double>("controller.v_pn_ref", 12.0);
        this->declare_parameter<double>("controller.max_att_rate", 5.0);
        this->declare_parameter<double>("controller.att_ref_tc", 0.15);
        this->declare_parameter<double>("controller.max_tilt_terminal_deg", 45.0);
        this->declare_parameter<double>("controller.terminal_tilt_range", 15.0);
        this->declare_parameter<double>("controller.terminal_decel_range", 30.0);
        this->declare_parameter<double>("controller.v_terminal", 12.0);
        this->declare_parameter<double>("estimation.dt_cam", 0.033);
        this->declare_parameter<double>("estimation.W_guess", 1.5);
        this->declare_parameter<double>("system.control_rate", 100.0);
        this->declare_parameter<double>("control.rate_tc", 0.02);
        this->declare_parameter<bool>("control.rate_control_attenuation", true);
        this->declare_parameter<bool>("use_simulation", false);

        load_parameters();

        focal_length_ = (cam_width_ / 2.0) / std::tan((cam_fov_deg_ * M_PI / 180.0) / 2.0);

        estimator_ = std::make_unique<RelativeIMMUKFEstimator>(dt_cam_, focal_length_, W_guess_, R_b2c_);

        mission_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/fpv/mission_launch", 10, std::bind(&InterceptorNodeRelativeIMMUKF::mission_callback, this, std::placeholders::_1));
        tracker_trigger_sub_ = this->create_subscription<std_msgs::msg::Header>(
            "/fpv/tracker_trigger", 10, std::bind(&InterceptorNodeRelativeIMMUKF::mission_launch_callback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/fpv/odom", rclcpp::SensorDataQoS(), std::bind(&InterceptorNodeRelativeIMMUKF::odom_callback, this, std::placeholders::_1), sub_options);

        bbox_sub_ = this->create_subscription<vision_msgs::msg::Detection2D>(
            "/fpv/target_bbox", 10, std::bind(&InterceptorNodeRelativeIMMUKF::bbox_callback, this, std::placeholders::_1), sub_options);

        control_pub_ = this->create_publisher<quadrotor_msgs::msg::ControlCommand>("/fpv/control_command", 10);
        mission_state_pub_ = this->create_publisher<std_msgs::msg::Bool>("/fpv/mission_state", 10);

        mission_state_.set(MissionState::IDLE);
        RCLCPP_INFO(this->get_logger(), "Relative IMM-UKF node ready (Pure Odometry mode). Odom=/fpv/odom BBox=/fpv/target_bbox Launch=/fpv/tracker_trigger Control=/fpv/control_command");
    }

private:
    void declare_parameters() {}

    void load_parameters() {
        cam_width_ = this->get_parameter("camera.width_px").as_int();
        cam_height_ = this->get_parameter("camera.height_px").as_int();
        cam_fov_deg_ = this->get_parameter("camera.fov_deg").as_double();
        const double pitch_deg = this->get_parameter("camera.mount_pitch_deg").as_double();
        const bool use_simulation = this->get_parameter("use_simulation").as_bool();
        std::vector<double> R_flat = this->get_parameter("extrinsics.body_to_camera_rotation").as_double_array();
        Eigen::Matrix3d R_b2c_base;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R_b2c_base(i, j) = R_flat[i * 3 + j];
            }
        }
        const double p_rad = -pitch_deg * M_PI / 180.0;
        Eigen::Matrix3d R_pitch;
        R_pitch << 1.0, 0.0, 0.0,
                   0.0, std::cos(p_rad), -std::sin(p_rad),
                   0.0, std::sin(p_rad), std::cos(p_rad);
        R_b2c_ = R_pitch * R_b2c_base;

        dt_cam_ = this->get_parameter("estimation.dt_cam").as_double();
        W_guess_ = this->get_parameter("estimation.W_guess").as_double();
        control_rate_ = this->get_parameter("system.control_rate").as_double();
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
        double K_v = this->get_parameter("controller.K_v").as_double();
        double K_ibvs = this->get_parameter("controller.K_ibvs").as_double();
        double K_att = this->get_parameter("controller.K_att").as_double();
        double K_yaw = this->get_parameter("controller.K_yaw").as_double();
        double K_z = this->get_parameter("controller.K_z").as_double();
        double K_zd = this->get_parameter("controller.K_zd").as_double();
        double v_des = this->get_parameter("controller.v_des").as_double();
        double max_tilt = this->get_parameter("controller.max_tilt_deg").as_double();
        double fov_keep = this->get_parameter("controller.fov_keep_margin").as_double();
        double ramp = this->get_parameter("controller.v_ramp_time").as_double();
        double jerk = this->get_parameter("controller.axial_jerk_max").as_double();
        double max_body_rate = this->get_parameter("controller.max_body_rate").as_double();
        double terminal_brake_range = this->get_parameter("controller.terminal_brake_range").as_double();
        double terminal_brake_floor = this->get_parameter("controller.terminal_brake_floor").as_double();
        double v_pn_ref = this->get_parameter("controller.v_pn_ref").as_double();
        double max_att_rate = this->get_parameter("controller.max_att_rate").as_double();
        double att_ref_tc = this->get_parameter("controller.att_ref_tc").as_double();
        double max_tilt_terminal_deg = this->get_parameter("controller.max_tilt_terminal_deg").as_double();
        double terminal_tilt_range = this->get_parameter("controller.terminal_tilt_range").as_double();
        double terminal_decel_range = this->get_parameter("controller.terminal_decel_range").as_double();
        double v_terminal = this->get_parameter("controller.v_terminal").as_double();
        double fov_half_v = (cam_height_ / 2.0) / std::max(1.0, focal_length_);
        double fov_half_h = (cam_width_ / 2.0) / std::max(1.0, focal_length_);
        // Compute terminal_bbox_size from camera intrinsics so the brake triggers at the correct
        // range regardless of focal length. bbox_size = focal * W / range, so at terminal_brake_range:
        // terminal_bbox_size = focal * W_guess / terminal_brake_range. For AirSim (focal=831, W=1.5,
        // range=5): 249px. For Python sim (focal=554, W=1.5, range=5): 166px. Previously hardcoded
        // 80px which triggered full brake at ~15m in AirSim (bbox=83px at R=15), stalling the UAV.
        const double terminal_bbox_size = focal_length_ * W_guess_ / std::max(0.1, terminal_brake_range);

        controller_ = std::make_unique<AdvancedHybridController>(
            N, K_v, K_ibvs, K_att, K_yaw, v_des, max_tilt,
            fov_half_v, fov_half_h, fov_keep, ramp, jerk, true, pitch_deg * M_PI / 180.0, 0.0,
            max_body_rate, terminal_brake_range, terminal_brake_floor, terminal_bbox_size, K_z, K_zd,
            v_pn_ref, max_att_rate, att_ref_tc,
            max_tilt_terminal_deg, terminal_tilt_range, terminal_decel_range, v_terminal);

    }

    void reset_estimator() {
        estimator_ = std::make_unique<RelativeIMMUKFEstimator>(dt_cam_, focal_length_, W_guess_, R_b2c_);
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
                "Waiting for Odometry to initialize...");
            return;
        }

        const double t_now = rclcpp::Time(msg->header.stamp).seconds();

        const bool has_bbox = !msg->header.frame_id.empty();
        const double u = msg->bbox.center.position.x - cam_width_ / 2.0;
        const double v = msg->bbox.center.position.y - cam_height_ / 2.0;
        const double s = std::max(msg->bbox.size_x, msg->bbox.size_y);

        Eigen::Vector3d uav_pos_use;
        Eigen::Matrix3d R_b2n;
        {
            std::lock_guard<std::mutex> ol(odom_mtx_);
            uav_pos_use = latest_odom_pos_;
            R_b2n = latest_R_b2n_;
        }

        const bool was_initialized = estimator_->initialized;
        const auto result = estimator_->process_camera(has_bbox, u, v, s, t_now, uav_pos_use, R_b2n);

        if (!was_initialized && estimator_->initialized) {
            transition_to_state(MissionState::TRACKING, "Target initialized");
        }
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS,
            "bbox=%d size=%.1f init=%d->%d", has_bbox, s, was_initialized, estimator_->initialized);

        if (!estimator_->initialized) return;

        Eigen::Vector3d target_pos_now = result.second.segment<3>(0);
        Eigen::Vector3d target_vel_now = result.second.segment<3>(3);

        Eigen::Vector3d r_est;
        Eigen::Vector3d v_est;
        {
            std::lock_guard<std::mutex> ol(odom_mtx_);
            r_est = target_pos_now - latest_odom_pos_;
            v_est = target_vel_now - latest_odom_vel_;
        }

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
        // Absolute target speed (clamped) feeds the speed-aware terminal-brake floor so the
        // interceptor never decelerates below a maneuvering target's speed in the final approach.
        cache.target_speed = std::min(target_vel_now.norm(), 15.0);

        {
            std::lock_guard<std::mutex> cache_lock(control_cache_mtx_);
            control_cache_ = cache;
        }

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS,
            "rel=[%.2f %.2f %.2f] vel=[%.2f %.2f %.2f] W_est=%.2f r_meas=%.1f r_est=%.1f",
            r_est(0), r_est(1), r_est(2), v_est(0), v_est(1), v_est(2),
            estimator_->imm_ukf.x(9),
            (has_bbox && s > 0.1) ? (focal_length_ * estimator_->W_guess / s) : 0.0,
            r_est.norm());

        // IMM diagnostic: mode probabilities, per-filter vel, and key P diagonals. Helps locate
        // which motion model is driving the vel jumps when the estimate oscillates.
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS*4,
            "IMM mu=[%.2f %.2f %.2f] v0=[%.1f %.1f %.1f] v1=[%.1f %.1f %.1f] v2=[%.1f %.1f %.1f] Pd=[%.1f %.1f %.1f %.1f]",
            estimator_->imm_ukf.mu(0), estimator_->imm_ukf.mu(1), estimator_->imm_ukf.mu(2),
            estimator_->imm_ukf.filters[0].x(3), estimator_->imm_ukf.filters[0].x(4), estimator_->imm_ukf.filters[0].x(5),
            estimator_->imm_ukf.filters[1].x(3), estimator_->imm_ukf.filters[1].x(4), estimator_->imm_ukf.filters[1].x(5),
            estimator_->imm_ukf.filters[2].x(3), estimator_->imm_ukf.filters[2].x(4), estimator_->imm_ukf.filters[2].x(5),
            estimator_->imm_ukf.P(0,0), estimator_->imm_ukf.P(3,3), estimator_->imm_ukf.P(6,6), estimator_->imm_ukf.P(9,9));
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
        Eigen::Vector3d odom_vel = Eigen::Vector3d::Zero();
        double t_now = 0.0;
        double sim_start_time = -1.0;
        {
            std::lock_guard<std::mutex> odom_lock(odom_mtx_);
            R_b2n = latest_R_b2n_;
            odom_vel = latest_odom_vel_;
            t_now = (last_imu_time_ > 0.0) ? last_imu_time_ : now.seconds();
            sim_start_time = sim_start_time_;
        }

        const double roll = std::atan2(R_b2n(2, 1), R_b2n(2, 2));
        const double pitch = std::atan2(-R_b2n(2, 0), std::sqrt(R_b2n(2, 1) * R_b2n(2, 1) + R_b2n(2, 2) * R_b2n(2, 2)));
        const double yaw = std::atan2(R_b2n(1, 0), R_b2n(0, 0));
        // Body-frame forward speed from odom velocity rotated into heading frame.
        const double v_fwd_body = std::cos(yaw) * odom_vel(0) + std::sin(yaw) * odom_vel(1);
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
            RCLCPP_INFO(this->get_logger(),
                "CAPTURE: holding (brake-to-hold) to end of mission (r_est=%.1fm los_az=%.0fdeg)",
                cache.r_est.norm(), los_az * 180.0 / M_PI);
        }
        if (captured_) {
            // Brake-to-hold: cancel forward velocity with nose-up pitch (a real quad brakes by
            // pitching), level the roll, hold altitude. theta_brake -> 0 as v_fwd -> 0, so this
            // naturally settles into a level hover once stopped (works in the drag-free sim AND on
            // real hardware; pure zero-rate "hover" sinks on a banked body and never stops a
            // drag-free body).
            constexpr double g_hover = 9.81;
            constexpr double max_brake_tilt = 0.436;   // 25 deg
            const double theta_brake = clamp(-0.12 * v_fwd_body, -max_brake_tilt, max_brake_tilt);
            const double tilt = std::sqrt(roll * roll + pitch * pitch);
            auto cmd = std::make_unique<quadrotor_msgs::msg::ControlCommand>();
            cmd->header.stamp = now;
            cmd->armed = true;
            cmd->control_mode = quadrotor_msgs::msg::ControlCommand::BODY_RATES;
            cmd->collective_thrust = g_hover / std::max(std::cos(tilt), 0.3);
            cmd->bodyrates.x = clamp(-3.0 * roll, -3.0, 3.0);
            cmd->bodyrates.y = clamp(5.0 * (theta_brake - pitch), -3.0, 3.0);
            cmd->bodyrates.z = 0.0;
            prev_bodyrate_x_ = cmd->bodyrates.x;
            prev_bodyrate_y_ = cmd->bodyrates.y;
            prev_bodyrate_z_ = cmd->bodyrates.z;
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
                constexpr double g_hover = 9.81;
                const double tilt = std::sqrt(roll * roll + pitch * pitch);
                cmd->collective_thrust = g_hover / std::max(std::cos(tilt), 0.3);
                cmd->bodyrates.x = clamp(-3.0 * roll, -3.0, 3.0);
                cmd->bodyrates.y = clamp(-3.0 * pitch, -3.0, 3.0);
                cmd->bodyrates.z = 0.0;
            }
            prev_bodyrate_x_ = cmd->bodyrates.x;
            prev_bodyrate_y_ = cmd->bodyrates.y;
            prev_bodyrate_z_ = cmd->bodyrates.z;
            control_pub_->publish(std::move(cmd));
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS/2,
                "%s (terminal dropout): age=%.2fs last_good_range=%.1fm",
                (dropout_age < coast_max_dur) ? "COAST" : "LEVELHOLD", dropout_age, last_good_range_);
            return;
        }

        const double v_z = odom_vel(2);
        const auto ctrl = controller_->compute_control(roll, pitch, yaw, cache.r_est, cache.v_est, los_az, cache.los_el, t_rel, v_fwd_body, cache.bbox_size, v_z, cache.target_speed, cache.los_rate);
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
        prev_bodyrate_x_ = cmd->bodyrates.x;
        prev_bodyrate_y_ = cmd->bodyrates.y;
        prev_bodyrate_z_ = cmd->bodyrates.z;
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
                has_odom_ = false;
            }
            last_imu_time_ = -1.0;
            sim_start_time_ = -1.0;
            prev_bodyrate_x_ = 0.0;
            prev_bodyrate_y_ = 0.0;
            prev_bodyrate_z_ = 0.0;
            has_prev_attitude_ = false;
            has_prev_los_ = false;
            los_rate_ema_ = 0.0;
            prev_los_t_ = -1.0;
            has_frozen_cmd_ = false;
            last_good_range_ = 1e9;
            last_bbox_time_ = 0.0;
            captured_ = false;
            min_range_seen_ = 1e9;
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

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> odom_lock(odom_mtx_);
        latest_odom_pos_(0) = msg->pose.pose.position.x;
        latest_odom_pos_(1) = msg->pose.pose.position.y;
        latest_odom_pos_(2) = msg->pose.pose.position.z;

        const Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                                   msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        latest_R_b2n_raw_ = q.normalized().toRotationMatrix();
        latest_R_b2n_ = latest_R_b2n_raw_;

        // mavros local_position/odom: pose in frame_id (map=ENU/world), twist in child_frame_id
        // (base_link=body FLU). Rotate body-frame velocity into world so v_est = target_vel_world
        // - odom_vel is consistent (both world). Without this, omega_los = r×v/r² mixes world r
        // with body v, corrupting PN lateral direction, and v_fwd_body double-rotates.
        const Eigen::Vector3d vel_body(msg->twist.twist.linear.x,
                                       msg->twist.twist.linear.y,
                                       msg->twist.twist.linear.z);
        const Eigen::Vector3d vel_world = latest_R_b2n_ * vel_body;
        latest_odom_vel_(0) = vel_world(0);
        latest_odom_vel_(1) = vel_world(1);
        latest_odom_vel_(2) = vel_world(2);

        yaw_align_init_ = true;
        has_odom_ = true;
        last_imu_time_ = rclcpp::Time(msg->header.stamp).seconds();

        if (!mission_state_.is_active()) {
            return;
        }

        ControlCacheData cache;
        {
            std::lock_guard<std::mutex> cache_lock(control_cache_mtx_);
            cache = control_cache_;
        }

        if (cache.valid && (last_imu_time_ - cache.source_time < 1.0)) {
            const Eigen::Vector3d target_abs = cache.r_est + latest_odom_pos_;
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS,
                "odom pos=[%.2f %.2f %.2f] vel=[%.2f %.2f %.2f] | target_abs=[%.2f %.2f %.2f] range=%.2f",
                latest_odom_pos_(0), latest_odom_pos_(1), latest_odom_pos_(2),
                latest_odom_vel_(0), latest_odom_vel_(1), latest_odom_vel_(2),
                target_abs(0), target_abs(1), target_abs(2), cache.r_est.norm());
        }
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
        auto cmd = std::make_unique<quadrotor_msgs::msg::ControlCommand>();
        cmd->header.stamp = this->get_clock()->now();
        cmd->armed = false;
        cmd->control_mode = quadrotor_msgs::msg::ControlCommand::BODY_RATES;
        control_pub_->publish(std::move(cmd));
    }

    int cam_width_ = 640;
    int cam_height_ = 480;
    double cam_fov_deg_ = 60.0;
    double focal_length_ = 554.256;
    Eigen::Matrix3d R_b2c_ = Eigen::Matrix3d::Identity();

    double dt_cam_ = 0.033;
    double W_guess_ = 1.5;
    double control_rate_ = 100.0;

    std::unique_ptr<RelativeIMMUKFEstimator> estimator_;
    std::mutex estimator_mtx_;

    std::unique_ptr<AdvancedHybridController> controller_;

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr mission_sub_;
    rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr tracker_trigger_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr bbox_sub_;
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
        double W_est = 0.0;
        double r_meas = 0.0;
        double los_az = 0.0;
        Eigen::Matrix3d R_b2n = Eigen::Matrix3d::Identity();
        double bbox_size = 0.0;
        double target_speed = 0.0;
    };
    ControlCacheData control_cache_;

    MissionStateMachine mission_state_;

    std::mutex odom_mtx_;
    bool has_odom_ = false;
    Eigen::Vector3d latest_odom_pos_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d latest_odom_vel_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d latest_R_b2n_raw_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d latest_R_b2n_ = Eigen::Matrix3d::Identity();

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
    // Min estimated range seen this mission (reopen detection): once the range bottoms out and
    // reopens, the UAV flew past the target. More robust than los_az under a noisy close-range
    // estimate (high-noise scenario), where the bearing never cleanly flips past 90 deg.
    double min_range_seen_ = 1e9;
    double frozen_thrust_ = 0.0;
    double frozen_rate_x_ = 0.0;
    double frozen_rate_y_ = 0.0;
    double frozen_rate_z_ = 0.0;

    double ctrl_coeff_ = 1.0;
    // Previous published body rates for EMA low-pass on rate output.
    double prev_bodyrate_x_ = 0.0;
    double prev_bodyrate_y_ = 0.0;
    double prev_bodyrate_z_ = 0.0;
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
