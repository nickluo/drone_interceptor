#pragma once

/**
 * @file bearing_angle_ekf.h
 * @brief Bearing-Angle EKF for 3D target tracking with adaptive width estimation
 * 
 * Ported from prototype/target_ekf.py
 * 
 * State vector: [px, py, pz, vx, vy, vz, W]^T (7D)
 *   - px, py, pz: target absolute position (world frame)
 *   - vx, vy, vz: target absolute velocity
 *   - W: target physical width (adaptive online estimation)
 * 
 * Measurement vector: [azimuth, elevation, visual_angle]^T (3D)
 *   - azimuth and elevation provide direction constraints
 *   - visual_angle α = 2·atan(W/(2·d)) with W joint estimation provides range observability
 */

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <deque>
#include <algorithm>

class BearingAngleEKF {
public:
    static constexpr int N_STATE = 7;
    static constexpr int N_MEAS = 3;

    using Vector7d = Eigen::Matrix<double, N_STATE, 1>;
    using Matrix7d = Eigen::Matrix<double, N_STATE, N_STATE>;
    using Vector3d = Eigen::Vector3d;
    using Matrix3d = Eigen::Matrix3d;
    using MatrixH = Eigen::Matrix<double, N_MEAS, N_STATE>;
    using VectorZ = Eigen::Matrix<double, N_MEAS, 1>;
    using MatrixS = Eigen::Matrix<double, N_MEAS, N_MEAS>;
    using MatrixK = Eigen::Matrix<double, N_STATE, N_MEAS>;

    /**
     * @brief Measurement struct for bearing-angle observations
     * Computed from bbox pixel coordinates and camera focal length
     */
    struct Measurement {
        double azimuth;       // arctan2(-v_pixel, focal_length)
        double elevation;     // arctan2(-u_pixel, focal_length)
        double visual_angle;  // 2 * arctan(bbox_size / (2 * focal_length))
    };

    /**
     * @brief History state snapshot for delayed measurement rollback
     * Stores EKF state at each predict step to enable out-of-sequence
     * measurement processing (compensating detection latency).
     */
    struct StateHistory {
        double timestamp;       // seconds
        Vector7d state;         // state vector snapshot
        Matrix7d covariance;    // covariance snapshot
    };

    /**
     * @brief Constructor
     * @param cam_focal_length Camera focal length in pixels
     * @param pos_dev Bbox position noise std dev in pixels
     * @param size_dev Bbox size noise std dev in pixels
     * @param initial_width_est Initial estimate of target physical width (m)
     * @param range_init_bias Bias factor for initial range (<1 breaks W/d coupling)
     * @param q_accel Process noise acceleration spectral density (m²/s⁴)/Hz
     * @param q_W Process noise for W random walk (m²/s)
     * @param alpha_R_inflate Inflation factor for visual angle measurement noise
     * @param estimated_predict_freq Estimated predict() call frequency in Hz (0 = disable history)
     * @param estimated_detection_latency Estimated detection pipeline latency in seconds (0 = disable history)
     */
    BearingAngleEKF(double cam_focal_length,
                    double pos_dev,
                    double size_dev,
                    double initial_width_est,
                    double range_init_bias = 0.5,
                    double q_accel = 1.0,
                    double q_W = 0.002,
                    double alpha_R_inflate = 2.0,
                    double estimated_predict_freq = 0.0,
                    double estimated_detection_latency = 0.0)
        : focal_length_(cam_focal_length),
          initial_width_est_(initial_width_est),
          range_init_bias_(range_init_bias),
          q_accel_(q_accel),
          q_W_(q_W),
          is_initialized_(false)
    {
        // History buffer for delayed measurement rollback
        if (estimated_predict_freq > 0.0 && estimated_detection_latency > 0.0) {
            max_buffer_size_ = static_cast<size_t>(estimated_predict_freq * estimated_detection_latency * 2.5);
            if (max_buffer_size_ < 10) max_buffer_size_ = 10;
        } else {
            max_buffer_size_ = 0;  // disabled
        }
        // Derive angular measurement noise from pixel noise
        double sigma_az = pos_dev / cam_focal_length;
        double sigma_el = pos_dev / cam_focal_length;
        double sigma_alpha = size_dev / cam_focal_length;

        // Measurement noise covariance R (3x3)
        R_ = VectorZ(sigma_az * sigma_az,
                      sigma_el * sigma_el,
                      sigma_alpha * sigma_alpha * alpha_R_inflate).asDiagonal();

        // Initial state
        x_.setZero();

        // Initial covariance
        // With range_init_bias, initial distance error is ~σ=30m → P_pos=900
        // P_W=4 (σ=2m) covers W_init ±2x uncertainty
        P_ = Vector7d(900.0, 900.0, 900.0,   // position σ≈30m
                       100.0, 100.0, 100.0,   // velocity σ=10m/s
                       4.0).asDiagonal();      // width σ=2m
    }

    /**
     * @brief Initialize the filter with first observation
     * @param bbox_u Pixel center horizontal offset from optical axis (pixels)
     * @param bbox_v Pixel center vertical offset from optical axis (pixels)
     * @param bbox_size Bbox size in pixels (width or diagonal)
     * @param drone_pos Drone position in world frame [x, y, z]
     * @param R_WC Rotation matrix from world to camera frame (3x3)
     */
    void initialize(double bbox_u, double bbox_v, double bbox_size,
                    const Vector3d& drone_pos, const Matrix3d& R_WC)
    {
        Vector3d initial_pos_est = drone_pos;
        if (bbox_size > 1e-6) {
            // range_init_bias < 1 breaks W/d coupling:
            // initial position is biased closer → α_pred > α_meas
            // → negative innovation → EKF drives W downward
            double distance = range_init_bias_ * focal_length_ * initial_width_est_ / bbox_size;
            Matrix3d R_CW = R_WC.transpose();
            Vector3d los_C_dir(1.0, bbox_u / focal_length_, bbox_v / focal_length_);
            los_C_dir.normalize();
            Vector3d los_W_dir = R_CW * los_C_dir;
            initial_pos_est += distance * los_W_dir;
        }

        x_.head<3>() = initial_pos_est;
        x_.segment<3>(3).setZero();  // velocity initialized to zero
        x_(6) = initial_width_est_;

        // Reset covariance
        P_ = Vector7d(900.0, 900.0, 900.0,
                       100.0, 100.0, 100.0,
                       4.0).asDiagonal();

        is_initialized_ = true;
        history_buffer_.clear();  // Fresh start for history
        std::cout << "[INFO] Bearing-Angle EKF initialized: pos=["
                  << initial_pos_est.transpose() << "], W_init=" << x_(6) << "m" << std::endl;
    }

    /**
     * @brief EKF prediction step (Constant Velocity model with kinematic process noise)
     * @param dt Time step in seconds
     * @param timestamp Current time in seconds (> 0 to save state to history buffer for
     *                  delayed measurement rollback; <= 0 skips history save, used during replay)
     */
    void predict(double dt, double timestamp = -1.0)
    {
        if (!is_initialized_) return;

        // State transition: CV model
        // F = I(7x7) with off-diagonal dt blocks for position
        Matrix7d F = Matrix7d::Identity();
        F(0, 3) = dt;
        F(1, 4) = dt;
        F(2, 5) = dt;

        // Kinematic process noise: white noise acceleration integration
        // Cross-covariance between position and velocity for physical consistency
        double qa = q_accel_;
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;

        Matrix7d Q = Matrix7d::Zero();
        for (int i = 0; i < 3; ++i) {
            Q(i, i)     = qa * dt3 / 3.0;    // position variance
            Q(i, i + 3) = qa * dt2 / 2.0;    // position-velocity cross
            Q(i + 3, i) = qa * dt2 / 2.0;    // velocity-position cross
            Q(i + 3, i + 3) = qa * dt;        // velocity variance
        }
        Q(6, 6) = q_W_ * dt;  // W random walk

        // Predict state and covariance
        x_ = F * x_;
        P_ = F * P_ * F.transpose() + Q;

        // Save to history buffer for delayed measurement rollback
        if (timestamp > 0.0 && max_buffer_size_ > 0) {
            history_buffer_.push_back({timestamp, x_, P_});
            while (history_buffer_.size() > max_buffer_size_) {
                history_buffer_.pop_front();
            }
        }
    }

    /**
     * @brief EKF update with delayed measurement rollback
     *
     * When measurement_timestamp > 0 and the history buffer is populated,
     * the filter rolls back state/covariance to the predict step closest to
     * the measurement time, applies the update there, then replays all
     * subsequent predict steps to bring state back to current time.
     * This compensates for detection pipeline latency.
     *
     * @param z Bearing-angle measurement
     * @param drone_pos Drone position in world frame
     * @param R_WC Rotation matrix from world to camera frame
     * @param measurement_timestamp Time the image was captured (seconds, > 0 to enable rollback)
     * @return true if update accepted
     */
    bool update(const Measurement& z,
                const Vector3d& drone_pos,
                const Matrix3d& R_WC,
                double measurement_timestamp = -1.0)
    {
        if (!is_initialized_) return false;

        // ---- Delayed-measurement rollback path ----
        if (measurement_timestamp > 0.0 && !history_buffer_.empty()) {
            // Binary search for closest historical state
            auto it = std::lower_bound(
                history_buffer_.begin(), history_buffer_.end(), measurement_timestamp,
                [](const StateHistory& entry, double t) { return entry.timestamp < t; });

            auto best_it = it;
            if (it == history_buffer_.end()) {
                best_it = std::prev(history_buffer_.end());
            } else if (it != history_buffer_.begin()) {
                auto prev_it = std::prev(it);
                if (std::abs(prev_it->timestamp - measurement_timestamp) <=
                    std::abs(it->timestamp - measurement_timestamp)) {
                    best_it = prev_it;
                }
            }

            // Save current state in case the update is rejected
            Vector7d saved_x = x_;
            Matrix7d saved_P = P_;

            // Roll back to historical state
            x_ = best_it->state;
            P_ = best_it->covariance;

            // Apply measurement update at the historical state
            bool accepted = update_internal(z, drone_pos, R_WC);

            if (!accepted) {
                // Rejected — restore current state, no side effects
                x_ = saved_x;
                P_ = saved_P;
                return false;
            }

            // Replay predictions forward, updating stale history entries
            double prev_time = best_it->timestamp;
            for (auto replay_it = std::next(best_it); replay_it != history_buffer_.end(); ++replay_it) {
                double dt_replay = replay_it->timestamp - prev_time;
                if (dt_replay > 0.0 && dt_replay < 0.5) {
                    predict(dt_replay);  // no timestamp → won't save to history
                }
                // Overwrite stale entry with corrected post-update trajectory
                replay_it->state = x_;
                replay_it->covariance = P_;
                prev_time = replay_it->timestamp;
            }
            return true;
        }

        // ---- Direct update path (no rollback) ----
        return update_internal(z, drone_pos, R_WC);
    }

private:
    /**
     * @brief Internal EKF update (measurement model + Kalman gain)
     * @return true if update was accepted, false if rejected
     */
    bool update_internal(const Measurement& z,
                         const Vector3d& drone_pos,
                         const Matrix3d& R_WC)
    {
        if (!is_initialized_) return false;

        Vector3d p_target = x_.head<3>();
        double W = x_(6);

        // Line-of-sight in camera frame
        Vector3d los_W = p_target - drone_pos;
        Vector3d los_C = R_WC * los_W;
        double x_c = los_C(0), y_c = los_C(1), z_c = los_C(2);

        // Target behind camera → skip
        if (x_c < 1.0) return false;

        // Singularity protection: push state away if too close
        double r = los_W.norm();
        if (r < 5.0) {
            double scale = 5.0 / (r + 1e-5);
            x_.head<3>() = drone_pos + los_W * scale;
            los_W = x_.head<3>() - drone_pos;
            los_C = R_WC * los_W;
            x_c = los_C(0); y_c = los_C(1); z_c = los_C(2);
        }

        // Predicted measurements
        double az_pred = std::atan2(-z_c, x_c);
        double el_pred = std::atan2(-y_c, x_c);
        double alpha_pred = 2.0 * std::atan(W / (2.0 * x_c));

        VectorZ z_pred(az_pred, el_pred, alpha_pred);
        VectorZ z_meas(z.azimuth, z.elevation, z.visual_angle);

        // Innovation with angle wrapping
        VectorZ y = z_meas - z_pred;
        y(0) = wrap_angle(y(0));
        y(1) = wrap_angle(y(1));

        // Jacobian H (3×7)
        MatrixH H = MatrixH::Zero();

        // 1. azimuth = atan2(-z_c, x_c)
        double den_az = x_c * x_c + z_c * z_c;
        if (den_az < 1e-10) den_az = 1e-10;
        Vector3d daz_dlosC(z_c / den_az, 0.0, -x_c / den_az);
        H.row(0).head<3>() = daz_dlosC.transpose() * R_WC;

        // 2. elevation = atan2(-y_c, x_c)
        double den_el = x_c * x_c + y_c * y_c;
        if (den_el < 1e-10) den_el = 1e-10;
        Vector3d del_dlosC(y_c / den_el, -x_c / den_el, 0.0);
        H.row(1).head<3>() = del_dlosC.transpose() * R_WC;

        // 3. visual_angle alpha = 2·atan(W/(2·x_c))
        double den_alpha = 4.0 * x_c * x_c + W * W;
        H.row(2).head<3>() = (-4.0 * W / den_alpha) * R_WC.row(0);
        H(2, 6) = 4.0 * x_c / den_alpha;  // d(α)/d(W)

        // Innovation covariance
        MatrixS S = H * P_ * H.transpose() + R_;

        // Innovation gate (chi² threshold, 3-DOF, 99.99% → ~18.5, use 20.0)
        MatrixS S_inv;
        bool invertible = false;
        double det = S.determinant();
        if (std::abs(det) > 1e-15) {
            S_inv = S.inverse();
            invertible = true;
        }
        if (!invertible) return false;

        double chi2 = y.transpose() * S_inv * y;
        if (chi2 > 20.0) return false;  // Reject extreme outliers

        // Kalman gain
        MatrixK K = P_ * H.transpose() * S_inv;

        // State update
        x_ = x_ + K * y;

        // Joseph form covariance update (numerically stable)
        Matrix7d I_KH = Matrix7d::Identity() - K * H;
        P_ = I_KH * P_ * I_KH.transpose() + K * R_ * K.transpose();

        // Physical constraint: W >= 0.1m
        if (x_(6) < 0.1) x_(6) = 0.1;

        return true;
    }

public:
    /**
     * @brief Get estimated target position in world frame
     */
    Vector3d get_position() const { return x_.head<3>(); }

    /**
     * @brief Get estimated target velocity in world frame
     */
    Vector3d get_velocity() const { return x_.segment<3>(3); }

    /**
     * @brief Get estimated target physical width
     */
    double get_width() const { return x_(6); }

    /**
     * @brief Get full state vector [px, py, pz, vx, vy, vz, W]
     */
    const Vector7d& get_state() const { return x_; }

    /**
     * @brief Get state covariance matrix
     */
    const Matrix7d& get_covariance() const { return P_; }

    /**
     * @brief Check if filter is initialized
     */
    bool is_initialized() const { return is_initialized_; }

    /**
     * @brief Check if filter state is healthy (no NaN/Inf)
     */
    bool is_healthy() const {
        return !x_.hasNaN() && x_.allFinite() && !P_.hasNaN() && P_.allFinite();
    }

    /**
     * @brief Reset filter to uninitialized state
     */
    void reset() {
        is_initialized_ = false;
        x_.setZero();
        P_ = Vector7d(900.0, 900.0, 900.0,
                       100.0, 100.0, 100.0,
                       4.0).asDiagonal();
        history_buffer_.clear();
    }

    /**
     * @brief Compute bearing-angle measurement from bbox pixel data
     * Static utility for converting raw pixel detections to angular measurements
     * 
     * @param bbox_u Horizontal offset of bbox center from image center (pixels)
     * @param bbox_v Vertical offset of bbox center from image center (pixels)
     * @param bbox_size Bbox size in pixels (typically width)
     * @param focal_length Camera focal length in pixels
     * @return Measurement struct with azimuth, elevation, visual_angle
     */
    static Measurement compute_measurement(double bbox_u, double bbox_v, double bbox_size,
                                           double focal_length)
    {
        Measurement m;
        m.azimuth = std::atan(-bbox_v / focal_length);
        m.elevation = std::atan(-bbox_u / focal_length);
        m.visual_angle = 2.0 * std::atan(bbox_size / (2.0 * focal_length));
        return m;
    }

    /**
     * @brief Compute R_WC (world-to-camera rotation) from drone yaw and gimbal Euler angles
     * Matches prototype: R_WC = R_G_C · R_W_G
     */
    static Matrix3d compute_R_WC(double psi, double gimbal_r, double gimbal_p, double gimbal_y)
    {
        double cpsi = std::cos(psi), spsi = std::sin(psi);
        Matrix3d R_W_G;
        R_W_G << cpsi, spsi, 0,
                 -spsi, cpsi, 0,
                 0, 0, 1;

        double c1 = std::cos(gimbal_y), s1 = std::sin(gimbal_y);
        double c2 = std::cos(gimbal_r), s2 = std::sin(gimbal_r);
        double c3 = std::cos(gimbal_p), s3 = std::sin(gimbal_p);

        Matrix3d R_G_C;
        R_G_C << c1*c3 - s1*s2*s3, s1*c3 + c1*s2*s3, -c2*s3,
                 -s1*c2,            c1*c2,              s2,
                 c1*s3 + s1*s2*c3, s1*s3 - c1*s2*c3,   c2*c3;

        return R_G_C * R_W_G;
    }

    /**
     * @brief Compute R_WC from camera-to-world quaternion
     * @param q_C_W Camera-to-world quaternion (q.toRotationMatrix() transforms camera→world)
     * @return R_WC world-to-camera rotation matrix
     * 
     * WARNING: This only works if q_C_W represents the FULL camera orientation in world frame.
     * If the gimbal quaternion doesn't include drone yaw (common for encoder-based gimbal yaw),
     * use compute_R_WC(psi, gimbal_r, gimbal_p, gimbal_y) instead.
     */
    static Matrix3d compute_R_WC_from_quaternion(const Eigen::Quaterniond& q_C_W)
    {
        // q_C_W.toRotationMatrix() = R_C_W (camera-to-world)
        // We need R_WC = R_C_W^T (world-to-camera)
        return q_C_W.toRotationMatrix().transpose();
    }

    /**
     * @brief Compute current LOS angles from state estimate (for PN switch decision)
     * @param drone_pos Current drone position
     * @param R_WC Current world-to-camera rotation
     * @return [azimuth, elevation] in radians. azimuth>0: target right, elevation>0: target down
     */
    Eigen::Vector2d compute_los_angles(const Vector3d& drone_pos, const Matrix3d& R_WC) const
    {
        if (!is_initialized_) return Eigen::Vector2d::Zero();
        
        Vector3d los_W = x_.head<3>() - drone_pos;
        Vector3d los_C = R_WC * los_W;
        
        Eigen::Vector2d angles;
        if (los_C(0) < 0.1) {
            angles.setZero();
        } else {
            angles(0) = std::atan2(-los_C(2), los_C(0));  // azimuth
            angles(1) = std::atan2(-los_C(1), los_C(0));  // elevation
        }
        return angles;
    }

    /**
     * @brief Compute full LOS state: angles and their analytical rates from 3D EKF state
     * 
     * LOS rates are computed by differentiating the camera-frame projection:
     *   d(az)/dt  = (-ż_c·x_c + z_c·ẋ_c) / (x_c² + z_c²)
     *   d(el)/dt  = (-ẏ_c·x_c + y_c·ẋ_c) / (x_c² + y_c²)
     * where los_C = R_WC * (p_target - p_drone), d(los_C)/dt ≈ R_WC * (v_target - v_drone)
     * (gimbal rotation rate ignored — stabilized platform assumption)
     *
     * @param drone_pos Current drone position in world frame
     * @param drone_vel Current drone velocity in world frame
     * @param R_WC World-to-camera rotation matrix
     * @return [azimuth, elevation, azimuth_rate, elevation_rate] (rad, rad/s)
     */
    Eigen::Vector4d get_los_state(const Vector3d& drone_pos,
                                   const Vector3d& drone_vel,
                                   const Matrix3d& R_WC) const
    {
        Eigen::Vector4d result = Eigen::Vector4d::Zero();
        if (!is_initialized_) return result;

        // LOS in camera frame
        Vector3d los_W = x_.head<3>() - drone_pos;
        Vector3d los_C = R_WC * los_W;
        double x_c = los_C(0), y_c = los_C(1), z_c = los_C(2);
        if (x_c < 1.0) return result;  // target behind camera

        // LOS angles
        result(0) = std::atan2(-z_c, x_c);  // azimuth
        result(1) = std::atan2(-y_c, x_c);  // elevation

        // Relative velocity in camera frame (ignoring dR_WC/dt for stabilized gimbal)
        Vector3d vel_rel_W = x_.segment<3>(3) - drone_vel;
        Vector3d vel_rel_C = R_WC * vel_rel_W;
        double xd = vel_rel_C(0), yd = vel_rel_C(1), zd = vel_rel_C(2);

        // Analytical LOS rates
        double den_az = x_c * x_c + z_c * z_c;
        double den_el = x_c * x_c + y_c * y_c;
        if (den_az > 1e-6) result(2) = (-zd * x_c + z_c * xd) / den_az;
        if (den_el > 1e-6) result(3) = (-yd * x_c + y_c * xd) / den_el;

        return result;
    }

private:
    static double wrap_angle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }

    // Parameters
    double focal_length_;
    double initial_width_est_;
    double range_init_bias_;
    double q_accel_;
    double q_W_;

    // State and covariance
    Vector7d x_;
    Matrix7d P_;

    // Measurement noise
    MatrixS R_;

    // Status
    bool is_initialized_;

    // History buffer for delayed measurement rollback
    std::deque<StateHistory> history_buffer_;
    size_t max_buffer_size_ = 0;
};


/**
 * @class ImagePlaneTracker
 * @brief Lightweight Kalman filter on normalized image coordinates for direct LOS rate estimation
 * 
 * Tracks target in image plane using a Constant Velocity model on normalized coordinates:
 *   State: [nu, nv, nu_dot, nv_dot]  (normalized coords = pixel_offset / focal_length)
 *   Measurement: [nu, nv]
 * 
 * LOS rates are derived analytically from the state:
 *   azimuth_rate  = d/dt[atan(-nv)] = (-nv_dot) / (1 + nv²)
 *   elevation_rate = d/dt[atan(-nu)] = (-nu_dot) / (1 + nu²)
 * 
 * Advantage over 3D-derived rates: range-independent, directly observable from image motion.
 * Mimics the old DroneEKF's pixel velocity tracking + stateToLosAngles() functionality.
 */
class ImagePlaneTracker {
public:
    using Vector4d = Eigen::Matrix<double, 4, 1>;
    using Matrix4d = Eigen::Matrix<double, 4, 4>;
    using Vector2d = Eigen::Vector2d;
    using Matrix2d = Eigen::Matrix2d;
    using Matrix2x4 = Eigen::Matrix<double, 2, 4>;
    using Matrix4x2 = Eigen::Matrix<double, 4, 2>;

    /**
     * @param sigma_pos   Measurement noise std in normalized coords (≈ pixel_std / focal_length)
     * @param q_vel       Process noise for velocity (normalized coords/s² spectral density)
     * @param ema_n       EMA smoothing window for output LOS rates
     */
    ImagePlaneTracker(double sigma_pos = 0.003, double q_vel = 0.5, int ema_n = 6)
        : sigma_pos_(sigma_pos), q_vel_(q_vel), ema_alpha_(2.0 / (ema_n + 1)),
          is_initialized_(false)
    {
        x_.setZero();
        P_ = Vector4d(0.01, 0.01, 1.0, 1.0).asDiagonal();

        // Measurement noise
        R_ = Vector2d(sigma_pos * sigma_pos, sigma_pos * sigma_pos).asDiagonal();

        // Observation matrix H = [I₂ 0₂]
        H_.setZero();
        H_(0, 0) = 1.0;
        H_(1, 1) = 1.0;

        los_rate_ema_.setZero();
        ema_initialized_ = false;
    }

    /**
     * @brief Initialize with first observation
     * @param nu  Normalized horizontal offset (u_pixel / focal_length, FLU left-positive)
     * @param nv  Normalized vertical offset (v_pixel / focal_length, FLU up-positive)
     */
    void initialize(double nu, double nv) {
        x_ << nu, nv, 0.0, 0.0;
        P_ = Vector4d(sigma_pos_ * sigma_pos_, sigma_pos_ * sigma_pos_,
                       1.0, 1.0).asDiagonal();
        is_initialized_ = true;
        ema_initialized_ = false;
        last_update_time_ = -1.0;
    }

    /**
     * @brief Predict step (Constant Velocity model)
     * @param dt Time step in seconds
     */
    void predict(double dt) {
        if (!is_initialized_ || dt <= 0) return;

        // State transition F
        Matrix4d F = Matrix4d::Identity();
        F(0, 2) = dt;
        F(1, 3) = dt;

        // Process noise Q (CV model: white noise jerk integration)
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        double qv = q_vel_;

        Matrix4d Q = Matrix4d::Zero();
        Q(0, 0) = qv * dt3 / 3.0;
        Q(0, 2) = qv * dt2 / 2.0;
        Q(2, 0) = qv * dt2 / 2.0;
        Q(2, 2) = qv * dt;
        Q(1, 1) = qv * dt3 / 3.0;
        Q(1, 3) = qv * dt2 / 2.0;
        Q(3, 1) = qv * dt2 / 2.0;
        Q(3, 3) = qv * dt;

        x_ = F * x_;
        P_ = F * P_ * F.transpose() + Q;
    }

    /**
     * @brief Update step with pixel measurement
     * @param nu  Normalized horizontal offset (FLU convention)
     * @param nv  Normalized vertical offset (FLU convention)
     * @param current_time  Current time in seconds (for dt computation if not using separate predict)
     * @return true if update accepted
     */
    bool update(double nu, double nv, double current_time = -1.0) {
        if (!is_initialized_) {
            initialize(nu, nv);
            last_update_time_ = current_time;
            return true;
        }

        // If time is provided and predict hasn't been called separately, do predict here
        if (current_time > 0 && last_update_time_ > 0) {
            double dt = current_time - last_update_time_;
            if (dt > 0 && dt < 1.0) {  // sanity check
                predict(dt);
            }
        }
        last_update_time_ = current_time;

        Vector2d z(nu, nv);
        Vector2d z_pred = H_ * x_;
        Vector2d y = z - z_pred;

        // Innovation covariance
        Matrix2d S = H_ * P_ * H_.transpose() + R_;
        Matrix2d S_inv = S.inverse();

        // Chi² gate (2-DOF, 99% → 9.21)
        double chi2 = y.transpose() * S_inv * y;
        if (chi2 > 15.0) return false;

        // Kalman gain
        Matrix4x2 K = P_ * H_.transpose() * S_inv;

        // State update
        x_ = x_ + K * y;

        // Joseph form covariance update
        Matrix4d I_KH = Matrix4d::Identity() - K * H_;
        P_ = I_KH * P_ * I_KH.transpose() + K * R_ * K.transpose();

        return true;
    }

    /**
     * @brief Get LOS state: [azimuth, elevation, azimuth_rate, elevation_rate]
     * 
     * Convention (matching BearingAngleEKF / camera FLU frame):
     *   azimuth  = atan(-nv)   (vertical angle, positive = target below boresight in standard coords)
     *   elevation = atan(-nu)  (horizontal angle, positive = target right of boresight in standard coords)
     *   
     * Rates:
     *   azimuth_rate  = d/dt[atan(-nv)]  = -nv_dot / (1 + nv²)
     *   elevation_rate = d/dt[atan(-nu)] = -nu_dot / (1 + nu²)
     * 
     * Output rates are EMA-filtered for smoothness.
     */
    Eigen::Vector4d get_los_state() const {
        Eigen::Vector4d result;
        double nu = x_(0), nv = x_(1);
        double nu_dot = x_(2), nv_dot = x_(3);

        result(0) = std::atan(-nv);    // azimuth
        result(1) = std::atan(-nu);    // elevation

        // Analytical LOS rates from normalized velocity
        double az_rate_raw  = -nv_dot / (1.0 + nv * nv);
        double el_rate_raw  = -nu_dot / (1.0 + nu * nu);

        // EMA smoothing
        if (!ema_initialized_) {
            los_rate_ema_(0) = az_rate_raw;
            los_rate_ema_(1) = el_rate_raw;
            ema_initialized_ = true;
        } else {
            los_rate_ema_(0) = ema_alpha_ * az_rate_raw + (1.0 - ema_alpha_) * los_rate_ema_(0);
            los_rate_ema_(1) = ema_alpha_ * el_rate_raw + (1.0 - ema_alpha_) * los_rate_ema_(1);
        }

        result(2) = los_rate_ema_(0);  // azimuth rate (filtered)
        result(3) = los_rate_ema_(1);  // elevation rate (filtered)

        return result;
    }

    /**
     * @brief Get raw pixel velocity in normalized coords [nu_dot, nv_dot]
     */
    Vector2d get_pixel_velocity() const { return x_.tail<2>(); }

    /**
     * @brief Get LOS rate magnitude (rad/s)
     */
    double get_los_rate_magnitude() const {
        auto los = get_los_state();
        return std::sqrt(los(2) * los(2) + los(3) * los(3));
    }

    bool is_initialized() const { return is_initialized_; }

    void reset() {
        is_initialized_ = false;
        ema_initialized_ = false;
        x_.setZero();
        P_ = Vector4d(0.01, 0.01, 1.0, 1.0).asDiagonal();
    }

    bool is_healthy() const {
        return !x_.hasNaN() && x_.allFinite() && !P_.hasNaN() && P_.allFinite();
    }

private:
    double sigma_pos_;
    double q_vel_;
    double ema_alpha_;

    Vector4d x_;
    Matrix4d P_;
    Matrix2d R_;
    Matrix2x4 H_;

    mutable Vector2d los_rate_ema_;
    mutable bool ema_initialized_;

    double last_update_time_ = -1.0;
    bool is_initialized_;
};
