#pragma once

#include <Eigen/Dense>

// Define fixed-size vector types for performance
using Vector5d = Eigen::Matrix<double, 5, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Vector9d = Eigen::Matrix<double, 9, 1>;
using Vector13d = Eigen::Matrix<double, 13, 1>;
using Vector16d = Eigen::Matrix<double, 16, 1>;

// ============================================================================
// EMAFilter Class (from exp_filter.py)
// ============================================================================
class EMAFilter {
public:
    EMAFilter(int n, int channels) : channels_(channels) {
        if (n <= 0 || channels <= 0) {
            throw std::invalid_argument("n and channels must be positive");
        }
        alpha_ = 2.0 / (n + 1);
        data_ = Eigen::VectorXd::Zero(channels);
        initialized_ = std::vector<uint8_t>(channels, false);
    }

    Eigen::VectorXd filter(const Eigen::VectorXd& values) {
        if (values.size() != channels_) {
            throw std::runtime_error("Input array size does not match filter channels");
        }
        for (int i = 0; i < channels_; ++i) {
            if (!initialized_[i]) {
                data_[i] = values[i];
                initialized_[i] = true;
            } else {
                data_[i] = alpha_ * values[i] + (1.0 - alpha_) * data_[i];
            }
        }
        return data_;
    }

private:
    double alpha_;
    int channels_;
    Eigen::VectorXd data_;
    std::vector<uint8_t> initialized_;
};

// ============================================================================
// TargetEKF Class (from target_ekf.py) - 与原版本相同
// ============================================================================
class TargetEKF {
public:
    TargetEKF(double dt, double focal_length, double target_size_est, double pos_dev, double size_dev)
        : focal_length_(focal_length), target_size_est_(target_size_est) {
        
        // Initialize state [px, py, pz, vx, vy, vz]
        x_ = Vector6d::Zero();
        
        // Process noise covariance
        Q_ = Eigen::MatrixXd::Zero(6, 6);
        Q_.block<3,3>(0, 0) = Eigen::Matrix3d::Identity() * (pos_dev * pos_dev * dt * dt);
        Q_.block<3,3>(3, 3) = Eigen::Matrix3d::Identity() * (pos_dev * pos_dev);
        
        // Measurement noise covariance
        R_ = Eigen::Matrix3d::Identity() * (size_dev * size_dev);
        
        // State transition matrix
        F_ = Eigen::MatrixXd::Identity(6, 6);
        F_.block<3,3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
        
        // Initial covariance
        P_ = Eigen::MatrixXd::Identity(6, 6) * 100.0;
        
        is_initialized_ = false;
    }

    void initialize(const Eigen::Vector3d& initial_pos_est) {
        x_.head<3>() = initial_pos_est;
        x_.tail<3>() = Eigen::Vector3d::Zero(); // Zero initial velocity
        is_initialized_ = true;
    }

    void predict(double dt) {
        if (!is_initialized_) return;
        
        // Update state transition matrix with actual dt
        F_.block<3,3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
        
        // Predict state
        x_ = F_ * x_;
        
        // Predict covariance
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    void update(const vision_msgs::msg::Detection2D& bbox, const Eigen::Vector3d& drone_pos, double psi) {
        if (!is_initialized_) return;
        
        // Convert bbox to position measurement
        Eigen::Vector3d z = noisy_bbox_to_pos(bbox, drone_pos, psi);
        
        // Measurement function H (position only)
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 6);
        H.block<3,3>(0, 0) = Eigen::Matrix3d::Identity();
        
        // Innovation
        Eigen::Vector3d y = z - H * x_;
        
        // Innovation covariance
        Eigen::Matrix3d S = H * P_ * H.transpose() + R_;
        
        // Kalman gain
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
        
        // Update state and covariance
        x_ = x_ + K * y;
        P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H) * P_;
    }
    
    Eigen::Vector3d noisy_bbox_to_pos(const vision_msgs::msg::Detection2D& bbox, const Eigen::Vector3d& drone_pos, double psi) {
        double cx = bbox.bbox.center.position.x;
        double cy = bbox.bbox.center.position.y;
        double size = std::max(bbox.bbox.size_x, bbox.bbox.size_y);
        
        // Estimate distance from bbox size
        double dist_est = focal_length_ * target_size_est_ / size;
        
        // Convert to world coordinates
        double cpsi = std::cos(psi), spsi = std::sin(psi);
        Eigen::Matrix3d R_C_W;
        R_C_W << cpsi, -spsi, 0,
                 spsi,  cpsi, 0,
                    0,     0, 1;
        
        Eigen::Vector3d pos_C(dist_est, cx * dist_est / focal_length_, cy * dist_est / focal_length_);
        Eigen::Vector3d pos_W = R_C_W * pos_C + drone_pos;
        
        return pos_W;
    }

    bool is_initialized() const { return is_initialized_; }
    const Vector6d& get_state() const { return x_; }

private:
    Vector6d x_;
    Eigen::MatrixXd P_;
    Eigen::MatrixXd F_;
    Eigen::MatrixXd Q_;
    Eigen::Matrix3d R_;
    double focal_length_;
    double target_size_est_;
    bool is_initialized_;
};