#pragma once

#include <cmath>
#include <stdexcept>
#include <Eigen/Geometry>

inline Eigen::Quaterniond integrate_body_gyro(const Eigen::Quaterniond& attitude,
                                             const Eigen::Vector3d& body_rate, double dt)
{
    if (!attitude.coeffs().allFinite() || attitude.norm() < 1e-6 ||
        !body_rate.allFinite() || !std::isfinite(dt) || dt < 0)
        throw std::invalid_argument("invalid gyro attitude integration input");
    const double speed = body_rate.norm();
    if (speed * dt < 1e-12)
        return attitude.normalized();
    // Body-frame increments postmultiply the body-to-world attitude.
    return (attitude * Eigen::Quaterniond(Eigen::AngleAxisd(speed * dt, body_rate / speed))).normalized();
}
