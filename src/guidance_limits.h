#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>

inline double braking_speed_limit(double distance, double deceleration, double delay)
{
    if (!std::isfinite(distance) || distance < 0 || !std::isfinite(deceleration) ||
        deceleration <= 0 || !std::isfinite(delay) || delay < 0)
        throw std::invalid_argument("invalid braking envelope");
    const double lag_velocity = deceleration * delay;
    return std::sqrt(lag_velocity * lag_velocity + 2 * deceleration * distance) - lag_velocity;
}

inline double vertical_closure_speed_limit(double horizontal_speed, double horizontal_distance,
                                          double height_error)
{
    if (!std::isfinite(horizontal_speed) || horizontal_speed < 0 ||
        !std::isfinite(horizontal_distance) || horizontal_distance < 0 ||
        !std::isfinite(height_error))
        throw std::invalid_argument("invalid closure geometry");
    if (horizontal_distance < 1e-3)
        return std::numeric_limits<double>::infinity();
    return horizontal_speed * std::abs(height_error) / horizontal_distance;
}
