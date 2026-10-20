#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

inline double bounded_target_axial_speed(double own_speed, double relative_speed, double limit)
{
    if (!std::isfinite(own_speed) || !std::isfinite(relative_speed) ||
        !std::isfinite(limit) || limit < 0)
        throw std::invalid_argument("invalid target axial speed inputs");
    const double target_speed = own_speed + relative_speed;
    // A monocular scale error is not evidence of a target moving at the configured bound.
    return std::abs(target_speed) <= limit ? target_speed : 0.0;
}

inline double terminal_vision_range_minimum(double previous, double distance, double bbox_size)
{
    if (!std::isfinite(previous) || previous < 0 || !std::isfinite(distance) || distance < 0 ||
        !std::isfinite(bbox_size) || bbox_size < 0)
        throw std::invalid_argument("invalid terminal range inputs");
    return bbox_size >= 80 ? std::min(previous, distance) : previous;
}

inline double braking_speed_limit(double distance, double deceleration, double delay)
{
    if (!std::isfinite(distance) || distance < 0 || !std::isfinite(deceleration) ||
        deceleration <= 0 || !std::isfinite(delay) || delay < 0)
        throw std::invalid_argument("invalid braking envelope");
    const double lag_velocity = deceleration * delay;
    return std::sqrt(lag_velocity * lag_velocity + 2 * deceleration * distance) - lag_velocity;
}

inline double terminal_speed_floor_cap(double cruise_speed, double capped_speed, double floor,
                                        double bbox_size, double bbox_threshold,
                                        double los_rate, double los_rate_threshold)
{
    if (!std::isfinite(cruise_speed) || cruise_speed < 0 ||
        !std::isfinite(capped_speed) || capped_speed < 0 ||
        !std::isfinite(floor) || floor < 0 ||
        !std::isfinite(bbox_size) || bbox_size < 0 ||
        !std::isfinite(bbox_threshold) || bbox_threshold <= 0 ||
        !std::isfinite(los_rate) || !std::isfinite(los_rate_threshold) ||
        los_rate_threshold < 0)
        throw std::invalid_argument("invalid terminal speed floor inputs");
    if (bbox_size < bbox_threshold || std::abs(los_rate) > los_rate_threshold)
        return capped_speed;
    return std::min(cruise_speed, std::max(capped_speed, floor));
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

// Terminal arrival vertical speed: the geometric rate that nulls the remaining height error
// exactly by arrival (t_go = range / closing_speed), scaled by `margin` (>1 leaves headroom for
// the vertical tracking lag of the K_az/a_z_limit loop). Sign matches height_error: positive
// height error (target above) demands a climb.
inline double terminal_arrival_vertical_speed(double height_error, double range,
                                              double closing_speed, double margin)
{
    if (!std::isfinite(height_error) || !std::isfinite(range) || range <= 0 ||
        !std::isfinite(closing_speed) || closing_speed <= 0 ||
        !std::isfinite(margin) || margin < 0)
        throw std::invalid_argument("invalid terminal arrival geometry");
    return margin * height_error * closing_speed / range;
}

// Terminal arrival axial speed: the dual of the law above. Given the vertical rate the airframe
// is ACTUALLY delivering, the horizontal closure that lets that climb null the height error by
// arrival is margin * climb_rate * horizontal_distance / |height_error| (the inverse of
// vertical_closure_speed_limit). Returns infinity when the geometry cannot bind (no horizontal
// distance, no height error), so the caller can apply it as a plain cap.
inline double axial_speed_for_arrival(double climb_rate, double horizontal_distance,
                                      double height_error, double margin)
{
    if (!std::isfinite(climb_rate) || climb_rate <= 0 ||
        !std::isfinite(horizontal_distance) || horizontal_distance < 0 ||
        !std::isfinite(height_error) || !std::isfinite(margin) || margin <= 0)
        throw std::invalid_argument("invalid arrival axial geometry");
    if (horizontal_distance < 1e-3 || std::abs(height_error) < 1e-6)
        return std::numeric_limits<double>::infinity();
    return margin * climb_rate * horizontal_distance / std::abs(height_error);
}
