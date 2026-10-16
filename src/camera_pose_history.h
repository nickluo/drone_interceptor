#pragma once

#include <cmath>
#include <deque>
#include <optional>
#include <Eigen/Geometry>

class CameraPoseHistory
{
public:
    bool add(double timestamp, const Eigen::Quaterniond& orientation)
    {
        if (!std::isfinite(timestamp) || timestamp <= 0 ||
            !orientation.coeffs().allFinite() || orientation.norm() < 1e-6 ||
            (!samples_.empty() && timestamp <= samples_.back().timestamp))
            return false;
        samples_.push_back({timestamp, orientation.normalized()});
        if (samples_.size() > 100)
            samples_.pop_front();
        return true;
    }

    std::optional<Eigen::Quaterniond> at(double timestamp) const
    {
        if (!std::isfinite(timestamp))
            return std::nullopt;
        for (const auto& sample : samples_)
            if (std::abs(sample.timestamp - timestamp) < 1e-6)
                return sample.orientation;
        // Delivery race between the gimbal and detector streams: a detection can be stamped a
        // few ms NEWER than the newest gimbal sample (both stamped at their respective publish
        // instants). No "after" sample exists yet, so strict bracketing would drop the frame —
        // hold the newest orientation instead. Margin small enough that a genuinely stalled
        // gimbal stream still fails the lookup.
        if (!samples_.empty() && timestamp > samples_.back().timestamp &&
            timestamp - samples_.back().timestamp <= .05)
            return samples_.back().orientation;
        for (std::size_t i = 1; i < samples_.size(); ++i) {
            const auto& before = samples_[i - 1];
            const auto& after = samples_[i];
            if (before.timestamp < timestamp && timestamp < after.timestamp) {
                const double gap = after.timestamp - before.timestamp;
                if (gap > .2)
                    return std::nullopt;
                return before.orientation.slerp((timestamp - before.timestamp) / gap,
                                                after.orientation).normalized();
            }
        }
        return std::nullopt;
    }

private:
    struct Sample {
        double timestamp;
        Eigen::Quaterniond orientation;
    };
    std::deque<Sample> samples_;
};
