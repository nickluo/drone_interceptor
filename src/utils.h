#pragma once

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Define fixed-size vector/matrix types for performance
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

// Clamp `value` to [low, high]. Shared by the IMM-UKF estimator, the controller and the node.
inline double clamp(double value, double low, double high) {
    return std::max(low, std::min(high, value));
}

// ============================================================================
// 任务状态枚举和线程安全状态机
// ============================================================================
enum class MissionState : uint8_t {
    IDLE,           // 空闲状态，等待任务启动
    INITIALIZING,   // 初始化中，等待传感器数据
    TRACKING,       // 跟踪目标中
    TERMINAL_GUIDANCE, // 末制导 (PN)
    BLIND_FLIGHT,   // 目标丢失，盲飞状态
    FAILSAFE,       // 故障安全模式
    COMPLETED,      // 任务完成
    ABORTED         // 任务中止
};

class MissionStateMachine {
public:
    MissionStateMachine() : state_(MissionState::IDLE) {}

    // 原子地获取当前状态
    MissionState get() const {
        return state_.load(std::memory_order_acquire);
    }

    // 原子地设置状态
    void set(MissionState new_state) {
        state_.store(new_state, std::memory_order_release);
    }

    // 尝试从from状态转换到to状态，成功返回true
    bool transition(MissionState from, MissionState to) {
        MissionState expected = from;
        return state_.compare_exchange_strong(expected, to,
            std::memory_order_acq_rel, std::memory_order_acquire);
    }

    // 检查是否可以进行控制
    bool can_control() const {
        MissionState s = get();
        return s == MissionState::TRACKING || s == MissionState::BLIND_FLIGHT || s == MissionState::TERMINAL_GUIDANCE;
    }

    // 检查任务是否活跃
    bool is_active() const {
        MissionState s = get();
        return s == MissionState::INITIALIZING ||
               s == MissionState::TRACKING ||
               s == MissionState::BLIND_FLIGHT ||
               s == MissionState::TERMINAL_GUIDANCE;
    }

    // 检查是否已终止
    bool is_terminated() const {
        MissionState s = get();
        return s == MissionState::COMPLETED ||
               s == MissionState::ABORTED ||
               s == MissionState::IDLE;
    }

    // 获取状态名称（用于日志）
    static const char* state_name(MissionState s) {
        switch (s) {
            case MissionState::IDLE: return "IDLE";
            case MissionState::INITIALIZING: return "INITIALIZING";
            case MissionState::TRACKING: return "TRACKING";
            case MissionState::TERMINAL_GUIDANCE: return "TERMINAL_GUIDANCE";
            case MissionState::BLIND_FLIGHT: return "BLIND_FLIGHT";
            case MissionState::FAILSAFE: return "FAILSAFE";
            case MissionState::COMPLETED: return "COMPLETED";
            case MissionState::ABORTED: return "ABORTED";
            default: return "UNKNOWN";
        }
    }

private:
    std::atomic<MissionState> state_;
};
