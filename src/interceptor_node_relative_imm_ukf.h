#pragma once

// InterceptorNodeRelativeIMMUKF: ROS2 node wiring the relative IMM-UKF estimator
// (relative_imm_ukf.h) and the PN/IBVS controller (advanced_hybrid_controller.h)
// to the aircraft. ODOMETRY-FREE: the IMU is the sole ego-motion source
// (orientation -> attitude, linear acceleration -> preintegrated dv_own);
// an optional radar relay provides mid-course relative [r, v_rel] measurements
// before the target enters the camera FOV. Publishes collective thrust + body
// rates as quadrotor_msgs/ControlCommand.
//
// Thread model: one MutuallyExclusive callback group for all subscriptions and
// one for the control timer; estimator_mtx_ guards the estimator + radar relay
// state, odom_mtx_ the IMU/strapdown state, control_cache_mtx_ the guidance
// cache written by the measurement callbacks and read by the control timer.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/header.hpp>
#include <quadrotor_msgs/msg/control_command.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "utils.h"
#include "camera_pose_history.h"
#include "imu_strapdown.h"
#include "relative_imm_ukf.h"
#include "advanced_hybrid_controller.h"

class InterceptorNodeRelativeIMMUKF : public rclcpp::Node {
public:
    InterceptorNodeRelativeIMMUKF();

private:
    void declare_parameters();
    void load_parameters();
    void reset_estimator();
    void reset_control_cache();
    void start_control_timer();

    // Output smoothing of the estimator's RELATIVE velocity (Fix B): EMA the raw v_rel so a
    // single-frame swing never reaches guidance. Shared by the vision and radar paths so a
    // mid-course -> terminal handover does not step the velocity fed to the controller.
    Eigen::Vector3d smooth_rel_vel(const Eigen::Vector3d& v_raw, double t_now);

    // Is vision currently delivering a consistent track? True when a radar-validated bbox has been
    // seen within vision_lost_timeout_. Used to decide guidance-cache ownership: vision owns the
    // cache as soon as it's delivering (even before the full handover completes), radar is the
    // fallback only when vision is absent/inconsistent for a definitive-loss period.
    bool vision_delivering(double now) const;

    // ---- Radar relay -------------------------------------------------------------------------
    // Each report carries ONE object (target or interceptor) as a world-ENU nav_msgs/Odometry,
    // stamped with the MEASUREMENT epoch: the radar publisher holds every sample for the sensor
    // latency before sending, so header.stamp is genuinely in the past and the lag is recoverable.
    // A relative measurement needs both objects, so samples are paired by stamp.
    void radar_callback(const nav_msgs::msg::Odometry::SharedPtr msg, bool is_target);
    void fuse_radar();

    // Mid-course guidance cache from the radar-driven estimate. Same contract as the vision path
    // (bbox_size = 0 so the terminal brake stays disarmed), but the LOS rate is taken ANALYTICALLY
    // from the filtered relative state instead of differencing the bearing: at 10 Hz with
    // metre-level position noise, bearing differencing would inject ~0.3 rad/s straight into the
    // PN lateral command.
    void fill_cache_from_radar(const Eigen::VectorXd& state, double t_now, double dt_step);

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
    bool bbox_matches_radar(const Eigen::Vector3d& los_nav, double bbox_s, double* d_az_out);

    // Called with only radar-validated frames. bbox_ok false means "no usable detection this frame"
    // — either the detector saw nothing or the detection failed the radar cross-check.
    void update_vision_handover(bool bbox_ok, double bbox_s, double t_now, double d_az);

    // ---- Post-lock width re-anchor (radar.width_reanchor_*) --------------------------------
    // Safety net for a badly locked W: with W inflated by k, the monocular range r = focal*W/s is
    // inflated by the same k, which (a) blinds every close-range gate keyed on r_est — dither
    // min-range, width freeze, size gate, coast, fly-through — and (b) wrecks terminal guidance.
    // While vision is locked the radar fusion is disengaged, but the radar cache still updates, so
    // the ratio r_vis/r_radar is an independent, continuously available consistency check. A
    // SUSTAINED disagreement (median over a window, both frame count and time span) re-anchors W
    // and rescales the estimator onto the radar-consistent scale. Bounded: cooldown after each
    // correction + width_reanchor_max total, and the window resets whenever any gate breaks.
    // Called from bbox_callback under estimator_mtx_ (radar_rel_pos_ writer side) BEFORE
    // process_camera so the rescale lands before this frame's update.
    void update_width_reanchor(bool bbox_ok, double bbox_s, double t_now);
    void apply_width_reanchor(double ratio, double w_locked, double r_vis, double r_radar,
                              double bbox_s, double t_now, int n_frames);

    void bbox_callback(const vision_msgs::msg::Detection2D::SharedPtr msg);
    void control_timer_callback();
    void mission_launch_callback(const std_msgs::msg::Header::SharedPtr msg);
    void mission_callback(const std_msgs::msg::Bool::SharedPtr msg);

    // IMU: sole ego-motion input (ODOMETRY-FREE). Orientation -> attitude R_b2n; angular velocity
    // -> collision detection; linear acceleration -> strapdown preintegration of the ownship
    // velocity change dv_own = ∫(R_b2n·(f_imu − bias) − g)dτ over the current camera window.
    // The accumulator is drained by bbox_callback each camera frame; it is a pure RELATIVE
    // quantity per window, so there is no long-term velocity drift.
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

    // /fpv/gimbal (Imu): the orientation field is q_C_W in the FLU camera frame (x=fwd, y=left,
    // z=up) — the SAME convention as airsim_tracker and the hardware gimbal driver. This node's
    // estimator uses the RDF optical frame internally (x=right, y=down, z=fwd), so convert
    // FLU->RDF at intake:
    //   q_C_W_rdf = q_C_W_flu * R(R_b2c_base_^T),  since (FLU->ENU)*(RDF->FLU) = RDF->ENU.
    // R_b2c_base_ is the FLU(body)->RDF(cam) extrinsic. angular_velocity / orientation_covariance
    // are NOT consumed here — this node runs its own IMM-UKF + body-IMU strapdown.
    void gimbal_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

    // Build R_b2c from the given gimbal cam-x pitch angle (positive = down-gaze).
    Eigen::Matrix3d gimbal_R_b2c(double angle) const;

    void transition_to_state(MissionState new_state, const char* reason);
    void publish_mission_state(bool active);
    void stop();

    // ---- Camera / estimator configuration ---------------------------------------------------
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

    // ---- ROS interfaces ----------------------------------------------------------------------
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr mission_sub_;
    rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr tracker_trigger_sub_;
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
    bool radar_vel_aid_after_lock_ = false;   // OFF: pure terminal vision by default (radar vel aid is an opt-in experiment)
    double radar_vel_aid_std_ = 2.5;         // m/s, 1-sigma of the vel-only aid
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
    // Dither suppression when a fresh radar track is inside this range (0 = off). Radar-only; see
    // compute_control's dither gate.
    double dither_radar_range_ = 20.0;
    // Post-lock width re-anchor (radar.width_reanchor_*): sustained r_vis/r_radar disagreement ->
    // re-lock W + rescale estimator. See update_width_reanchor.
    bool width_reanchor_enable_ = true;
    double width_reanchor_min_range_ = 10.0;
    double width_reanchor_max_range_ = 150.0;
    double width_reanchor_ratio_ = 1.25;
    double width_reanchor_window_ = 2.0;
    int width_reanchor_min_frames_ = 15;
    int width_reanchor_max_ = 2;
    int width_reanchor_count_ = 0;
    double width_reanchor_cooldown_until_ = -1.0;
    struct ReanchorSample { double ratio; double t; };
    std::vector<ReanchorSample> width_reanchor_samples_;

    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::CallbackGroup::SharedPtr timer_callback_group_;

    // ---- Guidance cache (written by measurement callbacks, read by the control timer) ---------
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
        double bbox_size = 0.0;
        double target_speed = 0.0;
        // Ownship world accel over the last camera window (IMU preintegration; geometric APN input).
        Eigen::Vector3d a_own = Eigen::Vector3d::Zero();
    };
    ControlCacheData control_cache_;

    MissionStateMachine mission_state_;

    // ---- IMU strapdown / ego state (guarded by odom_mtx_) -------------------------------------
    std::mutex odom_mtx_;
    // ODOMETRY-FREE ego state: attitude + preintegrated velocity change from the IMU only.
    Eigen::Matrix3d latest_R_b2n_ = Eigen::Matrix3d::Identity();   // body FLU -> world ENU
    CameraPoseHistory gimbal_history_;
    bool gyro_attitude_during_mission_ = false;
    bool mission_gyro_initialized_ = false;
    Eigen::Quaterniond mission_gyro_attitude_ = Eigen::Quaterniond::Identity();
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
    // Collision detection (AirSim/real HW only): a gyro spike vs the
    // last commanded body rate signals physical contact -> intercept success -> COMPLETED. Gated on
    // !use_simulation_: the ROS2 Python sim has ground-truth range and judges intercept externally.
    bool collision_detected_ = false;
    double collision_min_bbox_px_ = 0.0;
    bool use_imu_axial_speed_ = false;
    double target_axial_feedforward_limit_ = 0.0;
    double target_axial_feedforward_ = 0.0;
    double last_target_axial_time_ = -1.0;
    double capture_range_ = 2.0;
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
    double g_ = 9.81;              // gravity (from physics.g config)
    Eigen::Vector3d last_cmd_ = Eigen::Vector3d::Zero();
};
