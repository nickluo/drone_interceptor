// Main ROS2 node for the relative IMM-UKF interceptor. Class declaration and
// members live in interceptor_node_relative_imm_ukf.h; the estimator and
// controller stacks in relative_imm_ukf.h / advanced_hybrid_controller.h.

#include "interceptor_node_relative_imm_ukf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Throttled-log interval (ms) for the per-topic INFO_THROTTLE diagnostics.
constexpr int THROTTLE_INTERVAL_MS = 100;

InterceptorNodeRelativeIMMUKF::InterceptorNodeRelativeIMMUKF() : Node("interceptor_node_relative_imm_ukf") {
    declare_parameters();

    auto sub_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    timer_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = sub_group;

    load_parameters();

    estimator_ = std::make_unique<RelativeIMMUKFEstimator>(dt_cam_, cam_fx_, cam_fy_, w_guess_, R_b2c_);

    mission_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        this->get_parameter("topics.mission").as_string(), 10,
        std::bind(&InterceptorNodeRelativeIMMUKF::mission_callback, this, std::placeholders::_1));
    tracker_trigger_sub_ = this->create_subscription<std_msgs::msg::Header>(
        this->get_parameter("topics.mission_launch").as_string(), 10,
        std::bind(&InterceptorNodeRelativeIMMUKF::mission_launch_callback, this, std::placeholders::_1));

    // IMU: SOLE ego-motion source (ODOMETRY-FREE — the relative-state estimator + IMU strapdown
    // replaced the former odometry subscription). Orientation -> R_b2n (attitude for cam projection
    // + control); linear_acceleration -> preintegrated dv_own between camera frames (exogenous
    // a_own for the relativized process model); angular_velocity -> collision detection.
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        this->get_parameter("topics.imu").as_string(), rclcpp::SensorDataQoS(),
        std::bind(&InterceptorNodeRelativeIMMUKF::imu_callback, this, std::placeholders::_1), sub_options);

    bbox_sub_ = this->create_subscription<vision_msgs::msg::Detection2D>(
        this->get_parameter("topics.bbox").as_string(), 10,
        std::bind(&InterceptorNodeRelativeIMMUKF::bbox_callback, this, std::placeholders::_1), sub_options);

    // Gimbal orientation, IDENTICAL convention to the hardware gimbal
    // driver / AirSim: a sensor_msgs/Imu whose orientation field is the camera-to-world
    // quaternion q_C_W (camera attitude in world/ENU). angular_velocity and orientation_covariance
    // are ignored by this node (it has its own IMM-UKF + body-IMU strapdown); they are populated
    // by the publisher for contract consistency. Only consumed when camera.use_gimbal is true.
    gimbal_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        this->get_parameter("topics.gimbal").as_string(), rclcpp::SensorDataQoS(),
        std::bind(&InterceptorNodeRelativeIMMUKF::gimbal_callback, this, std::placeholders::_1), sub_options);

    control_pub_ = this->create_publisher<quadrotor_msgs::msg::ControlCommand>(
        this->get_parameter("topics.control").as_string(), 10);
    mission_state_pub_ = this->create_publisher<std_msgs::msg::Bool>(
        this->get_parameter("topics.mission_state").as_string(), 10);

    if (radar_enable_) {
        estimator_->max_range_ = radar_max_range_;
        radar_target_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            radar_topic_target_, 10,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) { radar_callback(msg, true); },
            sub_options);
        radar_own_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            radar_topic_own_, 10,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) { radar_callback(msg, false); },
            sub_options);
        RCLCPP_INFO(this->get_logger(),
            "Radar relay ENABLED: target=%s interceptor=%s pos_std=%.1fm vel_std=%.1fm/s handover=%d frames",
            radar_topic_target_.c_str(), radar_topic_own_.c_str(),
            radar_pos_std_, radar_vel_std_, handover_frames_);
    }

    mission_state_.set(MissionState::IDLE);
    RCLCPP_INFO(this->get_logger(),
        "Relative IMM-UKF node ready (ODOMETRY-FREE, IMU strapdown). "
        "IMU=%s BBox=%s Gimbal=%s Launch=%s Mission=%s Control=%s State=%s",
        imu_sub_->get_topic_name(), bbox_sub_->get_topic_name(), gimbal_sub_->get_topic_name(),
        tracker_trigger_sub_->get_topic_name(), mission_sub_->get_topic_name(),
        control_pub_->get_topic_name(), mission_state_pub_->get_topic_name());
}

void InterceptorNodeRelativeIMMUKF::declare_parameters() {
    this->declare_parameter<std::string>("topics.imu", "/mavros/imu/data");
    this->declare_parameter<std::string>("topics.bbox", "/fpv/target_bbox");
    this->declare_parameter<std::string>("topics.gimbal", "/fpv/gimbal");
    this->declare_parameter<std::string>("topics.mission_launch", "/fpv/tracker_trigger");
    this->declare_parameter<std::string>("topics.mission", "/fpv/mission_launch");
    this->declare_parameter<std::string>("topics.control", "/fpv/control_command");
    this->declare_parameter<std::string>("topics.mission_state", "/fpv/mission_state");
    this->declare_parameter<int>("camera.width_px", 640);
    this->declare_parameter<int>("camera.height_px", 480);
    // Intrinsics source — exactly one must be valid (see resolve_camera_intrinsics):
    //   fov_deg:    horizontal FOV in (0,180) -> fx = fy = (W/2)/tan(fov/2), principal point
    //               at the image center (0.0 = unset).
    //   intrinsics: 9-element row-major K matrix [fx 0 cx; 0 fy cy; 0 0 1], used directly.
    //               Takes precedence when both are set. No lens-distortion model.
    this->declare_parameter<double>("camera.fov_deg", 0.0);
    this->declare_parameter<std::vector<double>>("camera.intrinsics", std::vector<double>{});
    // Semi-strapdown single-axis pitch gimbal: gimbal_bias_deg is the UNIFIED camera down-gaze
    // angle (positive = down). Used for both the gimbal (world-stabilized boresight) and the
    // fixed mount (body-relative). The gimbal rebuilds R_b2c from q_C_W each frame; the fixed
    // mount uses the initial value.
    this->declare_parameter<bool>("camera.use_gimbal", false);
    this->declare_parameter<double>("camera.gimbal_bias_deg", 5.0);
    this->declare_parameter<double>("physics.g", 9.81);
    this->declare_parameter<double>("controller.N", 4.5);
    this->declare_parameter<double>("controller.K_att", 9.0);
    this->declare_parameter<double>("controller.K_yaw", 5.0);
    this->declare_parameter<double>("controller.v_des", 15.0);
    this->declare_parameter<double>("physics.max_tilt_deg", 35.0);
    this->declare_parameter<double>("controller.fov_keep_margin", 0.85);
    this->declare_parameter<double>("controller.axial_jerk_max", 15.0);
    this->declare_parameter<double>("physics.max_body_rate", 3.0);
    this->declare_parameter<double>("controller.terminal_brake_range", 5.0);
    this->declare_parameter<double>("controller.v_pn_ref", 12.0);
    this->declare_parameter<double>("controller.max_att_rate", 3.5);   // Fix G: clamp |d(phi/theta)_cmd/dt| (was 5)
    this->declare_parameter<double>("controller.att_ref_tc", 0.15);
    this->declare_parameter<double>("physics.max_tilt_terminal_deg", 45.0);
    this->declare_parameter<double>("controller.terminal_tilt_range", 15.0);
    this->declare_parameter<double>("controller.a_z_slew", 20.0);   // Fix C/E: |da_z/dt| limit (m/s^3)
    this->declare_parameter<bool>("controller.enable_fov_clamp", true);
    // Guidance tuning (set_guidance_tuning) — operator-facing knobs promoted from hardcoded.
    this->declare_parameter<double>("controller.K_fwd", 2.0);       // axial accel gain
    this->declare_parameter<double>("controller.K_az", 2.0);        // vertical-speed tracking gain
    this->declare_parameter<double>("physics.a_z_limit", 8.0);  // vertical accel clamp (m/s^2)
    this->declare_parameter<double>("controller.dither_amp", 0.8);  // m/s^2, lateral observability dither
    this->declare_parameter<double>("controller.dither_freq", 0.5); // Hz
    this->declare_parameter<double>("controller.dither_min_range", 10.0); // m, suppress dither inside
    this->declare_parameter<double>("controller.dither_max_bbox", 80.0); // px, ALSO suppress the dither when the bbox >= this size (large bbox = good size-SNR / close target; independent of the W-derived range estimate, which is exactly what blinds dither_min_range). 0 = off.
    this->declare_parameter<double>("controller.dither_radar_range", 20.0); // m, suppress the dither when a FRESH radar track shows the target closer than this (works after vision lock too — the radar cache keeps updating; only fusion is disengaged). 0 = off. Radar-only.
    this->declare_parameter<double>("controller.range_smooth_tc", 0.3); // s, EMA on the range feeding the turn-radius cap (0=off; radar smoothness)
    this->declare_parameter<double>("controller.turn_radius_cap_factor", 0.8); // turn-radius maneuverability cap factor (clamped [0.3,1.0]; 1.0=physics limit). The binding terminal-speed reducer — see compute_control.
    this->declare_parameter<double>("controller.axial_deadband", 1.5); // m/s, coast within ±this of v_des — kills radar-noise-driven pitch oscillation during blind flight
    this->declare_parameter<double>("radar.motion_gate_factor", 3.0); // widens the bbox-radar LOS gate for moving targets: bearing_tol += this * v_perp * delay / r
    this->declare_parameter<double>("controller.lateral_decay_floor", 0.4);  // Fix D: close-range a_lat floor
    this->declare_parameter<double>("controller.lateral_decay_start", 25.0); // Fix F: range lateral attenuation begins (m)
    // Vertical cascade (cascaded velocity-limited altitude track). Set v_climb/v_descent to the
    // airframe's real climb/sink capability (m/s) so the commanded rate is achievable; K_dz_up/dn
    // (1/s, closure tau=1/K) tune responsiveness independently for climb vs descent.
    this->declare_parameter<double>("physics.v_climb", 8.0);     // m/s, max commanded climb rate (cap)
    this->declare_parameter<double>("physics.v_descent", 8.0);   // m/s, max commanded sink rate (cap)
    this->declare_parameter<double>("controller.K_dz_up", 2.0);     // 1/s, climb responsiveness (upper target)
    this->declare_parameter<double>("controller.K_dz_dn", 1.0);     // 1/s, descent responsiveness (lower target; < K_up to avoid over-dive)
    this->declare_parameter<double>("estimation.w_guess", 1.5);
    this->declare_parameter<double>("estimation.vel_smooth_tc", 0.25);  // Fix B: target-vel EMA (s, 0=off)
    this->declare_parameter<bool>("estimation.gyro_attitude_during_mission", false);
    // Accelerometer bias in body frame, subtracted before the strapdown preintegration.
    // The Python-harness sim publishes a bias-free IMU (zeros); set calibrated values on real HW.
    this->declare_parameter<std::vector<double>>("imu_bias", {0.0, 0.0, 0.0});
    this->declare_parameter<double>("control.rate", 100.0);
    this->declare_parameter<double>("control.rate_tc", 0.02);
    this->declare_parameter<bool>("control.rate_control_attenuation", true);
    this->declare_parameter<bool>("use_simulation", false);
    this->declare_parameter<double>("collision_min_bbox_px", 0.0);
    this->declare_parameter<bool>("controller.use_imu_axial_speed", false);
    this->declare_parameter<double>("controller.target_axial_feedforward_limit", 0.0);
    this->declare_parameter<double>("controller.K_apn", 0.8);
    this->declare_parameter<double>("controller.vertical_brake_accel", 0.0);
    this->declare_parameter<double>("controller.vertical_brake_delay", 0.0);
    this->declare_parameter<double>("controller.capture_range", 2.0);
    this->declare_parameter<double>("controller.visual_speed_gain", 0.0);
    this->declare_parameter<double>("controller.visual_speed_floor", 0.0);
    this->declare_parameter<double>("controller.terminal_speed_floor", 0.0);
    this->declare_parameter<double>("controller.terminal_speed_bbox_threshold", 120.0);
    this->declare_parameter<double>("controller.terminal_speed_los_rate_threshold", 0.05);
    // ---- Radar relay (mid-course guidance before the target enters the camera FOV) --------
    // An external radar tracks BOTH the target and the interceptor and reports each as a
    // world-ENU nav_msgs/Odometry at ~10 Hz with metre-level noise and 50-100 ms latency. The
    // node differences the pair into a relative measurement, delay-compensates it, and feeds it
    // to the same IMM-UKF as a direct [r, v_rel] observation.
    this->declare_parameter<bool>("radar.enable", false);
    this->declare_parameter<std::string>("radar.topic_target", "/fpv/radar/target");
    this->declare_parameter<std::string>("radar.topic_interceptor", "/fpv/radar/interceptor");
    this->declare_parameter<double>("radar.pos_std", 3.0);        // m, 1-sigma per axis
    this->declare_parameter<double>("radar.vel_std", 1.0);        // m/s, 1-sigma per axis
    this->declare_parameter<double>("radar.pair_window", 0.15);   // s, max target/own stamp skew
    this->declare_parameter<double>("radar.timeout", 1.0);        // s, track considered stale
    this->declare_parameter<double>("radar.max_delay_comp", 0.3); // s, cap on latency extrapolation
    this->declare_parameter<double>("radar.max_range", 800.0);    // m, sanitizer ceiling with radar
    // Handover: consecutive vision frames required to declare a visual lock, and the window in
    // which the radar range + bbox size fixes the target width for terminal guidance.
    this->declare_parameter<int>("radar.handover_frames", 10);
    this->declare_parameter<double>("radar.handover_max_range", 200.0);
    this->declare_parameter<double>("radar.width_min", 0.3);
    this->declare_parameter<double>("radar.width_max", 5.0);
    this->declare_parameter<bool>("radar.vel_aid_after_lock", false); // OFF by default: terminal vision must be PURE vision (the radar node must not influence the endgame). Opt-in experiment: fuse radar VELOCITY (not position) while vision is locked to anchor v_closing — but in AirSim the lagged/10 Hz radar vel fought the 30 Hz vision update and HURT, so it is off.
    this->declare_parameter<double>("radar.vel_aid_std", 2.5);       // m/s, 1-sigma of the vel-only aid (large = trust the monocular solution more; steady-state P(vel) ~= vel_aid_std^2)
    this->declare_parameter<double>("radar.vision_lost_timeout", 1.5);
    // The radar reports the interceptor's OWN world velocity — the only drift-free correction
    // available for the IMU strapdown own_vel_ that damps the vertical loop.
    this->declare_parameter<bool>("radar.own_vel_aiding", true);
    this->declare_parameter<double>("radar.own_vel_tc", 5.0);
    // Yaw rotation from the radar world frame into the IMU-referenced estimator world frame.
    // Nonzero only when the FCU heading reference is offset from the radar ENU x-axis; the
    // handover bearing check measures exactly this angle and logs it.
    this->declare_parameter<double>("radar.yaw_offset_deg", 0.0);
    this->declare_parameter<double>("radar.handover_max_bearing_err", 0.20);
    // False-positive rejection. The detector fires on clouds/structures at long range and ONE
    // fused phantom pixel measurement is enough to drag the monocular width — and with it the
    // whole range scale — off. Every bbox is checked against the radar track before it reaches
    // the filter, not just before the handover.
    this->declare_parameter<bool>("radar.bbox_gate", true);
    this->declare_parameter<double>("radar.bbox_range_ratio", 2.0);
    this->declare_parameter<double>("radar.lock_min_duration", 0.7);
    this->declare_parameter<double>("radar.width_window", 1.5); // s, rolling window for the width median (survives transient misses)
    this->declare_parameter<double>("radar.width_min_bbox", 25.0); // px, only anchor the width from a bbox this large (a tiny long-range bbox is over-bound by the tracker -> inflated W). Shifts the lock closer where the bbox is accurate.
    this->declare_parameter<double>("radar.width_scale", 1.0);   // multiplicative correction for tracker over-bounding: locked_W *= scale. Calibrate as true_W / locked_W (e.g. 0.66 if it locks 2.26 for a 1.5 m target).
    // Re-intercept: on a terminal vision loss (a miss, NOT a clean hit), re-engage the radar
    // relay for another mid-course + terminal pass instead of brake-to-hold forever. Off by default.
    this->declare_parameter<bool>("radar.reintercept_enable", false);   // master switch
    this->declare_parameter<int>("radar.reintercept_max", 2);            // max re-intercept attempts (total terminal passes = 1 + this)
    this->declare_parameter<double>("radar.reintercept_kill_range", 1.0);// m; if the closest approach was <= this, treat as a clean hit (don't re-attempt). 0 = always re-attempt.
    this->declare_parameter<double>("radar.reintercept_min_range", 15.0);// m; re-engage only once radar shows the target back beyond this standoff (clean re-approach). 0 = immediate.
    // Post-lock width re-anchor (safety net for a badly locked W). If the locked W disagrees
    // with the still-tracking radar for a sustained window (r_vis = focal*W/bbox vs r_radar),
    // re-anchor W and rescale the estimator. Bounds: frame count + window span + cooldown +
    // width_reanchor_max total corrections. No-ops without a fresh radar track.
    this->declare_parameter<bool>("radar.width_reanchor_enable", true);
    this->declare_parameter<double>("radar.width_reanchor_min_range", 10.0);  // m, only trust the ratio beyond this radar range (metro noise dominates closer)
    this->declare_parameter<double>("radar.width_reanchor_max_range", 150.0); // m, only trust the ratio inside this radar range
    this->declare_parameter<double>("radar.width_reanchor_ratio", 1.25);      // fire when |median r_vis/r_radar| outside [1/x, x]
    this->declare_parameter<double>("radar.width_reanchor_window", 2.0);      // s, sustained-disagreement window
    this->declare_parameter<int>("radar.width_reanchor_min_frames", 15);      // frames inside the window (>=5)
    this->declare_parameter<int>("radar.width_reanchor_max", 2);              // max corrections per mission
}

void InterceptorNodeRelativeIMMUKF::load_parameters() {
    cam_width_ = this->get_parameter("camera.width_px").as_int();
    cam_height_ = this->get_parameter("camera.height_px").as_int();
    cam_fov_deg_ = this->get_parameter("camera.fov_deg").as_double();
    // Must resolve before the controller FOV halves and the estimator are built below/after.
    resolve_camera_intrinsics();
    const bool use_simulation = this->get_parameter("use_simulation").as_bool();
    use_simulation_ = use_simulation;   // gates gyro collision detect
    collision_min_bbox_px_ = this->get_parameter("collision_min_bbox_px").as_double();
    use_imu_axial_speed_ = this->get_parameter("controller.use_imu_axial_speed").as_bool();
    target_axial_feedforward_limit_ = this->get_parameter(
        "controller.target_axial_feedforward_limit").as_double();
    if (!std::isfinite(target_axial_feedforward_limit_) || target_axial_feedforward_limit_ < 0 ||
        (target_axial_feedforward_limit_ > 0 && !use_imu_axial_speed_))
        throw std::invalid_argument("target axial feedforward requires IMU axial speed and a nonnegative finite limit");
    capture_range_ = this->get_parameter("controller.capture_range").as_double();
    if (!std::isfinite(capture_range_) || capture_range_ <= 0)
        throw std::invalid_argument("controller.capture_range must be finite and positive");
    if (!std::isfinite(collision_min_bbox_px_) || collision_min_bbox_px_ < 0.0)
        throw std::invalid_argument("collision_min_bbox_px must be finite and nonnegative");
    // FLU(body)→RDF(cam) standard extrinsic — a frame convention, not a tuning param.
    R_b2c_base_ << 0.0, -1.0, 0.0,
                   0.0,  0.0, -1.0,
                   1.0,  0.0,  0.0;
    g_ = this->get_parameter("physics.g").as_double();
    use_gimbal_ = this->get_parameter("camera.use_gimbal").as_bool();
    gimbal_bias_rad_ = this->get_parameter("camera.gimbal_bias_deg").as_double() * M_PI / 180.0;
    // gimbal_bias_deg (positive = down-gaze) is the unified camera tilt for BOTH the gimbal
    // (world-stabilized) and the fixed mount (body-relative). The gimbal rebuilds R_b2c_ from
    // q_C_W each frame; the fixed mount keeps this initial value.
    R_b2c_ = gimbal_R_b2c(gimbal_bias_rad_);

    w_guess_ = this->get_parameter("estimation.w_guess").as_double();
    control_rate_ = this->get_parameter("control.rate").as_double();
    const double rate_tc = this->get_parameter("control.rate_tc").as_double();
    const bool rate_attenuation = this->get_parameter("control.rate_control_attenuation").as_bool();
    // First-order low-pass coefficient on body-rate output.
    // EMA form: out_new = out_old + ctrl_coeff_ * (raw - out_old). Attenuates high-frequency
    // cmd jumps from estimator/guidance noise before they reach the attitude controller.
    if (!use_simulation && rate_attenuation && rate_tc > 1e-6 && control_rate_ > 1e-6) {
        ctrl_coeff_ = 1.0 - std::exp(-1.0 / (control_rate_ * rate_tc));
    } else {
        ctrl_coeff_ = 1.0;
    }
    double N = this->get_parameter("controller.N").as_double();
    double K_att = this->get_parameter("controller.K_att").as_double();
    double K_yaw = this->get_parameter("controller.K_yaw").as_double();
    double v_des = this->get_parameter("controller.v_des").as_double();
    double max_tilt = this->get_parameter("physics.max_tilt_deg").as_double();
    double fov_keep = this->get_parameter("controller.fov_keep_margin").as_double();
    double jerk = this->get_parameter("controller.axial_jerk_max").as_double();
    double max_body_rate = this->get_parameter("physics.max_body_rate").as_double();
    double terminal_brake_range = this->get_parameter("controller.terminal_brake_range").as_double();
    double v_pn_ref = this->get_parameter("controller.v_pn_ref").as_double();
    double max_att_rate = this->get_parameter("controller.max_att_rate").as_double();
    double att_ref_tc = this->get_parameter("controller.att_ref_tc").as_double();
    double max_tilt_terminal_deg = this->get_parameter("physics.max_tilt_terminal_deg").as_double();
    double terminal_tilt_range = this->get_parameter("controller.terminal_tilt_range").as_double();
    // tan(half-FOV) for the controller's keep-in-view clamp, from the RESOLVED intrinsics.
    // With an off-center principal point the FOV is asymmetric; the nearer edge binds first
    // when keeping the target inside the image, so use the smaller extent per axis. (Equals
    // the centered (W/2)/fx in the fov_deg-derived case.)
    double fov_half_v = std::min(cam_cy_, cam_height_ - cam_cy_) / std::max(1.0, cam_fy_);
    double fov_half_h = std::min(cam_cx_, cam_width_ - cam_cx_) / std::max(1.0, cam_fx_);

    const bool enable_fov_clamp = this->get_parameter("controller.enable_fov_clamp").as_bool();
    controller_ = std::make_unique<AdvancedHybridController>(
        N, K_att, K_yaw, v_des, max_tilt,
        fov_half_v, fov_half_h, fov_keep, jerk, enable_fov_clamp, gimbal_bias_rad_,
        max_body_rate, terminal_brake_range,
        v_pn_ref, max_att_rate, att_ref_tc,
        max_tilt_terminal_deg, terminal_tilt_range);
    controller_->set_a_z_slew(this->get_parameter("controller.a_z_slew").as_double());
    controller_->set_lateral_decay_floor(this->get_parameter("controller.lateral_decay_floor").as_double());
    controller_->set_lat_decay_start(this->get_parameter("controller.lateral_decay_start").as_double());
    controller_->set_gimbal(use_gimbal_, gimbal_bias_rad_);
    controller_->set_g(g_);
    controller_->set_vertical_climb(
        this->get_parameter("physics.v_climb").as_double(),
        this->get_parameter("physics.v_descent").as_double(),
        this->get_parameter("controller.K_dz_up").as_double(),
        this->get_parameter("controller.K_dz_dn").as_double());
    controller_->set_guidance_tuning(
        this->get_parameter("controller.K_fwd").as_double(),
        this->get_parameter("controller.K_az").as_double(),
        this->get_parameter("physics.a_z_limit").as_double(),
        this->get_parameter("controller.dither_amp").as_double(),
        this->get_parameter("controller.dither_freq").as_double(),
        this->get_parameter("controller.dither_min_range").as_double(),
        this->get_parameter("controller.dither_max_bbox").as_double());
    controller_->set_range_smooth_tc(this->get_parameter("controller.range_smooth_tc").as_double());
    controller_->set_apn_gain(this->get_parameter("controller.K_apn").as_double());
    controller_->set_vertical_braking(
        this->get_parameter("controller.vertical_brake_accel").as_double(),
        this->get_parameter("controller.vertical_brake_delay").as_double());
    controller_->set_visual_speed_gain(
        this->get_parameter("controller.visual_speed_gain").as_double(),
        this->get_parameter("controller.visual_speed_floor").as_double());
    controller_->set_terminal_speed_floor(
        this->get_parameter("controller.terminal_speed_floor").as_double(),
        this->get_parameter("controller.terminal_speed_bbox_threshold").as_double(),
        this->get_parameter("controller.terminal_speed_los_rate_threshold").as_double());
    {
        const double trcf = this->get_parameter("controller.turn_radius_cap_factor").as_double();
        if (trcf < 0.3 - 1e-9 || trcf > 1.0 + 1e-9) {
            RCLCPP_WARN(this->get_logger(),
                "controller.turn_radius_cap_factor=%.3f is outside [0.3, 1.0] -> clamped. "
                "The cap is the binding terminal-speed reducer; >1.0 would disable it.", trcf);
        }
        controller_->set_turn_radius_cap_factor(trcf);
    }
    controller_->set_axial_deadband(this->get_parameter("controller.axial_deadband").as_double());
    controller_->set_motion_gate_factor(this->get_parameter("radar.motion_gate_factor").as_double());
    dither_radar_range_ = std::max(0.0, this->get_parameter("controller.dither_radar_range").as_double());
    vel_smooth_tc_ = this->get_parameter("estimation.vel_smooth_tc").as_double();
    gyro_attitude_during_mission_ = this->get_parameter("estimation.gyro_attitude_during_mission").as_bool();
    const std::vector<double> bias = this->get_parameter("imu_bias").as_double_array();
    if (bias.size() == 3) {
        imu_bias_ = Eigen::Vector3d(bias[0], bias[1], bias[2]);
    }

    radar_enable_ = this->get_parameter("radar.enable").as_bool();
    radar_topic_target_ = this->get_parameter("radar.topic_target").as_string();
    radar_topic_own_ = this->get_parameter("radar.topic_interceptor").as_string();
    radar_pos_std_ = std::max(0.1, this->get_parameter("radar.pos_std").as_double());
    radar_vel_std_ = std::max(0.05, this->get_parameter("radar.vel_std").as_double());
    radar_pair_window_ = this->get_parameter("radar.pair_window").as_double();
    radar_timeout_ = this->get_parameter("radar.timeout").as_double();
    radar_max_delay_comp_ = this->get_parameter("radar.max_delay_comp").as_double();
    radar_max_range_ = this->get_parameter("radar.max_range").as_double();
    handover_frames_ = std::max(1, static_cast<int>(this->get_parameter("radar.handover_frames").as_int()));
    handover_max_range_ = this->get_parameter("radar.handover_max_range").as_double();
    width_min_ = this->get_parameter("radar.width_min").as_double();
    width_max_ = this->get_parameter("radar.width_max").as_double();
    radar_vel_aid_after_lock_ = this->get_parameter("radar.vel_aid_after_lock").as_bool();
    radar_vel_aid_std_ = std::max(0.5, this->get_parameter("radar.vel_aid_std").as_double());
    vision_lost_timeout_ = this->get_parameter("radar.vision_lost_timeout").as_double();
    radar_own_vel_aiding_ = this->get_parameter("radar.own_vel_aiding").as_bool();
    radar_own_vel_tc_ = std::max(0.05, this->get_parameter("radar.own_vel_tc").as_double());
    handover_max_bearing_err_ = this->get_parameter("radar.handover_max_bearing_err").as_double();
    radar_bbox_gate_ = this->get_parameter("radar.bbox_gate").as_bool();
    radar_bbox_range_ratio_ = std::max(1.1, this->get_parameter("radar.bbox_range_ratio").as_double());
    lock_min_duration_ = this->get_parameter("radar.lock_min_duration").as_double();
    width_window_s_ = std::max(0.1, this->get_parameter("radar.width_window").as_double());
    width_min_bbox_ = std::max(1.0, this->get_parameter("radar.width_min_bbox").as_double());
    width_scale_ = this->get_parameter("radar.width_scale").as_double();
    reintercept_enable_ = this->get_parameter("radar.reintercept_enable").as_bool();
    reintercept_max_ = std::max(0, static_cast<int>(this->get_parameter("radar.reintercept_max").as_int()));
    reintercept_kill_range_ = std::max(0.0, this->get_parameter("radar.reintercept_kill_range").as_double());
    reintercept_min_range_ = std::max(0.0, this->get_parameter("radar.reintercept_min_range").as_double());
    width_reanchor_enable_ = this->get_parameter("radar.width_reanchor_enable").as_bool();
    width_reanchor_min_range_ = std::max(1.0, this->get_parameter("radar.width_reanchor_min_range").as_double());
    width_reanchor_max_range_ = std::max(width_reanchor_min_range_ + 10.0,
                                         this->get_parameter("radar.width_reanchor_max_range").as_double());
    width_reanchor_ratio_ = clamp(this->get_parameter("radar.width_reanchor_ratio").as_double(), 1.05, 3.0);
    width_reanchor_window_ = std::max(0.5, this->get_parameter("radar.width_reanchor_window").as_double());
    width_reanchor_min_frames_ = std::max(5, static_cast<int>(this->get_parameter("radar.width_reanchor_min_frames").as_int()));
    width_reanchor_max_ = std::max(0, static_cast<int>(this->get_parameter("radar.width_reanchor_max").as_int()));
    const double radar_yaw_off = this->get_parameter("radar.yaw_offset_deg").as_double() * M_PI / 180.0;
    radar_R_yaw_ = Eigen::AngleAxisd(radar_yaw_off, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    vision_locked_ = !radar_enable_;
}

// Resolve the pinhole intrinsics (no distortion model). camera.intrinsics — the 9-element
// row-major K matrix [fx 0 cx; 0 fy cy; 0 0 1] — is used directly and takes precedence; a
// nonzero skew entry is ignored with a warning. Otherwise a valid camera.fov_deg (horizontal,
// in (0,180)) keeps the legacy derivation fx = fy = (W/2)/tan(fov/2) with the principal point
// at the image center. Neither source valid is a hard error: every pixel<->metric conversion
// downstream (LOS bearings, monocular range r = f*W/s) rides on the focal, so a silently
// guessed intrinsics would corrupt guidance instead of failing fast.
void InterceptorNodeRelativeIMMUKF::resolve_camera_intrinsics() {
    if (cam_width_ <= 0 || cam_height_ <= 0) {
        throw std::invalid_argument("camera.width_px and camera.height_px must be positive");
    }

    const std::vector<double> K = this->get_parameter("camera.intrinsics").as_double_array();
    const char* source = "fov_deg";
    if (!K.empty()) {
        if (K.size() != 9) {
            throw std::invalid_argument(
                "camera.intrinsics must be the 9-element row-major K matrix [fx 0 cx; 0 fy cy; 0 0 1]");
        }
        if (std::abs(K[1]) > 1e-9) {
            RCLCPP_WARN(this->get_logger(),
                "camera.intrinsics skew K(0,1)=%.3f ignored (skew-free pinhole assumed)", K[1]);
        }
        if (std::abs(K[6]) > 1e-9 || std::abs(K[7]) > 1e-9 || std::abs(K[8] - 1.0) > 1e-9) {
            RCLCPP_WARN(this->get_logger(),
                "camera.intrinsics last row must be [0 0 1]; got [%.3f %.3f %.3f] (ignored)",
                K[6], K[7], K[8]);
        }
        cam_fx_ = K[0];
        cam_fy_ = K[4];
        cam_cx_ = K[2];
        cam_cy_ = K[5];
        if (!std::isfinite(cam_fx_) || !std::isfinite(cam_fy_) || cam_fx_ <= 0.0 || cam_fy_ <= 0.0) {
            throw std::invalid_argument("camera.intrinsics fx/fy must be finite and positive");
        }
        if (!std::isfinite(cam_cx_) || !std::isfinite(cam_cy_) ||
            cam_cx_ <= 0.0 || cam_cx_ >= cam_width_ || cam_cy_ <= 0.0 || cam_cy_ >= cam_height_) {
            throw std::invalid_argument("camera.intrinsics cx/cy must be finite and inside the image");
        }
        source = "intrinsics matrix";
        if (cam_fov_deg_ > 0.0 && cam_fov_deg_ < 180.0) {
            RCLCPP_INFO(this->get_logger(),
                "camera.intrinsics set -> overriding camera.fov_deg=%.1f", cam_fov_deg_);
        }
    } else if (std::isfinite(cam_fov_deg_) && cam_fov_deg_ > 0.0 && cam_fov_deg_ < 180.0) {
        cam_fx_ = cam_fy_ = (cam_width_ / 2.0) / std::tan(cam_fov_deg_ * M_PI / 360.0);
        cam_cx_ = cam_width_ / 2.0;
        cam_cy_ = cam_height_ / 2.0;
    } else {
        throw std::invalid_argument(
            "camera intrinsics unresolved: set camera.fov_deg to a horizontal FOV in (0,180) or "
            "camera.intrinsics to the 9-element row-major K matrix [fx 0 cx; 0 fy cy; 0 0 1]");
    }
    RCLCPP_INFO(this->get_logger(),
        "Camera intrinsics (%s): fx=%.1f fy=%.1f cx=%.1f cy=%.1f (%dx%d)",
        source, cam_fx_, cam_fy_, cam_cx_, cam_cy_, cam_width_, cam_height_);
}

void InterceptorNodeRelativeIMMUKF::reset_estimator() {
    estimator_ = std::make_unique<RelativeIMMUKFEstimator>(dt_cam_, cam_fx_, cam_fy_, w_guess_, R_b2c_);
    if (radar_enable_) estimator_->max_range_ = radar_max_range_;
}

void InterceptorNodeRelativeIMMUKF::reset_control_cache() {
    std::lock_guard<std::mutex> cache_lock(control_cache_mtx_);
    control_cache_.valid = false;
}

void InterceptorNodeRelativeIMMUKF::start_control_timer() {
    if (control_timer_) return;
    const auto period = std::chrono::duration<double>(1.0 / control_rate_);
    control_timer_ = this->create_wall_timer(
        period, std::bind(&InterceptorNodeRelativeIMMUKF::control_timer_callback, this), timer_callback_group_);
}

Eigen::Vector3d InterceptorNodeRelativeIMMUKF::smooth_rel_vel(const Eigen::Vector3d& v_raw, double t_now) {
    if (vel_smooth_tc_ <= 1e-6) return v_raw;
    if (has_target_vel_smoothed_ && t_now > last_vel_smooth_t_ &&
        (t_now - last_vel_smooth_t_) < 1.0) {
        const double a = clamp((t_now - last_vel_smooth_t_) / vel_smooth_tc_, 0.0, 1.0);
        target_vel_smoothed_ += a * (v_raw - target_vel_smoothed_);
    } else {
        target_vel_smoothed_ = v_raw;
    }
    last_vel_smooth_t_ = t_now;
    has_target_vel_smoothed_ = true;
    return target_vel_smoothed_;
}

bool InterceptorNodeRelativeIMMUKF::vision_delivering(double now) const {
    return last_vision_time_ > 0.0 && (now - last_vision_time_) < vision_lost_timeout_;
}

void InterceptorNodeRelativeIMMUKF::radar_callback(const nav_msgs::msg::Odometry::SharedPtr msg, bool is_target) {
    RadarSample s;
    s.stamp = rclcpp::Time(msg->header.stamp).seconds();
    s.pos = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y,
                            msg->pose.pose.position.z);
    s.vel = Eigen::Vector3d(msg->twist.twist.linear.x, msg->twist.twist.linear.y,
                            msg->twist.twist.linear.z);
    if (!s.pos.allFinite() || !s.vel.allFinite()) return;
    s.valid = true;
    {
        std::lock_guard<std::mutex> lock(radar_mtx_);
        if (is_target) { radar_target_ = s; ++radar_msg_target_; }
        else { radar_own_ = s; ++radar_msg_own_; }
    }
    fuse_radar();
}

void InterceptorNodeRelativeIMMUKF::fuse_radar() {
    if (!mission_state_.is_active()) return;

    RadarSample tgt, own;
    {
        std::lock_guard<std::mutex> lock(radar_mtx_);
        tgt = radar_target_;
        own = radar_own_;
    }
    if (!tgt.valid || !own.valid) return;
    if (std::abs(tgt.stamp - own.stamp) > radar_pair_window_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Radar pair rejected: target/interceptor stamps %.3fs apart (radar.pair_window=%.3f)",
            std::abs(tgt.stamp - own.stamp), radar_pair_window_);
        return;
    }

    const double t_meas = std::min(tgt.stamp, own.stamp);
    const double now = this->get_clock()->now().seconds();

    std::lock_guard<std::mutex> lock(estimator_mtx_);
    if (t_meas <= radar_last_fused_t_ + 1e-6) return;   // this pair was already consumed
    radar_last_fused_t_ = t_meas;

    // Latency compensation: extrapolate the relative position from the measurement epoch to
    // `now` along the measured relative velocity. Both objects are sampled at the same instant,
    // so the ownship leg of the extrapolation is carried by v_meas as well.
    // radar_R_yaw_ rotates the radar world frame onto the IMU-referenced estimator frame; the
    // guidance mixes the two (los_az = atan2(r) - imu_yaw), so a yaw offset here flies the
    // interceptor at a constant angle off the target.
    const double lag = clamp(now - t_meas, 0.0, radar_max_delay_comp_);
    // Online yaw correction: rotate the radar measurement by the EMA of the camera-radar az diff
    // (bearing_err_ema_). This auto-calibrates the systematic frame misalignment that
    // radar.yaw_offset_deg is meant to correct — no manual calibration needed. Applied here
    // (before the UKF) so a wrong correction makes the gate TIGHTER (conservative, self-correcting)
    // rather than looser (divergent, as a gate-side correction would be).
    Eigen::Matrix3d R_online_yaw = Eigen::Matrix3d::Identity();
    if (has_bearing_err_ && std::abs(bearing_err_ema_) < 0.5) {  // sanity cap: < 28.6°
        R_online_yaw = Eigen::AngleAxisd(bearing_err_ema_, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    }
    const Eigen::Vector3d v_meas = R_online_yaw * radar_R_yaw_ * (tgt.vel - own.vel);
    const Eigen::Vector3d r_meas = R_online_yaw * radar_R_yaw_ * (tgt.pos - own.pos) + v_meas * lag;

    radar_rel_pos_ = r_meas;
    radar_rel_time_ = now;
    radar_has_rel_ = true;

    // Ownship velocity aiding. own_vel_ is otherwise an unbounded IMU integral running since the
    // MISSION trigger, and it damps the vertical loop — a long wait for the FCU to accept
    // external control is enough to poison it. This is a slow DRIFT TRIM, not a replacement:
    // the vertical loop needs own_vel_'s high-frequency content, and blending in a 10 Hz,
    // lag-delayed radar velocity at a short time constant adds enough phase lag to make that
    // loop oscillate (throttle bang-bangs between its clamps and takes the whole attitude loop
    // with it). Keep own_vel_tc long compared with the vertical-loop time constant.
    if (radar_own_vel_aiding_) {
        const Eigen::Vector3d v_own_radar = radar_R_yaw_ * own.vel;
        const double dt_aid = (radar_own_vel_t_ > 0.0)
            ? clamp(now - radar_own_vel_t_, 0.001, 1.0) : radar_own_vel_tc_;
        radar_own_vel_t_ = now;
        std::lock_guard<std::mutex> ol(odom_mtx_);
        if (!own_vel_aided_) {
            own_vel_ = v_own_radar;   // strapdown has no absolute anchor: hard-seed the first fix
            own_vel_aided_ = true;
        } else {
            own_vel_ += clamp(dt_aid / radar_own_vel_tc_, 0.0, 1.0) * (v_own_radar - own_vel_);
        }
    }

    // Drain the IMU preintegration window for this radar epoch (shared by both branches).
    Eigen::Vector3d dv_own = Eigen::Vector3d::Zero();
    {
        std::lock_guard<std::mutex> ol(odom_mtx_);
        if (last_imu_time_ < 0.0 || !yaw_align_init_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Radar track received but IMU not ready yet");
            return;
        }
        dv_own = dv_own_accum_;
        dv_own_accum_.setZero();
        last_dv_own_ = dv_own;
    }

    // Terminal vision is PURE vision: once vision is locked the radar node must not influence
    // the estimator/guidance at all (the radar is for MID-COURSE only). So while vision_locked_,
    // fully disengage — no radar fusion (the radar's lagged 10 Hz vel fought the 30 Hz vision
    // update and HURT in AirSim, and its position's metro noise corrupts the monocular range).
    // The radar re-engages full pos+vel only once vision is DEFINITIVELY lost (vision_locked_ ->
    // false via vision_lost_timeout), restarting mid-course relay. vel_aid_after_lock is an
    // opt-in experiment (off by default): fuse radar VELOCITY only to anchor v_closing.
    if (vision_locked_) {
        if (radar_vel_aid_after_lock_) {
            estimator_->process_radar_vel(v_meas, now, dv_own, radar_vel_aid_std_);
        }
        return;   // vision owns the guidance cache while locked
    }

    const bool was_initialized = estimator_->initialized;
    // Window the IMU accumulator actually covers — the interval since the previous estimator
    // step, whichever source produced it. Needed for a_own below: dividing by the camera dt
    // would treble it whenever radar is the only measurement source.
    const double dt_step = estimator_->has_last_cap_time
        ? clamp(now - estimator_->last_cap_time, 0.001, 0.3) : dt_cam_;
    const auto result = estimator_->process_radar(r_meas, v_meas, now, dv_own,
                                                  radar_pos_std_, radar_vel_std_);
    if (!result.first) return;
    if (!was_initialized) {
        transition_to_state(MissionState::BLIND_FLIGHT, "Radar relay acquired target");
        RCLCPP_INFO(this->get_logger(),
            "Radar handover start: r=[%.1f %.1f %.1f] |r|=%.1fm |v_rel|=%.1fm/s lag=%.0fms",
            r_meas(0), r_meas(1), r_meas(2), r_meas.norm(), v_meas.norm(), lag * 1000.0);
    }

    // Radar owns the guidance cache only when vision is NOT delivering a consistent track
    // (mid-course before visual acquisition, or after a definitive vision loss). When vision is
    // delivering the vision path owns the cache — the radar's only role then is the one-time
    // width anchor (block in update_vision_handover), not guidance.
    if (!vision_delivering(now)) {
        fill_cache_from_radar(result.second, now, dt_step);
    }

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS * 5,
        "[RADAR] meas=[%.1f %.1f %.1f] est=[%.1f %.1f %.1f] |r|=%.1f lag=%.0fms",
        r_meas(0), r_meas(1), r_meas(2),
        result.second(0), result.second(1), result.second(2),
        result.second.segment<3>(0).norm(), lag * 1000.0);
}

void InterceptorNodeRelativeIMMUKF::fill_cache_from_radar(const Eigen::VectorXd& state, double t_now, double dt_step) {
    const Eigen::Vector3d r_est = state.segment<3>(0);
    const Eigen::Vector3d v_est = smooth_rel_vel(state.segment<3>(3), t_now);

    const double los_az_world = std::atan2(r_est(1), r_est(0));
    const double los_el = std::asin(clamp(r_est(2) / std::max(r_est.norm(), 0.1), -1.0, 1.0));
    const double r_xy2 = std::max(r_est(0) * r_est(0) + r_est(1) * r_est(1), 1.0);
    const double raw_rate = (r_est(0) * v_est(1) - r_est(1) * v_est(0)) / r_xy2;
    constexpr double alpha_radar_los = 0.3;
    los_rate_ema_ = has_prev_los_
        ? (1.0 - alpha_radar_los) * los_rate_ema_ + alpha_radar_los * raw_rate
        : raw_rate;
    prev_los_az_world_ = los_az_world;
    prev_los_t_ = t_now;
    has_prev_los_ = true;

    ControlCacheData cache;
    cache.valid = true;
    cache.has_bbox = false;
    cache.r_est = r_est;
    cache.v_est = v_est;
    cache.los_az_world = los_az_world;
    cache.los_el = los_el;
    cache.los_rate = los_rate_ema_;
    cache.source_time = t_now;
    cache.update_time = this->get_clock()->now().seconds();
    cache.bbox_size = 0.0;
    // Transverse (escape) speed — see the vision path. (Not |v_rel|, which is the closing speed
    // and would defeat the brake floor during mid-course closure.)
    {
        const double r_xy = std::sqrt(r_est(0) * r_est(0) + r_est(1) * r_est(1));
        const double v_transverse = (r_xy > 1e-3)
            ? std::abs(r_est(0) * v_est(1) - r_est(1) * v_est(0)) / r_xy : 0.0;
        cache.target_speed = std::min(v_transverse, 15.0);
    }
    cache.a_own = last_dv_own_ / std::max(dt_step, 1e-3);
    {
        std::lock_guard<std::mutex> cache_lock(control_cache_mtx_);
        control_cache_ = cache;
    }
}

bool InterceptorNodeRelativeIMMUKF::bbox_matches_radar(const Eigen::Vector3d& los_nav, double bbox_s, double* d_az_out) {
    if (!radar_bbox_gate_) return true;
    const bool radar_fresh = radar_has_rel_ &&
        (this->get_clock()->now().seconds() - radar_rel_time_) < radar_timeout_;
    const double r_radar = radar_rel_pos_.norm();
    if (!radar_fresh || r_radar < 1.0 || los_nav.norm() < 0.5) return true;

    const Eigen::Vector3d u_radar = radar_rel_pos_ / r_radar;
    const double az_cam = std::atan2(los_nav(1), los_nav(0));
    const double az_radar = std::atan2(u_radar(1), u_radar(0));
    if (d_az_out) {
        *d_az_out = std::atan2(std::sin(az_cam - az_radar), std::cos(az_cam - az_radar));
    }

    const double d_ang = std::acos(clamp(los_nav.normalized().dot(u_radar), -1.0, 1.0));
    // Range- and motion-dependent bearing tolerance. At close range with a fast-crossing target,
    // the residual radar delay introduces a bearing error ~ v_perp * lag / r. Scale by a safety
    // factor and add to the base tolerance. This also accommodates the radar yaw offset + noise
    // (the base is widened to cover the ~6° steady offset the [FRAME] log reports). Together they
    // break the reject->no-fusion->drift->reject feedback loop that grows the error to 35°.
    double v_perp_gate = 0.0;
    if (estimator_ && estimator_->initialized) {
        const auto& x = estimator_->imm_ukf->x;
        const double r_xy = std::sqrt(x(0)*x(0) + x(1)*x(1));
        v_perp_gate = (r_xy > 1e-3)
            ? std::abs(x(0)*x(4) - x(1)*x(3)) / r_xy : 0.0;   // |r_xy × v_xy| / |r_xy|
    }
    const double motion_tol = controller_->motion_gate_factor_ * v_perp_gate *
        radar_max_delay_comp_ / std::max(r_radar, 1.0);
    // pos_noise_tol DISABLED: the yaw auto-correction above removes the systematic 6° offset,
    // leaving only ~5-7° random noise — well within the 11.5° base gate. A range-dependent
    // pos-noise term was tried but it over-widened at close range and let false positives through.
    const double pos_noise_tol = 0.0;
    const double bearing_limit = handover_max_bearing_err_ + motion_tol + pos_noise_tol;
    if (d_ang > bearing_limit) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "[BBOX-REJECT] LOS %.1fdeg off (limit %.1f+%.1f+%.1f=%.1f, r=%.0f v_perp=%.1f bbox=%.0fpx)",
            d_ang * 180.0 / M_PI, handover_max_bearing_err_ * 180.0 / M_PI,
            motion_tol * 180.0 / M_PI, pos_noise_tol * 180.0 / M_PI,
            bearing_limit * 180.0 / M_PI, r_radar, v_perp_gate, bbox_s);
        return false;
    }
    if (width_locked_ && bbox_s > 1.0) {
        const double r_vis = cam_fx_ * estimator_->imm_ukf->x(9) / bbox_s;
        const double ratio = r_vis / r_radar;
        if (ratio > radar_bbox_range_ratio_ || ratio < 1.0 / radar_bbox_range_ratio_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                "[BBOX-REJECT] implied range %.0fm vs radar %.0fm, bbox=%.0fpx",
                r_vis, r_radar, bbox_s);
            return false;
        }
    }
    return true;
}

void InterceptorNodeRelativeIMMUKF::update_vision_handover(bool bbox_ok, double bbox_s, double t_now, double d_az) {
    if (!bbox_ok) {
        vision_frames_ = 0;   // the vision LOCK still requires an unbroken run (false-positive guard)
        vision_lock_t0_ = -1.0;
        // The width buffer is NOT cleared here: a single transient missed frame (radar stall,
        // one rejected bbox) no longer throws away the accumulated width samples. The rolling
        // window below ages them out instead, so the width lock survives brief gaps.
        if (vision_locked_ && (t_now - last_vision_time_) > vision_lost_timeout_) {
            vision_locked_ = false;
            RCLCPP_WARN(this->get_logger(),
                "Vision gone %.1fs — radar relay owns guidance again",
                t_now - last_vision_time_);
        }
        return;
    }
    last_vision_time_ = t_now;

    const bool radar_fresh = radar_has_rel_ &&
        (this->get_clock()->now().seconds() - radar_rel_time_) < radar_timeout_;

    // Track the SYSTEMATIC yaw offset (constant frame misalignment). Subtract the expected
    // motion-induced az diff (v_perp * delay / r) from d_az before updating the EMA — this
    // removes the target-motion component so the EMA converges to the constant offset only.
    // The EMA is then applied in fuse_radar as an online yaw correction (rotating the radar
    // measurement to match the camera frame), auto-calibrating radar.yaw_offset_deg.
    if (radar_fresh && std::isfinite(d_az)) {
        double d_az_sys = d_az;
        if (estimator_ && estimator_->initialized) {
            const auto& xs = estimator_->imm_ukf->x;
            const double rxy = std::sqrt(xs(0)*xs(0) + xs(1)*xs(1));
            if (rxy > 1e-3) {
                const double vp = std::abs(xs(0)*xs(4) - xs(1)*xs(3)) / rxy;
                const double motion_az = vp * radar_max_delay_comp_ / rxy;
                d_az_sys = d_az - std::copysign(motion_az, d_az);
            }
        }
        bearing_err_ema_ = has_bearing_err_ ? (0.92 * bearing_err_ema_ + 0.08 * d_az_sys) : d_az_sys;
        has_bearing_err_ = true;
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "[FRAME] cam-radar az diff=%.1f deg (motion-corrected %.1f, yaw_ema %.1f)",
            d_az * 180.0 / M_PI, d_az_sys * 180.0 / M_PI, bearing_err_ema_ * 180.0 / M_PI);
    }
    if (!width_locked_ && radar_fresh && bbox_s >= width_min_bbox_) {
        // Anchor W on the RADAR relative range, NOT the estimator blend. Before the width locks,
        // the camera size channel is scale-degenerate (s = focal*W/z), so the blend — although
        // radar-fused — can sit anywhere along the W/r hyperbola that its pixel updates dragged
        // it to (observed: blend 19.7 m while the radar measured ~56 m -> W locked 0.58 m for a
        // 1.5 m target; the error is permanent once locked). radar_rel_pos_ is the raw
        // delay-compensated radar pair — unbiased; its per-sample noise (sigma ~4 m) is removed
        // by the median/MAD window below, which is exactly what it is for.
        const double r_anchor = radar_rel_pos_.norm();
        if (r_anchor > 1.0 && r_anchor < handover_max_range_) {
            const double w_meas = clamp(bbox_s * r_anchor / cam_fx_, width_min_, width_max_);
            width_samples_.push_back({w_meas, t_now});
            // Rolling window: age out stale samples so a transient missed frame (which no longer
            // wipes the buffer) does not leave stale data, and the median tracks a moving range.
            width_samples_.erase(
                std::remove_if(width_samples_.begin(), width_samples_.end(),
                               [&](const WidthSample& s){ return (t_now - s.t) > width_window_s_; }),
                width_samples_.end());
            if (estimator_) estimator_->w_guess = w_meas;   // helps a monocular-only re-init
            if (static_cast<int>(width_samples_.size()) >= handover_frames_) {
                // MAD outlier rejection: a phantom bbox that passes the bearing gate but has a wild
                // size is rejected instead of contaminating the median (observed w_est 1.5 -> 0.66).
                std::vector<double> ws;
                for (const auto& s : width_samples_) ws.push_back(s.w);
                std::sort(ws.begin(), ws.end());
                const double med = ws[ws.size() / 2];
                double mad = 0.0;
                for (double w : ws) mad += std::abs(w - med);
                mad /= std::max((size_t)1, ws.size());
                constexpr double kappa = 3.0;
                std::vector<double> kept;
                for (const auto& s : width_samples_) {
                    if (mad < 1e-6 || std::abs(s.w - med) <= kappa * 1.4826 * mad) kept.push_back(s.w);
                }
                if (!kept.empty()) std::sort(kept.begin(), kept.end());
                const double w_med = kept.empty() ? med : kept[kept.size() / 2];
                // Apply the detector over-bounding correction (width_scale) and clamp. A visual
                // tracker systematically over-bounds small targets (observed W~2.26 vs ~1.5 at
                // 186 m / 16 px); width_scale = true_W / locked_W calibrates that multiplicative
                // bias. The width_min_bbox gate above already shifted the anchor closer (larger,
                // more accurate bboxes) which shrinks any additive padding bias.
                const double w_fused = clamp(width_scale_ * w_med, width_min_, width_max_);
                if (estimator_) estimator_->lock_width(w_fused);
                width_locked_ = true;
                RCLCPP_INFO(this->get_logger(),
                    "[HANDOVER] Target width fixed from radar range: W=%.2fm "
                    "(median %.2f, scale %.3f, kept %zu/%zu, r_radar=%.1fm bbox=%.1fpx fx=%.0f)",
                    w_fused, w_med, width_scale_, kept.size(), ws.size(), r_anchor, bbox_s, cam_fx_);
            }
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Width fix skipped: radar range %.0fm outside (1, %.0f]m",
                r_anchor, handover_max_range_);
        }
    }

    if (!vision_locked_) {
        if (vision_frames_ == 0) vision_lock_t0_ = t_now;
        ++vision_frames_;
        // Frame count AND elapsed time: a burst of consecutive false positives can clear a pure
        // frame count in a fraction of a second, which is how a phantom took the terminal phase.
        const bool run_ok = vision_frames_ >= handover_frames_ &&
                            (t_now - vision_lock_t0_) >= lock_min_duration_;
        // Handing the endgame to vision before the radar has fixed the width would run the
        // terminal range r = focal*W/s on the configured GUESS, and nothing would pin W
        // afterwards — it drifts with every update. Only skip this when there is no radar at all.
        const bool width_ready = width_locked_ || !radar_has_rel_;
        if (run_ok && width_ready) {
            vision_locked_ = true;
            had_terminal_attempt_ = true;   // a terminal phase began — gates re-intercept on a later vision loss
            RCLCPP_INFO(this->get_logger(),
                "[HANDOVER] Visual lock: %d validated frames over %.1fs, W=%.2fm%s "
                "— terminal guidance on vision, radar standing by",
                vision_frames_, t_now - vision_lock_t0_,
                estimator_ ? estimator_->imm_ukf->x(9) : 0.0,
                width_locked_ ? " (radar-fixed)" : " (guess, no radar)");
            transition_to_state(MissionState::TRACKING, "Vision lock acquired");
        } else if (run_ok) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Visual run stable (%d frames); width not radar-fixed yet (vision guidance "
                "already active on w_guess, W anchor pending)", vision_frames_);
        }
    }
}

void InterceptorNodeRelativeIMMUKF::update_width_reanchor(bool bbox_ok, double bbox_s, double t_now) {
    if (!width_reanchor_enable_ || width_reanchor_max_ <= 0) return;
    const bool radar_fresh = radar_has_rel_ &&
        (this->get_clock()->now().seconds() - radar_rel_time_) < radar_timeout_;
    const double r_radar = radar_rel_pos_.norm();
    const bool gates = vision_locked_ && width_locked_ && radar_fresh && bbox_ok &&
                       bbox_s >= width_min_bbox_ &&
                       width_reanchor_count_ < width_reanchor_max_ &&
                       t_now >= width_reanchor_cooldown_until_;
    if (!gates || r_radar < width_reanchor_min_range_ || r_radar > width_reanchor_max_range_) {
        width_reanchor_samples_.clear();   // any broken gate resets the window
        return;
    }
    const double w_locked = estimator_ ? estimator_->imm_ukf->x(9) : 0.0;
    if (w_locked < 0.05 || bbox_s <= 1.0) return;
    const double r_vis = cam_fx_ * w_locked / bbox_s;   // vision-implied range
    const double ratio = r_vis / r_radar;                     // >1: W inflated (range over-estimated)
    // Radar glitch / bbox outlier: DROP the sample but keep the window (a single wild ratio
    // must neither trigger nor starve the detector).
    if (!(ratio > 0.2 && ratio < 5.0)) return;
    width_reanchor_samples_.push_back({ratio, t_now});
    width_reanchor_samples_.erase(
        std::remove_if(width_reanchor_samples_.begin(), width_reanchor_samples_.end(),
                       [&](const ReanchorSample& s){ return (t_now - s.t) > width_reanchor_window_; }),
        width_reanchor_samples_.end());
    const int n = static_cast<int>(width_reanchor_samples_.size());
    if (n < width_reanchor_min_frames_) return;
    if (t_now - width_reanchor_samples_.front().t < 0.75 * width_reanchor_window_) return;  // need a real span
    std::vector<double> rs;
    for (const auto& s : width_reanchor_samples_) rs.push_back(s.ratio);
    std::sort(rs.begin(), rs.end());
    const double med = rs[rs.size() / 2];
    if (med > width_reanchor_ratio_ || med < 1.0 / width_reanchor_ratio_) {
        apply_width_reanchor(med, w_locked, r_vis, r_radar, bbox_s, t_now, n);
    }
}

void InterceptorNodeRelativeIMMUKF::apply_width_reanchor(double ratio, double w_locked, double r_vis, double r_radar,
                                          double bbox_s, double t_now, int n_frames) {
    ++width_reanchor_count_;
    const double s_scale = 1.0 / ratio;                       // == r_radar / r_vis
    const double w_new = clamp(w_locked / ratio, width_min_, width_max_);
    // Rescale every IMM filter: x' = D x, P' = D P D^T with D = diag(s,s,s,1,...,1). The error
    // is a pure SCALE error (W and pos both off by the same factor), so position-only scaling
    // is the physically consistent correction. Velocity is NOT stepped (v_perp is scale-free
    // via bearing; a v_los step would jolt PN mid-flight) — the 30 Hz pixel update re-converges
    // v_los within a few frames, and the P scaling lets the update absorb the innovation.
    for (auto& filter : estimator_->imm_ukf->filters) {
        filter.x.segment<3>(0) *= s_scale;
        filter.P.topLeftCorner<3, 3>() *= s_scale * s_scale;
        filter.P.block(0, 3, 3, kStateDim - 3) *= s_scale;
        filter.P.block(3, 0, kStateDim - 3, 3) *= s_scale;
    }
    estimator_->lock_width(w_new);   // re-freezes W at the corrected value + recomputes the blend
    // Keep the estimated-range monotone trackers on the NEW scale: a post-correction growth of
    // r_est (relative to the old minimum) would false-trigger the range_reopened fly-through
    // capture at a moment when the interceptor is mid-approach.
    if (min_range_seen_ < 1e8) min_range_seen_ *= s_scale;
    if (min_vision_range_ < 1e8) min_vision_range_ *= s_scale;
    if (last_good_range_ < 1e8) last_good_range_ *= s_scale;
    width_reanchor_samples_.clear();
    width_reanchor_cooldown_until_ = t_now + width_reanchor_window_;   // hysteresis
    RCLCPP_WARN(this->get_logger(),
        "[W-REANCHOR] %d/%d: locked W=%.2fm inconsistent with radar "
        "(r_vis=%.1fm vs r_radar=%.1fm, ratio %.2f over %d frames) -> W=%.2fm, "
        "rel pos rescaled x%.2f, bbox=%.1fpx",
        width_reanchor_count_, width_reanchor_max_, w_locked, r_vis, r_radar, ratio,
        n_frames, w_new, s_scale, bbox_s);
}

void InterceptorNodeRelativeIMMUKF::bbox_callback(const vision_msgs::msg::Detection2D::SharedPtr msg) {
    if (!mission_state_.is_active()) {
        return;
    }

    std::lock_guard<std::mutex> lock(estimator_mtx_);

    if (last_imu_time_ < 0.0 || !yaw_align_init_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Waiting for IMU to initialize...");
        return;
    }

    const double t_now = rclcpp::Time(msg->header.stamp).seconds();

    const bool has_bbox = !msg->header.frame_id.empty();
    // Principal-point-relative pixel coordinates (the estimator's measurement frame).
    const double u = msg->bbox.center.position.x - cam_cx_;
    const double v = msg->bbox.center.position.y - cam_cy_;
    // Use the bbox WIDTH only. The pinhole size model s = focal*W/depth treats s as the
    // projection of a FIXED physical width W; the target's image height varies with viewing
    // angle and attitude (a banked/tilted drone), so max(size_x,size_y) can latch onto the
    // unstable height and inflate/corrupt the scale. size_x is the stable width dimension and
    // is what every consumer of s (the UKF size channel, the radar width anchor, the terminal
    // brake, the size gate) actually models. (The Python sim publishes size_x == size_y, so
    // this is a no-op there; it matters for AirSim's real tracker bbox.)
    const double s = msg->bbox.size_x;

    Eigen::Matrix3d R_b2n;
    Eigen::Vector3d dv_own;
    Eigen::Quaterniond gimbal_q_now = Eigen::Quaterniond::Identity();
    bool gimbal_rcvd = false;
    {
        std::lock_guard<std::mutex> ol(odom_mtx_);
        R_b2n = latest_R_b2n_;
        gimbal_q_now = latest_gimbal_q_;
        gimbal_rcvd = gimbal_received_;
        if (use_gimbal_ && gimbal_rcvd) {
            const auto matched = gimbal_history_.at(t_now);
            if (!matched) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Skipping bbox without a timestamp-matched camera pose");
                return;
            }
            gimbal_q_now = *matched;
        }
        // Drain the IMU-preintegration window for this camera frame: dv_own is the ownship
        // velocity change since the last camera frame (world frame G, gravity-corrected).
        dv_own = dv_own_accum_;
        dv_own_accum_.setZero();
        last_dv_own_ = dv_own;
    }

    if (last_bbox_cam_t_ > 0.0 && t_now > last_bbox_cam_t_) {
        const double dt_measured = t_now - last_bbox_cam_t_;
        if (dt_measured > 1e-3 && dt_measured < 0.2) {
            dt_cam_ = dt_measured;
            estimator_->dt_cam = dt_measured;
        }
    }
    last_bbox_cam_t_ = t_now;

    // Semi-strapdown gimbal: rebuild R_b2c from the ACTUAL (lagged) camera-to-world
    // orientation q_C_W so the measurement model and the LOS reconstruction match the physical
    // camera orientation at the bbox's measurement/prediction timestamp, not the latest
    // callback. Mixing epochs during a pitch transient invents target LOS motion and scale.
    //
    // NOTE: in gimbal mode R_b2c_ is a SYNTHETIC matrix, not a true body->camera rotation. Its
    // only purpose is to make the estimator's R_n2c = R_b2c*R_b2n^T and pixel_to_los_nav's
    // R_b2n*R_b2c^T*ray reproduce the camera projection from q_C_W: with R_b2c_ = q_C_W^T * R_b2n,
    // R_n2c = q_C_W^T and LOS = q_C_W*ray_cam. The same local R_b2n read here is passed to
    // process_camera below (so the two cancel exactly). Until the first /fpv/gimbal msg arrives,
    // fall back to the bias-derived body->camera mount (R_n2c then includes attitude via R_b2n^T).
    if (use_gimbal_) {
        if (gimbal_rcvd) {
            R_b2c_ = gimbal_q_now.normalized().toRotationMatrix().transpose() * R_b2n;
        } else {
            R_b2c_ = gimbal_R_b2c(gimbal_bias_rad_);
        }
        estimator_->R_b2c = R_b2c_;
    }

    // Needs the rebuilt R_b2c_, so it runs here rather than at message intake.
    bool bbox_ok = has_bbox;
    if (radar_enable_) {
        double d_az = std::numeric_limits<double>::quiet_NaN();
        // Radar cross-check is a HANDOVER-time false-positive guard only. Once vision is locked
        // the target is terminal-authoritative: at close range the radar's metre-level noise
        // makes the camera-radar bearing disagree by 12-30 deg for the REAL target, so gating
        // then REJECTS good bboxes and forces a vision dropout right before intercept. Trust
        // vision unconditionally once locked; the estimator's close-range size gate backstops FPs.
        if (has_bbox && !vision_locked_) {
            const Eigen::Vector3d los_nav_meas =
                RelativeIMMUKFEstimator::pixel_to_los_nav(u, v, cam_fx_, cam_fy_, R_b2n, R_b2c_);
            bbox_ok = bbox_matches_radar(los_nav_meas, s, &d_az);
        }
        update_vision_handover(bbox_ok, s, t_now, d_az);
        // Post-lock W/radar consistency check (safety net for a badly locked W). Must run before
        // process_camera so a rescale lands before this frame's update.
        update_width_reanchor(bbox_ok, s, t_now);
    }

    const bool was_initialized = estimator_->initialized;
    const auto result = estimator_->process_camera(bbox_ok, u, v, s, t_now, R_b2n, dv_own);

    if (!was_initialized && estimator_->initialized) {
        transition_to_state(MissionState::TRACKING, "Target initialized");
        had_terminal_attempt_ = true;   // camera-only path: vision owns guidance from frame 0
    }
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS,
        "bbox=%d%s size=%.1f init=%d->%d", has_bbox, (has_bbox && !bbox_ok) ? "(rejected)" : "",
        s, was_initialized, estimator_->initialized);

    if (!estimator_->initialized) return;

    // Vision owns the guidance cache as soon as a CONSISTENT bbox is in hand (the radar is for
    // mid-course when vision can't see the target, and for the one-time width anchor). Once
    // vision_locked_ the cache fills every frame (even dropouts, has_bbox=false -> COAST). Pre-
    // lock, fill only when a consistent bbox is available — the radar owns the cache otherwise.
    if (!vision_locked_ && !bbox_ok) return;

    // Relative-state estimator: x[0:3] IS the relative position, x[3:6] the relative velocity.
    Eigen::Vector3d r_est = result.second.segment<3>(0);
    const Eigen::Vector3d rel_vel_now = smooth_rel_vel(result.second.segment<3>(3), t_now);
    const Eigen::Vector3d v_est = rel_vel_now;   // relative velocity (no ownship vel available)

    double los_az_world = std::atan2(r_est(1), r_est(0));
    double los_el = std::asin(clamp(r_est(2) / std::max(r_est.norm(), 0.1), -1.0, 1.0));

    if (bbox_ok) {
        // Use the same timestamp-matched camera pose as the UKF measurement.
        const Eigen::Vector3d los_nav = RelativeIMMUKFEstimator::pixel_to_los_nav(u, v, cam_fx_, cam_fy_, R_b2n, R_b2c_);
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
    cache.has_bbox = bbox_ok;
    cache.r_est = r_est;
    cache.v_est = v_est;
    cache.los_az_world = los_az_world;
    cache.los_el = los_el;
    cache.los_rate = los_rate_ema_;
    cache.source_time = t_now;
    cache.update_time = this->get_clock()->now().seconds();
    cache.bbox_size = bbox_ok ? s : 0.0;
    // TRANSVERSE (escape) component of the relative velocity — the part that actually justifies
    // not braking (a target sliding sideways out of the FOV). The full |v_rel| is dominated by
    // the CLOSING speed (≈ own speed), so using it for the brake floor was circular: the faster
    // the closure, the higher the floor, the less it braked — the interceptor barrelled in at
    // v_des and collided. Use only the transverse component (r_xy × v_xy)/|r_xy|.
    {
        const double r_xy = std::sqrt(r_est(0) * r_est(0) + r_est(1) * r_est(1));
        const double v_transverse = (r_xy > 1e-3)
            ? std::abs(r_est(0) * rel_vel_now(1) - r_est(1) * rel_vel_now(0)) / r_xy : 0.0;
        cache.target_speed = std::min(v_transverse, 15.0);
    }
    // Ownship world accel over the last camera window (for the controller's geometric APN).
    cache.a_own = last_dv_own_ / std::max(dt_cam_, 1e-3);

    {
        std::lock_guard<std::mutex> cache_lock(control_cache_mtx_);
        control_cache_ = cache;
    }

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS,
        "rel=[%.2f %.2f %.2f] vel=[%.2f %.2f %.2f] w_est=%.2f r_meas=%.1f r_est=%.1f",
        r_est(0), r_est(1), r_est(2), v_est(0), v_est(1), v_est(2),
        estimator_->imm_ukf->x(9),
        (has_bbox && s > 0.1) ? (cam_fx_ * estimator_->w_guess / s) : 0.0,
        r_est.norm());

    // IMM diagnostic: mode probabilities, per-filter vel, and key P diagonals. Helps locate
    // which motion model is driving the vel jumps when the estimate oscillates.
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS*4,
        "IMM mu=[%.2f %.2f %.2f] v0=[%.1f %.1f %.1f] v1=[%.1f %.1f %.1f] v2=[%.1f %.1f %.1f] Pd=[%.1f %.1f %.1f %.1f]",
        estimator_->imm_ukf->mu(0), estimator_->imm_ukf->mu(1), estimator_->imm_ukf->mu(2),
        estimator_->imm_ukf->filters[0].x(3), estimator_->imm_ukf->filters[0].x(4), estimator_->imm_ukf->filters[0].x(5),
        estimator_->imm_ukf->filters[1].x(3), estimator_->imm_ukf->filters[1].x(4), estimator_->imm_ukf->filters[1].x(5),
        estimator_->imm_ukf->filters[2].x(3), estimator_->imm_ukf->filters[2].x(4), estimator_->imm_ukf->filters[2].x(5),
        estimator_->imm_ukf->P(0,0), estimator_->imm_ukf->P(3,3), estimator_->imm_ukf->P(6,6), estimator_->imm_ukf->P(9,9));
}

void InterceptorNodeRelativeIMMUKF::control_timer_callback() {
    if (!mission_state_.is_active() || !controller_) return;

    // Silent radar is otherwise invisible: the node just behaves like the vision-only variant.
    if (radar_enable_ && !radar_has_rel_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Radar ENABLED but no usable track yet (%s: %lu msgs, %s: %lu msgs)",
            radar_topic_target_.c_str(), radar_msg_target_,
            radar_topic_own_.c_str(), radar_msg_own_);
    }

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
    if (cache.has_bbox) {
        // The image and its matched pose share an old epoch; guide from the
        // short-horizon relative prediction, not the delayed camera position.
        const double delay = clamp(now.seconds() - cache.source_time, 0.0, .25);
        const double cos_el = std::cos(cache.los_el);
        Eigen::Vector3d measured_ray(cos_el * std::cos(cache.los_az_world),
                                     cos_el * std::sin(cache.los_az_world), std::sin(cache.los_el));
        measured_ray = cache.r_est.norm() * measured_ray + cache.v_est * delay;
        cache.r_est += cache.v_est * delay;
        cache.los_az_world = std::atan2(measured_ray.y(), measured_ray.x());
        cache.los_el = std::asin(clamp(measured_ray.z() / std::max(measured_ray.norm(), .1), -1.0, 1.0));
    }

    Eigen::Matrix3d R_b2n;
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    Eigen::Quaterniond gimbal_q_now = Eigen::Quaterniond::Identity();
    bool gimbal_rcvd = false;
    Eigen::Vector3d own_vel_now = Eigen::Vector3d::Zero();
    double t_now = 0.0;
    double sim_start_time = -1.0;
    {
        std::lock_guard<std::mutex> odom_lock(odom_mtx_);
        R_b2n = latest_R_b2n_;
        gyro = latest_gyro_;
        gimbal_q_now = latest_gimbal_q_;
        gimbal_rcvd = gimbal_received_;
        own_vel_now = own_vel_;
        t_now = (last_imu_time_ > 0.0) ? last_imu_time_ : now.seconds();
        // Anchor mission-relative time on the first control tick: the controller's jerk/slew/
        // EMA states and the observability DITHER (sin(2*pi*f*t)) all need a running t.
        if (sim_start_time_ < 0.0) {
            sim_start_time_ = t_now;
        }
        sim_start_time = sim_start_time_;
    }
    // Actual camera boresight world elevation from the measured q_C_W (camera->world): the
    // camera forward axis is +Z (RDF), so its world z-component is sin(boresight_elev). Fed to
    // the controller's vertical-FOV clamp so it tracks the TRUE boresight (incl. servo lag).
    // NaN when the gimbal is disabled or no /fpv/gimbal msg has arrived yet (clamp then falls
    // back to the -gimbal_bias_ assumption).
    double boresight_el_meas = std::numeric_limits<double>::quiet_NaN();
    if (use_gimbal_ && gimbal_rcvd) {
        const Eigen::Matrix3d R_c2n = gimbal_q_now.normalized().toRotationMatrix();
        boresight_el_meas = std::asin(clamp(R_c2n(2, 2), -1.0, 1.0));
    }

    const double roll = std::atan2(R_b2n(2, 1), R_b2n(2, 2));
    const double pitch = std::atan2(-R_b2n(2, 0), std::sqrt(R_b2n(2, 1) * R_b2n(2, 1) + R_b2n(2, 2) * R_b2n(2, 2)));
    const double yaw = std::atan2(R_b2n(1, 0), R_b2n(0, 0));
    // Closing speed along LOS from relative quantities only (no ownship vel):
    // v_closing = -d|r|/dt = -v_rel·r̂ (positive = approaching). Replaces odom v_fwd_body.
    const double v_closing = cache.valid
        ? -cache.v_est.dot(cache.r_est) / std::max(cache.r_est.norm(), 1e-3)
        : 0.0;
    const double measured_axial_speed = use_imu_axial_speed_
        ? own_vel_now.x() * std::cos(yaw) + own_vel_now.y() * std::sin(yaw)
        : v_closing;
    if (target_axial_feedforward_limit_ > 0 && cache.has_bbox && cache.bbox_size >= 80) {
        // IMU own velocity + UKF relative velocity estimates target motion.
        // Keep image braking a closing-speed reference, not an own-speed cap
        // below a moving target's speed. Bound/filter uncertain monocular scale.
        const double relative_axial = cache.v_est.x() * std::cos(yaw)
            + cache.v_est.y() * std::sin(yaw);
        const double estimate = bounded_target_axial_speed(
            measured_axial_speed, relative_axial, target_axial_feedforward_limit_);
        if (last_target_axial_time_ > 0 && t_now > last_target_axial_time_) {
            const double alpha = -std::expm1(-std::min(t_now - last_target_axial_time_, .1) / .5);
            target_axial_feedforward_ += alpha * (estimate - target_axial_feedforward_);
        }
        last_target_axial_time_ = t_now;
    }
    const double axial_speed = measured_axial_speed - target_axial_feedforward_;
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

    // Collision detection (gyro spike) — AirSim / real HW only.
    // The ROS2 Python sim (use_simulation_=true) has ground-truth range and judges the intercept
    // externally (ros2_sim_node.py), so the gyro check is skipped there. On contact the body rate
    // spikes well beyond what was commanded -> declare COMPLETED (intercept success) and stop.
    if (!use_simulation_ && !collision_detected_ &&
        (collision_min_bbox_px_ == 0.0 ||
         (cache.has_bbox && cache.bbox_size >= collision_min_bbox_px_))) {
        const Eigen::Vector3d rate_error = gyro - last_cmd_;
        const double rate_error_norm = rate_error.norm();
        if (rate_error_norm > 4.0) {
            collision_detected_ = true;
            RCLCPP_WARN(this->get_logger(),
                "[COLLISION] Gyro spike! |err|=%.1f rad/s gyro=[%.2f,%.2f,%.2f] "
                "cmd=[%.2f,%.2f,%.2f]",
                rate_error_norm,
                gyro.x(), gyro.y(), gyro.z(),
                last_cmd_.x(), last_cmd_.y(), last_cmd_.z());
            transition_to_state(MissionState::COMPLETED, "Collision detected via gyro");
            stop();
            return;
        }
    }

    // ---- Re-intercept (radar.reintercept_enable) ----------------------------------------
    // On a terminal vision-loss MISS (target gone > vision_lost_timeout AFTER a real terminal
    // phase) that was NOT a clean hit (closest approach > kill_range), if the radar has
    // re-acquired the target at a standoff and attempts remain: clear the capture/dropout
    // latches and resume radar mid-course guidance for another pass. The radar cache is already
    // being filled again (vision_locked_ flipped false at vision_lost_timeout, reopening fuse_radar
    // + fill_cache_from_radar). Placed BEFORE the capture/dropout blocks so the reset clears both,
    // then execution falls through to compute_control. Self-debouncing: had_terminal_attempt_ is
    // cleared here and only re-armed on the next vision lock.
    {
        const double prev_min = min_range_seen_;
        const bool terminal_miss = had_terminal_attempt_ && !vision_delivering(now.seconds()) &&
                                   min_range_seen_ > reintercept_kill_range_;
        const bool radar_reacquired = radar_enable_ && radar_has_rel_ &&
                                      radar_rel_pos_.norm() > reintercept_min_range_;
        if (reintercept_enable_ && terminal_miss && radar_reacquired &&
            reintercept_count_ < reintercept_max_) {
            ++reintercept_count_;
            captured_ = false;
            has_frozen_cmd_ = false;
            last_good_range_ = 1e9;
            min_range_seen_ = 1e9;
            min_vision_range_ = 1e9;
            last_bbox_time_ = 0.0;
            {
                std::lock_guard<std::mutex> ol(odom_mtx_);
                v_own_brake_.setZero();
                brake_vel_seeded_ = false;
            }
            vision_locked_ = false;          // radar owns guidance again (force; likely already false)
            had_terminal_attempt_ = false;   // re-arms on the next vision lock
            transition_to_state(MissionState::BLIND_FLIGHT, "Re-intercept: radar relay re-engaged");
            RCLCPP_WARN(this->get_logger(),
                "[REINTERCEPT] attempt %d/%d after terminal miss (min %.1fm > kill %.1fm); "
                "radar standoff %.1fm — resuming mid-course relay",
                reintercept_count_, reintercept_max_, prev_min, reintercept_kill_range_,
                radar_rel_pos_.norm());
            // fall through to normal guidance (the radar-filled cache drives compute_control)
        }
    }

    // Capture lockout: latch true once the UAV has intercepted, then BRAKE-TO-HOLD for the rest
    // of the mission. Resuming normal guidance after a close approach — bbox flickered back or
    // dead-reckon chased a reacquired target — flew the UAV 70-250 m off after an otherwise-clean
    // intercept at v_des=20. Two reliable triggers (the close approach happens during a bbox
    // dropout, when the range ESTIMATE is biased and useless, so do NOT key off r_est alone):
    //   (1) dead-reckoned range < capture_range (only fires if the estimate happens to be good), OR
    //   (2) target now BEHIND the UAV (|los_az|>90°, observable) after being close at the last
    //       detection (last_good_range < coast_range) — i.e. the UAV flew past the target.
    const double capture_range = capture_range_;
    const bool overshot = std::abs(los_az) > (M_PI / 2.0);
    if (cache.valid) {
        min_range_seen_ = std::min(min_range_seen_, cache.r_est.norm());
        // Vision-only running min: anchors the fly-through (range_reopened) detector. The
        // estimator/radar cache range is INCONSISTENT with vision during a radar re-engage — radar
        // drives min_range_seen_ down as the interceptor closes, but vision at handover still
        // measures the standoff range, so min_range_seen_ would false-trigger a reopen the instant
        // vision handsover. min_vision_range_ advances only on real vision frames, so a reopen
        // requires a genuine vision close approach first.
        if (cache.has_bbox) {
            // Startup monocular scale can dip below 15 m at an 80 m standoff.
            // A sufficiently large image must anchor a terminal pass first.
            min_vision_range_ = terminal_vision_range_minimum(
                min_vision_range_, cache.r_est.norm(), cache.bbox_size);
        }
    }
    // Reopen: range bottomed out and has since grown past it — the UAV flew through the target.
    // Robust under a noisy close-range estimate (high-noise scenario) where los_az never cleanly
    // flips past 90 deg. Keyed on min_vision_range_ (vision-confirmed) so it only arms after a real
    // vision close approach — immune to the radar-cache inconsistency during radar re-engage.
    const bool range_reopened = has_frozen_cmd_ && min_vision_range_ < 15.0 &&
        cache.valid && cache.r_est.norm() > min_vision_range_ + 8.0;
    // Capture is a terminal-phase concept: only arm it once a confirmed vision terminal pass has
    // begun this engagement (had_terminal_attempt_). This suppresses a spurious range_reopened
    // capture during radar re-engage right after a re-intercept, where the noisy post-dropout
    // estimate dips min_range_seen_ < 15 then jumps (reads as a fly-through but is just the
    // re-approach). Legitimate captures (close approach / overshot / real fly-through) always
    // occur after vision engaged, so had_terminal_attempt_ is true for them.
    if (had_terminal_attempt_ && !captured_ && cache.valid &&
        (cache.r_est.norm() < capture_range ||
         (overshot && has_frozen_cmd_ && last_good_range_ < 12.0) ||
         range_reopened)) {
        captured_ = true;
        // Seed the brake-scoped ownship velocity: the estimator's v_closing flips sign once
        // the dead-reckoned r_est passes behind (fly-through), so braking on it runs AWAY
        // (positive feedback). Instead, take the ownship velocity at capture ≈ last good
        // closing speed along the current heading, then track it by IMU strapdown integration
        // (imu_callback) — the brake then always opposes the REAL ownship velocity.
        {
            std::lock_guard<std::mutex> ol(odom_mtx_);
            const double v_seed = std::max(v_closing, 0.0);
            v_own_brake_ = Eigen::Vector3d(v_seed * std::cos(yaw), v_seed * std::sin(yaw), 0.0);
            brake_vel_seeded_ = true;
        }
        RCLCPP_INFO(this->get_logger(),
            "CAPTURE: holding (brake-to-hold) to end of mission (r_est=%.1fm los_az=%.0fdeg v_seed=%.1f)",
            cache.r_est.norm(), los_az * 180.0 / M_PI, std::max(v_closing, 0.0));
    }
    if (captured_) {
        // Brake-to-hold on the strapdown brake velocity (NOT the estimator v_closing — its
        // sign is unreliable once the target is behind): nose-up pitch cancels the forward
        // component, roll cancels the lateral component, tilt-compensated hover thrust holds
        // altitude. Both angles -> 0 as the velocity -> 0, settling into a level hover.
        const double g_hover = g_;
        constexpr double max_brake_tilt = 0.436;   // 25 deg
        Eigen::Vector3d v_brake;
        {
            std::lock_guard<std::mutex> ol(odom_mtx_);
            v_brake = v_own_brake_;
        }
        const double v_fwd_brake = std::cos(yaw) * v_brake(0) + std::sin(yaw) * v_brake(1);
        const double v_lat_brake = -std::sin(yaw) * v_brake(0) + std::cos(yaw) * v_brake(1);
        const double theta_brake = clamp(-0.12 * v_fwd_brake, -max_brake_tilt, max_brake_tilt);
        const double phi_brake = clamp(0.12 * v_lat_brake, -max_brake_tilt, max_brake_tilt);
        const double tilt = std::sqrt(roll * roll + pitch * pitch);
        auto cmd = std::make_unique<quadrotor_msgs::msg::ControlCommand>();
        cmd->header.stamp = now;
        cmd->armed = true;
        cmd->control_mode = quadrotor_msgs::msg::ControlCommand::BODY_RATES;
        cmd->collective_thrust = g_hover / std::max(std::cos(tilt), 0.3);
        cmd->bodyrates.x = clamp(5.0 * (phi_brake - roll), -3.0, 3.0) * ctrl_coeff_;
        cmd->bodyrates.y = clamp(5.0 * (theta_brake - pitch), -3.0, 3.0) * ctrl_coeff_;
        cmd->bodyrates.z = 0.0;
        last_cmd_ = Eigen::Vector3d(cmd->bodyrates.x, cmd->bodyrates.y, cmd->bodyrates.z);
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
            const double g_hover = g_;
            const double tilt = std::sqrt(roll * roll + pitch * pitch);
            cmd->collective_thrust = g_hover / std::max(std::cos(tilt), 0.3);
            cmd->bodyrates.x = clamp(-3.0 * roll, -3.0, 3.0) * ctrl_coeff_;
            cmd->bodyrates.y = clamp(-3.0 * pitch, -3.0, 3.0) * ctrl_coeff_;
            cmd->bodyrates.z = 0.0;
        }
        last_cmd_ = Eigen::Vector3d(cmd->bodyrates.x, cmd->bodyrates.y, cmd->bodyrates.z);
        control_pub_->publish(std::move(cmd));
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS/2,
            "%s (terminal dropout): age=%.2fs last_good_range=%.1fm",
            (dropout_age < coast_max_dur) ? "COAST" : "LEVELHOLD", dropout_age, last_good_range_);
        return;
    }

    // Vertical damping from the IMU-integrated ownship vertical velocity (robust; the
    // monocular v_est(2) used before is poorly observable and left the vertical PD to over-dive
    // on below-FOV targets — own_vel_(2) is the actual own climb/sink rate).
    const double v_z = own_vel_now(2);
    // Radar-known close range suppresses the dither even while vision owns guidance. The radar
    // cache keeps updating after the vision lock (fuse_radar returns early but writes
    // radar_rel_pos_ first), so this is an independent range opinion the whole terminal phase.
    // Copy under the estimator mutex (fuse_radar's writer side); no other lock is held here.
    bool dither_radar_close = false;
    if (radar_enable_ && dither_radar_range_ > 0.0) {
        Eigen::Vector3d r_radar;
        double t_radar;
        bool has_rel;
        {
            std::lock_guard<std::mutex> lock(estimator_mtx_);
            r_radar = radar_rel_pos_;
            t_radar = radar_rel_time_;
            has_rel = radar_has_rel_;
        }
        dither_radar_close = has_rel &&
            (now.seconds() - t_radar) < radar_timeout_ &&
            r_radar.norm() < dither_radar_range_;
    }
    const auto ctrl = controller_->compute_control(roll, pitch, yaw, cache.r_est, cache.v_est,
                                                   los_az, cache.los_el, t_rel, axial_speed,
                                                   cache.bbox_size, v_z, cache.target_speed,
                                                   cache.los_rate, cache.a_own, boresight_el_meas,
                                                   (radar_enable_ && radar_has_rel_ && !vision_delivering(now.seconds())),
                                                   dither_radar_close);
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
    // controller, suppressing pitch/roll/yaw oscillation.
    // Rate scaling: ctrl_coeff_ scales the raw body-rate command. This acts as a
    // fixed gain reduction (~0.39 with rate_tc=0.02) that damps the attitude loop.
    // Correct for AirSim/real hardware (inner attitude loop adds gain). For the Python
    // sim (bodyrates execute directly) this under-powers the UAV, but retuning K_att
    // for EMA full authority is a separate task. Keep gain form as the stable baseline.
    cmd->bodyrates.x = ctrl_coeff_ * ctrl[1];
    cmd->bodyrates.y = ctrl_coeff_ * ctrl[2];
    cmd->bodyrates.z = ctrl_coeff_ * ctrl[3];
    last_cmd_ = Eigen::Vector3d(cmd->bodyrates.x, cmd->bodyrates.y, cmd->bodyrates.z);
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
        "ctrl rel=[%.2f %.2f %.2f] vel=[%.2f %.2f %.2f] cmd=[%.2f %.2f %.2f %.2f] pitch=%.2f los_el=%.2f own_axial=%.2f own_vz=%.2f bbox=%.1f target_axial_ff=%.2f",
        cache.r_est(0), cache.r_est(1), cache.r_est(2), cache.v_est(0), cache.v_est(1), cache.v_est(2),
        ctrl[0], ctrl[1], ctrl[2], ctrl[3], pitch, cache.los_el,
        measured_axial_speed, v_z, cache.bbox_size, target_axial_feedforward_);

    // Attitude-loop diagnostic: actual Euler angles + measured body rates (finite-differenced)
    // vs commanded body rates. Reveals whether oscillation is from cmd jumps (estimator/guidance)
    // or from P-gain too high (cmd smooth but rate overshoots).
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), THROTTLE_INTERVAL_MS,
        "att rpy=[%.3f %.3f %.3f] rate=[%.2f %.2f %.2f] cmd_rate=[%.2f %.2f %.2f]",
        roll, pitch, yaw, rate_roll, rate_pitch, rate_yaw, ctrl[1], ctrl[2], ctrl[3]);
}

void InterceptorNodeRelativeIMMUKF::mission_launch_callback(const std_msgs::msg::Header::SharedPtr msg) {
    // OFF stops everything (disarm edge, RTH); DISMISS only cancels interception
    // control while the tracker keeps running (pilot stop pulse). Both end the
    // mission here.
    if ((msg->frame_id == "OFF" || msg->frame_id == "DISMISS") && mission_state_.is_active()) {
        RCLCPP_INFO(this->get_logger(), "Mission %s received! Stopping control...",
                    msg->frame_id == "DISMISS" ? "dismiss" : "abort");
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
        vision_locked_ = !radar_enable_;
        vision_frames_ = 0;
        last_vision_time_ = -1.0;
        width_locked_ = false;
        width_samples_.clear();
        radar_last_fused_t_ = -1.0;
        radar_rel_time_ = -1.0;
        radar_has_rel_ = false;
        radar_rel_pos_.setZero();
        radar_own_vel_t_ = -1.0;
        own_vel_aided_ = false;
        bearing_err_ema_ = 0.0;
        has_bearing_err_ = false;
        vision_lock_t0_ = -1.0;
        radar_msg_target_ = 0;
        radar_msg_own_ = 0;
        {
            std::lock_guard<std::mutex> rl(radar_mtx_);
            radar_target_ = RadarSample();
            radar_own_ = RadarSample();
        }
        {
            std::lock_guard<std::mutex> ol(odom_mtx_);
            dv_own_accum_.setZero();
            last_dv_own_.setZero();
            last_imu_time_ = -1.0;
            mission_gyro_initialized_ = false;
            sim_start_time_ = -1.0;
            // Keep measured body/camera orientation for the first post-trigger bbox.
            last_bbox_cam_t_ = -1.0;
            dt_cam_ = 0.033;
        }
        last_cmd_ = Eigen::Vector3d::Zero();
        has_target_vel_smoothed_ = false;
        has_prev_attitude_ = false;
        has_prev_los_ = false;
        los_rate_ema_ = 0.0;
        prev_los_t_ = -1.0;
        has_frozen_cmd_ = false;
        target_axial_feedforward_ = 0.0;
        last_target_axial_time_ = -1.0;
        last_good_range_ = 1e9;
        last_bbox_time_ = 0.0;
        captured_ = false;
        min_range_seen_ = 1e9;
        min_vision_range_ = 1e9;
        reintercept_count_ = 0;
        width_reanchor_samples_.clear();
        width_reanchor_count_ = 0;
        width_reanchor_cooldown_until_ = -1.0;
        had_terminal_attempt_ = false;
        collision_detected_ = false;
        yaw_align_init_ = false;
        {
            std::lock_guard<std::mutex> ol(odom_mtx_);
            v_own_brake_.setZero();
            own_vel_.setZero();
            brake_vel_seeded_ = false;
        }
        transition_to_state(MissionState::INITIALIZING, "Mission launch command");
        publish_mission_state(true);
        start_control_timer();
        RCLCPP_INFO(this->get_logger(), "Mission started: odometry tracking active");
    }
}

void InterceptorNodeRelativeIMMUKF::mission_callback(const std_msgs::msg::Bool::SharedPtr msg) {
    auto header = std::make_shared<std_msgs::msg::Header>();
    header->stamp = this->get_clock()->now();
    header->frame_id = msg->data ? "MISSION" : "OFF";
    mission_launch_callback(header);
}

void InterceptorNodeRelativeIMMUKF::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    const double t = rclcpp::Time(msg->header.stamp).seconds();
    std::lock_guard<std::mutex> odom_lock(odom_mtx_);
    const Eigen::Vector3d gyro(msg->angular_velocity.x, msg->angular_velocity.y,
                               msg->angular_velocity.z);
    const Eigen::Quaterniond q(msg->orientation.w, msg->orientation.x,
                               msg->orientation.y, msg->orientation.z);
    if (!q.coeffs().allFinite() || q.norm() < 1e-6 || !gyro.allFinite()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Ignoring invalid IMU orientation or angular velocity");
        return;
    }
    if (last_imu_time_ > 0.0 && t <= last_imu_time_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Ignoring non-increasing IMU timestamp");
        return;
    }
    if (gyro_attitude_during_mission_ && mission_state_.is_active()) {
        if (!mission_gyro_initialized_) {
            mission_gyro_attitude_ = q.normalized();
            mission_gyro_initialized_ = true;
        } else if (last_imu_time_ > 0.0) {
            mission_gyro_attitude_ = integrate_body_gyro(mission_gyro_attitude_,
                0.5 * (latest_gyro_ + gyro), std::min(t - last_imu_time_, 0.05));
        }
        latest_R_b2n_ = mission_gyro_attitude_.toRotationMatrix();
    } else {
        latest_R_b2n_ = q.normalized().toRotationMatrix();
        mission_gyro_initialized_ = false;
    }
    latest_gyro_ = gyro;
    if (last_imu_time_ > 0.0 && t > last_imu_time_) {
        const double dt = std::min(t - last_imu_time_, 0.05);   // guard against stamp gaps
        const Eigen::Vector3d f_body(msg->linear_acceleration.x - imu_bias_(0),
                                     msg->linear_acceleration.y - imu_bias_(1),
                                     msg->linear_acceleration.z - imu_bias_(2));
        const Eigen::Vector3d a_world = latest_R_b2n_ * f_body - Eigen::Vector3d(0.0, 0.0, g_);
        dv_own_accum_ += a_world * dt;
        own_vel_ += a_world * dt;   // always-on ownship vel (world) for vertical damping
        // Brake-scoped ownship velocity (capture hold only): seeded from the last good
        // v_closing at capture, then propagated by pure strapdown integration. Short-horizon
        // (seconds), so accel-noise drift is negligible.
        if (brake_vel_seeded_) {
            v_own_brake_ += a_world * dt;
        }
    }
    last_imu_time_ = t;
    yaw_align_init_ = true;
}

void InterceptorNodeRelativeIMMUKF::gimbal_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    const Eigen::Quaterniond q_flu(msg->orientation.w, msg->orientation.x,
                                   msg->orientation.y, msg->orientation.z);
    if (q_flu.norm() > 1e-6) {
        const Eigen::Quaterniond q_rdf = q_flu.normalized() *
                                         Eigen::Quaterniond(R_b2c_base_.transpose());
        std::lock_guard<std::mutex> odom_lock(odom_mtx_);
        if (!gimbal_history_.add(rclcpp::Time(msg->header.stamp).seconds(), q_rdf)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Ignoring invalid or non-increasing gimbal pose");
            return;
        }
        latest_gimbal_q_ = q_rdf.normalized();
        gimbal_received_ = true;
    }
}

Eigen::Matrix3d InterceptorNodeRelativeIMMUKF::gimbal_R_b2c(double angle) const {
    Eigen::Matrix3d R_g;
    R_g << 1.0, 0.0, 0.0,
           0.0, std::cos(angle), -std::sin(angle),
           0.0, std::sin(angle), std::cos(angle);
    return R_g * R_b2c_base_;
}

void InterceptorNodeRelativeIMMUKF::transition_to_state(MissionState new_state, const char* reason) {
    const MissionState old_state = mission_state_.get();
    if (old_state == new_state) return;
    mission_state_.set(new_state);
    RCLCPP_INFO(this->get_logger(), "Mission state: %s -> %s (%s)",
                MissionStateMachine::state_name(old_state),
                MissionStateMachine::state_name(new_state), reason);
}

void InterceptorNodeRelativeIMMUKF::publish_mission_state(bool active) {
    auto state_msg = std::make_unique<std_msgs::msg::Bool>();
    state_msg->data = active;
    mission_state_pub_->publish(std::move(state_msg));
}

void InterceptorNodeRelativeIMMUKF::stop() {
    if (control_timer_) {
        control_timer_->cancel();
        control_timer_ = nullptr;
    }
    reset_control_cache();
    publish_mission_state(false);
    const double g_hover = g_;
    auto cmd = std::make_unique<quadrotor_msgs::msg::ControlCommand>();
    cmd->header.stamp = this->get_clock()->now();
    cmd->armed = true;
    cmd->control_mode = quadrotor_msgs::msg::ControlCommand::ATTITUDE;
    cmd->collective_thrust = g_hover;
    control_pub_->publish(std::move(cmd));
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InterceptorNodeRelativeIMMUKF>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
