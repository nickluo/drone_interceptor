#pragma once

// AdvancedHybridController: PN + IBVS guidance and attitude/thrust allocation for
// the relative IMM-UKF interceptor node. Analytical (no optimal control): lateral
// proportional navigation with APN lead, an axial speed law with maneuverability
// and visual proximity caps, a cascaded vertical track, and a P-law onto the
// attitude reference with slew/EMA shaping. Mirrors the verified Python
// implementation in prototype/sim_test_relative_imm_ukf.py.

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "guidance_limits.h"
#include "utils.h"

class AdvancedHybridController {
public:
    double N;
    double K_att;
    double K_yaw;
    double v_des;
    double max_tilt_rad;
    double g = 9.81;             // gravity (set from physics.g config via set_g)
    double fov_half_v_ = M_PI;
    double fov_half_h_ = M_PI;
    double fov_keep_margin_ = 0.85;
    double axial_jerk_max_ = 1e9;
    bool enable_fov_clamp_ = false;
    double mount_pitch_ = 0.0;
    double last_axial_accel_ = 0.0;
    double last_t_ = -1.0;
    bool has_last_ = false;
    // Vertical-accel slew-limit state (Fix C): bounds |da_z/dt|.
    double last_a_z_ = 0.0;
    double last_az_t_ = -1.0;
    bool has_last_az_ = false;
    double a_z_slew_ = 8.0;   // m/s^3, rate limit on the vertical accel command
    double lateral_decay_floor_ = 0.4;  // Fix D: a_lat_mag floor at terminal_brake_range (close-range weave damp)
    double lat_decay_start_ = 25.0;     // Fix F: range at which the close-range lateral attenuation begins (m)
    double max_body_rate_ = 3.0;
    double terminal_brake_range_ = 5.0;
    // Decoupled gains (config-driven via set_guidance_tuning).
    double K_fwd_ = 2.0;        // axial: a_fwd = K_fwd * (v_des - v_fwd_body)
    double K_az_ = 2.0;         // vertical-speed tracking gain: a_z = K_az*(v_z_desired - v_z)
    double a_z_limit_ = 8.0;    // vertical accel clamp (m/s^2) — airframe thrust/accel limit
    double vertical_brake_accel_ = 0.0;
    double vertical_brake_delay_ = 0.0;
    double visual_speed_gain_ = 0.0;
    double visual_speed_floor_ = 0.0;
    double terminal_speed_floor_ = 0.0;
    double terminal_speed_bbox_threshold_ = 120.0;
    double terminal_speed_los_rate_threshold_ = 0.05;
    // Turn-radius cap scaling factor. The cap v ≤ factor·sqrt(g·tan(tilt)·r) is the adaptive
    // mid-course speed reducer. The terminal floor is enabled only from current visual
    // observability, so it does not depend on the target scenario.
    double turn_radius_cap_factor_ = 0.8;
    // Axial deadband: coast (a_fwd=0) within ±this of v_des_eff — kills radar-noise-driven pitch oscillation.
    double axial_deadband_ = 1.5;
    // Motion-dependent LOS gate widening: bearing_tol += motion_gate_factor * v_perp * delay / r.
    double motion_gate_factor_ = 3.0;
    // Semi-strapdown single-axis pitch gimbal: keeps the camera boresight near-horizontal
    // (world elev = -gimbal_bias_) regardless of body pitch, so the vertical FOV is DECOUPLED
    // from theta_cmd. When true, the fixed-mount pitch FOV clamp is skipped and vertical
    // keep-in-frame is done through ALTITUDE (a_z) instead (see compute_control).
    bool has_gimbal_ = false;
    double gimbal_bias_ = 0.0;   // rad, down-gaze bias: boresight elev = -gimbal_bias_
    // Vertical cascade (cascaded velocity-limited altitude track): v_climb_/v_descent_ = max
    // commanded climb/sink rate [m/s] (the cap); K_dz_up_/dn_ = dz->rate gain [1/s] (closure
    // tau = 1/K). Asymmetric: climb more responsive than descent (avoid below-FOV over-dive).
    double v_climb_   = 8.0;
    double v_descent_ = 8.0;
    double K_dz_up_   = 2.0;
    double K_dz_dn_   = 1.0;
    // Terminal arrival-margin law (see compute_control section B): inside terminal_arrival_range_,
    // demand |v_z| >= terminal_arrival_margin_ * |dz| / t_go so the residual vertical error is
    // removed BY arrival. margin <= 0 (default) disables — pure profile tracking as before.
    double terminal_arrival_margin_ = 0.0;
    double terminal_arrival_range_  = 0.0;
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
    // Terminal authority (maneuvering-target tracking): relax the roll clamp inside terminal_tilt_range_.
    // max_tilt_terminal_rad_ is ALSO the bank used by the turn-radius speed cap in that band, so raising
    // it both permits tighter turns AND raises the close-range speed cap (win-win for terminal maneuverability).
    double max_tilt_terminal_rad_ = 45.0 * M_PI / 180.0;  // roll clamp + cap tilt inside terminal_tilt_range_
    double terminal_tilt_range_ = 15.0;  // m, below this range phi_cmd and the speed cap use max_tilt_terminal_rad_
    // Active monocular-observability dither: a small lateral sinusoidal acceleration that gives
    // the ownship a horizontal parallax baseline. With x != 0 the pinhole u=fx*x/z becomes
    // sensitive to z (depth), so the depth/width degeneracy (size=fx*W/z) is broken and the
    // along-LOS v_rel — otherwise only weakly observable via the noisy size channel — becomes
    // observable through the bearing channel. The IMU preintegration captures the dither motion
    // as a_own, so the estimator predict carries it; the update resolves depth from the parallax.
    double dither_amp_ = 0.8;          // m/s^2, lateral dither amplitude
    double dither_freq_ = 0.5;         // Hz, lateral dither frequency
    double dither_min_range_ = 10.0;   // m, suppress dither inside this range (don't disturb terminal)
    // SECOND, W-independent close gate: suppress the dither when the bbox is already LARGE in px.
    // The dither exists to lift the size-channel SNR while the bbox is small; a large bbox means
    // good SNR and/or a close target, where the dither only disturbs terminal precision. bbox px
    // is a DIRECT observable, immune to the monocular (W, range) scale error that blinds
    // dither_min_range_: W over-estimated by k over-estimates range by k, so the range gate never
    // trips while the serpentine persists into true close range and causes a miss. 80 px chosen
    // empirically (wrong_w x10 FOV sweep: 80px 7/10 vs 60px 5/10 vs off 6/10 — 60px cuts the
    // dither while the W/scale estimate still needs its observability aid). At focal 1057:
    // W 1.5 -> 20 m, W 0.8 -> 10.6 m; an over-bound tracker trips even earlier — safer. 0 = off.
    double dither_max_bbox_ = 80.0;    // px, ALSO suppress the dither when bbox >= this size
    double last_bbox_size_ = 0.0;      // px, last live bbox (stale-hold across dropout frames)
    double last_bbox_t_ = -1.0;        // s, mission-relative time of last_bbox_size_
    // Axial-only range EMA (radar smoothness): isolates the turn-radius cap + terminal-decel ramp
    // from 10 Hz radar single-sample range noise. r_norm elsewhere (PN, dz, APN) stays unsmoothed.
    // Lag is conservative during closing (over-estimates range -> cap allows slightly more speed,
    // decel engages slightly later), so it biases against the over-braking we are fixing.
    double r_norm_smooth_tc_ = 0.0;    // s, 0 = off (passthrough)
    double r_norm_ema_ = 0.0;
    double last_rn_ema_t_ = -1.0;
    bool   has_rn_ema_ = false;
    // Geometric APN: recover the target lateral accel from TRUE-PN geometry + IMU (NOT the
    // estimator a_t state — unobservable with a single bbox): a_t_perp = range*ydot +
    // 2*lambda_dot*v_range + a_own_perp, where ydot = d(lambda_dot)/dt. Leads a turning target.
    double K_apn_ = 0.8;
    double K_apn_min_at_ = 0.5;        // m/s^2, gate on |a_t_perp_geom|
    double ydot_tc_ = 0.8;             // s, EMA time constant on d(lambda_dot)/dt
    double atp_tc_ = 0.6;              // s, EMA time constant on a_t_perp_geom
    double prev_los_rate_ = 0.0;
    double prev_lr_t_ = -1.0;
    bool   has_prev_lr_ = false;
    double ydot_ema_ = 0.0;
    double atp_ema_ = 0.0;
    double aown_perp_ema_ = 0.0;
    double last_los_dt_ = 0.02;   // s, last valid LOS-sample dt (for atp EMA alpha); updated each camera frame

    AdvancedHybridController(double N_val,
                             double K_att_val, double K_yaw_val, double v_des_val, double max_tilt_deg,
                             double fov_half_v, double fov_half_h, double fov_keep_margin,
                             double axial_jerk_max, bool enable_fov_clamp,
                             double mount_pitch_rad,
                             double max_body_rate, double terminal_brake_range,
                             double v_pn_ref, double max_att_rate, double att_ref_tc,
                             double max_tilt_terminal_deg, double terminal_tilt_range)
        : N(N_val), K_att(K_att_val),
          K_yaw(K_yaw_val), v_des(v_des_val), max_tilt_rad(max_tilt_deg * M_PI / 180.0),
          fov_half_v_(fov_half_v), fov_half_h_(fov_half_h), fov_keep_margin_(fov_keep_margin),
          axial_jerk_max_(axial_jerk_max), enable_fov_clamp_(enable_fov_clamp),
          mount_pitch_(mount_pitch_rad),
          max_body_rate_(max_body_rate), terminal_brake_range_(terminal_brake_range),
          v_pn_ref_(v_pn_ref), max_att_rate_(max_att_rate), att_ref_tc_(att_ref_tc),
          max_tilt_terminal_rad_(max_tilt_terminal_deg * M_PI / 180.0),
          terminal_tilt_range_(terminal_tilt_range) {}

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
        last_a_z_ = 0.0;
        last_az_t_ = -1.0;
        has_last_az_ = false;
        prev_los_rate_ = 0.0;
        prev_lr_t_ = -1.0;
        has_prev_lr_ = false;
        ydot_ema_ = 0.0;
        atp_ema_ = 0.0;
        aown_perp_ema_ = 0.0;
        last_los_dt_ = 0.02;
        r_norm_ema_ = 0.0;
        last_rn_ema_t_ = -1.0;
        has_rn_ema_ = false;
        last_bbox_size_ = 0.0;
        last_bbox_t_ = -1.0;
    }
    void set_a_z_slew(double v) { a_z_slew_ = (v > 0.0) ? v : 1e9; }  // <=0 disables the slew limit
    void set_lateral_decay_floor(double v) { lateral_decay_floor_ = clamp(v, 0.0, 1.0); }
    void set_lat_decay_start(double v) { lat_decay_start_ = std::max(v, terminal_brake_range_ + 1.0); }
    void set_g(double g_val) { g = g_val; }
    void set_gimbal(bool has_gimbal, double gimbal_bias_rad) {
        has_gimbal_ = has_gimbal;
        gimbal_bias_ = gimbal_bias_rad;
    }
    // Vertical-cascade tuning (cascaded velocity-limited altitude track). Independent so the
    // operator can set the climb/sink RATE (m/s, the cap — calibrate to the airframe's real
    // capability so the commanded rate is achievable) and the RESPONSIVENESS (K_dz 1/s, closure
    // tau = 1/K) separately for climb vs descent.
    void set_vertical_climb(double v_climb, double v_descent, double K_dz_up, double K_dz_dn) {
        v_climb_   = std::max(0.5, v_climb);
        v_descent_ = std::max(0.5, v_descent);
        K_dz_up_   = std::max(0.1, K_dz_up);
        K_dz_dn_   = std::max(0.1, K_dz_dn);
    }
    // Terminal arrival-margin law: inside `range`, demand |v_z| >= margin * |dz| / t_go
    // (t_go = range / closing speed) so the vertical error is removed by arrival instead of
    // asymptotically at K_dz. margin <= 0 disables.
    void set_terminal_arrival(double margin, double range) {
        if (!std::isfinite(margin) || margin < 0 || !std::isfinite(range) || range < 0)
            throw std::invalid_argument("invalid terminal arrival parameters");
        terminal_arrival_margin_ = margin;
        terminal_arrival_range_  = range;
    }
    // Guidance tuning (operator-facing): axial gain, vertical tracking gain + accel limit, and the
    // lateral observability dither. All independent config params.
    void set_guidance_tuning(double K_fwd, double K_az, double a_z_limit,
                             double dither_amp, double dither_freq, double dither_min_range,
                             double dither_max_bbox) {
        K_fwd_           = std::max(0.1, K_fwd);
        K_az_            = std::max(0.1, K_az);
        a_z_limit_       = std::max(1.0, a_z_limit);
        dither_amp_      = std::max(0.0, dither_amp);
        dither_freq_     = std::max(0.01, dither_freq);
        dither_min_range_= std::max(1.0, dither_min_range);
        dither_max_bbox_ = std::max(0.0, dither_max_bbox);   // 0 disables the px gate
    }
    // Axial-only range EMA time constant (radar smoothness). <=0 disables (passthrough r_norm).
    void set_range_smooth_tc(double tc) { r_norm_smooth_tc_ = std::max(0.0, tc); }
    void set_apn_gain(double gain) {
        if (!std::isfinite(gain) || gain < 0)
            throw std::invalid_argument("controller.K_apn must be finite and nonnegative");
        K_apn_ = gain;
    }
    void set_vertical_braking(double deceleration, double delay) {
        if (!std::isfinite(deceleration) || deceleration < 0 ||
            !std::isfinite(delay) || delay < 0)
            throw std::invalid_argument("invalid vertical braking parameters");
        vertical_brake_accel_ = deceleration;
        vertical_brake_delay_ = delay;
    }
    void set_visual_speed_gain(double gain, double floor) {
        if (!std::isfinite(gain) || gain < 0 || !std::isfinite(floor) || floor < 0)
            throw std::invalid_argument("visual speed gain and floor must be finite and nonnegative");
        visual_speed_gain_ = gain;
        visual_speed_floor_ = floor;
    }
    void set_terminal_speed_floor(double floor, double bbox_threshold, double los_rate_threshold) {
        if (!std::isfinite(floor) || floor < 0 ||
            !std::isfinite(bbox_threshold) || bbox_threshold <= 0 ||
            !std::isfinite(los_rate_threshold) || los_rate_threshold < 0)
            throw std::invalid_argument("invalid terminal speed floor parameters");
        terminal_speed_floor_ = floor;
        terminal_speed_bbox_threshold_ = bbox_threshold;
        terminal_speed_los_rate_threshold_ = los_rate_threshold;
    }
    // Turn-radius cap factor: clamped to [0.3, 1.0] (1.0 = physics limit; <1 brakes sooner). The cap is
    // the maneuverability-based speed reducer (see compute_control). Out-of-range is WARNed at the
    // call site so a stale yaml "3.0 = disabled" is not silently swallowed.
    void set_turn_radius_cap_factor(double v) { turn_radius_cap_factor_ = clamp(v, 0.3, 1.0); }
    void set_axial_deadband(double v) { axial_deadband_ = std::max(0.0, v); }
    void set_motion_gate_factor(double v) { motion_gate_factor_ = std::max(0.0, v); }

    // Odometry-free control with optional coordinated terminal speed limits. Inputs:
    //   r_est, v_est : RELATIVE target pos/vel in world frame G (estimator state directly)
    //   los_az       : target azimuth in body heading frame (rad, 0 = dead ahead)
    //   los_el       : target elevation in world frame (rad, 0 = level, negative = below)
    //   v_fwd_body   : closing speed proxy = -v_rel·r̂ (positive = approaching); NOT odom
    //   bbox_size    : target bbox size in px (observable proximity cue)
    //   v_z          : ownship vertical speed proxy = -v_rel(2) (target-level approximation)
    //   target_speed : retained for call-site compatibility; terminal speed uses observed LOS instead.
    //   los_rate     : EMA-smoothed measured bearing rate (lambda_dot)
    //   a_own_world  : ownship world accel over the last camera window (IMU preintegration)
    //   t            : mission-relative time (s)
    std::vector<double> compute_control(double roll, double pitch, double yaw,
                                        const Eigen::Vector3d& r_est, const Eigen::Vector3d& v_est,
                                        double los_az, double los_el, double t,
                                        double v_fwd_body = 0.0, double bbox_size = 0.0,
                                        double v_z = 0.0, double target_speed = 0.0,
                                        double los_rate = 0.0,
                                        const Eigen::Vector3d& a_own_world = Eigen::Vector3d::Zero(),
                                        double boresight_el_meas = std::numeric_limits<double>::quiet_NaN(),
                                        bool radar_owns_guidance = false,
                                        bool radar_close_range = false) {
        (void)v_est;
        (void)target_speed;
        const double r_norm = r_est.norm();
        // Axial-only range EMA (radar smoothness): r_norm carries +-~4 m of 10 Hz radar noise; fed
        // 1:1 into the turn-radius cap (sqrt(g*tan*r)) it swings v_des_eff and brakes every cycle.
        // Smooth it ONLY for the axial chain (cap + terminal decel). Lag is conservative on closing.
        double r_norm_axial = r_norm;
        if (r_norm_smooth_tc_ > 1e-6) {
            if (has_rn_ema_ && t > last_rn_ema_t_ && (t - last_rn_ema_t_) < 1.0) {
                const double dt_rn = t - last_rn_ema_t_;
                const double a_rn = clamp(dt_rn / r_norm_smooth_tc_, 0.0, 1.0);
                r_norm_ema_ += a_rn * (r_norm - r_norm_ema_);
                r_norm_axial = r_norm_ema_;
            } else {
                r_norm_ema_ = r_norm;
                has_rn_ema_ = true;
            }
        }
        last_rn_ema_t_ = t;

        // ---- (A) Bearing-only lateral guidance -------------------------------------------
        // LOS rate from the measured bearing ONLY (clean, no 1/r blowup, no v_est noise).
        double lambda_dot = los_rate;
        // Deadband: a steady bearing (<0.02 rad/s) yields ZERO lateral lead. Stops residual bearing
        // noise around zero from sign-flipping a_lat.
        if (std::abs(lambda_dot) < 0.02) {
            lambda_dot = 0.0;
        }
        const double omega_los_mag = std::min(std::abs(lambda_dot), 1.0);
        double N_eff = N;
        if (r_norm > 15.0) {
            N_eff = N - (N - 2.0) * clamp((r_norm - 15.0) / 35.0, 0.0, 1.0);
        }
        // Speed-scheduled PN: use v_des (OPEN-LOOP) for the v in a_lat = N*v*lambda_dot, NOT the
        // measured v_closing. v_closing is under-observable along-LOS until the estimator anchors
        // scale; using it starves PN lead in the first seconds and lets a crossing target escape
        // the FOV. v_des is the intended ownship speed — the correct PN proxy — available from t=0.
        const double v_pn = std::min(v_des, v_pn_ref_);
        double a_lat_mag = N_eff * v_pn * omega_los_mag;
        // Close-range lateral attenuation (Fix D/F): gently scale a_lat_mag toward
        // lateral_decay_floor_ over [terminal_brake_range, lat_decay_start_] to damp the
        // close-approach PN weave. NOT a hard zero — the floor keeps PN lead for turning targets.
        const double lat_f = clamp((r_norm - terminal_brake_range_) /
                                   std::max(lat_decay_start_ - terminal_brake_range_, 1.0), 0.0, 1.0);
        a_lat_mag *= lateral_decay_floor_ + (1.0 - lateral_decay_floor_) * lat_f;
        // Lateral direction in world horizontal plane: perpendicular to LOS projected to xy.
        Eigen::Vector2d e_los_xy(r_est(0), r_est(1));
        if (e_los_xy.norm() < 1e-3) e_los_xy << 1.0, 0.0;
        e_los_xy.normalize();
        Eigen::Vector2d e_perp_xy(-e_los_xy(1), e_los_xy(0));  // 90 deg CCW
        const double turn_sign = (lambda_dot >= 0.0) ? 1.0 : -1.0;
        Eigen::Vector2d a_lat_world = turn_sign * a_lat_mag * e_perp_xy;
        // Active observability dither (mid-course only): lateral sinusoid for monocular scale/depth.
        // Suppressed while the radar owns guidance: the radar gives absolute scale directly, so the
        // monocular-observability job the dither does is redundant during radar-only flight, and the
        // 0.3 Hz lateral sinusoid only adds an airspeed oscillation. Re-engages at/after vision lock.
        // TWO additional independent close gates (a W-over-estimated range estimate keeps r_norm
        // above dither_min_range_ forever, so the serpentine used to persist into true close range
        // and cause misses):
        //  (a) bbox px: a large bbox = good size-SNR / close target — the dither's job is done. This
        //      is a DIRECT observable, immune to the monocular (W, range) scale error. Stale-hold
        //      the last live bbox for 0.5 s (~vision_hold) so a dropout frame cannot re-arm it.
        //  (b) radar range: a fresh radar track showing the target inside dither_radar_range — works
        //      after the vision lock too (the radar cache keeps updating; only fusion disengages).
        bool dither_suppressed_by_bbox = false;
        if (dither_max_bbox_ > 0.0) {
            double s_eff = bbox_size;
            if (s_eff <= 0.0 && last_bbox_t_ >= 0.0 && (t - last_bbox_t_) < 0.5) {
                s_eff = last_bbox_size_;
            }
            dither_suppressed_by_bbox = s_eff >= dither_max_bbox_;
        }
        if (bbox_size > 0.0) {
            last_bbox_size_ = bbox_size;
            last_bbox_t_ = t;
        }
        if (dither_amp_ > 0.0 && r_norm > dither_min_range_ && !radar_owns_guidance &&
            !dither_suppressed_by_bbox && !radar_close_range) {
            const double a_dither = dither_amp_ * std::sin(2.0 * M_PI * dither_freq_ * t);
            a_lat_world += a_dither * e_perp_xy;
        }
        // Geometric APN: classic PN lags a turning target. Recover the target lateral accel from
        // TRUE-PN geometry + IMU: a_t_perp = range*ydot + 2*lambda_dot*v_range + a_own_perp.
        // EMA a_own_perp with the SAME tc as ydot so the ownship-dither component cancels for a
        // static/CV target; EMA a_t_perp_geom so pixel-noise spikes average toward 0.
        if (has_prev_lr_ && t > prev_lr_t_) {
            const double dt_lr = t - prev_lr_t_;
            if (dt_lr > 1e-3 && dt_lr < 1.0) {
                const double ydot_raw = (los_rate - prev_los_rate_) / dt_lr;
                const double a_ydot = clamp(dt_lr / ydot_tc_, 0.0, 1.0);
                ydot_ema_ += a_ydot * (ydot_raw - ydot_ema_);
                const double a_ao = clamp(dt_lr / ydot_tc_, 0.0, 1.0);
                const double a_own_perp = a_own_world.head<2>().dot(e_perp_xy);
                aown_perp_ema_ += a_ao * (a_own_perp - aown_perp_ema_);
                last_los_dt_ = dt_lr;
            }
        }
        prev_los_rate_ = los_rate;
        prev_lr_t_ = t;
        has_prev_lr_ = true;
        const double a_t_perp_geom = r_norm * ydot_ema_ + 2.0 * los_rate * v_fwd_body + aown_perp_ema_;
        {
            const double a_atp = clamp(last_los_dt_ / atp_tc_, 0.0, 1.0);
            atp_ema_ += a_atp * (a_t_perp_geom - atp_ema_);
        }
        if (std::abs(a_t_perp_geom) > K_apn_min_at_) {
            a_lat_world += K_apn_ * a_t_perp_geom * e_perp_xy;
        }

        // ---- (A) Axial guidance: cruise ceiling with maneuverability and visual proximity caps ----
        double v_des_eff = v_des;
        // Turn-radius maneuverability cap: v ≤ factor·sqrt(g·tan(tilt)·r). Uses the range-appropriate
        // bank (terminal tilt inside terminal_tilt_range) so the raised terminal bank translates into
        // sustained close-range speed AND tighter turns.
        const double tilt_for_cap = (r_norm_axial < terminal_tilt_range_) ? max_tilt_terminal_rad_ : max_tilt_rad;
        v_des_eff = std::min(v_des_eff, turn_radius_cap_factor_ * std::sqrt(g * std::tan(tilt_for_cap) * r_norm_axial));
        // The raw image cue can brake before an overestimated monocular range catches up.
        if (visual_speed_gain_ > 0 && bbox_size > 0)
            v_des_eff = std::min(v_des_eff,
                std::max(visual_speed_floor_, visual_speed_gain_ / bbox_size));
        // Once the target is large and the measured LOS is stable, preserve a useful terminal
        // closing speed. This is driven by current observations and is therefore shared by
        // stationary and moving-target missions. A large LOS rate keeps the maneuverability cap.
        v_des_eff = terminal_speed_floor_cap(v_des, v_des_eff, terminal_speed_floor_,
                                             bbox_size, terminal_speed_bbox_threshold_,
                                             los_rate, terminal_speed_los_rate_threshold_);
        // Vertical-reachability axial cap — the dual of the terminal arrival law in section (B).
        // The arrival law DEMANDS the climb that nulls dz by arrival, but when the realized
        // vertical rate cannot follow (thrust shared with 14-17 m/s of axial drag leaves
        // ~1.5-3 m/s of vertical-rate tracking lag), the demand alone still passes ~1-2 m low.
        // Inside the arrival envelope, ALSO slow the axial closure to what the REALIZED
        // vertical rate can match, so both channels converge dz/r_xy onto the arrival
        // geometry instead of the axial channel outrunning the vertical one. Floored at a
        // maneuvering closure (target lateral motion), dies out as dz -> 0, and never engages
        // outside the envelope (launch dash, mid-course) or when the height error is small.
        if (terminal_arrival_margin_ > 0.0 && terminal_arrival_range_ > 0.0 &&
            r_norm < terminal_arrival_range_ && std::abs(r_est(2)) > 0.5) {
            const double r_xy_arr = r_est.head<2>().norm();
            if (r_xy_arr > 0.5) {
                // margin 1.15 with the envelope at 35 m: the early engagement manages the
                // closure down to ~10 m/s before the terminal, so 1.15 no longer lets a
                // 15-17 m/s far-side dash outrun the climb (that failure was with the
                // envelope at 20 m); 1.0 starved the endgame closure and left ~0.85 m of
                // horizontal shortfall at the pass.
                const double v_axial_cap = axial_speed_for_arrival(
                    std::max(std::abs(v_z), 2.0), r_xy_arr, r_est(2), 1.15);
                v_des_eff = std::max(std::min(v_des_eff, v_axial_cap), 4.0);
            }
        }
        // v_fwd_body (= v_closing from v_rel) is under-observable along-LOS until scale converges;
        // a low v_closing inflates a_fwd -> ownship over-accelerates ON-AXIS, starving lateral PN
        // lead. Scene-adaptive 3-level floor: turning target (|atp_ema| large) -> 0.88 (don't blast
        // through a circle); laterally-moving target (v_perp > 1) -> 0.82; static/slow -> 0.8.
        // Braking (v_closing above the floor) passes through unchanged.
        const double v_perp = std::abs(r_norm * los_rate);
        const double v_floor_frac = (std::abs(atp_ema_) > 0.5) ? 0.88 : ((v_perp > 1.0) ? 0.82 : 0.8);
        const double v_fwd_for_afwd = std::max(v_fwd_body, v_floor_frac * v_des_eff);
        // Axial deadband: coast (a_fwd=0) within ±axial_deadband_ of v_des_eff. Kills the pitch
        // oscillation from radar v_closing noise (±1 m/s) during blind flight — the interceptor
        // cruises at v_des±deadband (all plenty for a kinetic-energy intercept) instead of chasing
        // every noise sample. Outside the band K_fwd applies, shifted by the deadband (no step).
        const double v_error = v_des_eff - v_fwd_for_afwd;
        double a_fwd;
        if (std::abs(v_error) <= axial_deadband_) {
            a_fwd = 0.0;
        } else {
            a_fwd = K_fwd_ * (v_error - std::copysign(axial_deadband_, v_error));
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

        // ---- (B) Decoupled vertical: cascaded velocity-limited altitude track ------------
        const double dz_target = r_est(2);
        // Command a vertical SPEED toward the target altitude (capped), then accelerate to track it.
        // (1) the cap prevents runaway dive/overshoot; (2) cap saturation makes a biased monocular
        // dz (wrong target width) NOT inflate the descent — robust to wrong_w. Damping rides on the
        // IMU own vertical velocity. v_climb/v_descent (m/s, max commanded climb/sink rate) and
        // K_dz_up/dn (1/s, responsiveness; closure tau = 1/K) are INDEPENDENT config params
        // (set_vertical_climb). CLIMB uses a higher gain than DESCENT so an upper-FOV target does not
        // asymptotically stall ~1 m short, while a below-FOV target still brakes without over-dive.
        // Set v_climb/v_descent to the airframe's real climb/sink capability so the commanded rate
        // is achievable (else lag -> stall).
        // K_az (vertical-speed tracking gain) and a_z_limit (vertical accel clamp) are config-driven
        // members (set_guidance_tuning); see the controller params in the yaml.
        double v_z_desired;
        if (dz_target > 0.0) {
            v_z_desired = clamp(dz_target * K_dz_up_, 0.0, v_climb_);
        } else {
            v_z_desired = clamp(dz_target * K_dz_dn_, -v_descent_, 0.0);
        }
        if (visual_speed_gain_ > 0 && bbox_size > 0) {
            // Axial image braking must also slow the climb, or the vehicle reaches the
            // target altitude early and passes above it while still closing horizontally.
            const double closure_limit = vertical_closure_speed_limit(
                v_des_eff, r_est.head<2>().norm(), dz_target);
            v_z_desired = clamp(v_z_desired, -closure_limit, closure_limit);
        }
        // Terminal arrival margin: the two laws above converge the climb gently ONTO the
        // simultaneous-arrival profile — the proportional term shrinks with the residual error
        // and the profile cap carries no margin for the vertical tracking lag (K_az, a_z_limit,
        // thrust shared with the axial channel). A target engaged from below is therefore
        // arrived at with roughly the vertical error it had a couple of loop time-constants out
        // (observed in AirSim: a systematic ~1 m low pass at 12 m/s closure). Inside the
        // terminal envelope DEMAND at least margin x dz/t_go so the residual is removed by
        // arrival; margin > 1 leaves headroom for the tracking lag. Bounded by v_climb/v_descent
        // here and by a_z_limit downstream. Gated on measured closure, and on the range window
        // where the monocular estimate is reliable.
        if (terminal_arrival_margin_ > 0.0 && terminal_arrival_range_ > 0.0 &&
            r_norm < terminal_arrival_range_ && v_fwd_body > 1.0) {
            const double v_z_arrival = terminal_arrival_vertical_speed(
                dz_target, r_norm, v_fwd_body, terminal_arrival_margin_);
            if (dz_target > 0.0)
                v_z_desired = clamp(std::max(v_z_desired, v_z_arrival), 0.0, v_climb_);
            else if (dz_target < 0.0)
                v_z_desired = clamp(std::min(v_z_desired, v_z_arrival), -v_descent_, 0.0);
        }
        if (vertical_brake_accel_ > 0) {
            const double speed_limit = braking_speed_limit(std::abs(dz_target),
                                                           vertical_brake_accel_, vertical_brake_delay_);
            v_z_desired = clamp(v_z_desired, -speed_limit, speed_limit);
        }
        double a_z = clamp(K_az_ * (v_z_desired - v_z), -a_z_limit_, a_z_limit_);
        // Slew limit |da_z/dt|: the z-estimate can jump O(10 m) in one cycle; ramp a_z instead of
        // stepping it (which would pitch the body over).
        if (has_last_az_ && t > last_az_t_) {
            const double dt_az = t - last_az_t_;
            if (dt_az > 0.0 && dt_az < 1.0) {
                a_z = clamp(a_z, last_a_z_ - a_z_slew_ * dt_az,
                            last_a_z_ + a_z_slew_ * dt_az);
            }
        }
        last_a_z_ = a_z;
        last_az_t_ = t;
        has_last_az_ = true;

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

        const bool target_behind = std::abs(los_az) > (M_PI / 2.0);
        if (enable_fov_clamp_ && has_gimbal_ && !target_behind) {
            // Semi-strapdown gimbal: the boresight is decoupled from body pitch (gimbal cancels
            // it; elev ~= -gimbal_bias_), so a theta clamp cannot keep the target in the vertical
            // FOV — ALTITUDE can. If the target approaches the bottom edge (los_el well below the
            // boresight), command a descent so the frame re-centers; symmetric for the top edge.
            // Overrides the PD a_z only near the edge (outer 30% of the half-FOV). Mirrors the
            // verified Python implementation (sim_test_relative_imm_ukf->py).
            const double fov_keep = fov_half_v_ * fov_keep_margin_;
            // Boresight world elevation: prefer the ACTUAL camera orientation measured by the gimbal
            // (q_C_W) — this tracks the true boresight INCLUDING servo lag, so the vertical-FOV clamp
            // reacts to where the camera really points during pitch transients (closer to real HW).
            // Fall back to the constant -gimbal_bias_ assumption when no measurement is available.
            const double boresight_el = std::isfinite(boresight_el_meas)
                ? boresight_el_meas : -gimbal_bias_;
            const double el_err = los_el - boresight_el;   // negative = target below boresight
            const double edge = 0.7 * fov_keep;
            if (el_err < -edge) {
                a_z = std::min(a_z, -8.0 * clamp((-el_err - edge) /
                                                 std::max(fov_keep - edge, 1e-3), 0.0, 1.0));
            } else if (el_err > edge) {
                a_z = std::max(a_z, 8.0 * clamp((el_err - edge) /
                                                std::max(fov_keep - edge, 1e-3), 0.0, 1.0));
            }
        } else if (enable_fov_clamp_ && !target_behind && r_norm < 15.0) {
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
        // Throttle from the ACTUAL attitude, not the commanded tilt: during a hard brake the
        // attitude lags theta_cmd, and (a_z+g)/cos(commanded tilt) then over-thrusts vertically ->
        // brake-climb -> target exits the bottom of the (gimbal-horizontal) frame at 5-7 m.
        // T*cos(actual tilt) = a_z + g  =>  vertical accel tracks a_z regardless of the attitude
        // lag. Floor the divisor slightly beyond max_tilt so a transient over-bank cannot blow
        // the throttle up. Mirrors the verified Python implementation.
        const double cos_tilt_actual = std::max(std::cos(roll) * std::cos(pitch),
                                                std::cos(max_tilt_rad + 0.2));
        const double throttle = clamp((a_z + g) / cos_tilt_actual, 2.0, 25.0);

        // ---- Attitude rate commands (P on attitude error) -------------------------------
        double omega_x = clamp(K_att * (phi_cmd - roll), -max_body_rate_, max_body_rate_);
        double omega_y = clamp(K_att * (theta_cmd - pitch), -max_body_rate_, max_body_rate_);
        double omega_z = clamp(K_yaw * los_az, -max_body_rate_, max_body_rate_);
        return {throttle, omega_x, omega_y, omega_z};
    }
};
