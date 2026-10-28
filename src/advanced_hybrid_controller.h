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

// ============================================================================
// Attitude-geometry helpers (shared by the controller and the node fallbacks).
// Quaternion/two-channel attitude interface: valid through +-90 deg of body pitch,
// where the former Euler (theta, phi, yaw) interface was singular and its no-wrap
// P-laws produced full-authority transients on representation flips.
// ============================================================================

// Wrap an angle to (-pi, pi].
inline double wrapAngle(double a)
{
    return std::atan2(std::sin(a), std::cos(a));
}

// Two-channel TILT rates: rotate the thrust axis z_meas onto z_cmd about the axis perpendicular
// to both, carrying no heading (rotated-camera-plane) component. At low pitch this is the former
// Euler roll/pitch P-law pair; at high pitch it stays decoupled from the heading channel.
// Non-singular at any attitude.
inline Eigen::Vector3d tiltChannelRates(const Eigen::Matrix3d& R_b2n, const Eigen::Vector3d& z_cmd,
                                        double k_tilt, double max_rate)
{
    const Eigen::Vector3d z_meas = R_b2n.col(2);
    const Eigen::Vector3d cr = z_meas.cross(z_cmd);
    const double crn = cr.norm();
    Eigen::Vector3d omega_world = Eigen::Vector3d::Zero();
    if (crn > 1e-9) {
        omega_world = (k_tilt * std::atan2(crn, z_meas.dot(z_cmd))) * (cr / crn);
    }
    const Eigen::Vector3d omega_body = R_b2n.transpose() * omega_world;
    return Eigen::Vector3d(clamp(omega_body.x(), -max_rate, max_rate),
                           clamp(omega_body.y(), -max_rate, max_rate),
                           clamp(omega_body.z(), -max_rate, max_rate));
}

// Camera boresight azimuth in the world frame. Prefers the actual gimbal pose (R_c2n:
// camera->world rotation; boresight = camera +Z) — non-singular at any body pitch, since the
// semi-strapdown pitch gimbal keeps the boresight near horizontal. Without a gimbal, falls back
// to the body forward axis, then to the thrust axis as the forward axis goes vertical (the
// pitch-90 degeneracy of the Euler heading).
inline double boresightAzimuth(const Eigen::Matrix3d& R_b2n, const Eigen::Matrix3d* R_c2n)
{
    if (R_c2n != nullptr) {
        const Eigen::Vector3d b = R_c2n->col(2);
        if (Eigen::Vector2d(b.x(), b.y()).norm() > 0.1) {
            return std::atan2(b.y(), b.x());
        }
    }
    const Eigen::Vector3d bx = R_b2n.col(0);
    if (Eigen::Vector2d(bx.x(), bx.y()).norm() > 0.2) {
        return std::atan2(bx.y(), bx.x());
    }
    const Eigen::Vector3d bz = R_b2n.col(2);
    return std::atan2(bz.y(), bz.x());
}

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
    double max_att_rate_ = 1e9;   // max |d(tilt_mag)/dt|, |d(tilt_az)/dt| [rad/s] (slew governor)
    double last_tilt_mag_ = 0.0;  // governor state (dedicated; axial block reuses last_t_/has_last_)
    Eigen::Vector2d last_az_ = Eigen::Vector2d(1.0, 0.0);
    double last_att_t_ = -1.0;
    bool   has_last_att_ = false;
    // Attitude-reference EMA low-pass (suppresses camera-rate content / roll limit-cycle).
    double att_ref_tc_ = 0.15;    // EMA time constant on (tilt_mag, tilt_az) [s]
    double last_tilt_mag_f_ = 0.0;  // EMA-filtered reference state
    Eigen::Vector2d last_az_f_ = Eigen::Vector2d(1.0, 0.0);
    // sin(pitch)-keyed lateral-authority attenuation (a_lat -> 0 over the window): the single-axis
    // pitch gimbal cannot compensate roll, so banking swings the boresight azimuth by ~sin(pitch)
    // per radian of roll — lateral PN increasingly costs pointing as the nose-down pitch deepens.
    // Dormant in the validated <=68 deg flight envelope; authoritative pointing at the bench/missile
    // 90 deg regime, where the bearing rate cannot self-extinguish without ownship translation.
    double a_lat_tilt_start_ = 65.0 * M_PI / 180.0;   // window start, in sin(pitch) units
    double a_lat_tilt_end_ = 85.0 * M_PI / 180.0;     // ... end (a_lat -> 0)
    // Max body NOSE-UP pitch the semi-strapdown gimbal can compensate (its up-travel minus the
    // down-gaze bias): braking harder nose-up than this saturates the servo and drops the
    // boresight below the target. 0 = unlimited.
    double gimbal_nose_up_max_ = 35.0 * M_PI / 180.0;
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
    // APN range gate: a_t_perp = r*ydot + ... scales the LOS-acceleration NOISE by r, so at
    // long range (80-90 m standoff) a converging monocular estimate injects spurious lateral
    // accel and banks the vehicle early (observed: camera boresight error to 80 deg within
    // 3 s of launch, mission camera-guard abort). APN is a TERMINAL lead term — apply it only
    // inside this radius where the lead matters and the noise is bounded. 0 disables the gate.
    double K_apn_max_range_ = 40.0;    // m
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
        last_tilt_mag_ = 0.0;
        last_az_ = Eigen::Vector2d(1.0, 0.0);
        last_att_t_ = -1.0;
        has_last_att_ = false;
        last_tilt_mag_f_ = 0.0;
        last_az_f_ = Eigen::Vector2d(1.0, 0.0);
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
    void set_apn_gain(double gain, double max_range) {
        if (!std::isfinite(gain) || gain < 0 || !std::isfinite(max_range) || max_range < 0)
            throw std::invalid_argument("controller.K_apn must be finite and nonnegative");
        K_apn_ = gain;
        K_apn_max_range_ = max_range;
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
    void set_lateral_tilt_attenuation(double start_deg, double end_deg) {
        a_lat_tilt_start_ = std::max(0.0, start_deg) * M_PI / 180.0;
        a_lat_tilt_end_ = std::max(a_lat_tilt_start_ + 1e-3, end_deg) * M_PI / 180.0;
    }
    void set_gimbal_nose_up_max(double deg) { gimbal_nose_up_max_ = std::max(0.0, deg) * M_PI / 180.0; }

    // Odometry-free control with optional coordinated terminal speed limits. Inputs:
    //   R_b2n        : measured body->world attitude (rotation matrix). The attitude interface is
    //                  quaternion/two-channel — no Euler extraction, valid through +-90 deg of
    //                  body pitch (bench rig operating point / missile-mode dives), where the
    //                  former (roll, pitch, yaw) Euler interface was singular.
    //   az_bore      : measured camera boresight azimuth in the world frame (rad; from the gimbal
    //                  pose when available — non-singular heading reference at any body pitch).
    //   los_az_err   : bearing error relative to the boresight, wrap(az_los - az_bore) (rad).
    //   r_est, v_est : RELATIVE target pos/vel in world frame G (estimator state directly)
    //   los_el       : target elevation in world frame (rad, 0 = level, negative = below)
    //   v_fwd_body   : closing speed proxy = -v_rel·r̂ (positive = approaching); NOT odom
    //   bbox_size    : target bbox size in px (observable proximity cue)
    //   v_z          : ownship vertical speed proxy = -v_rel(2) (target-level approximation)
    //   target_speed : retained for call-site compatibility; terminal speed uses observed LOS instead.
    //   los_rate     : EMA-smoothed measured bearing rate (lambda_dot)
    //   a_own_world  : ownship world accel over the last camera window (IMU preintegration)
    //   t            : mission-relative time (s)
    std::vector<double> compute_control(const Eigen::Matrix3d& R_b2n, double az_bore, double los_az_err,
                                        const Eigen::Vector3d& r_est, const Eigen::Vector3d& v_est,
                                        double los_el, double t,
                                        double v_fwd_body = 0.0, double bbox_size = 0.0,
                                        double v_z = 0.0, double target_speed = 0.0,
                                        double los_rate = 0.0,
                                        const Eigen::Vector3d& a_own_world = Eigen::Vector3d::Zero(),
                                        double boresight_el_meas = std::numeric_limits<double>::quiet_NaN(),
                                        bool radar_owns_guidance = false,
                                        bool radar_close_range = false) {
        (void)v_est;
        (void)target_speed;
        const double az_los = wrapAngle(az_bore + los_az_err);
        // sin(pitch) is the roll->boresight-azimuth coupling factor (a roll of phi at nose-down
        // pitch theta swings the boresight azimuth by ~sin(theta)*phi): 0 at level (roll is pure
        // image roll), ~1 at the 90 deg dive. Used by the lateral attenuation and the
        // thrust-azimuth budget below. R(2,0) is the world-z component of the body forward axis
        // (negative when nose-down), so sin(pitch) = -R(2,0) with nose-down positive.
        const double sin_pitch = clamp(-R_b2n(2, 0), -1.0, 1.0);
        // Lateral-authority tilt attenuation: banking costs pointing increasingly with nose-down
        // pitch (the single-axis pitch gimbal cannot compensate roll). In free flight PN
        // self-extinguishes (ownship translation converges the bearing) before deep pitch, so
        // this is dormant in the validated <=68 deg envelope; on a translation-locked bench the
        // bearing rate NEVER decays and full lateral authority destabilizes pointing.
        const double att_lat = clamp((std::sin(a_lat_tilt_end_) - sin_pitch) /
                                      std::max(std::sin(a_lat_tilt_end_) - std::sin(a_lat_tilt_start_), 1e-6),
                                      0.0, 1.0);
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
        a_lat_mag *= att_lat;
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
        if (K_apn_max_range_ <= 0.0 || r_norm < K_apn_max_range_) {
            if (std::abs(a_t_perp_geom) > K_apn_min_at_) {
                a_lat_world += K_apn_ * att_lat * a_t_perp_geom * e_perp_xy;
            }
        }

        // ---- (A) Axial guidance: cruise ceiling with maneuverability and visual proximity caps ----
        double v_des_eff = v_des;
        // Turn-radius maneuverability cap: v ≤ factor·sqrt(g·tan(tilt)·r). Uses the range-appropriate
        // bank (terminal tilt inside terminal_tilt_range) so the raised terminal bank translates into
        // sustained close-range speed AND tighter turns.
        const double tilt_for_cap = (r_norm_axial < terminal_tilt_range_) ? max_tilt_terminal_rad_ : max_tilt_rad;
        // tan() is negative past 90 deg of tilt (sqrt of a negative): clamp the cap tilt into the
        // valid turn-radius domain — past 90 deg of bank the "turn radius" concept has already
        // conceded all maneuverability anyway.
        const double tilt_for_cap_valid = std::min(tilt_for_cap, 89.0 * M_PI / 180.0);
        v_des_eff = std::min(v_des_eff, turn_radius_cap_factor_ * std::sqrt(g * std::tan(tilt_for_cap_valid) * r_norm_axial));
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

        // ---- (C) Thrust-vector construction (quaternion / two-channel) -------------------
        // The desired attitude is built from the world-frame specific-force direction
        // d = [a_h, a_z+g] directly: body Z aligns with d, and the tilt is expressed in the
        // BORESIGHT-azimuth frame (tilt_fwd, tilt_left) — the exact former theta_cmd / -phi_cmd
        // at low tilt, still well-defined to +-180 deg. This replaces the Euler theta/phi
        // clamps + Euler P-laws, whose representation singularity at pitch +-90 deg and no-wrap
        // P-laws made the >65 deg regime (bench rig operating point, missile-mode dives)
        // unusable.
        const bool target_behind = std::abs(los_az_err) > (M_PI / 2.0);
        Eigen::Vector3d d(a_h_world(0), a_h_world(1), a_z + g);
        // Tilt cap (flight envelope): cap the tilt of d from vertical, preserving the tilt
        // azimuth; valid beyond 90 deg (cap > 90 lets the thrust axis pass horizontal for
        // missile-mode dives). Terminal range relaxes the cap (the old max_tilt_terminal roll
        // relaxation).
        const double tilt_cap = (r_norm < terminal_tilt_range_) ? max_tilt_terminal_rad_ : max_tilt_rad;
        const double a_h_mag = a_h_world.norm();
        if (a_h_mag > 1e-9) {
            const double tilt_raw = std::atan2(a_h_mag, d(2));
            // Thrust-azimuth pointing budget: a thrust-azimuth deviation delta from the LOS
            // swings the boresight azimuth by ~sin(pitch)*delta (the roll coupling above), so
            // the affordable deviation is the remaining keep-in-frame margin over that coupling.
            // At level pitch sin->0 and the budget is unbounded (flight banking untouched); at
            // the 90 deg dive the boresight ~ the thrust axis and the budget collapses onto the
            // LOS — exactly where pointing must be authoritative (bench: the bearing rate never
            // self-extinguishes, and a wandering thrust azimuth walks the target out of frame).
            const double az_ah = std::atan2(a_h_world(1), a_h_world(0));
            const double d_az = wrapAngle(az_ah - az_los);
            // Engage only in the deep-pitch regime (sin_pitch > 0.75, ~49 deg nose-down): with
            // the two-channel decomposition a STEADY bank does not rotate the camera plane (the
            // tilt channel's rotation is in-plane below the degeneracy), so the boresight cost of
            // a bank is transient only — clamping at moderate pitch just starves terminal lateral
            // authority. In the deep-pitch regime the tilt channel's axis rotates toward the
            // vertical and the boresight ~ the thrust axis, where the budget keeps pointing
            // authoritative (bench rig, missile-mode dives).
            const double fov_az_budget = fov_half_h_ * fov_keep_margin_ - std::abs(los_az_err);
            // clamp to [0, pi]: a NEGATIVE budget (error already past the keep-in-frame margin)
            // must PIN the thrust azimuth onto the LOS (d_az_max -> 0), not unbind it — at deep
            // pitch the boresight ~ the thrust axis, and an unbound PN azimuth makes the tilt
            // and heading channels fight over the same DOF (observed: a stuck compromise with
            // ~60 deg pointing error and ~100 deg residual roll on the rig).
            const double d_az_max = (sin_pitch > 0.75)
                ? clamp(fov_az_budget / std::max(sin_pitch, 0.75), 0.0, M_PI) : M_PI;
            Eigen::Vector2d a_h_eff = a_h_world;
            if (std::abs(d_az) > d_az_max) {
                const double d_az_c = clamp(d_az, -d_az_max, d_az_max);
                a_h_eff = a_h_mag * Eigen::Vector2d(std::cos(az_los + d_az_c), std::sin(az_los + d_az_c));
                d(0) = a_h_eff(0);
                d(1) = a_h_eff(1);
            }
            if (tilt_raw > tilt_cap) {
                const Eigen::Vector2d s_xy = a_h_eff / a_h_mag;
                d = Eigen::Vector3d(std::sin(tilt_cap) * s_xy(0), std::sin(tilt_cap) * s_xy(1),
                                    std::cos(tilt_cap));
            }
        }
        // Spherical tilt parametrization of the desired thrust direction, in the
        // boresight-azimuth frame: (tilt_mag in [0, pi], tilt_az). Valid over the WHOLE sphere —
        // a (tilt_fwd, tilt_left) tangent parametrization z = z_hat + fwd*tan(tf) + left*tan(tl)
        // silently INVERTS the direction past 90 deg of tilt (tan flips sign), which a >90 deg
        // missile-dive command turned into a full-backward attitude flip.
        const Eigen::Vector2d fwd_h(std::cos(az_bore), std::sin(az_bore));
        const Eigen::Vector2d left_h(-std::sin(az_bore), std::cos(az_bore));
        const Eigen::Vector3d left_h3(-std::sin(az_bore), std::cos(az_bore), 0.0);
        // Rotate d about the horizontal left axis (exact at any tilt) — the pitch-plane clamp
        // operator for the nose-up cap and the fixed-mount FOV window.
        auto rotate_about_left = [&left_h3](const Eigen::Vector3d& v, double angle) {
            return v * std::cos(angle) + left_h3.cross(v) * std::sin(angle)
                 + left_h3 * left_h3.dot(v) * (1.0 - std::cos(angle));
        };
        // fwd-plane tilt (the former theta_cmd).
        double tf = std::atan2(d.head<2>().dot(fwd_h), d(2));
        // Nose-up cap from the gimbal's up-travel: the semi-strapdown servo (cmd = -pitch + bias)
        // saturates at its mechanical max; braking harder nose-up than that drops the boresight
        // below the target.
        if (has_gimbal_ && gimbal_nose_up_max_ > 0.0) {
            const double tf_min = -(gimbal_nose_up_max_ - gimbal_bias_);
            if (tf < tf_min) {
                d = rotate_about_left(d, tf_min - tf);
                tf = tf_min;
            }
        }

        if (enable_fov_clamp_ && has_gimbal_ && !target_behind) {
            // Semi-strapdown gimbal: the boresight is decoupled from body pitch (gimbal cancels
            // it; elev ~= -gimbal_bias_), so a tilt clamp cannot keep the target in the vertical
            // FOV — ALTITUDE can. If the target approaches the bottom edge (los_el well below the
            // boresight), command a descent so the frame re-centers; symmetric for the top edge.
            // Overrides the PD a_z only near the edge (outer 30% of the half-FOV). Affects only
            // the throttle below (the tilt was built from the pre-override a_z, as before).
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
            // tf_hi (nose-down): kept conservative (fov_hard - los_el - mount_pitch). The
            // geometrically correct form (mount_pitch + fov_hard - los_el) allows large nose-down
            // and was tested -> aggressive dive on low targets -> image smear -> overshoot.
            const double theta_hi = std::min(max_tilt_rad, fov_hard - los_el - mount_pitch_);
            // tf_lo (nose-up): CORRECT form to keep the target inside the vertical FOV.
            // boresight elev = mount_pitch - theta, so boresight <= los_el + fov_keep =>
            // theta >= mount_pitch - fov_keep - los_el. fov_keep leaves headroom so attitude
            // errors don't re-lose it.
            const double fov_keep_v = fov_half_v_ * fov_keep_margin_;
            const double theta_lo = std::max(-max_tilt_rad, mount_pitch_ - fov_keep_v - los_el);
            const double tf_lo = std::min(theta_lo, theta_hi);
            const double tf_hi2 = std::max(theta_lo, theta_hi);
            if (tf < tf_lo) {
                d = rotate_about_left(d, tf_lo - tf);
                tf = tf_lo;
            } else if (tf > tf_hi2) {
                d = rotate_about_left(d, tf_hi2 - tf);
                tf = tf_hi2;
            }
        }

        // Recompute the spherical parametrization after the clamps.
        const double d_h_mag = d.head<2>().norm();
        double tilt_mag = std::atan2(d_h_mag, d(2));      // 0..pi (past 90 deg = dive)
        Eigen::Vector2d s_az = (d_h_mag > 1e-9) ? (d.head<2>() / d_h_mag) : fwd_h;

        // ---- Attitude reference EMA low-pass ------------------------------------------------
        // Suppress camera-rate content / roll limit-cycle before the rate law. An EMA (vs the slew
        // governor below) is the right tool for a steady oscillation: it attenuates high-freq tilt
        // swings while preserving the slow PN turn bias. Applied to the tilt magnitude and the
        // tilt azimuth (ANGULAR EMA — a vector lerp passes through zero on a bank reversal and
        // snaps the direction). Distinct from the body-rate decay coeff ctrl_coeff_; this filters
        // the attitude REFERENCE, not the rate output.
        if (has_last_att_ && att_ref_tc_ > 1e-6) {
            const double dt = t - last_att_t_;
            if (dt > 0.0 && dt < 1.0) {
                const double a_ref = clamp(dt / att_ref_tc_, 0.0, 1.0);
                tilt_mag = last_tilt_mag_f_ + a_ref * (tilt_mag - last_tilt_mag_f_);
                const double az_now = std::atan2(s_az(1), s_az(0));
                const double az_last = std::atan2(last_az_f_(1), last_az_f_(0));
                const double az_f = az_last + a_ref * wrapAngle(az_now - az_last);
                s_az = Eigen::Vector2d(std::cos(az_f), std::sin(az_f));
            }
        }
        last_tilt_mag_f_ = tilt_mag;
        last_az_f_ = s_az;

        // ---- Attitude reference slew governor ----------------------------------------------
        // Bound |d(tilt_mag)/dt| and |d(az)/dt| so the desired thrust direction cannot flip
        // +-max_tilt in a single cycle. That flip is what bang-bangs the tilt channel into
        // +-max_body_rate saturation and drives the roll limit-cycle at high v_des. Mirrors the
        // axial_jerk_max_ slew limit above; uses dedicated state because the axial block already
        // overwrote last_t_/has_last_ for this cycle. Does NOT change steady-state authority.
        if (has_last_att_) {
            const double dt = t - last_att_t_;
            if (dt > 0.0 && dt < 1.0) {
                const double max_dtilt = max_att_rate_ * dt;
                tilt_mag = clamp(tilt_mag, last_tilt_mag_ - max_dtilt, last_tilt_mag_ + max_dtilt);
                const double az_now = std::atan2(s_az(1), s_az(0));
                const double az_last = std::atan2(last_az_(1), last_az_(0));
                const double d_az_s = clamp(wrapAngle(az_now - az_last), -max_dtilt, max_dtilt);
                const double az_new = az_last + d_az_s;
                s_az = Eigen::Vector2d(std::cos(az_new), std::sin(az_new));
            }
        }
        last_tilt_mag_ = tilt_mag;
        last_az_ = s_az;
        last_att_t_     = t;
        has_last_att_   = true;

        // ---- (C) Rate commands: TWO-CHANNEL decomposition ---------------------------------
        //   heading channel: rotation about the world vertical — moves the camera plane (and so
        //     the gimbal-compensated boresight) onto the LOS azimuth; never disturbs the thrust
        //     axis' tilt geometry. Gain K_yaw on the boresight-relative bearing error.
        //   tilt channel: rotation about the axis perpendicular to the boresight plane taking
        //     z_meas onto z_cmd (gain K_att) — tilts the thrust WITHIN the plane; never rotates
        //     the plane, so it never moves the boresight (below the 90 deg degeneracy where the
        //     two merge). At low pitch this reproduces the former Euler P-laws term by term
        //     (omega_x ~ K_att*roll err, omega_y ~ K_att*pitch err, omega_z ~ K_yaw*los_az).
        Eigen::Vector3d z_cmd(std::sin(tilt_mag) * s_az(0), std::sin(tilt_mag) * s_az(1),
                              std::cos(tilt_mag));
        const Eigen::Vector3d z_meas = R_b2n.col(2);
        const Eigen::Vector3d cr = z_meas.cross(z_cmd);
        const double crn = cr.norm();
        Eigen::Vector3d omega_world = Eigen::Vector3d::Zero();
        if (crn > 1e-9) {
            omega_world = (K_att * std::atan2(crn, z_meas.dot(z_cmd))) * (cr / crn);
        }
        omega_world(2) += K_yaw * los_az_err;
        const Eigen::Vector3d omega_body = R_b2n.transpose() * omega_world;

        // ---- (B) Thrust: magnitude to realize the vertical component + tilt overhead --------
        // Throttle from the ACTUAL attitude, not the commanded tilt: during a hard brake the
        // attitude lags the command, and (a_z+g)/cos(commanded tilt) then over-thrusts
        // vertically -> brake-climb -> target exits the bottom of the (gimbal-horizontal) frame
        // at 5-7 m. T*cos(actual tilt) = a_z + g  =>  vertical accel tracks a_z regardless of
        // the attitude lag. R(2,2) = cos(actual tilt) with no Euler extraction; the floor
        // follows the tilt cap (cos(cap+0.2)) so an over-tilted body cannot blow the throttle
        // up, floored at 0.25 for caps beyond 90 deg.
        const double cos_tilt_actual = std::max(R_b2n(2, 2),
                                                std::max(std::cos(max_tilt_rad + 0.2), 0.25));
        const double throttle = clamp((a_z + g) / cos_tilt_actual, 2.0, 25.0);

        return {throttle,
                clamp(omega_body(0), -max_body_rate_, max_body_rate_),
                clamp(omega_body(1), -max_body_rate_, max_body_rate_),
                clamp(omega_body(2), -max_body_rate_, max_body_rate_)};
    }
};
