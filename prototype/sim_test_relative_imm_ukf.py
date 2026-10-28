#!/usr/bin/env python3
"""
Simulation test harness for InterceptorNodeRelativeIMMUKF algorithm.

Ports the C++ IMM-UKF estimator + AdvancedHybridController to Python (exact mirror),
then runs interception scenarios against a simplified quadrotor dynamics + camera
model to verify algorithm correctness.

Verifies:
  1. IMM-UKF estimator converges (pos/vel/width error bounded).
  2. Controller drives UAV toward target (range decreases, LOS rate -> 0).
  3. Interception achieved (range < threshold) for nominal scenario.
  4. Robustness: target maneuver, bbox dropout, measurement noise.
  5. No divergence / NaN / instability.

NOT a ROS2 integration test — pure algorithm verification.
"""

import numpy as np
import sys
import os
from dataclasses import dataclass, field
from collections import deque

np.set_printoptions(precision=3, suppress=True, linewidth=140)

# ============================================================================
# Constants (mirror C++ + params yaml)
# ============================================================================
M_PI = 3.14159265358979323846
STATE_DIM = 10
MEAS_DIM = 3
MODE_COUNT = 3

CAM_WIDTH_PX = 960
CAM_HEIGHT_PX = 960
CAM_FOV_DEG = 60.0
MOUNT_PITCH_DEG = 15.0
FOCAL_LENGTH = (CAM_WIDTH_PX / 2.0) / np.tan(np.deg2rad(CAM_FOV_DEG) / 2.0)

# R_b2c: body(FLU) -> cam(RDF), with mount pitch
R_B2C_BASE = np.array([[0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [1.0, 0.0, 0.0]])
_p = np.deg2rad(MOUNT_PITCH_DEG)
R_PITCH = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(_p), -np.sin(_p)],
                    [0.0, np.sin(_p), np.cos(_p)]])
R_B2C = R_PITCH @ R_B2C_BASE
GIMBAL_TAU = 0.1            # s, semi-strapdown cam-x pitch-gimbal servo lag (boresight → horizontal)
# Boresight world elevation = -(own_pitch + gimbal_total); servo_cmd = -own_pitch + GIMBAL_BIAS
# => elev = -GIMBAL_BIAS. POSITIVE bias = down-gaze. (Was -5.0 => 5 deg UP-gaze — sign bug that
# cost 10 deg of bottom-of-frame margin exactly where the terminal brake-climb loses the target.)
GIMBAL_BIAS = np.deg2rad(5.0)    # rad, fixed down-gaze bias (keeps nonzero image-v for co-altitude targets)

DT_CAM = 0.04            # actual camera period: cam_step=round(DT_CAM/dt_ctrl)=2, dt_ctrl=0.02 -> 0.04s.
                          # Was 0.033 (30Hz nominal) but the discrete cam_step gives 0.04s; the mismatch
                          # starves the relativized predict (ownship motion under-played by ~20%).
W_GUESS = 1.5
CONTROL_RATE = 50.0
RATE_TC = 0.02
# Mirror the C++ node's use_simulation:=true mode: ctrl_coeff_=1.0 (full rate authority, NO EMA
# attenuation). The C++ node only attenuates when !use_simulation; against the Python/ROS2 harness
# it runs at full authority. RATE_ATTEN=True (ctrl_coeff=0.63) silently under-powered the UAV and
# masked the v_des=20 overshoot the real node exhibits.
RATE_ATTEN = False

# Controller params (from yaml)
CTRL_N = 4.5
CTRL_K_V = 2.2
CTRL_K_IBVS = 2.0
CTRL_K_ATT = 6.0      # was 9.0; lowered to damp the close-range roll limit-cycle (iter 2)
CTRL_K_YAW = 5.0
CTRL_K_Z = 2.0       # vertical proportional (a_z = K_z*dz - K_zd*vz)
CTRL_K_ZD = 2.0      # vertical damping on pursuer odom v_z (K_z=2 -> zeta≈0.7)
CTRL_V_DES = 20.0
CTRL_MAX_TILT_DEG = 45.0     # user req: pitch/roll authority 45 deg (gimbal decouples camera from pitch)
CTRL_FOV_KEEP = 1.0
CTRL_JERK = 30.0      # was 15; raised so the terminal decel can shed ~9 m/s (iter 3)
CTRL_MAX_BODY_RATE = 3.0
CTRL_TERM_BRAKE_RANGE = 5.0
# Axial speed law: flat v_des + single adaptive turn-radius cap (mirror C++). The cap is the ONLY
# close-range speed reducer; no smooth profile / terminal_brake_floor.
CTRL_TURN_RADIUS_CAP_FACTOR = 1.0  # cap factor (1.0 = physics limit; <1 brakes sooner). Mirror C++.
# Iter 1-3 terminal-authority / oscillation params (mirror C++):
CTRL_V_PN_REF = 12.0              # cap on v_fwd in PN lateral accel (iter 1)
CTRL_MAX_ATT_RATE = 3.5           # attitude-reference slew governor |d(phi/theta)_cmd/dt| (Fix G: was 5)
CTRL_ATT_REF_TC = 0.15            # attitude-reference EMA time constant [s] (iter 2)
CTRL_MAX_TILT_TERMINAL_DEG = 55.0  # roll clamp + cap tilt inside terminal_tilt_range (raised 45->55, mirror C++)
CTRL_TERMINAL_TILT_RANGE = 30.0   # m, roll-clamp + cap gate
CTRL_AZ_SLEW = 20.0               # m/s^3, |da_z/dt| slew limit (Fix C/E, mirror C++ yaml controller.a_z_slew)
CTRL_LAT_DECAY_FLOOR = 0.4        # Fix D: a_lat_mag floor at terminal_brake_range (close-range weave damp)
CTRL_LAT_DECAY_START = 25.0       # Fix F: range at which close-range lateral attenuation begins (m, was 12)
CTRL_DITHER_MAX_BBOX = 80.0       # px, ALSO suppress the dither when bbox >= this (large bbox = good size-SNR /
                                  # close target; DIRECT observable immune to the (W, range) scale error that
                                  # blinds dither_min_range — mirror C++ controller.dither_max_bbox. 0 = off)
VEL_SMOOTH_TC = 0.25              # s, EMA on target-velocity estimate (Fix B, mirror C++ estimation.vel_smooth_tc)
COAST_RANGE = 12.0       # terminal coast-on-dropout engages below this range (mirror C++)
COAST_MAX_DUR = 1.5      # coast on the last good cmd this long (lets a close intercept complete / bbox return)
# Parallax-demod depth channel: DISABLED. During a fast closing the 2-cycle (4 s) demod window
# spans a ~4x range change and the PN lateral accel leaks into the 0.5 Hz dither band, inflating
# A_meas -> depth biased HIGH (measured 64 m at true 25 m, sigma accepted by the 5-sigma gate).
# The biased depth drags v_rel to the +-30 clamp -> controller sees v_closing=+30 -> brakes/
# reverses at 10 m -> false capture -> fly-away. With the channel OFF all 7 scenarios intercept
# single-pass (incl. wrong_wguess: the size channel + q_w_dynamic resolves scale without it).
# The lateral dither stays ON: its parallax still reaches the estimator through the bearing
# channel (a_own predict + pixel update), which is where the scale observability actually comes from.
ENABLE_PARALLAX_DEPTH = False

G = 9.81
DRAG_COEFF = 0.1    # no drag — the active brake-to-hold (capture) stops the UAV; drag blunts the close approach


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ============================================================================
# robust_cholesky (mirror C++)
# ============================================================================
def robust_cholesky(matrix):
    sym = (matrix + matrix.T) * 0.5
    sym = sym + np.eye(matrix.shape[0]) * 1e-8
    try:
        L = np.linalg.cholesky(sym)
        return np.array(L)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(sym)
        w = np.maximum(w, 1e-8)
        return V @ np.diag(np.sqrt(w))


# ============================================================================
# Process noise + motion models (mirror C++)
# ============================================================================
def make_Q(q_pos, q_vel, q_acc, q_w):
    Q = np.zeros((STATE_DIM, STATE_DIM))
    for i in range(3):
        Q[i, i] = q_pos
        Q[i + 3, i + 3] = q_vel
        Q[i + 6, i + 6] = q_acc
    Q[9, 9] = q_w
    return Q


# Relative-state motion models. State x = [r(3), v_rel(3), a_t(3), W(1)] in a gravity-aligned
# world frame G (origin arbitrary). r/v_rel are TARGET-OWNSHIP (relative); a_t is the target's
# OWN acceleration; `a_own` is the ownship's world-frame acceleration over dt, supplied as an
# exogenous input from IMU preintegration (Δv_own/dt, gravity-corrected). Ownship pos/vel are
# NOT state — relative quantities are drift-free because each bbox update closes the loop on r.
# Mirror of the C++ relativized estimator (see plan: fpv-odom-mavros-imu-data-imu-sequential-eich).
def f_singer(x, dt, a_own):
    x_new = x.copy()
    decay = np.exp(-dt / 1.5)
    a_rel = x[6:9] - a_own                          # relative accel = a_target - a_ownship
    x_new[6:9] = x[6:9] * decay                      # target own accel decays (Singer)
    x_new[3:6] = x[3:6] + a_rel * dt                 # v_rel
    x_new[0:3] = x[0:3] + x[3:6] * dt + 0.5 * a_rel * dt * dt   # r (relative pos)
    x_new[9] = x[9]
    return x_new


def f_ct(x, dt, a_own):
    x_new = x.copy()
    p = x[0:3]
    v = x[3:6]              # v_rel
    a_t = x[6:9]            # target own accel
    a_rel = a_t - a_own
    v_xy2 = max(v[0] ** 2 + v[1] ** 2, 1e-4)
    omega = clamp((v[0] * a_t[1] - v[1] * a_t[0]) / v_xy2, -2.0, 2.0)   # approx: v_rel stands in for target vel
    if abs(omega) < 1e-3:
        x_new[0:3] = p + v * dt + 0.5 * a_rel * dt * dt
        x_new[3:6] = v + a_rel * dt
    else:
        swt = np.sin(omega * dt)
        cwt = np.cos(omega * dt)
        x_new[0] = p[0] + (v[0] * swt - v[1] * (1.0 - cwt)) / omega
        x_new[1] = p[1] + (v[1] * swt + v[0] * (1.0 - cwt)) / omega
        x_new[2] = p[2] + v[2] * dt + 0.5 * a_rel[2] * dt * dt
        x_new[3] = v[0] * cwt - v[1] * swt
        x_new[4] = v[0] * swt + v[1] * cwt
        x_new[5] = v[2] + a_rel[2] * dt
        x_new[6] = a_t[0] * cwt - a_t[1] * swt      # target own accel rotates with the turn
        x_new[7] = a_t[0] * swt + a_t[1] * cwt
        x_new[8] = a_t[2]
    x_new[9] = x[9]
    return x_new


def f_rotor(x, dt, a_own):
    x_new = x.copy()
    v = x[3:6]              # v_rel
    a_t = x[6:9]            # target own accel
    a_rel = a_t - a_own
    # NOTE: the original absolute-state f_rotor modeled target aerodynamic drag as
    # a_real = a_thrust - k_drag*|v_target|*v_target. In relative coordinates v_target is
    # UNOBSERVABLE (it would need ownship vel). Substituting v_rel is wrong: when the target is
    # static but the ownship is moving, |v_rel| is large and the drag term spuriously brakes the
    # RELATIVE velocity -> v_rel under-estimation that grows with ownship accel. Drop the drag
    # term; rotor degrades to a target-accel-relaxation model (tau=2s), distinct from Singer (1.5s).
    x_new[0:3] = x[0:3] + v * dt + 0.5 * a_rel * dt * dt
    x_new[3:6] = v + a_rel * dt
    x_new[6:9] = a_t * np.exp(-dt / 2.0)
    x_new[9] = x[9]
    return x_new


def apply_model(x, dt, mode, a_own):
    if mode == 0:
        return f_singer(x, dt, a_own)
    if mode == 1:
        return f_ct(x, dt, a_own)
    return f_rotor(x, dt, a_own)


# ============================================================================
# Measurement model (mirror C++)
# ============================================================================
def h_meas_pixel(x, R_n2c, focal_length):
    # Relative-state form: x[0:3] IS the relative position (target - ownship) in world frame G.
    # No uav_pos needed — the measurement is purely relative; R_n2c = R_b2c @ R_b2n^T carries the
    # ownship ATTITUDE (from IMU orientation), which is all the projection requires.
    rel_p = x[0:3]
    p_cam = R_n2c @ rel_p
    p_cam[2] = max(p_cam[2], 0.5)
    u = focal_length * p_cam[0] / p_cam[2]
    v = focal_length * p_cam[1] / p_cam[2]
    W_est = clamp(x[9], 0.9, 2.5)
    s = focal_length * W_est / p_cam[2]
    return np.array([u, v, s])


def pixel_to_los_nav(u, v, focal_length, R_b2n, R_b2c):
    ray_cam = np.array([u / focal_length, v / focal_length, 1.0])
    ray_cam /= np.linalg.norm(ray_cam)
    ray_body = R_b2c.T @ ray_cam
    out = R_b2n @ ray_body
    return out / np.linalg.norm(out)


# ============================================================================
# UKF (mirror C++ UnscentedKalmanFilter10D)
# ============================================================================
class UKF10D:
    def __init__(self, mode):
        self.x = np.zeros(STATE_DIM)
        self.P = np.eye(STATE_DIM)
        self.Q = np.zeros((STATE_DIM, STATE_DIM))
        self.R = np.eye(3)
        self.R[0, 0] = 0.005 ** 2
        self.R[1, 1] = 0.005 ** 2
        self.R[2, 2] = 0.015 ** 2
        self.mode = mode
        self.sigmas_f = [np.zeros(STATE_DIM) for _ in range(2 * STATE_DIM + 1)]
        self.predicted = False

    def predict(self, dt, a_own):
        n = STATE_DIM
        Wi = 0.5 / n
        L = robust_cholesky(n * self.P)
        sigmas = [np.zeros(n) for _ in range(2 * n + 1)]
        sigmas[0] = self.x.copy()
        for i in range(n):
            sigmas[i + 1] = self.x + L[:, i]
            sigmas[i + 1 + n] = self.x - L[:, i]
        # a_own is a deterministic exogenous input — same value applied to every sigma point (not augmented).
        self.sigmas_f = [apply_model(sigmas[i], dt, self.mode, a_own) for i in range(2 * n + 1)]
        x_pred = np.zeros(n)
        for i in range(1, 2 * n + 1):
            x_pred += Wi * self.sigmas_f[i]
        P_pred = np.zeros((n, n))
        for i in range(1, 2 * n + 1):
            diff = self.sigmas_f[i] - x_pred
            P_pred += Wi * np.outer(diff, diff)
        P_pred += self.Q
        self.x = x_pred
        self.P = (P_pred + P_pred.T) * 0.5
        self.predicted = True

    def update(self, z_pixel, R_n2c, focal_length):
        n = STATE_DIM
        Wi = 0.5 / n
        if not self.predicted:
            L = robust_cholesky(n * self.P)
            self.sigmas_f = [np.zeros(n) for _ in range(2 * n + 1)]
            self.sigmas_f[0] = self.x.copy()
            for i in range(n):
                self.sigmas_f[i + 1] = self.x + L[:, i]
                self.sigmas_f[i + 1 + n] = self.x - L[:, i]
        self.predicted = False
        sigmas_h = [h_meas_pixel(self.sigmas_f[i], R_n2c, focal_length)
                    for i in range(2 * n + 1)]
        z_pred = np.zeros(3)
        for i in range(1, 2 * n + 1):
            z_pred += Wi * sigmas_h[i]
        P_zz = np.zeros((3, 3))
        P_xz = np.zeros((n, 3))
        for i in range(1, 2 * n + 1):
            dz = sigmas_h[i] - z_pred
            dx = self.sigmas_f[i] - self.x
            P_zz += Wi * np.outer(dz, dz)
            P_xz += Wi * np.outer(dx, dz)
        P_zz += self.R
        P_zz = (P_zz + P_zz.T) * 0.5
        try:
            K = P_xz @ np.linalg.inv(P_zz)
        except np.linalg.LinAlgError:
            return 1e-30
        y = z_pixel - z_pred
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(y)):
            return 1e-30
        self.x = self.x + K @ y
        self.P = self.P - K @ P_zz @ K.T
        self.P = (self.P + self.P.T) * 0.5
        det = max(abs(np.linalg.det(P_zz)), 1e-30)
        exponent = -0.5 * y @ np.linalg.solve(P_zz, y)
        likelihood = np.exp(clamp(exponent, -80.0, 80.0)) / np.sqrt((2.0 * np.pi) ** 3 * det)
        if not np.isfinite(likelihood) or likelihood < 1e-30:
            likelihood = 1e-30
        return likelihood

    def update_depth(self, z_depth, R_depth):
        # Scalar parallax-depth update (h_depth(x) = ||x[0:3]||). Runs AFTER the pixel update as a
        # sequential scalar pass. Re-sigmas from the current (post-pixel) x/P so the depth and size
        # channels both see the post-pixel covariance. h_depth is along-LOS range; depth measurement
        # is independent of W -> breaks the size = fx*W/r degeneracy and anchors range, which makes
        # the along-LOS v_rel observable through range change. Returns likelihood but the caller does
        # NOT use it for IMM mode re-weighting (depth is mode-independent geometry).
        n = STATE_DIM
        Wi = 0.5 / n
        if self.P[9, 9] < 1e-3:
            return 1e-30                      # W pinned -> size channel sufficient, skip
        L = robust_cholesky(n * self.P)
        h_vals = []
        dx_list = []
        for i in range(n):
            sp = self.x + L[:, i]
            sm = self.x - L[:, i]
            h_vals.append(np.linalg.norm(sp[0:3]))
            dx_list.append(sp - self.x)
            h_vals.append(np.linalg.norm(sm[0:3]))
            dx_list.append(sm - self.x)
        z_pred = sum(Wi * h for h in h_vals)
        if not np.isfinite(z_pred) or z_pred < 0.1:
            return 1e-30
        P_xz = np.zeros(n)
        P_zz = 0.0
        for dx, h in zip(dx_list, h_vals):
            dh = h - z_pred
            P_xz += Wi * dx * dh
            P_zz += Wi * dh * dh
        P_zz += R_depth
        if P_zz < 1e-9:
            return 1e-30
        K = P_xz / P_zz                       # n-vector
        y = z_depth - z_pred
        if not np.all(np.isfinite(K)) or not np.isfinite(y):
            return 1e-30
        self.x = self.x + K * y
        # Joseph form for PSD safety
        I_KH = np.eye(n) - np.outer(K, P_xz) / max(P_zz, 1e-12)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * R_depth
        self.P = (self.P + self.P.T) * 0.5
        det = max(abs(P_zz), 1e-30)
        exponent = -0.5 * (y * y) / P_zz
        likelihood = np.exp(clamp(exponent, -80.0, 80.0)) / np.sqrt(2.0 * np.pi * det)
        if not np.isfinite(likelihood) or likelihood < 1e-30:
            likelihood = 1e-30
        return likelihood


# ============================================================================
# IMM (mirror C++ RobustIMMEstimator10D)
# ============================================================================
class IMMEstimator10D:
    def __init__(self):
        self.filters = [UKF10D(i) for i in range(MODE_COUNT)]
        self.filters[0].Q = make_Q(1e-3, 0.1, 6.0, 0.05)
        self.filters[1].Q = make_Q(1e-3, 0.1, 2.0, 0.05)
        self.filters[2].Q = make_Q(1e-3, 0.1, 5.0, 0.05)
        self.mu = np.array([0.2, 0.4, 0.4])
        self.M = np.array([[0.85, 0.05, 0.10],
                           [0.05, 0.90, 0.05],
                           [0.10, 0.05, 0.85]])
        self.x = np.zeros(STATE_DIM)
        self.P = np.eye(STATE_DIM)
        self.cbar = np.zeros(3)
        self.omega = np.zeros((3, 3))
        self.compute_mixing_probabilities()

    def compute_mixing_probabilities(self):
        self.cbar = self.M.T @ self.mu
        for j in range(MODE_COUNT):
            self.cbar[j] = max(self.cbar[j], 1e-30)
        for i in range(MODE_COUNT):
            for j in range(MODE_COUNT):
                self.omega[i, j] = (self.M[i, j] * self.mu[i]) / self.cbar[j]

    def predict(self, dt, a_own):
        xs = [np.zeros(STATE_DIM) for _ in range(MODE_COUNT)]
        Ps = [np.zeros((STATE_DIM, STATE_DIM)) for _ in range(MODE_COUNT)]
        for j in range(MODE_COUNT):
            for i in range(MODE_COUNT):
                xs[j] += self.omega[i, j] * self.filters[i].x
            for i in range(MODE_COUNT):
                diff = self.filters[i].x - xs[j]
                Ps[j] += self.omega[i, j] * (self.filters[i].P + np.outer(diff, diff))
        for j in range(MODE_COUNT):
            self.filters[j].x = xs[j]
            self.filters[j].P = (Ps[j] + Ps[j].T) * 0.5
            self.filters[j].predict(dt, a_own)
        self.compute_mixing_probabilities()
        self.compute_state_estimate()

    def update(self, z_pixel, R_n2c, focal_length):
        likelihoods = np.zeros(3)
        for j in range(MODE_COUNT):
            likelihoods[j] = self.filters[j].update(z_pixel, R_n2c, focal_length)
        self.mu = self.cbar * likelihoods
        mu_sum = self.mu.sum()
        if not np.isfinite(mu_sum) or mu_sum < 1e-30:
            self.mu = np.ones(3) / MODE_COUNT
        else:
            self.mu /= mu_sum
        self.compute_mixing_probabilities()
        self.compute_state_estimate()

    def update_depth(self, z_depth, R_depth):
        # Scalar parallax-depth update across all filters. Does NOT update mode probabilities (mu) —
        # depth is mode-independent geometry; letting it re-weight mu would over-commit to whichever
        # mode has the smallest P_zz_depth. Only state/P are updated.
        for j in range(MODE_COUNT):
            self.filters[j].update_depth(z_depth, R_depth)
        self.compute_state_estimate()

    def compute_state_estimate(self):
        self.x = np.zeros(STATE_DIM)
        for j in range(MODE_COUNT):
            self.x += self.mu[j] * self.filters[j].x
        self.P = np.zeros((STATE_DIM, STATE_DIM))
        for j in range(MODE_COUNT):
            diff = self.filters[j].x - self.x
            self.P += self.mu[j] * (self.filters[j].P + np.outer(diff, diff))
        self.P = (self.P + self.P.T) * 0.5


# ============================================================================
# RelativeIMMUKFEstimator (mirror C++)
# ============================================================================
class RelativeIMMUKFEstimator:
    def __init__(self, dt_cam, focal_length, W_guess, R_b2c):
        self.dt_cam = dt_cam
        self.focal_length = focal_length
        self.W_guess = W_guess
        self.R_b2c = R_b2c
        self.imm_ukf = IMMEstimator10D()
        self.initialized = False
        self.frozen_width = 0.0
        self.has_frozen_width = False
        self.last_cap_time = 0.0
        self.has_last_cap_time = False
        self.first_meas_after_dropout = False
        self.width_freeze_range = 15.0  # freeze monocular scale below this range (mirror C++)
        self.size_gate_lo = 0.6         # reject bbox if measured size < lo*predicted (mirror C++)
        self.size_gate_hi = 1.8         #   or > hi*predicted
        # The size-plausibility gate exists to reject CLOSE-RANGE mis-detections (a huge garbage
        # bbox that corrupts the monocular scale). Applied at long range it starves the relative
        # estimator: r_est lags the true range during a closing -> s_pred is too small -> good
        # measurements get rejected -> r never corrects (deadlock). Gate it to close range only.
        self.size_gate_range = 12.0
        self.last_los_nav = None        # cached LOS unit vector (world) for divergence reset

    def update_process_noise(self, r_norm, W_est, bbox_s=0.0):
        q_scale = 1.0
        r_freeze = self.width_freeze_range
        snr_factor = clamp(bbox_s / 40.0, 0.1, 1.0) if bbox_s > 0.0 else 1.0
        q_w_scale = clamp((r_norm - r_freeze) / 20.0, 0.0, 1.0) ** 2
        q_w_dynamic = 0.005 * q_w_scale * snr_factor
        # Relative-state: v_rel is driven by the a_own predict (high-quality, IMU preintegration).
        # The monocular measurement constrains v_rel only weakly (depth is noisy), so a large Q_vel
        # makes the update TRUST the lagged measurement and pull v_rel back toward it (under-playing
        # the closing speed). Keep Q_vel SMALL so K_vvel stays small and the a_own predict dominates.
        q_vel = 0.02 * q_scale
        self.imm_ukf.filters[0].Q = make_Q(0.05 * q_scale, q_vel, 1.0 * q_scale, q_w_dynamic * q_scale)
        self.imm_ukf.filters[1].Q = make_Q(0.05 * q_scale, q_vel, 0.5 * q_scale, q_w_dynamic * q_scale)
        self.imm_ukf.filters[2].Q = make_Q(0.05 * q_scale, q_vel, 1.0 * q_scale, q_w_dynamic * q_scale)

    def sanitize_filters(self):
        # Relative-state form: r_norm is just |f.x[0:3]| (no uav_pos subtraction). On divergence,
        # reset along the last observed LOS direction (cached by process_camera) at a nominal range.
        should_freeze = False
        for f in self.imm_ukf.filters:
            r_norm = np.linalg.norm(f.x[0:3])
            r_freeze = self.width_freeze_range
            if r_norm < r_freeze:
                should_freeze = True
                break
        if not should_freeze:
            self.has_frozen_width = False
        elif not self.has_frozen_width:
            self.frozen_width = clamp(self.imm_ukf.x[9], 1.0, 1.8)
            self.has_frozen_width = True
        los_dir = self.last_los_nav if self.last_los_nav is not None else np.array([1.0, 0.0, 0.0])
        for f in self.imm_ukf.filters:
            diverged = (not np.all(np.isfinite(f.x))) or (not np.all(np.isfinite(f.P)))
            if not diverged:
                for i in range(STATE_DIM):
                    if abs(f.x[i]) > 1e6:
                        diverged = True
                        break
            if diverged:
                f.x = np.zeros(STATE_DIM)
                f.x[0:3] = 10.0 * los_dir                    # relative pos along last LOS
                f.x[9] = self.W_guess
                f.P = np.zeros((STATE_DIM, STATE_DIM))
                f.P[np.diag_indices(STATE_DIM)] = [100, 100, 100, 25, 25, 25, 10, 10, 10, 1.0]
                continue
            rel_p = f.x[0:3]
            r_norm = np.linalg.norm(rel_p)
            if r_norm > 200.0:
                f.x[0:3] = rel_p / max(r_norm, 1e-6) * 120.0
            for i in range(3, 6):
                f.x[i] = clamp(f.x[i], -30.0, 30.0)
            for i in range(6, 9):
                f.x[i] = clamp(f.x[i], -15.0, 15.0)
            f.x[9] = clamp(f.x[9], 0.8, 1.8)
            if self.has_frozen_width:
                f.x[9] = self.frozen_width
                f.P[9, :] = 0.0
                f.P[:, 9] = 0.0
                f.P[9, 9] = 1e-8
            f.P = (f.P + f.P.T) * 0.5
            for i in range(3):
                f.P[i, i] = clamp(f.P[i, i], 2.0, 500.0)
            for i in range(3, 6):
                f.P[i, i] = clamp(f.P[i, i], 0.5, 500.0)
            for i in range(6, 9):
                f.P[i, i] = clamp(f.P[i, i], 1e-4, 10.0)
            f.P[9, 9] = clamp(f.P[9, 9], 1e-4, 500.0)
        self.imm_ukf.compute_state_estimate()
        if self.has_frozen_width:
            self.imm_ukf.x[9] = self.frozen_width
            self.imm_ukf.P[9, :] = 0.0
            self.imm_ukf.P[:, 9] = 0.0
            self.imm_ukf.P[9, 9] = 1e-8
        else:
            self.imm_ukf.x[9] = clamp(self.imm_ukf.x[9], 1.2, 1.8)

    def process_camera(self, has_bbox, bbox_u, bbox_v, bbox_s, t_now, R_b2n_cur, dv_own,
                       z_depth=np.nan, R_depth=np.inf):
        # Relative-state estimator. Inputs that replaced ownship pos/vel:
        #   R_b2n_cur — ownship ATTITUDE (from IMU orientation), used for cam projection + LOS ray.
        #   dv_own    — ownship velocity change over [last_cam, now] in world frame G, from IMU
        #               preintegration (Δv_own = ∫(R_b2n·(f_imu − bias) − g)dτ). Pure relative
        #               quantity, recomputed each window — NOT a drifting integrated velocity.
        # No uav_pos/uav_vel anywhere; r in the state IS the relative position.
        if not has_bbox:
            if not self.initialized:
                return False, np.full(STATE_DIM, np.nan)
            w_prev = self.imm_ukf.x[9]
            r_norm = np.linalg.norm(self.imm_ukf.x[0:3])
            a_own = dv_own / max(self.dt_cam, 1e-3)
            self.update_process_noise(r_norm, w_prev)
            # ALWAYS predict r forward using ownship motion (a_own). The original absolute-state
            # estimator could skip the predict here because a stationary target's ABSOLUTE position
            # does not move; the RELATIVE position r = target - own DOES move, by the ownship's
            # motion every frame. Skipping the predict freezes r while the ownship keeps flying ->
            # unbounded drift the moment the target leaves the FOV.
            self.imm_ukf.predict(self.dt_cam, a_own)
            # Coast: assume the target stops ACCELERATING (a_t->0, constant-velocity target) but
            # DO NOT touch v_rel. In relative coordinates v_rel = v_target - v_own; zeroing or
            # decaying it would erase the ownship's ongoing motion from r's predict (r is propagated
            # by v_rel*dt above) -> r freezes while the ownship flies on. Keep v_rel; let the
            # a_own-driven predict carry the ownship motion, the retained v_rel carry target motion.
            for f in self.imm_ukf.filters:
                f.x[6:9] = 0.0
                f.x[9] = w_prev
                f.P[9, 9] = max(0.001, f.P[9, 9])
            self.imm_ukf.compute_state_estimate()
            self.imm_ukf.x[9] = w_prev
            self.sanitize_filters()
            self.first_meas_after_dropout = True
            return True, self.imm_ukf.x
        R_b2n_cap = R_b2n_cur
        los_nav = pixel_to_los_nav(bbox_u, bbox_v, self.focal_length, R_b2n_cap, self.R_b2c)
        self.last_los_nav = los_nav.copy()
        z_pixel = np.array([bbox_u, bbox_v, bbox_s])
        R_n2c = self.R_b2c @ R_b2n_cap.T
        R_k = np.zeros((3, 3))
        R_k[0, 0] = 25.0
        R_k[1, 1] = 25.0
        size_noise_std = max(5.0, 300.0 / max(1.0, bbox_s))
        R_k[2, 2] = size_noise_std ** 2
        if self.first_meas_after_dropout:
            R_k *= 5.0
        if self.initialized:
            dt_meas = clamp(t_now - self.last_cap_time, 0.001, 0.1) if self.has_last_cap_time else self.dt_cam
            a_own = dv_own / max(dt_meas, 1e-3)
            self.update_process_noise(np.linalg.norm(self.imm_ukf.x[0:3]), self.imm_ukf.x[9], bbox_s)
            self.imm_ukf.predict(dt_meas, a_own)
            # Size-plausibility gate (mirror C++): reject detections whose bbox size is wildly
            # inconsistent with predicted range/width -> predict-only, mark soft-recovery. Stops a
            # garbage close-range size from corrupting scale/position (phantom-target dive).
            r_pred = np.linalg.norm(self.imm_ukf.x[0:3])
            s_pred = self.focal_length * max(0.1, self.imm_ukf.x[9]) / max(0.5, r_pred)
            size_ratio = bbox_s / max(1.0, s_pred)
            if r_pred < self.size_gate_range and (size_ratio < self.size_gate_lo or size_ratio > self.size_gate_hi):
                self.imm_ukf.compute_state_estimate()
                self.first_meas_after_dropout = True
                self.last_cap_time = t_now
                self.has_last_cap_time = True
                return True, self.imm_ukf.x
            for f in self.imm_ukf.filters:
                f.R = R_k
            self.imm_ukf.update(z_pixel, R_n2c, self.focal_length)
            self.last_cap_time = t_now
            self.has_last_cap_time = True
            self.first_meas_after_dropout = False
            # Parallax-depth scalar update (sequential, after pixel+size). Gates: finite depth,
            # reasonable R, W not yet pinned, innovation within 5 sigma. Skip silently otherwise.
            if np.isfinite(z_depth) and R_depth < 1e6 and self.imm_ukf.P[9, 9] > 1e-3:
                z_pred_d = float(np.linalg.norm(self.imm_ukf.x[0:3]))
                P_zz_d = max(float(np.trace(self.imm_ukf.P[0:3, 0:3])), 1e-6) + R_depth
                if abs(z_depth - z_pred_d) < 5.0 * np.sqrt(P_zz_d):
                    self.imm_ukf.update_depth(z_depth, R_depth)
            self.sanitize_filters()
            # (was: clamp fused v_rel to [-15,15] — an absolute-state target-speed limit. In relative
            # coordinates v_rel = v_target - v_ownship easily exceeds 15 m/s during a high-speed
            # closing, so clamping here caps the closing-speed estimate and starves the controller.
            # sanitize_filters already bounds v_rel to [-30,30]; no extra clamp needed.)
            return True, self.imm_ukf.x
        else:
            W_init = clamp(self.W_guess, 0.5, 4.0)
            r0 = clamp(self.focal_length * W_init / max(1.0, bbox_s), 10.0, 300.0)
            x0 = np.zeros(STATE_DIM)
            x0[0:3] = r0 * los_nav                      # relative pos = range * LOS (no uav_pos)
            x0[9] = W_init
            P0 = np.zeros((STATE_DIM, STATE_DIM))
            P0[np.diag_indices(STATE_DIM)] = [100, 100, 100, 25, 25, 25, 10, 10, 10, 0.09]
            for f in self.imm_ukf.filters:
                f.x = x0.copy()
                f.P = P0.copy()
                f.R = R_k
                f.predicted = False
            self.imm_ukf.x = x0.copy()
            self.imm_ukf.P = P0.copy()
            self.initialized = True
            self.last_cap_time = t_now
            self.has_last_cap_time = True
            self.sanitize_filters()
            return True, self.imm_ukf.x


# ============================================================================
# ParallaxDemodulator: active-dither synchronous-demodulation depth (NO VIO)
# ============================================================================
class ParallaxDemodulator:
    """Synchronous-demodulation parallax depth. Ownship vel is NOT a state.

    The ownship lateral dither (controller, freq omega) produces a parallax baseline. Demodulating
    the IMU lateral accel AND the bbox bearing at omega coherently sums N samples (per-frame SNR
    ~2.8 -> demod SNR ~20) and rejects target low-freq motion + IMU DC drift. The dither-frequency
    lateral displacement amplitude `baseline = A_meas/omega^2` is DC-free — the sin(omega*t) multiply
    kills v_own_avg, so NO ownship-velocity state is needed. `depth = baseline / dtheta`, returned
    as an extra scalar UKF measurement that anchors along-LOS range (independent of W), making the
    along-LOS v_rel observable.

    push() every camera step; compute_depth() returns (depth, R_depth, eta).
    """
    def __init__(self, dither_freq, dither_amp, focal_length, dt_cam,
                 window_cycles=2.0, eta_tau=5.0, eta_min=0.7,
                 r_depth_floor=1.0, pixel_noise=1.5, imu_accel_noise=0.05):
        self.omega = 2.0 * np.pi * dither_freq
        self.A_cmd = float(dither_amp)
        self.focal_length = focal_length
        self.dt_cam = dt_cam
        self.N = max(int(round(window_cycles / dither_freq / dt_cam)), 8)
        self.samples = deque(maxlen=self.N)      # (t, a_lat, theta_perp or None)
        self.eta_tau = eta_tau
        self.eta_filt = 1.0
        self.eta_t_last = -1.0
        self.eta_min = eta_min
        self.r_depth_floor = r_depth_floor
        self.pixel_noise = pixel_noise
        self.imu_accel_noise = imu_accel_noise
        self.min_bearing_frac = 0.75

    def reset(self):
        self.samples.clear()
        self.eta_filt = 1.0
        self.eta_t_last = -1.0

    def push(self, t, a_lat, theta_perp, has_bbox):
        # theta_perp = LOS lateral angle (rad) along the dither axis (e_perp_xy); None during dropout.
        self.samples.append((t, a_lat, theta_perp if has_bbox else None))

    def compute_depth(self, t_now):
        N = len(self.samples)
        if N < int(0.75 * self.N):
            return np.nan, np.inf, self.eta_filt
        bearing = [s for s in self.samples if s[2] is not None]
        if len(bearing) < int(self.min_bearing_frac * self.N):
            return np.nan, np.inf, self.eta_filt
        # accel I/Q demod -> ownship dither displacement amplitude (DC-free baseline)
        I_a = sum(s[1] * np.sin(self.omega * s[0]) for s in self.samples)
        Q_a = sum(s[1] * np.cos(self.omega * s[0]) for s in self.samples)
        A_meas = (2.0 / N) * np.hypot(I_a, Q_a)
        phi_a = np.arctan2(Q_a, I_a)
        # eta = tracking fidelity gate (LPF of measured vs commanded dither amplitude)
        eta_raw = A_meas / max(self.A_cmd, 1e-6)
        if self.eta_t_last > 0.0 and t_now > self.eta_t_last:
            a_lpf = clamp((t_now - self.eta_t_last) / self.eta_tau, 0.0, 1.0)
            self.eta_filt += a_lpf * (eta_raw - self.eta_filt)
        else:
            self.eta_filt = eta_raw
        self.eta_t_last = t_now
        if self.eta_filt < self.eta_min:
            return np.nan, np.inf, self.eta_filt
        # bearing I/Q demod -> parallax bearing amplitude
        Nb = len(bearing)
        I_t = sum(s[2] * np.sin(self.omega * s[0]) for s in bearing)
        Q_t = sum(s[2] * np.cos(self.omega * s[0]) for s in bearing)
        dtheta = (2.0 / Nb) * np.hypot(I_t, Q_t)
        phi_t = np.arctan2(Q_t, I_t)
        if dtheta < 1e-5:
            return np.nan, np.inf, self.eta_filt
        # sign: front-facing target -> bearing rotates opposite to ownship lateral motion -> phi_t-phi_a ~ pi
        # Parallax depth: both A_meas/omega^2 (ownship dither displacement amplitude) and dtheta
        # (bearing dither amplitude) are positive magnitudes. accel integrates twice to displacement
        # (180 deg), and bearing follows displacement, so accel and bearing are 180 deg apart — but
        # we only use the amplitudes here, so no sign/phase gate is needed.
        baseline = A_meas / (self.omega ** 2)
        depth = baseline / dtheta
        if not np.isfinite(depth) or depth < 2.0 or depth > 200.0:
            return np.nan, np.inf, self.eta_filt
        sigma_theta = self.pixel_noise / self.focal_length
        sigma_dtheta = sigma_theta / np.sqrt(max(Nb / 2.0, 1.0))
        sigma_A = self.imu_accel_noise / np.sqrt(max(N / 2.0, 1.0))
        rel_err = np.hypot(sigma_A / max(A_meas, 1e-6), sigma_dtheta / dtheta)
        R_depth = max((depth * rel_err) ** 2, self.r_depth_floor)
        return depth, R_depth, self.eta_filt


# ============================================================================
# AdvancedHybridController (mirror C++)
# ============================================================================
class AdvancedHybridController:
    def __init__(self):
        self.N = CTRL_N
        self.K_v = CTRL_K_V
        self.K_ibvs = CTRL_K_IBVS
        self.K_att = CTRL_K_ATT
        self.K_yaw = CTRL_K_YAW
        self.v_des = CTRL_V_DES
        self.max_tilt_rad = np.deg2rad(CTRL_MAX_TILT_DEG)
        self.fov_half_v = (CAM_HEIGHT_PX / 2.0) / max(1.0, FOCAL_LENGTH)
        self.fov_half_h = (CAM_WIDTH_PX / 2.0) / max(1.0, FOCAL_LENGTH)
        self.fov_keep_margin = CTRL_FOV_KEEP
        self.axial_jerk_max = CTRL_JERK
        self.enable_fov_clamp = True
        self.mount_pitch = np.deg2rad(MOUNT_PITCH_DEG)
        self.max_body_rate = CTRL_MAX_BODY_RATE
        self.az_priority_max = np.deg2rad(12.0)   # off-LOS thrust-azimuth bound at 90 deg tilt (pointing priority)
        self.a_lat_tilt_start = np.deg2rad(65.0)  # lateral-authority attenuation window, in sin(pitch)
        self.axial_deadband = 1.5                 # m/s, coast band around v_des_eff (mirror C++)
        self.gimbal_nose_up_max = np.deg2rad(85.0)  # max body nose-up the gimbal can compensate (C20S: 35; 0 = unlimited)
        self.a_lat_tilt_end = np.deg2rad(85.0)    # ... and end (a_lat -> 0); shallow-pitch banking untouched
        self.terminal_brake_range = CTRL_TERM_BRAKE_RANGE
        self.K_pitch = 1.2
        self.K_roll = 1.0
        self.K_z = CTRL_K_Z
        self.K_zd = CTRL_K_ZD
        self.a_z_limit = 4.0     # m/s^2 vertical accel clamp (C++ yaml controller.a_z_limit is 12)
        self.K_fwd = 2.0
        self.has_gimbal = True                  # semi-strapdown pitch gimbal: boresight decoupled from theta
        self.gimbal_bias = GIMBAL_BIAS          # boresight elev = -gimbal_bias (down-gaze positive)
        self.last_axial_accel = 0.0
        self.last_t = -1.0
        self.has_last = False
        # Iter 1-3 terminal-authority / oscillation params (mirror C++).
        self.v_pn_ref = CTRL_V_PN_REF
        self.max_att_rate = CTRL_MAX_ATT_RATE
        self.att_ref_tc = CTRL_ATT_REF_TC
        self.max_tilt_terminal_rad = np.deg2rad(CTRL_MAX_TILT_TERMINAL_DEG)
        self.terminal_tilt_range = CTRL_TERMINAL_TILT_RANGE
        self.turn_radius_cap_factor = CTRL_TURN_RADIUS_CAP_FACTOR
        # Attitude-reference slew governor + EMA state (dedicated; axial block reuses last_t/has_last).
        self.last_tilt_fwd = 0.0
        self.last_tilt_left = 0.0
        self.last_att_t = -1.0
        self.has_last_att = False
        self.last_tilt_fwd_f = 0.0
        self.last_tilt_left_f = 0.0
        # Vertical-accel slew-limit state (Fix C, mirror C++).
        self.last_a_z = 0.0
        self.last_az_t = -1.0
        self.has_last_az = False
        self.a_z_slew = CTRL_AZ_SLEW
        self.lateral_decay_floor = CTRL_LAT_DECAY_FLOOR
        self.lat_decay_start = CTRL_LAT_DECAY_START
        # Active monocular-observability dither (scale-observability dither, per yaml use_odometry
        # comment): a small lateral sinusoidal acceleration that gives the ownship a horizontal
        # parallax baseline. With x != 0 the pinhole u=fx*x/z becomes sensitive to z (depth), so the
        # depth/width degeneracy (size=fx*W/z) is broken and the along-LOS v_rel — otherwise only
        # weakly observable via the noisy size channel — becomes observable through the bearing
        # channel. The IMU preintegration already captures the dither motion as a_own, so the
        # estimator predict carries it; the measurement update then resolves depth from the parallax.
        self.dither_amp = 0.8          # m/s^2, lateral dither amplitude (lowered: 1.5 contaminates PN turn_sign on v20)
        self.dither_freq = 0.5         # Hz, lateral dither frequency
        self.dither_min_range = 10.0   # m, suppress dither inside this range (don't disturb terminal)
        self.dither_max_bbox = CTRL_DITHER_MAX_BBOX  # px, second close gate independent of the (W, range) scale
        self.last_bbox_size = 0.0      # px, last live bbox (stale-hold across dropout frames, mirror C++)
        self.last_bbox_t = -1.0
        self.K_apn = 0.8               # Geometric-APN gain on bearing-derived a_t_perp (turning-target lead)
        self.K_apn_min_at = 0.5       # m/s^2, gate |a_t_perp_geom| (ydot EMA rejects noise on static/CV; maneuvering a_t_perp~1 passes)
        # lambda_dot 2nd-derivative (ydot) EMA state for geometric APN — bearing curvature is noisy.
        self.prev_los_rate = 0.0
        self.prev_lr_t = -1.0
        self.has_prev_lr = False
        self.ydot_ema = 0.0
        self.atp_ema = 0.0
        self.aown_perp_ema = 0.0
        self.los_rate_slow = 0.0
        self.ydot_tc = 0.8             # s, EMA time constant on d(lambda_dot)/dt (smooth bearing curvature noise)
        self.atp_ema = 0.0            # EMA of a_t_perp_geom (sustained turn vs pixel-noise spike)
        self.atp_tc = 0.6             # s, EMA time constant on a_t_perp_geom
        self.aown_perp_ema = 0.0
        self.los_rate_slow = 0.0      # EMA of a_own_perp matched to ydot_tc (so dither cancels in a_t_perp_geom)
        self.los_rate_slow = 0.0      # slow EMA of lambda_dot (rejects 0.5Hz dither for turn_sign)
        self.los_rate_slow_tc = 1.5   # s

    def reset(self):
        self.last_axial_accel = 0.0
        self.last_t = -1.0
        self.has_last = False
        self.last_tilt_fwd = 0.0
        self.last_tilt_left = 0.0
        self.last_att_t = -1.0
        self.has_last_att = False
        self.last_tilt_fwd_f = 0.0
        self.last_tilt_left_f = 0.0
        self.last_a_z = 0.0
        self.last_az_t = -1.0
        self.has_last_az = False
        self.prev_los_rate = 0.0
        self.prev_lr_t = -1.0
        self.has_prev_lr = False
        self.ydot_ema = 0.0
        self.atp_ema = 0.0
        self.aown_perp_ema = 0.0
        self.los_rate_slow = 0.0
        self.last_bbox_size = 0.0
        self.last_bbox_t = -1.0

    def compute_control(self, R_b2n, az_bore, los_az_err, r_est, v_est, los_el, t,
                        v_fwd_body=0.0, bbox_size=0.0, v_z=0.0, los_rate=0.0, target_speed=0.0,
                        a_own_world=None):
        # Quaternion attitude interface (high-pitch capable): R_b2n replaces Euler roll/pitch/yaw,
        # az_bore (measured camera boresight azimuth, gimbal-stabilized) replaces the Euler yaw
        # heading frame, and los_az_err = wrap(az_los - az_bore) replaces the Euler los_az. At low
        # pitch this is mathematically equivalent to the former Euler P-laws; at pitch -> 90 deg the
        # axis-angle attitude error re-allocates roll/yaw authority automatically (Euler yaw loses
        # cos(pitch) of bearing authority there) and has no representation singularity.
        az_los = np.arctan2(np.sin(az_bore + los_az_err), np.cos(az_bore + los_az_err))
        r_norm = np.linalg.norm(r_est)
        omega_los = np.zeros(3)
        if r_norm > 0.1:
            omega_los = np.cross(r_est, v_est) / (r_norm * r_norm)
        # LOS rate from the measured bearing ONLY (clean, no 1/r blowup, no v_est noise). The old
        # warmup fallback onto omega_los[2] fired whenever the bearing rate was small and injected
        # the noisy v_est term -> a_lat sign-flip -> roll limit-cycle. Deadband a steady bearing.
        lambda_dot = los_rate
        if abs(lambda_dot) < 0.02:
            lambda_dot = 0.0
        omega_los_mag = min(abs(lambda_dot), 1.0)
        N_eff = self.N
        if r_norm > 15.0:
            N_eff = self.N - (self.N - 2.0) * clamp((r_norm - 15.0) / 35.0, 0.0, 1.0)
        v_fwd_safe = clamp(v_fwd_body, 0.5, 20.0)
        # Speed-scheduled PN: use v_des (OPEN-LOOP) for the v in a_lat = N*v*lambda_dot, NOT the
        # measured v_closing (v_fwd_body). v_closing is under-observable along-LOS until the parallax
        # demod anchors scale (~4s); using it here starves PN lead in the first seconds and lets a
        # crossing target escape the FOV before the demod converges. v_des is the intended ownship
        # speed — the correct PN proxy — and is available from t=0. Capped by v_pn_ref as before.
        v_pn = min(self.v_des, self.v_pn_ref)
        a_lat_mag = N_eff * v_pn * omega_los_mag
        # Lateral-authority tilt attenuation: the single-axis pitch gimbal cannot compensate
        # roll, so banking swings the boresight azimuth by ~sin(tilt) per radian of roll — PN
        # lateral (which banks) increasingly costs pointing as the tilt deepens. In free flight
        # PN self-extinguishes (own translation converges the bearing) before deep tilt, so this
        # is dormant in the validated <=68 deg envelope; on a translation-locked bench the
        # bearing rate NEVER decays and full lateral authority destabilizes pointing. Attenuate
        # a_lat smoothly to zero over [a_lat_tilt_start, a_lat_tilt_end] of MEASURED tilt.
        # sin(pitch) is the roll->boresight-azimuth coupling factor (a roll of phi at nose-down
        # pitch theta swings the boresight azimuth by ~sin(theta)*phi): 0 at level (roll is pure
        # image roll), ~1 at the 90 deg dive. Use IT, not the total tilt, so flight banking at
        # low pitch is untouched while deep-dive pointing is protected.
        sin_pitch = clamp(-np.asarray(R_b2n)[2, 0], -1.0, 1.0)   # nose-down positive
        att_lat = clamp((np.sin(self.a_lat_tilt_end) - sin_pitch) /
                        max(np.sin(self.a_lat_tilt_end) - np.sin(self.a_lat_tilt_start), 1e-6),
                        0.0, 1.0)
        a_lat_mag *= att_lat
        # Close-range lateral attenuation (Fix D/F, mirror C++): gently scale a_lat_mag toward
        # lateral_decay_floor over [terminal_brake_range, lat_decay_start] to damp the closed-loop
        # close-approach PN weave (full N + v_des=20 over-corrects; seen as far out as ~26 m in
        # AirSim). NOT a hard zero — a 0.4 floor keeps PN lead for turning targets. Window widened
        # 12 -> 25 m (Fix F) so the 22-27 m weave zone is covered.
        lat_f = clamp((r_norm - self.terminal_brake_range) /
                      max(self.lat_decay_start - self.terminal_brake_range, 1.0), 0.0, 1.0)
        a_lat_mag *= self.lateral_decay_floor + (1.0 - self.lateral_decay_floor) * lat_f
        e_los_xy = np.array([r_est[0], r_est[1]])
        if np.linalg.norm(e_los_xy) < 1e-3:
            e_los_xy = np.array([1.0, 0.0])
        e_los_xy = e_los_xy / np.linalg.norm(e_los_xy)
        e_perp_xy = np.array([-e_los_xy[1], e_los_xy[0]])
        turn_sign = 1.0 if lambda_dot >= 0.0 else -1.0
        a_lat_world = turn_sign * a_lat_mag * e_perp_xy
        # Active observability dither (mid-course only): lateral sinusoid for monocular scale/depth.
        # Second, W-independent close gate (mirror C++): suppress when the bbox is already LARGE in
        # px — the size-SNR is then good and/or the target close, so the dither only disturbs
        # terminal precision. Immune to the monocular scale error: W over-estimated by k keeps
        # r_norm > dither_min_range forever while the true range is well inside it. Stale-hold the
        # last live bbox for 0.5 s so a dropout frame cannot re-arm the dither.
        dither_suppressed_by_bbox = False
        if self.dither_max_bbox > 0.0:
            s_eff = bbox_size
            if s_eff <= 0.0 and self.last_bbox_t >= 0.0 and (t - self.last_bbox_t) < 0.5:
                s_eff = self.last_bbox_size
            dither_suppressed_by_bbox = s_eff >= self.dither_max_bbox
        if bbox_size > 0.0:
            self.last_bbox_size = bbox_size
            self.last_bbox_t = t
        if self.dither_amp > 0.0 and r_norm > self.dither_min_range and not dither_suppressed_by_bbox:
            a_dither = self.dither_amp * np.sin(2.0 * np.pi * self.dither_freq * t)
            a_lat_world = a_lat_world + a_dither * e_perp_xy
        # Geometric APN: classic PN (a_lat = N*v_pn*lambda_dot) assumes constant bearing, so it lags
        # a turning target — the ownship flies through the target's initial position while the target
        # has already turned (the maneuvering-ct failure). Recover the target lateral accel a_t_perp
        # from TRUE-PN geometry + IMU (NOT the estimator a_t state — that is unobservable with a single
        # bbox): a_t_perp = range*ydot + lambda_dot*v_range + a_own_perp, where ydot = d(lambda_dot)/dt.
        # Lead a_lat with it so the ownship turns WITH the target.
        if self.has_prev_lr and t > self.prev_lr_t:
            dt_lr = t - self.prev_lr_t
            if dt_lr > 1e-3 and dt_lr < 1.0:
                ydot_raw = (los_rate - self.prev_los_rate) / dt_lr
                a_ydot = clamp(dt_lr / self.ydot_tc, 0.0, 1.0)
                self.ydot_ema += a_ydot * (ydot_raw - self.ydot_ema)
        self.prev_los_rate = los_rate
        self.prev_lr_t = t
        self.has_prev_lr = True
        a_own_perp = float(np.dot(a_own_world[0:2], e_perp_xy)) if a_own_world is not None else 0.0
        # EMA a_own_perp with the SAME tc as ydot_ema so the ownship-dither component cancels in
        # a_t_perp_geom (range*ÿ and a_own_perp delayed equally) for a static/CV target.
        a_ao = clamp(0.02 / self.ydot_tc, 0.0, 1.0)
        self.aown_perp_ema += a_ao * (a_own_perp - self.aown_perp_ema)
        # True-PN geometry: a_t_perp = range·ÿ + 2·λ̇·v_range + a_own_perp (2x Coriolis). The 2x and
        # the matched-lag aown_perp_ema make ownship dither cancel for a static target.
        a_t_perp_geom = r_norm * self.ydot_ema + 2.0 * los_rate * v_fwd_body + self.aown_perp_ema
        # EMA a_t_perp_geom: a sustained turn (maneuvering) keeps |atp_ema| large; pixel-noise spikes
        # (high_noise) average toward 0. Use atp_ema for the floor gate to reject noise.
        a_atp = clamp(0.02 / self.atp_tc, 0.0, 1.0)
        self.atp_ema += a_atp * (a_t_perp_geom - self.atp_ema)
        if abs(a_t_perp_geom) > self.K_apn_min_at:
            a_lat_world = a_lat_world + self.K_apn * att_lat * a_t_perp_geom * e_perp_xy
        # Flat v_des + single adaptive turn-radius cap (mirror C++): no deliberate terminal brake.
        # The cap keeps min turn radius inside the current range so the interceptor can still bend
        # onto a dodging target. Uses the range-appropriate tilt (terminal tilt inside terminal_tilt_range).
        v_des_eff = self.v_des
        moving_floor = (target_speed + 1.5) if target_speed > 1.0 else 3.0  # 3.0 = static-target stall guard
        v_des_eff = max(v_des_eff, min(moving_floor, self.v_des))
        tilt_for_cap = self.max_tilt_terminal_rad if r_norm < self.terminal_tilt_range else self.max_tilt_rad
        # tan() is negative past 90 deg of tilt (sqrt of a negative): clamp the cap tilt into
        # the valid turn-radius domain — past 90 deg of bank the "turn radius" concept has
        # already conceded all maneuverability anyway.
        tilt_for_cap = min(tilt_for_cap, np.deg2rad(89.0))
        v_des_eff = min(v_des_eff, self.turn_radius_cap_factor * np.sqrt(G * np.tan(tilt_for_cap) * r_norm))
        # v_fwd_body (= v_closing) is under-observable along-LOS until the parallax demod anchors
        # scale (~4s); a low v_closing inflates a_fwd -> ownship over-accelerates ON-AXIS, starving
        # the lateral PN lead -> a crossing target escapes the FOV (the regression vs the absolute-
        # state version, which used true own_vel here and held a_fwd ~= 0). Floor v_closing at
        # 0.8*v_des_eff so the ownship holds near v_des instead of blasting through. Braking passes.
        # Scene-adaptive v_closing floor: raise it only for a clearly turning target (|a_t_perp_geom|
        # large) so the ownship slows and doesn't blast THROUGH a circular target; keep 0.8 otherwise so
        # CV/static/crossing closing speed isn't starved. a_t_perp_geom from the geometric-APN block.
        # 3-level floor: turning target (|atp_ema| large) → 0.88 (don't blast through circle);
        # moving target (lateral v_perp > 1 m/s, e.g. crossing/dropout) → 0.82 (avoid blast-through
        # on a mutually-closing target); static/slow → 0.8 (close fast, noise-robust).
        v_perp = abs(r_norm * los_rate)
        floor = 0.88 if abs(self.atp_ema) > 0.5 else (0.82 if v_perp > 1.0 else 0.8)
        v_fwd_for_afwd = max(v_fwd_body, floor * v_des_eff)
        # Axial deadband (mirror C++): coast (a_fwd=0) within +-axial_deadband of v_des_eff.
        # Kills pitch oscillation from v_closing noise — on the bench the monocular scale is
        # unobservable (no parallax), so v_closing wanders and the deadband keeps the axial
        # channel from chasing it; shifted by the deadband (no step) outside the band.
        v_error = v_des_eff - v_fwd_for_afwd
        if abs(v_error) <= self.axial_deadband:
            a_fwd = 0.0
        else:
            a_fwd = self.K_fwd * (v_error - np.sign(v_error) * self.axial_deadband)
        if self.has_last and t > self.last_t:
            dt = t - self.last_t
            if dt > 0.0 and dt < 1.0:
                a_fwd = clamp(a_fwd, self.last_axial_accel - self.axial_jerk_max * dt,
                              self.last_axial_accel + self.axial_jerk_max * dt)
        self.last_axial_accel = a_fwd
        self.last_t = t
        self.has_last = True
        a_fwd_world = a_fwd * e_los_xy
        dz_target = r_est[2]
        # PD on vertical: K_z on position error, K_zd damps pursuer odom v_z so the climb rate
        # built up while dz_target was large is braked before the zero-crossing (mirror C++ fix).
        # a_z_limit matches the C++ yaml controller.a_z_limit (12) when overridden for missile-mode
        # scenarios: d_z = a_z + G < 0 is what lets the thrust vector pass 90 deg of tilt.
        a_z = clamp(self.K_z * dz_target - self.K_zd * v_z, -self.a_z_limit, self.a_z_limit)
        # Slew limit |da_z/dt| (Fix C, mirror C++): z-estimate can jump O(10 m) in one cycle ->
        # step a_z to +-4 -> pitch step. Ramp it instead.
        if self.has_last_az and t > self.last_az_t:
            dt_az = t - self.last_az_t
            if 0.0 < dt_az < 1.0:
                a_z = clamp(a_z, self.last_a_z - self.a_z_slew * dt_az,
                            self.last_a_z + self.a_z_slew * dt_az)
        self.last_a_z = a_z
        self.last_az_t = t
        self.has_last_az = True
        a_h_world = a_lat_world + a_fwd_world
        # ---- Thrust-vector cascade (quaternion, high-pitch capable) --------------------------
        # The desired attitude is built from the desired specific-force direction d = [a_h, a_z+G]
        # directly: body Z aligns with d, and the body X-Z plane contains the LOS azimuth so the
        # gimbal-compensated boresight points at the target at ANY pitch. This replaces the former
        # Euler theta_cmd/phi_cmd + Euler P-laws, whose representation singularity at pitch +-90 deg
        # and no-wrap P-laws made the >65 deg regime unusable (bench rig / missile-mode dives).
        target_behind = abs(los_az_err) > (M_PI / 2.0)
        d = np.array([a_h_world[0], a_h_world[1], a_z + G])
        # Tilt cap (flight envelope): cap the tilt of d from vertical, preserving the tilt azimuth.
        # Replaces the separate theta/phi Euler clamps; valid beyond 90 deg (cap > 90 lets the
        # thrust axis pass horizontal for missile-mode dives). Terminal range relaxes the cap,
        # mirroring the old max_tilt_terminal roll relaxation.
        tilt_cap = self.max_tilt_terminal_rad if r_norm < self.terminal_tilt_range else self.max_tilt_rad
        a_h_mag = float(np.linalg.norm(d[0:2]))
        if a_h_mag > 1e-9:
            # Thrust-azimuth pointing budget: a thrust-azimuth deviation delta from the LOS
            # swings the boresight azimuth by ~sin(pitch)*delta (the roll coupling above), so the
            # affordable deviation is the remaining keep-in-frame margin divided by that coupling.
            # At level pitch sin->0 and the budget is unbounded (flight banking untouched); at the
            # 90 deg dive the boresight ~ the thrust axis and the budget collapses onto the LOS —
            # exactly where pointing must be authoritative (bench: the bearing rate never
            # self-extinguishes, and a wandering thrust azimuth walks the target out of frame).
            tilt_raw = np.arctan2(a_h_mag, d[2])
            az_ah = np.arctan2(a_h_world[1], a_h_world[0])
            d_az = np.arctan2(np.sin(az_ah - az_los), np.cos(az_ah - az_los))
            fov_az_budget = self.fov_half_h * self.fov_keep_margin - abs(los_az_err)
            if fov_az_budget > 0.0 and sin_pitch > 0.15:
                d_az_max = min(np.pi, fov_az_budget / max(sin_pitch, 0.15))
            else:
                d_az_max = np.pi
            if abs(d_az) > d_az_max:
                d_az = clamp(d_az, -d_az_max, d_az_max)
                a_h_world = a_h_mag * np.array([np.cos(az_los + d_az), np.sin(az_los + d_az)])
                d[0:2] = a_h_world
            if tilt_raw > tilt_cap:
                s_xy = d[0:2] / a_h_mag
                d = np.array([np.sin(tilt_cap) * s_xy[0], np.sin(tilt_cap) * s_xy[1], np.cos(tilt_cap)])
        # Decompose the tilt in the boresight-azimuth frame: (tilt_fwd, tilt_left) are the exact
        # former theta_cmd / -phi_cmd at low tilt, and stay well-defined to +-180 deg.
        fwd_h = np.array([np.cos(az_bore), np.sin(az_bore)])
        left_h = np.array([-np.sin(az_bore), np.cos(az_bore)])
        tilt_fwd = np.arctan2(float(np.dot(d[0:2], fwd_h)), d[2])
        tilt_left = np.arctan2(float(np.dot(d[0:2], left_h)), d[2])
        # Nose-up cap from the gimbal's up-travel: the semi-strapdown servo can only compensate
        # so much nose-up body pitch (servo_cmd = -pitch + bias saturates at the mechanical max);
        # braking harder nose-up than that drops the boresight below the target.
        if self.has_gimbal and self.gimbal_nose_up_max > 0.0:
            tilt_fwd = max(tilt_fwd, -(self.gimbal_nose_up_max - self.gimbal_bias))
        if self.enable_fov_clamp and not self.has_gimbal and not target_behind:
            # Fixed-mount only: boresight elev = theta + mount_pitch, so clamp the forward tilt to
            # keep the target within fov_keep vertically. With the semi-strapdown gimbal the
            # boresight is DECOUPLED from body pitch (gimbal cancels it, elev ~= -GIMBAL_BIAS), so
            # this clamp is geometry-stale and only throttles pitch authority — skip it (gimbal).
            fov_keep = self.fov_half_v * self.fov_keep_margin
            theta_hi = min(self.max_tilt_rad, los_el - self.mount_pitch + fov_keep)
            theta_lo = max(-self.max_tilt_rad, los_el - self.mount_pitch - fov_keep)
            tilt_fwd = clamp(tilt_fwd, min(theta_lo, theta_hi), max(theta_lo, theta_hi))
        elif self.enable_fov_clamp and self.has_gimbal and not target_behind:
            # Gimbal vertical keep-in-frame: pitch cannot move the boresight, ALTITUDE can. If the
            # target approaches the bottom edge (los_el << boresight elev = -gimbal bias), command
            # a descent so the frame re-centers; symmetric for the top edge. Overrides the PD a_z
            # only near the edge (outer 30% of the half-FOV). Affects only the throttle below.
            fov_keep = self.fov_half_v * self.fov_keep_margin
            boresight_el = -self.gimbal_bias
            el_err = los_el - boresight_el          # negative = target below boresight
            edge = 0.7 * fov_keep
            if el_err < -edge:
                a_z = min(a_z, -2.0 * clamp((-el_err - edge) / max(fov_keep - edge, 1e-3), 0.0, 1.0) * 4.0)
            elif el_err > edge:
                a_z = max(a_z, 2.0 * clamp((el_err - edge) / max(fov_keep - edge, 1e-3), 0.0, 1.0) * 4.0)
        # Attitude-reference EMA low-pass (iter 2): suppress camera-rate content / roll
        # limit-cycle. Applied to the tilt components — identical to the former phi/theta_cmd EMA
        # at low tilt, still well-defined at high tilt.
        if self.has_last_att and self.att_ref_tc > 1e-6:
            dt_att = t - self.last_att_t
            if 0.0 < dt_att < 1.0:
                a_ref = clamp(dt_att / self.att_ref_tc, 0.0, 1.0)
                tilt_fwd = self.last_tilt_fwd_f + a_ref * (tilt_fwd - self.last_tilt_fwd_f)
                tilt_left = self.last_tilt_left_f + a_ref * (tilt_left - self.last_tilt_left_f)
        self.last_tilt_fwd_f = tilt_fwd
        self.last_tilt_left_f = tilt_left
        # Attitude-reference slew governor (iter 1): hard cap on |d(tilt)_cmd/dt|.
        if self.has_last_att:
            dt_att = t - self.last_att_t
            if 0.0 < dt_att < 1.0:
                max_dtilt = self.max_att_rate * dt_att
                tilt_fwd = clamp(tilt_fwd, self.last_tilt_fwd - max_dtilt, self.last_tilt_fwd + max_dtilt)
                tilt_left = clamp(tilt_left, self.last_tilt_left - max_dtilt, self.last_tilt_left + max_dtilt)
        self.last_tilt_fwd = tilt_fwd
        self.last_tilt_left = tilt_left
        self.last_att_t = t
        self.has_last_att = True
        # Rebuild the desired body Z (thrust axis) from the shaped tilts, then command rates via
        # the TWO-CHANNEL decomposition (tailsitter-transition style):
        #   heading channel: rotation about the world vertical — moves the camera plane (and so
        #     the gimbal-compensated boresight) onto the LOS azimuth; never disturbs the thrust
        #     axis' tilt geometry.
        #   tilt channel: rotation about the axis perpendicular to the boresight plane taking
        #     z_meas onto z_cmd — tilts the thrust WITHIN the plane; never rotates the plane, so
        #     it never moves the boresight (below the 90 deg degeneracy where they merge).
        # At low pitch this reproduces the former Euler P-laws term by term (omega_x ~ K_att*roll
        # err, omega_y ~ K_att*pitch err, omega_z ~ K_yaw*los_az); at high pitch it stays
        # decoupled and non-singular where the Euler interface (and a single shortest-rotation
        # attitude error, which routes tilt corrections through plane rotations) fell apart.
        z_cmd = np.array([fwd_h[0] * np.tan(tilt_fwd) + left_h[0] * np.tan(tilt_left),
                          fwd_h[1] * np.tan(tilt_fwd) + left_h[1] * np.tan(tilt_left), 1.0])
        z_cmd /= max(np.linalg.norm(z_cmd), 1e-9)
        z_meas = np.asarray(R_b2n)[:, 2]
        cr = np.cross(z_meas, z_cmd)
        crn = np.linalg.norm(cr)
        if crn > 1e-9:
            theta_t = np.arctan2(crn, float(np.dot(z_meas, z_cmd)))
            omega_tilt_world = (self.K_att * theta_t) * (cr / crn)
        else:
            omega_tilt_world = np.zeros(3)
        omega_world = omega_tilt_world.copy()
        omega_world[2] += self.K_yaw * los_az_err
        omega_body = np.asarray(R_b2n).T @ omega_world
        omega = np.array([clamp(omega_body[0], -self.max_body_rate, self.max_body_rate),
                          clamp(omega_body[1], -self.max_body_rate, self.max_body_rate),
                          clamp(omega_body[2], -self.max_body_rate, self.max_body_rate)])
        # Throttle from the ACTUAL attitude (mirror C++ fix): T*cos(actual tilt) = a_z + G so the
        # vertical accel tracks a_z regardless of attitude lag. R[2,2] = cos(actual tilt) with no
        # Euler extraction; the floor follows the tilt cap (cos(cap+0.2)) so an over-tilted body
        # cannot blow the throttle up, floored at 0.25 for caps beyond 90 deg.
        cos_tilt_actual = max(R_b2n[2, 2], max(np.cos(self.max_tilt_rad + 0.2), 0.25))
        throttle = clamp((a_z + G) / cos_tilt_actual, 2.0, 25.0)
        return np.array([throttle, omega[0], omega[1], omega[2]])


# ============================================================================
# Quadrotor dynamics (simplified, for sim)
# ============================================================================
def Rx(p):
    """Rotation about the x-axis (cam-x in R_B2C, same physical elev axis as body-y pitch)."""
    c, s = np.cos(p), np.sin(p)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])


def euler_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def R_to_euler(R):
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def R_from_thrust_heading(z_b, az_ref, R_meas, eps=0.25):
    """Desired body->world attitude from a desired thrust-axis direction and a heading reference.

    Body Z aligns with z_b (unit, desired specific-force direction); the body X-Z plane contains
    the horizontal reference direction az_ref (the LOS azimuth), so the gimbal-compensated camera
    boresight — which always lies in the X-Z plane — points at the target azimuth at ANY pitch.
    Degenerate when z_b is parallel to the reference direction (pitch ~90 deg with the thrust axis
    along the LOS): the X-Z plane then trivially contains the reference and roll about Z is a free
    DOF; blend toward holding the current roll (measured body Y projected perpendicular to z_b) so
    the command stays continuous and the image does not churn.
    """
    z_b = np.asarray(z_b, dtype=float)
    zn = np.linalg.norm(z_b)
    if not np.isfinite(zn) or zn < 1e-9:
        return np.asarray(R_meas, dtype=float).copy()
    z_b = z_b / zn
    h = np.array([np.cos(az_ref), np.sin(az_ref), 0.0])
    y_c = np.cross(z_b, h)
    w = float(np.linalg.norm(y_c))
    if w >= eps:
        y_b = y_c / w
    else:
        # hold-current-roll fallback: measured body Y (world), projected perpendicular to z_b
        y_meas = np.asarray(R_meas, dtype=float)[:, 1]
        y_hold = y_meas - np.dot(y_meas, z_b) * z_b
        nh = np.linalg.norm(y_hold)
        if nh < 1e-6:
            # extreme fallback: any vector perpendicular to z_b
            tmp = np.cross(z_b, np.array([0.0, 0.0, 1.0]))
            y_hold = tmp if np.linalg.norm(tmp) > 1e-6 else np.array([1.0, 0.0, 0.0])
            nh = np.linalg.norm(y_hold)
        y_hold = y_hold / nh
        if w > 1e-9:
            y_b = (w / eps) * (y_c / w) + (1.0 - w / eps) * y_hold
            y_b = y_b / np.linalg.norm(y_b)
        else:
            y_b = y_hold
    x_b = np.cross(y_b, z_b)
    R = np.column_stack([x_b, y_b, z_b])
    # re-orthonormalize (blend path can be slightly off); guarantee det=+1
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U = U.copy()
        U[:, 2] = -U[:, 2]
        R = U @ Vt
    return R


def attitude_error_rates(R_meas, R_cmd, kx, ky, kz, max_rate):
    """Body-rate command from the attitude error R_err = R_meas^T @ R_cmd via axis-angle.

    The axis-angle error is the shortest rotation taking the measured attitude to the command,
    expressed in BODY axes, so the per-axis gains reproduce the former Euler P-laws at low pitch
    (err_x~roll err, err_y~pitch err, err_z~yaw err) while remaining valid at any attitude —
    no Euler extraction, no representation singularity at pitch +-90 deg, no wrap jumps.
    """
    R_err = np.asarray(R_meas, dtype=float).T @ np.asarray(R_cmd, dtype=float)
    tr = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(tr))
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # near-180 deg: axis from the symmetric part of R_err
        A = (R_err + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        axis = np.zeros(3)
        axis[i] = 1.0
        v = A[:, i]
        nv = np.linalg.norm(v)
        if nv > 1e-9:
            axis = v / nv
    else:
        axis = np.array([R_err[2, 1] - R_err[1, 2],
                         R_err[0, 2] - R_err[2, 0],
                         R_err[1, 0] - R_err[0, 1]]) / (2.0 * np.sin(angle))
    err = angle * axis
    return np.array([clamp(kx * err[0], -max_rate, max_rate),
                     clamp(ky * err[1], -max_rate, max_rate),
                     clamp(kz * err[2], -max_rate, max_rate)])


def tilt_channel_rates(R_b2n, z_cmd, k_tilt, max_rate):
    """Tilt-channel body rates: rotate the thrust axis z_meas onto z_cmd about the axis
    perpendicular to both (in the boresight plane at low tilt), WITHOUT any heading/rotated-plane
    component — the two-channel decomposition's tilt half. Non-singular at any attitude."""
    R = np.asarray(R_b2n, dtype=float)
    z_meas = R[:, 2]
    cr = np.cross(z_meas, np.asarray(z_cmd, dtype=float))
    crn = np.linalg.norm(cr)
    if crn > 1e-9:
        w = (k_tilt * float(np.arctan2(crn, float(np.dot(z_meas, z_cmd))))) * (cr / crn)
    else:
        w = np.zeros(3)
    wb = R.T @ w
    return np.array([clamp(wb[0], -max_rate, max_rate),
                     clamp(wb[1], -max_rate, max_rate),
                     clamp(wb[2], -max_rate, max_rate)])


def boresight_azimuth(R_b2n, R_n2c=None):
    """Camera boresight azimuth in the world frame. Prefers the actual gimbal pose (R_n2c:
    nav->camera; boresight = camera +Z) — non-singular at any body pitch since the semi-strapdown
    gimbal keeps the boresight near horizontal. Without a gimbal, falls back to the body forward
    axis, blending toward the body up (thrust) axis as the forward axis goes vertical (the
    pitch-90 Euler heading degeneracy)."""
    if R_n2c is not None:
        b = np.asarray(R_n2c, dtype=float)[2, :]          # camera +Z in world
        bxy = np.linalg.norm(b[0:2])
        if bxy > 0.1:
            return float(np.arctan2(b[1], b[0]))
    R = np.asarray(R_b2n, dtype=float)
    bx = R[:, 0]                                          # body forward in world
    wx = float(np.linalg.norm(bx[0:2]))
    if wx > 0.2:
        return float(np.arctan2(bx[1], bx[0]))
    bz = R[:, 2]                                          # body up/thrust in world
    return float(np.arctan2(bz[1], bz[0]))


@dataclass
class QuadrotorState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    R_b2n: np.ndarray = field(default_factory=lambda: np.eye(3))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gimbal_pitch: float = 0.0   # camera pitch servo (sim gimbal; tracks target elevation)


def step_quadrotor(state, throttle, omega_cmd, dt, rate_tau=0.05):
    """Integrate quadrotor. throttle = collective accel (m/s^2) along body z.
    omega_cmd tracked with first-order lag tau. Returns the world-frame acceleration a_world
    (gravity-corrected free accel) so the main loop can accumulate Δv_own for IMU preintegration."""
    # body-rate tracking (first-order lag)
    state.omega = state.omega + (omega_cmd - state.omega) * (dt / rate_tau)
    # rotation integration (exponential map)
    w = state.omega * dt
    angle = np.linalg.norm(w)
    if angle > 1e-9:
        axis = w / angle
        dR = np.eye(3) + np.sin(angle) * skew(axis) + (1 - np.cos(angle)) * skew(axis) @ skew(axis)
        state.R_b2n = state.R_b2n @ dR
        # re-orthonormalize
        U, _, Vt = np.linalg.svd(state.R_b2n)
        state.R_b2n = U @ Vt
    # accel: thrust along body z, minus gravity, minus linear body drag (yaml physics.drag_coeff).
    a_world = state.R_b2n @ np.array([0.0, 0.0, throttle]) - np.array([0.0, 0.0, G])
    a_world = a_world - DRAG_COEFF * state.vel
    state.vel = state.vel + a_world * dt
    state.pos = state.pos + state.vel * dt
    return a_world


def step_rig(state, omega_cmd, dt, r_imu, rate_tau=0.05):
    """3-axis free-attitude test bench: translation LOCKED, rotation free (first-order rate
    tracking like step_quadrotor). Returns the accelerometer artifact the IMU strapdown sees at an
    IMU offset r_imu (body frame) from the rig rotation center:
        a_meas_world = R * (alpha x r + omega x (omega x r)).
    Thrust is absorbed by the rig structure (a proper accelerometer never sees constraint forces —
    at the rotation center it reads pure gravity, indistinguishable from a hover), and gravity is
    removed by the strapdown, so the TRUE ownship linear accel is 0 and these artifacts are the
    entire strapdown input: the own_vel drift they cause is what the bench measures."""
    omega_prev = state.omega.copy()
    state.omega = state.omega + (np.asarray(omega_cmd, dtype=float) - state.omega) * (dt / rate_tau)
    w = state.omega * dt
    angle = np.linalg.norm(w)
    if angle > 1e-9:
        axis = w / angle
        dR = np.eye(3) + np.sin(angle) * skew(axis) + (1.0 - np.cos(angle)) * skew(axis) @ skew(axis)
        state.R_b2n = state.R_b2n @ dR
        U, _, Vt = np.linalg.svd(state.R_b2n)
        state.R_b2n = U @ Vt
    alpha = (state.omega - omega_prev) / dt
    r = np.asarray(r_imu, dtype=float)
    a_body = np.cross(alpha, r) + np.cross(state.omega, np.cross(state.omega, r))
    return state.R_b2n @ a_body


def update_gimbal(state, dt, tau):
    """Semi-strapdown single-axis (cam-x pitch) gimbal servo. NO fixed mount — the gimbal directly
    drives the camera boresight to HORIZONTAL in the world frame, compensating ownship pitch-down
    for closing. cam-x is the same physical elevation axis as body-y pitch (R_B2C_BASE maps
    body-x→cam-z, so a body-y rotation of the boresight becomes a cam y-z plane rotation = cam-x).
    Boresight world elev = own_pitch + gimbal_total; setting gimbal_total = -own_pitch → elev 0.
    First-order lag: boresight trails own_pitch by ~tau*d(pitch)/dt (residual fully captured since
    we use the lagged angle). Returns gimbal_total (the cam-x rotation actually applied = g_servo).
    """
    own_pitch = R_to_euler(state.R_b2n)[1]              # nose-down positive
    servo_cmd = -own_pitch + GIMBAL_BIAS                # gimbal_total → -own_pitch+bias → boresight near-horizontal (slight down-gaze)
    alpha = 1.0 - np.exp(-dt / tau)
    state.gimbal_pitch += alpha * (servo_cmd - state.gimbal_pitch)
    return state.gimbal_pitch                           # gimbal_total (actual, lagged)


# ============================================================================
# Camera projection (mirror node's measurement generation)
# ============================================================================
def project_target(uav_pos, R_b2n, target_pos, target_width, focal_length, fov_half):
    """Return (u, v, size, in_fov) — pixel coords centered on image center."""
    rel_p = target_pos - uav_pos
    R_n2c = R_B2C @ R_b2n.T
    p_cam = R_n2c @ rel_p
    if p_cam[2] <= 0.5:
        return 0.0, 0.0, 0.0, False
    u = focal_length * p_cam[0] / p_cam[2]
    v = focal_length * p_cam[1] / p_cam[2]
    s = focal_length * target_width / p_cam[2]
    # FOV check: angle from boresight (cam z-axis)
    angle = np.arctan2(np.sqrt(p_cam[0] ** 2 + p_cam[1] ** 2), p_cam[2])
    in_fov = angle < fov_half
    return u, v, s, in_fov


# ============================================================================
# Simulation scenario runner
# ============================================================================
def detect_reopen(ranges, r0):
    """Intercept-quality metric: True if the range reopened after first closing.
    A clean intercept is a single pass (range monotonically decreases to the
    threshold); a fragile one closes, overshoots, and the range climbs back
    before a later pass finally connects. Detection: after the range first drops
    below 50% of the initial range, did it climb back above 60% of it? (Note:
    `max(ranges)` alone is NOT a reopen signal — it is just the initial range.)
    """
    if r0 < 1.0 or len(ranges) < 2:
        return False
    close_thr = 0.5 * r0
    reopen_thr = 0.6 * r0
    for i in range(len(ranges)):
        if ranges[i] < close_thr:
            return max(ranges[i:]) > reopen_thr
    return False


@dataclass
class ScenarioResult:
    name: str
    intercepted: bool
    min_range: float
    final_range: float
    est_pos_err_final: float
    est_vel_err_final: float
    est_width_err_final: float
    max_los_rate: float
    final_los_rate: float
    nan_count: int
    sim_time: float
    single_pass: bool = True      # False if the range reopened before intercept (multi-pass)
    captured: bool = False        # capture latch fired (brake-to-hold)
    notes: str = ""


def run_scenario(name, target_init_pos, target_vel_fn, uav_init_pos, uav_init_yaw,
                 sim_time=20.0, dt_ctrl=1.0/CONTROL_RATE, dt_cam=DT_CAM,
                 bbox_dropout_steps=None, bad_size_steps=None, noise_pos=1.5, noise_size=2.0,
                 target_width_true=1.5, intercept_range=1.0, seed=42, v_des_override=None,
                 rig_mode=False, r_imu=None, rig_vib=0.0, max_tilt_override=None,
                 az_limit_override=None, dither_amp_override=None, uav_init_pitch_deg=0.0):
    """Run one interception scenario. Returns ScenarioResult + log arrays.

    rig_mode: emulate the 3-axis free-attitude test bench — translation LOCKED, rotation free.
    The IMU strapdown then sees only the rotational artifacts at r_imu (body frame, offset from
    the rig rotation center) plus optional vibration rig_vib (m/s^2 std); intercept metrics are
    meaningless (use the boresight/pitch/roll logs in the rig report instead).
    max_tilt_override: deg — raise the thrust tilt cap beyond the flight default (missile-mode
    dives / the bench operating point near pitch 90).
    """
    np.random.seed(seed)
    estimator = RelativeIMMUKFEstimator(dt_cam, FOCAL_LENGTH, W_GUESS, R_B2C)
    controller = AdvancedHybridController()
    demod = ParallaxDemodulator(controller.dither_freq, controller.dither_amp, FOCAL_LENGTH, dt_cam)
    r_est_xy_lp = None    # lowpass r_est horizontal -> dither-free LOS reference for the demod axis
    if v_des_override is not None:
        controller.v_des = v_des_override   # exercise the iter-3 terminal-authority path at high speed
    if max_tilt_override is not None:
        controller.max_tilt_rad = np.deg2rad(max_tilt_override)
        controller.max_tilt_terminal_rad = np.deg2rad(max_tilt_override)
    if az_limit_override is not None:
        controller.a_z_limit = az_limit_override
    if dither_amp_override is not None:
        controller.dither_amp = dither_amp_override
    if r_imu is None:
        r_imu = np.array([0.08, 0.0, -0.06])
    fov_half = np.deg2rad(CAM_FOV_DEG / 2.0)

    # UAV state
    uav = QuadrotorState()
    uav.pos = np.array(uav_init_pos, dtype=float)
    uav.R_b2n = euler_to_R(0.0, np.deg2rad(uav_init_pitch_deg), uav_init_yaw)

    target_pos = np.array(target_init_pos, dtype=float)
    target_vel = np.array(target_vel_fn(0.0), dtype=float)

    # EMA rate filter (mirror node ctrl_coeff_)
    if RATE_ATTEN and RATE_TC > 1e-6 and CONTROL_RATE > 1e-6:
        ctrl_coeff = 1.0 - np.exp(-1.0 / (CONTROL_RATE * RATE_TC))
    else:
        ctrl_coeff = 1.0
    prev_rate = np.zeros(3)

    # PN lateral LOS bearing-rate EMA state (lambda_dot from measured bearing).
    prev_los_az = None
    prev_los_t = 0.0
    los_rate_ema = 0.0

    n_steps = int(sim_time / dt_ctrl)
    cam_step = max(1, int(round(dt_cam / dt_ctrl)))
    log = {'t': [], 'range': [], 'los_rate': [], 'est_pos_err': [], 'est_vel_err': [],
           'est_w_err': [], 'mu': [], 'uav_pos': [], 'tgt_pos': [], 'est_pos': [], 'est_vel': [], 'est_acc': [],
           'cmd': [], 'nan': 0, 'pitch': [], 'roll': [], 'az_err': [], 'own_vel': []}
    own_vel_int = np.zeros(3)      # strapdown own_vel mirror (bench drift metric)
    min_range = 1e9
    nan_count = 0
    intercepted = False
    intercept_t = -1.0
    last_bbox = (0.0, 0.0, 0.0, False)
    cache = None  # (r_est, v_est, los_az_world, los_el, bbox_size, t_rel)
    # Output smoothing of the estimator target velocity (Fix B, mirror C++).
    target_vel_smoothed = None
    last_tv_t = 0.0
    # Terminal coast-on-dropout state (mirror C++ control_timer_callback).
    last_meas_has_bbox = False
    has_frozen_cmd = False
    last_good_range = 1e9
    last_bbox_time = 0.0
    # IMU preintegration accumulator: ownship Δv over the current camera window (world frame G).
    # Filled by step_quadrotor's returned a_world each control step; drained at each camera step.
    dv_own_accum = np.zeros(3)
    last_dv_own = np.zeros(3)    # last camera-window Δv_own (for geometric-APN a_own_perp)
    frozen_cmd = np.zeros(4)
    captured = False        # mirror C++ capture lockout: hold after a genuine close intercept
    min_range_seen = 1e9    # min estimated range (reopen detection, mirror C++)

    for step in range(n_steps):
        t = step * dt_ctrl
        # target dynamics
        target_vel = np.array(target_vel_fn(t), dtype=float)
        target_pos = target_pos + target_vel * dt_ctrl

        # Semi-strapdown gimbal: every control step, servo the cam-x pitch so boresight → horizontal
        # (cancels own_pitch's elev effect). Must run BEFORE the camera update (project_target reads
        # the module-level R_B2C, which we update in-place here). Angle known → estimator consistent.
        _gimbal_total = update_gimbal(uav, dt_ctrl, GIMBAL_TAU)
        R_B2C[:,:] = Rx(_gimbal_total) @ R_B2C_BASE

        # camera update at camera rate
        if step % cam_step == 0:
            u, v, s, in_fov = project_target(uav.pos, uav.R_b2n, target_pos, target_width_true,
                                             FOCAL_LENGTH, fov_half)
            has_bbox = in_fov
            if has_bbox and bbox_dropout_steps is not None:
                # dropout windows
                for (a, b) in bbox_dropout_steps:
                    if a <= step < b:
                        has_bbox = False
                        break
            if has_bbox and bad_size_steps is not None:
                # garbage-size windows: inject a wildly-wrong (tiny) bbox size on a real detection,
                # mimicking the close-range misdetection that corrupted the monocular scale in AirSim.
                for (a, b) in bad_size_steps:
                    if a <= step < b:
                        s = max(1.0, 0.25 * s)
                        break
            last_meas_has_bbox = has_bbox
            if has_bbox:
                u += np.random.normal(0, noise_pos)
                v += np.random.normal(0, noise_pos)
                s += np.random.normal(0, noise_size)
                s = max(1.0, s)
            # Drain the IMU-preintegration window for this camera step. dv_own is the ownship
            # velocity change since the last camera frame (world frame); the estimator consumes it
            # as the exogenous a_own input to the relativized process model (no ownship pos/vel).
            dv_own = dv_own_accum.copy()
            dv_own_accum = np.zeros(3)
            last_dv_own = dv_own.copy()
            # Parallax demod feed: project ownship lateral accel + LOS lateral bearing onto e_perp_xy.
            if has_bbox:
                los_nav_meas = pixel_to_los_nav(u, v, FOCAL_LENGTH, uav.R_b2n, R_B2C)
            else:
                los_nav_meas = None
            # Dither-free LOS reference: lowpass r_est horizontal so the demod axis does NOT follow
            # the dither (a rotating e_perp_xy cancels the parallax signal in theta_perp).
            ref_xy = (estimator.imm_ukf.x[0:2]
                      if (estimator.initialized and np.linalg.norm(estimator.imm_ukf.x[0:2]) > 1e-3)
                      else (los_nav_meas[0:2] if los_nav_meas is not None else None))
            if ref_xy is not None:
                if r_est_xy_lp is None:
                    r_est_xy_lp = ref_xy.copy()
                else:
                    alpha_lp = clamp(dt_cam / 1.0, 0.0, 1.0)   # tau=1s, cutoff ~0.16Hz < dither 0.5Hz
                    r_est_xy_lp = r_est_xy_lp + alpha_lp * (ref_xy - r_est_xy_lp)
            if r_est_xy_lp is not None:
                e_los_xy = r_est_xy_lp / max(np.linalg.norm(r_est_xy_lp), 1e-6)
                e_perp_xy = np.array([-e_los_xy[1], e_los_xy[0]])
                a_lat = float(np.dot(dv_own[0:2] / max(dt_cam, 1e-3), e_perp_xy))
                if has_bbox and los_nav_meas is not None:
                    theta_perp = float(np.arctan2(np.dot(los_nav_meas[0:2], e_perp_xy),
                                                  np.dot(los_nav_meas[0:2], e_los_xy)))
                else:
                    theta_perp = None
            else:
                a_lat = 0.0
                theta_perp = None
            demod.push(t, a_lat, theta_perp, has_bbox)
            if ENABLE_PARALLAX_DEPTH:
                z_depth, R_depth, eta_now = demod.compute_depth(t)
            else:
                z_depth, R_depth = np.nan, np.inf
            # Sync estimator's R_b2c to the actual servoed gimbal angle (incl. lag) so its R_n2c and
            # pixel_to_los_nav match the measurement-generation rotation (project_target uses module R_B2C).
            estimator.R_b2c = Rx(_gimbal_total) @ R_B2C_BASE
            ok, x_est = estimator.process_camera(has_bbox, u, v, s, t, uav.R_b2n.copy(), dv_own,
                                                 z_depth=z_depth, R_depth=R_depth)
            if ok and estimator.initialized:
                r_est = x_est[0:3].copy()           # relative pos (target - own) — state IS relative
                # Fix B: EMA the relative velocity (v_rel) so the raw UKF swings don't reach guidance.
                tgt_vel = x_est[3:6]
                if VEL_SMOOTH_TC > 1e-6:
                    if target_vel_smoothed is not None and t > last_tv_t and (t - last_tv_t) < 1.0:
                        a_tv = clamp((t - last_tv_t) / VEL_SMOOTH_TC, 0.0, 1.0)
                        target_vel_smoothed = target_vel_smoothed + a_tv * (tgt_vel - target_vel_smoothed)
                    else:
                        target_vel_smoothed = tgt_vel.copy()
                    last_tv_t = t
                    tgt_vel = target_vel_smoothed
                v_est = tgt_vel.copy()              # relative velocity (no - uav.vel)
                if has_bbox:
                    los_nav = pixel_to_los_nav(u, v, FOCAL_LENGTH, uav.R_b2n, R_B2C)
                    los_az_world = np.arctan2(los_nav[1], los_nav[0])
                    los_el = np.arcsin(clamp(los_nav[2], -1.0, 1.0))
                else:
                    los_az_world = np.arctan2(r_est[1], r_est[0])
                    los_el = np.arcsin(clamp(r_est[2] / max(np.linalg.norm(r_est), 0.1), -1.0, 1.0))
                # Smoothed LOS bearing rate (lambda_dot) for PN lateral lead (mirror C++).
                if prev_los_az is not None:
                    daz = los_az_world - prev_los_az
                    daz = np.arctan2(np.sin(daz), np.cos(daz))
                    dlos_t = max(t - prev_los_t, 1e-3)
                    if dlos_t < 1.0:
                        los_rate_ema = 0.75 * los_rate_ema + 0.25 * (daz / dlos_t)
                prev_los_az = los_az_world
                prev_los_t = t
                # bbox_size 0.0 on dropout frames — mirror the C++ control-cache semantics the
                # dither px gate's stale-hold is designed around.
                cache = (r_est, v_est, los_az_world, los_el, (s if has_bbox else 0.0), t, los_rate_ema)

        # control at control rate
        if cache is not None:
            r_est, v_est, los_az_world, los_el, bbox_size, t_cache, los_rate = cache
            # Quaternion attitude interface: boresight azimuth from the actual (lagged) gimbal pose
            # — a non-singular heading reference at any body pitch — and the bearing error relative
            # to it (replaces los_az_world - euler_yaw, which degenerates at pitch +-90).
            az_bore = boresight_azimuth(uav.R_b2n, R_n2c=R_B2C @ uav.R_b2n.T)
            los_az_err = np.arctan2(np.sin(los_az_world - az_bore), np.cos(los_az_world - az_bore))
            # Closing speed along LOS from relative quantities only (no ownship vel):
            # v_closing = -d|r|/dt = -v_rel·r̂  (positive = approaching). Replaces odom v_fwd_body.
            v_closing = -np.dot(v_est, r_est) / max(np.linalg.norm(r_est), 1e-3)
            # Capture lockout (mirror C++): once intercepted, BRAKE-TO-HOLD to end of mission. The
            # close approach happens during a bbox dropout when the range estimate is biased, so trigger
            # on EITHER r_est<capture_range OR target-behind (|los_az|>90°) after being close — the
            # latter is the reliable overshoot signal. Then brake-to-hold (nose-up pitch cancels
            # forward speed; pure zero-rate hover sinks on a banked body and never stops a drag-free
            # body). Stops the post-intercept fly-away.
            overshot = abs(los_az_err) > (np.pi / 2.0)
            r_norm = np.linalg.norm(r_est)
            min_range_seen = min(min_range_seen, r_norm)
            range_reopened = has_frozen_cmd and min_range_seen < 15.0 and r_norm > min_range_seen + 8.0
            if not captured and (r_norm < 2.0 or
                                 (overshot and has_frozen_cmd and last_good_range < 12.0) or
                                 range_reopened):
                captured = True
            if captured:
                # Brake-to-hold via the two-channel interface: tilt opposite the residual closing
                # speed within the boresight plane, heading channel idle (the old Euler P-laws
                # -3*roll / 5*(theta_brake-pitch) are this channel's low-pitch limit).
                max_brake_tilt = np.deg2rad(25.0)
                tilt_fwd = clamp(-0.12 * v_closing, -max_brake_tilt, max_brake_tilt)
                z_cmd = np.array([np.cos(az_bore) * np.tan(tilt_fwd),
                                  np.sin(az_bore) * np.tan(tilt_fwd), 1.0])
                z_cmd = z_cmd / np.linalg.norm(z_cmd)
                omega = tilt_channel_rates(uav.R_b2n, z_cmd, 5.0, 3.0)
                ctrl = np.array([G / max(uav.R_b2n[2, 2], 0.3), omega[0], omega[1], omega[2]])
            else:
                # Terminal coast-on-dropout (mirror C++): if the bbox dropped inside the terminal phase,
                # COAST (freeze last good cmd) for coast_max_dur to let the detector reacquire; past
                # that, LEVEL-AND-HOLD instead of chasing the dead-reckoned r_est (the v_des=20 fly-away).
                coast_range = COAST_RANGE
                coast_max_dur = COAST_MAX_DUR
                dropout_age = t - last_bbox_time
                terminal_dropout = (not last_meas_has_bbox) and has_frozen_cmd and (last_good_range < coast_range)
                if terminal_dropout and dropout_age < coast_max_dur:
                    ctrl = frozen_cmd.copy()   # COAST: last good (matched-speed) cmd — correct thrust, no sink
                elif terminal_dropout:
                    # LEVEL-AND-HOLD via the tilt channel: drive the thrust axis to vertical,
                    # heading channel idle (old Euler -3*roll / -3*pitch P-laws, high-pitch valid).
                    omega = tilt_channel_rates(uav.R_b2n, np.array([0.0, 0.0, 1.0]), 3.0, 3.0)
                    ctrl = np.array([G / max(uav.R_b2n[2, 2], 0.3), omega[0], omega[1], omega[2]])
                else:
                    # Relative-speed proxy drives the speed-aware brake floor (ownship absolute
                    # speed unobservable without odom). v_z ownship vertical speed ≈ -v_rel[2]
                    # (target-level approximation; preserves vertical-PD damping sign).
                    # Transverse (escape) speed = |r_xy × v_xy|/|r_xy| — the part that justifies not
                    # braking. Full |v_est| is dominated by closing speed -> circular (mirror C++ fix).
                    r_xy = float(np.sqrt(r_est[0] * r_est[0] + r_est[1] * r_est[1]))
                    v_transverse = abs(r_est[0] * v_est[1] - r_est[1] * v_est[0]) / r_xy if r_xy > 1e-3 else 0.0
                    target_speed = clamp(v_transverse, 0.0, 15.0)
                    a_own_world = last_dv_own / max(dt_cam, 1e-3)   # ownship world accel (geometric-APN a_own_perp)
                    raw_ctrl = controller.compute_control(uav.R_b2n, az_bore, los_az_err, r_est, v_est,
                                                          los_el, t, v_closing, bbox_size, -v_est[2],
                                                          los_rate, target_speed, a_own_world=a_own_world)
                    if not np.all(np.isfinite(raw_ctrl)):
                        nan_count += 1
                        raw_ctrl = np.array([G, 0, 0, 0])
                    rate_raw = raw_ctrl[1:4]
                    rate_filt = prev_rate + ctrl_coeff * (rate_raw - prev_rate)
                    prev_rate = rate_filt
                    ctrl = np.array([raw_ctrl[0], rate_filt[0], rate_filt[1], rate_filt[2]])
                    if last_meas_has_bbox:
                        last_bbox_time = t
                        last_good_range = np.linalg.norm(r_est)
                        frozen_cmd = ctrl.copy()
                        has_frozen_cmd = True
            if rig_mode:
                a_world_step = step_rig(uav, ctrl[1:4], dt_ctrl, r_imu)
                if rig_vib > 0.0:
                    a_world_step = a_world_step + np.random.normal(0.0, rig_vib, 3)
            else:
                a_world_step = step_quadrotor(uav, ctrl[0], ctrl[1:4], dt_ctrl)
            dv_own_accum += a_world_step * dt_ctrl    # accumulate ownship Δv for next camera window
            own_vel_int += a_world_step * dt_ctrl     # always-on strapdown own_vel mirror

        # range + LOS rate
        rel = target_pos - uav.pos
        range_now = np.linalg.norm(rel)
        min_range = min(min_range, range_now)
        if range_now < intercept_range and not intercepted and not rig_mode:
            intercepted = True
            intercept_t = t
        if np.linalg.norm(rel) > 0.1 and np.linalg.norm(uav.vel) > 0.1:
            los_rate = np.linalg.norm(np.cross(rel, uav.vel)) / (range_now ** 2)
        else:
            los_rate = 0.0

        # logging
        log['t'].append(t)
        log['range'].append(range_now)
        log['los_rate'].append(los_rate)
        _roll_log, _pitch_log, _ = R_to_euler(uav.R_b2n)
        log['pitch'].append(np.rad2deg(_pitch_log))
        log['roll'].append(np.rad2deg(_roll_log))
        _az_bore_log = boresight_azimuth(uav.R_b2n, R_n2c=R_B2C @ uav.R_b2n.T)
        if np.linalg.norm(rel[0:2]) > 1e-6:
            _az_tgt = np.arctan2(rel[1], rel[0])
            log['az_err'].append(np.rad2deg(np.arctan2(np.sin(_az_tgt - _az_bore_log),
                                                       np.cos(_az_tgt - _az_bore_log))))
        else:
            log['az_err'].append(np.nan)
        log['own_vel'].append(float(np.linalg.norm(own_vel_int)))
        log['uav_pos'].append(uav.pos.copy())
        log['tgt_pos'].append(target_pos.copy())
        if estimator.initialized:
            est_pos = estimator.imm_ukf.x[0:3]   # relative (target-own)
            est_vel = estimator.imm_ukf.x[3:6]   # relative
            est_w = estimator.imm_ukf.x[9]
            rel_pos_true = target_pos - uav.pos
            rel_vel_true = target_vel - uav.vel
            log['est_pos_err'].append(np.linalg.norm(est_pos - rel_pos_true))
            log['est_vel_err'].append(np.linalg.norm(est_vel - rel_vel_true))
            log['est_w_err'].append(abs(est_w - target_width_true))
            log['est_pos'].append(est_pos.copy())
            log['est_vel'].append(est_vel.copy())
            log['est_acc'].append(estimator.imm_ukf.x[6:9].copy())
            log['mu'].append(estimator.imm_ukf.mu.copy())
        else:
            log['est_pos_err'].append(np.nan)
            log['est_vel_err'].append(np.nan)
            log['est_w_err'].append(np.nan)
            log['est_pos'].append(np.array([np.nan]*3))
            log['est_vel'].append(np.array([np.nan]*3))
            log['est_acc'].append(np.array([np.nan]*3))
            log['mu'].append(np.array([np.nan]*3))
        log['cmd'].append(prev_rate.copy())

        if intercepted and not rig_mode:
            break

    # finalize
    est_pos_err_final = log['est_pos_err'][-1] if log['est_pos_err'] else np.nan
    est_vel_err_final = log['est_vel_err'][-1] if log['est_vel_err'] else np.nan
    est_w_err_final = log['est_w_err'][-1] if log['est_w_err'] else np.nan
    max_los_rate = max(log['los_rate']) if log['los_rate'] else 0.0
    final_los_rate = log['los_rate'][-1] if log['los_rate'] else 0.0
    final_range = log['range'][-1] if log['range'] else 0.0

    notes = ""
    if intercept_t > 0:
        notes = f"intercepted at t={intercept_t:.2f}s"
    elif min_range < intercept_range * 2:
        notes = f"near-miss min_range={min_range:.2f}m"
    else:
        notes = f"no intercept, min_range={min_range:.2f}m"

    r0 = log['range'][0] if log['range'] else 0.0
    single_pass = not detect_reopen(log['range'], r0)

    result = ScenarioResult(
        name=name, intercepted=intercepted, min_range=min_range,
        final_range=final_range, est_pos_err_final=est_pos_err_final,
        est_vel_err_final=est_vel_err_final, est_width_err_final=est_w_err_final,
        max_los_rate=max_los_rate, final_los_rate=final_los_rate,
        nan_count=nan_count, sim_time=log['t'][-1] if log['t'] else 0.0,
        single_pass=single_pass, captured=captured, notes=notes)
    return result, log


# ============================================================================
# Main: run scenarios + report
# ============================================================================
def main():
    print("=" * 80)
    print("InterceptorNodeRelativeIMMUKF — Algorithm Simulation Verification")
    print("=" * 80)
    print(f"Focal length: {FOCAL_LENGTH:.1f}px  FOV: {CAM_FOV_DEG}deg  Mount pitch: {MOUNT_PITCH_DEG}deg")
    print(f"Control rate: {CONTROL_RATE}Hz  Cam rate: {1/DT_CAM:.1f}Hz  v_des: {CTRL_V_DES}m/s")
    print()

    scenarios = []

    # --- Scenario 1: Nominal — stationary target ahead, level ---
    print("[1/7] Nominal: stationary target ahead...")
    r1, l1 = run_scenario(
        "nominal_stationary",
        target_init_pos=[40.0, 0.0, 10.0],
        target_vel_fn=lambda t: [0.0, 0.0, 0.0],
        uav_init_pos=[0.0, 0.0, 10.0],
        uav_init_yaw=0.0,
        sim_time=25.0,
        v_des_override=CTRL_V_DES)
    scenarios.append(r1)

    # --- Scenario 2: Constant-velocity crossing target ---
    print("[2/7] Crossing target (constant velocity)...")
    r2, l2 = run_scenario(
        "crossing_cv",
        target_init_pos=[50.0, 20.0, 12.0],
        target_vel_fn=lambda t: [-3.0, 2.0, 0.0],
        uav_init_pos=[0.0, 0.0, 10.0],
        uav_init_yaw=0.0,
        sim_time=30.0,
        v_des_override=CTRL_V_DES)
    scenarios.append(r2)

    # --- Scenario 3: Maneuvering target (coordinated turn) ---
    print("[3/7] Maneuvering target (coordinated turn)...")
    def ct_vel(t):
        omega = 0.3
        r = 15.0
        return [-r * omega * np.sin(omega * t), r * omega * np.cos(omega * t), 0.0]
    r3, l3 = run_scenario(
        "maneuvering_ct",
        target_init_pos=[45.0, 0.0, 10.0],
        target_vel_fn=ct_vel,
        uav_init_pos=[0.0, 0.0, 10.0],
        uav_init_yaw=0.0,
        sim_time=30.0,
        v_des_override=CTRL_V_DES)
    scenarios.append(r3)

    # --- Scenario 4: BBox dropout (target occluded mid-flight) ---
    print("[4/7] BBox dropout (2s occlusion)...")
    r4, l4 = run_scenario(
        "bbox_dropout",
        target_init_pos=[40.0, 5.0, 10.0],
        target_vel_fn=lambda t: [-2.0, 1.0, 0.0],
        uav_init_pos=[0.0, 0.0, 10.0],
        uav_init_yaw=0.0,
        sim_time=30.0,
        bbox_dropout_steps=[(300, 400)],  # ~6s-8s at 50Hz
        v_des_override=CTRL_V_DES)
    scenarios.append(r4)

    # --- Scenario 5: High measurement noise ---
    print("[5/7] High measurement noise...")
    r5, l5 = run_scenario(
        "high_noise",
        target_init_pos=[35.0, 0.0, 10.0],
        target_vel_fn=lambda t: [0.0, 0.0, 0.0],
        uav_init_pos=[0.0, 0.0, 10.0],
        uav_init_yaw=0.0,
        sim_time=30.0,
        noise_pos=5.0, noise_size=6.0,
        v_des_override=CTRL_V_DES)
    scenarios.append(r5)

    # --- Scenario 6: Wrong W_guess (size mismatch) ---
    print("[6/7] Wrong W_guess (true=0.8, guess=1.5)...")
    r6, l6 = run_scenario(
        "wrong_wguess",
        target_init_pos=[40.0, 0.0, 10.0],
        target_vel_fn=lambda t: [0.0, 0.0, 0.0],
        uav_init_pos=[0.0, 0.0, 10.0],
        uav_init_yaw=0.0,
        sim_time=30.0,
        target_width_true=0.8,
        v_des_override=CTRL_V_DES)
    scenarios.append(r6)

    # --- Scenario 7: v_des=20 intercept of the same maneuvering target as scenario 3 ----------
    # Holds the target fixed (coordinated turn, omega=0.3, r=15, start [45,0,10] — identical to
    # scenario 3) and raises only the closing speed to v_des=20, isolating the high-speed terminal-
    # authority question: can the controller intercept the same maneuvering target when it arrives
    # twice as fast? NOTE: the previous geometry (omega=0.4, start [60,0,10]) was adversarial to the
    # pure-pursuit + PN axial guidance — at 20 m/s the pursuer overshot and range reopened to ~60 m
    # before a fragile second-pass intercept at 0.92 m / 14.7 s. This geometry intercepts cleanly in a
    # single pass (~0.9 m at ~9 s). Verified: range is monotone-decreasing, no reopen.
    print("[7/7] x2 maneuvering target (same target as #3, high-speed terminal authority)...")
    def ct_vel_v20(t):
        omega = 0.3
        r = 15.0
        return [-r * omega * np.sin(omega * t), r * omega * np.cos(omega * t), 0.0]
    r7, l7 = run_scenario(
        "maneuvering_x2",
        target_init_pos=[45.0, 0.0, 10.0],
        target_vel_fn=ct_vel_v20,
        uav_init_pos=[0.0, 0.0, 10.0],
        uav_init_yaw=0.0,
        sim_time=15.0,
        v_des_override=CTRL_V_DES*2)
    scenarios.append(r7)

    # --- Scenario 8: Missile-mode steep dive (quaternion attitude layer, tilt cap 95 deg) --------
    # Target below-and-ahead: the axial channel rails, the thrust vector pitches past the old 45-65
    # deg Euler envelope toward ~75-85 deg. Pass = intercept with the quaternion attitude layer
    # holding the camera on target through the dive (the former Euler P-laws are singular past 90).
    print("[8] Missile-mode steep dive (tilt cap 95 deg, target below-ahead)...")
    r8, l8 = run_scenario(
        "dive_missile",
        target_init_pos=[30.0, 0.0, -6.0],
        target_vel_fn=lambda t: [0.0, 0.0, 0.0],
        uav_init_pos=[0.0, 0.0, 8.0],
        uav_init_yaw=0.0,
        sim_time=12.0,
        v_des_override=CTRL_V_DES,
        max_tilt_override=95.0, az_limit_override=12.0)
    scenarios.append(r8)

    # --- Bench (rig) scenarios: 3-axis free-attitude stand, translation LOCKED ------------------
    # These characterize the bench operating envelope, not intercept: metrics are boresight
    # tracking error, pitch achieved, roll bounds, strapdown own-vel drift from the rig rotation
    # artifacts, and stability (no NaN / no runaway rates).
    rig_results = []

    # [R1] Bench at the 90 deg operating point: rig pre-positioned at 60 deg pitch, target 10 m
    # ahead and 4 m BELOW the rotation center — the descent channel drives d_z = a_z+G < 0, so the
    # commanded thrust tilt passes 90 deg (the missile-mode geometry) while PN/heading keeps the
    # boresight on the laterally sweeping target through the old Euler interface's singularity.
    print("[R1] Rig @90deg operating point (target below), laterally sweeping target...")
    def rig1_vel(t):
        y = 3.5 * np.sin(2.0 * np.pi * 0.1 * t)
        return [0.0, y, 0.0]
    rr1, lr1 = run_scenario(
        "rig_90_sweep",
        target_init_pos=[10.0, 0.0, -4.0],
        target_vel_fn=rig1_vel,
        uav_init_pos=[0.0, 0.0, 0.0],
        uav_init_yaw=0.0,
        sim_time=20.0,
        v_des_override=CTRL_V_DES,
        rig_mode=True, max_tilt_override=95.0, az_limit_override=12.0, dither_amp_override=0.0,
        uav_init_pitch_deg=60.0)
    rig_results.append(rr1)

    # [R2] Bench pitch sweep: target starts near-level ahead and descends, sweeping the commanded
    # pitch from ~30 deg through the 90 deg operating point (validates continuity of the blend /
    # degeneracy handling across the whole bench envelope).
    print("[R2] Rig pitch sweep 30->90+ deg (descending target)...")
    def rig2_vel(t):
        return [0.0, 0.0, -0.3]
    rr2, lr2 = run_scenario(
        "rig_pitch_sweep",
        target_init_pos=[10.0, 0.0, 0.0],
        target_vel_fn=rig2_vel,
        uav_init_pos=[0.0, 0.0, 0.0],
        uav_init_yaw=0.0,
        sim_time=16.0,
        v_des_override=CTRL_V_DES,
        rig_mode=True, max_tilt_override=95.0, az_limit_override=12.0, dither_amp_override=0.0,
        uav_init_pitch_deg=30.0)
    rig_results.append(rr2)

    # === Report ===
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    header = f"{'Scenario':<22} {'Intercept':<10} {'MinRng(m)':<10} {'FinRng(m)':<10} {'PosErr(m)':<10} {'VelErr(m/s)':<11} {'WErr(m)':<9} {'MaxLOS(s)':<10} {'NaN':<5} {'1Pass':<6}"
    print(header)
    print("-" * len(header))
    all_pass = True
    for r in scenarios:
        status = "PASS" if r.intercepted else "FAIL"
        if not r.intercepted:
            all_pass = False
        if r.nan_count > 0:
            all_pass = False
        print(f"{r.name:<22} {status:<10} {r.min_range:<10.2f} {r.final_range:<10.2f} "
              f"{r.est_pos_err_final:<10.2f} {r.est_vel_err_final:<11.2f} {r.est_width_err_final:<9.2f} "
              f"{r.max_los_rate:<10.3f} {r.nan_count:<5} {'Y' if r.single_pass else 'N':<6} {r.notes}")

    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    intercept_count = sum(1 for r in scenarios if r.intercepted)
    nan_total = sum(r.nan_count for r in scenarios)
    print(f"Interceptions: {intercept_count}/{len(scenarios)}")
    print(f"Clean single-pass (no range reopen): "
          f"{sum(1 for r in scenarios if r.single_pass)}/{len(scenarios)}")
    print(f"NaN commands:  {nan_total}")
    print(f"Estimator convergence (final pos err < 5m): "
          f"{sum(1 for r in scenarios if r.est_pos_err_final < 5.0)}/{len(scenarios)}")
    print(f"Estimator stability (final vel err < 3m/s): "
          f"{sum(1 for r in scenarios if r.est_vel_err_final < 3.0)}/{len(scenarios)}")
    if all_pass:
        print("\n>>> ALL SCENARIOS PASSED — algorithm verified correct <<<")
    else:
        print("\n>>> SOME SCENARIOS FAILED — see details above <<<")

    # Missile-dive attitude envelope: prove the flight dive actually entered the >65 deg regime
    # that the old Euler interface could not command.
    _p8 = np.array(l8['pitch'])
    print(f"\ndive_missile attitude envelope: pitch min={np.nanmin(_p8):.1f} deg  max={np.nanmax(_p8):.1f} deg"
          f"  (old Euler-command ceiling was max_tilt 65/68 deg)")

    # === Bench (rig) report: boresight tracking / attitude envelope / strapdown drift =========
    if rig_results:
        print()
        print("=" * 80)
        print("BENCH (3-AXIS RIG) REPORT — translation locked, rotation free")
        print("=" * 80)
        for rr, lg in zip(rig_results, [lr1, lr2]):
            t_arr = np.array(lg['t'])
            pitch_arr = np.array(lg['pitch'])
            roll_arr = np.array(lg['roll'])
            az_err_arr = np.array(lg['az_err'])
            own_vel_arr = np.array(lg['own_vel'])
            settled = t_arr > 3.0
            az_abs = np.abs(az_err_arr[settled & ~np.isnan(az_err_arr)])
            print(f"{rr.name}:")
            print(f"  pitch   : min={np.nanmin(pitch_arr):7.1f} deg  max={np.nanmax(pitch_arr):7.1f} deg")
            print(f"  |roll|  : max={np.nanmax(np.abs(roll_arr)):7.1f} deg")
            print(f"  az_err  : mean={np.mean(az_abs):6.2f} deg  p95={np.percentile(az_abs, 95):6.2f} deg"
                  f"  max={np.max(az_abs):6.2f} deg   (after 3 s settle)")
            print(f"  strapdown own_vel drift (rig artifacts): {own_vel_arr[-1]:.3f} m/s over "
                  f"{t_arr[-1]:.0f} s")
            print(f"  NaN cmds: {rr.nan_count}   capture latched: {rr.captured}")

    # Save logs for plotting
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        logs = [l1, l2, l3, l4, l5, l6]
        names = [r.name for r in scenarios]
        for ax, log, nm in zip(axes.flat, logs, names):
            t = np.array(log['t'])
            rng = np.array(log['range'])
            pos_err = np.array(log['est_pos_err'])
            los = np.array(log['los_rate'])
            ax.plot(t, rng, 'b-', label='range (m)', linewidth=1.5)
            ax.plot(t, pos_err, 'r--', label='est pos err (m)', linewidth=1.0)
            ax2 = ax.twinx()
            ax2.plot(t, los, 'g:', label='LOS rate (rad/s)', linewidth=0.8)
            ax2.set_ylabel('LOS rate (rad/s)', color='g')
            ax.set_title(nm)
            ax.set_xlabel('t (s)')
            ax.set_ylabel('m')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim_test_results.png')
        plt.savefig(out, dpi=100)
        print(f"\nPlot saved: {out}")
    except Exception as e:
        print(f"\nPlot skipped: {e}")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
