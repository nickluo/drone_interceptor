#!/usr/bin/env python3
"""
ROS2 simulation node for testing InterceptorNodeRelativeIMMUKF (real C++ node).

Publishes:
  /fpv/odom          (nav_msgs/Odometry, BEST_EFFORT)  @ 100 Hz — UAV ego state
  /mavros/imu/data   (sensor_msgs/Imu, BEST_EFFORT) @ 100 Hz — body IMU (attitude + spec force)
  /fpv/gimbal        (sensor_msgs/Imu, BEST_EFFORT) @ 100 Hz — camera-to-world orientation q_C_W
  /fpv/target_bbox   (vision_msgs/Detection2D, BEST_EFFORT) @ 30 Hz — camera projection
  /fpv/radar/target      (nav_msgs/Odometry) @ 10 Hz — radar relay track (noisy + delayed)
  /fpv/radar/interceptor (nav_msgs/Odometry) @ 10 Hz — radar relay track (noisy + delayed)
  /fpv/tracker_trigger (std_msgs/Header "MISSION") — one-shot mission launch

Subscribes:
  /fpv/control_command (quadrotor_msgs/ControlCommand) — applies thrust+bodyrates
  /fpv/mission_state   (std_msgs/Bool) — mission active flag

Internal quadrotor dynamics: first-order body-rate tracking + thrust along body z - gravity.
Camera: pinhole projection on a semi-strapdown single-axis pitch gimbal (boresight servo-held
near-horizontal, 5° down-gaze bias, servo lag 0.1 s) + Gaussian noise. The actual camera-to-world
orientation q_C_W is published on /fpv/gimbal (sensor_msgs/Imu), matching the C++ interceptor node.

Usage:
  ros2 launch drone_interceptor launch_relative_imm_ukf.py   # terminal 1
  python3 ros2_sim_node.py --scenario nominal                 # terminal 2

  # radar relay mid-course guidance (target starts outside the camera FOV):
  ros2 launch drone_interceptor launch_relative_imm_ukf.py use_radar:=True
  python3 ros2_sim_node.py --scenario radar_relay

Scenarios: nominal, crossing, maneuvering, dropout, noise, wrong_w,
           radar_relay, radar_far, radar_maneuver, radar_only, radar_overbound
"""

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import yaml

import rclpy
import rclpy.qos
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from vision_msgs.msg import Detection2D
from quadrotor_msgs.msg import ControlCommand
from std_msgs.msg import Header, Bool


# ============================================================================
# Constants — loaded from config/relative_imm_ukf_params.yaml (sim2real consistency)
# ============================================================================
M_PI = math.pi


def _load_yaml_config():
    """Load the C++ node's config yaml so the sim stays in sync without manual alignment.
    Tries: (1) next to this script (../config/), (2) the install share dir. Falls back to
    the hardcoded defaults below if neither is found."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config',
                     'relative_imm_ukf_params.yaml'),
        '/media/nick/data/workspace/fpv_ws/install/drone_interceptor/share/drone_interceptor/'
        'config/relative_imm_ukf_params.yaml',
    ]
    for path in candidates:
        try:
            with open(path) as f:
                doc = yaml.safe_load(f)
            return doc.get('interceptor_node_relative_imm_ukf', {}).get('ros__parameters', {})
        except (FileNotFoundError, OSError, yaml.YAMLError):
            continue
    print("[sim] WARNING: config yaml not found — using hardcoded fallback defaults")
    return {}


_CFG = _load_yaml_config()

# Physics / dynamics
G = _CFG.get('physics', {}).get('g', 9.81)
DRAG_COEFF = _CFG.get('physics', {}).get('drag_coeff', 0.1625)

# Control loop
CONTROL_RATE = _CFG.get('control', {}).get('rate', 100.0)
RATE_TAU = _CFG.get('control', {}).get('rate_tc', 0.02)

# Camera intrinsics — fov_deg is the HORIZONTAL FOV; focal from width, vertical FOV derived.
CAM_WIDTH_PX = int(_CFG.get('camera', {}).get('width_px', 960))
CAM_HEIGHT_PX = int(_CFG.get('camera', {}).get('height_px', 960))
CAM_FOV_DEG = _CFG.get('camera', {}).get('fov_deg', 60.0)
FOCAL_LENGTH = (CAM_WIDTH_PX / 2.0) / math.tan(math.radians(CAM_FOV_DEG) / 2.0)
FOV_HALF_H = math.atan2(CAM_WIDTH_PX / 2.0, FOCAL_LENGTH)   # horizontal half-FOV (rad)
FOV_HALF_V = math.atan2(CAM_HEIGHT_PX / 2.0, FOCAL_LENGTH)  # vertical half-FOV (rad, < FOV_HALF_H for wide sensors)

# Camera frame rate (fixed for the sim — the C++ node measures dt_cam adaptively from timestamps)
CAM_RATE = 30.0

# ---------------------------------------------------------------------------
# Simulated radar relay. Mirrors airsim_tracker/radar_relay_node.py: ground truth for BOTH
# vehicles, corrupted with Gaussian noise and held for a sampled latency before publishing, with
# header.stamp = the MEASUREMENT epoch so the consumer can recover the lag.
# ---------------------------------------------------------------------------
RADAR_RATE = 10.0            # Hz
RADAR_POS_NOISE = 2.0        # m, constant 1-sigma per axis
RADAR_RANGE_NOISE_FRAC = 0.005   # extra 1-sigma as a fraction of slant range
RADAR_POS_NOISE_MAX = 5.0    # m, cap on the total position 1-sigma
RADAR_VEL_NOISE = 0.5        # m/s, 1-sigma per axis
RADAR_DELAY_MIN = 0.05       # s
RADAR_DELAY_MAX = 0.10       # s

# Gimbal (bias from config; tau is sim-only — models the servo lag, no C++ counterpart)
GIMBAL_BIAS = math.radians(_CFG.get('camera', {}).get('gimbal_bias_deg', 5.0))

# R_b2c base: body(FLU) -> cam(RDF). Standard frame convention (hardcoded in C++ too).
# Semi-strapdown single-axis pitch gimbal: R_B2C = Rx(gimbal_total) @ R_B2C_BASE.
R_B2C_BASE = np.array([[0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [1.0, 0.0, 0.0]])
GIMBAL_TAU = 0.1                       # s, servo lag (sim-only; no C++ config counterpart)
R_B2C = R_B2C_BASE.copy()              # updated in-place each control step from the gimbal state


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
INTERCEPT_RANGE = 1.0   # sync with sim_test_relative_imm_ukf.py (was 2.0)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def detect_reopen(ranges, r0):
    """Intercept-quality metric: True if the range reopened after first closing.
    A clean intercept is a single pass (range monotonically decreases to the
    threshold); a fragile one closes, overshoots, and the range climbs back before
    a later pass finally connects. After the range first drops below 50% of the
    initial range, did it climb back above 60% of it? (max(ranges) alone is NOT a
    reopen signal — it is just the initial range.) Mirrors sim_test_relative_imm_ukf.py.
    """
    if r0 < 1.0 or len(ranges) < 2:
        return False
    close_thr = 0.5 * r0
    reopen_thr = 0.6 * r0
    for i in range(len(ranges)):
        if ranges[i] < close_thr:
            return max(ranges[i:]) > reopen_thr
    return False


def skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])


def euler_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def R_to_euler(R):
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def quat_to_R(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    """Return (w, x, y, z) scalar-first."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return w, x, y, z


# ============================================================================
# Quadrotor dynamics
# ============================================================================
@dataclass
class QuadState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    R_b2n: np.ndarray = field(default_factory=lambda: np.eye(3))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gimbal_pitch: float = GIMBAL_BIAS   # semi-strapdown cam-x pitch servo state (actual angle)


def step_rig(state, omega_cmd, dt, r_imu, rate_tau=0.05):
    """3-axis free-attitude test bench: translation LOCKED, rotation free (first-order rate
    tracking like step_quadrotor). Returns the accelerometer artifact the IMU strapdown sees at
    an IMU offset r_imu (body frame) from the rig rotation center:
        a_meas_world = R * (alpha x r + omega x (omega x r)).
    Thrust is absorbed by the rig structure (a proper accelerometer never sees constraint
    forces — at the rotation center it reads pure gravity, indistinguishable from a hover), and
    gravity is removed by the strapdown, so the TRUE ownship linear accel is 0 and these
    artifacts are the entire strapdown input: the own_vel drift they cause is what the bench
    measures."""
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


def update_gimbal(state, dt, tau=GIMBAL_TAU):
    """Semi-strapdown single-axis (cam-x pitch) gimbal servo. The gimbal drives the camera
    boresight to near-horizontal in the world frame, compensating ownship pitch. Boresight world
    elev = -(own_pitch + gimbal_total); servo_cmd = -own_pitch + GIMBAL_BIAS => boresight settles
    at -GIMBAL_BIAS (down-gaze). First-order lag tau: the boresight trails fast pitch changes;
    the residual is fully captured because the ACTUAL lagged angle is what gets published/applied.
    Returns gimbal_total (the cam-x rotation actually applied)."""
    own_pitch = R_to_euler(state.R_b2n)[1]              # nose-down positive
    servo_cmd = -own_pitch + GIMBAL_BIAS
    alpha = 1.0 - math.exp(-dt / tau)
    state.gimbal_pitch += alpha * (servo_cmd - state.gimbal_pitch)
    return state.gimbal_pitch


def step_quadrotor(state, throttle, omega_cmd, dt):
    """Integrate quadrotor. throttle = collective accel (m/s^2) along body z.
    omega_cmd tracked with first-order lag tau. Returns a_world (gravity-corrected free accel)
    so the IMU publisher can derive the body-frame specific force."""
    state.omega = state.omega + (omega_cmd - state.omega) * (dt / RATE_TAU)
    w = state.omega * dt
    angle = np.linalg.norm(w)
    if angle > 1e-9:
        axis = w / angle
        dR = np.eye(3) + math.sin(angle) * skew(axis) + (1 - math.cos(angle)) * skew(axis) @ skew(axis)
        state.R_b2n = state.R_b2n @ dR
        U, _, Vt = np.linalg.svd(state.R_b2n)
        state.R_b2n = U @ Vt
    a_world = state.R_b2n @ np.array([0.0, 0.0, throttle]) - np.array([0.0, 0.0, G])
    a_world = a_world - DRAG_COEFF * state.vel
    state.vel = state.vel + a_world * dt
    state.pos = state.pos + state.vel * dt
    return a_world


# ============================================================================
# Camera projection (mirror C++ h_meas_pixel + project)
# ============================================================================
def project_target(uav_pos, R_b2n, target_pos, target_width, focal_length, fov_half_h, fov_half_v):
    """Return (u, v, size, in_fov) — pixel coords centered on image center."""
    rel_p = target_pos - uav_pos
    R_n2c = R_B2C @ R_b2n.T
    p_cam = R_n2c @ rel_p
    if p_cam[2] <= 0.5:
        return 0.0, 0.0, 0.0, False
    u = focal_length * p_cam[0] / p_cam[2]
    v = focal_length * p_cam[1] / p_cam[2]
    s = focal_length * target_width / p_cam[2]
    # Separate h/v FOV check (non-square sensors have v_fov < h_fov)
    angle_h = abs(math.atan2(p_cam[0], p_cam[2]))
    angle_v = abs(math.atan2(p_cam[1], p_cam[2]))
    in_fov = (angle_h < fov_half_h) and (angle_v < fov_half_v)
    return u, v, s, in_fov


# ============================================================================
# Scenarios
# ============================================================================
SCENARIOS = {
    'nominal': dict(
        target_init=[40.0, 0.0, 10.0],
        vel_fn=lambda t: [0.0, 0.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0,
        sim_time=25.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None),
    'crossing': dict(
        target_init=[50.0, 20.0, 12.0],
        vel_fn=lambda t: [-3.0, 2.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0,
        sim_time=30.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None),
    # 'maneuvering' == sim_test_relative_imm_ukf.py scenario 7 (maneuvering_v20): same coordinated-
    # turn target (omega=0.3, r=15, start [45,0,10]). In this harness the pursuer's v_des comes from
    # the C++ node's yaml (20 m/s), which is exactly scenario 7's high-speed intent — so this single
    # entry already covers scenario 7 (no separate duplicate needed). sim_test's scenario 3 uses the
    # same target at v_des=10, a speed distinction that lives in the node config, not this harness.
    'maneuvering': dict(
        target_init=[45.0, 0.0, 10.0],
        vel_fn=lambda t: [-15 * 0.3 * math.sin(0.3 * t),
                          15 * 0.3 * math.cos(0.3 * t), 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0,
        sim_time=30.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None),
    'dropout': dict(
        target_init=[40.0, 5.0, 10.0],
        vel_fn=lambda t: [-2.0, 1.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0,
        sim_time=30.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=(6.0, 8.0)),
    'noise': dict(
        target_init=[35.0, 0.0, 10.0],
        vel_fn=lambda t: [0.0, 0.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0,
        sim_time=30.0, noise_pos=5.0, noise_size=6.0,
        target_width=1.5, dropout=None),
    'wrong_w': dict(
        target_init=[40.0, 0.0, 10.0],
        vel_fn=lambda t: [0.0, 0.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0,
        sim_time=30.0, noise_pos=1.5, noise_size=2.0,
        target_width=0.8, dropout=None),
    # --- Radar relay scenarios -------------------------------------------------------------
    # The target starts OUTSIDE the camera FOV (bearing beyond the horizontal half-FOV) and at a
    # range where a real detector could not resolve it anyway (detect_min_size gates the bbox on
    # apparent size). Guidance must therefore run on the radar relay until the target is both in
    # frame and large enough, at which point the width handover arms the vision terminal phase.
    'radar_relay': dict(
        target_init=[120.0, 95.0, 25.0],
        vel_fn=lambda t: [-4.0, -3.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0,
        sim_time=45.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None,
        detect_min_size=25.0, radar=True),
    'radar_far': dict(
        target_init=[210.0, -160.0, 40.0],
        vel_fn=lambda t: [-6.0, 4.0, -0.5],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0,
        sim_time=60.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None,
        detect_min_size=25.0, radar=True),
    'radar_maneuver': dict(
        target_init=[110.0, 85.0, 30.0],
        vel_fn=lambda t: [-12 * 0.25 * math.sin(0.25 * t) - 3.0,
                          12 * 0.25 * math.cos(0.25 * t) - 2.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0,
        sim_time=50.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None,
        detect_min_size=25.0, radar=True),
    # Pure radar guidance: the detector never fires, so the whole engagement runs on the relay.
    # Bounds how close radar-only guidance can get (no visual terminal phase).
    'radar_only': dict(
        target_init=[120.0, 95.0, 25.0],
        vel_fn=lambda t: [-4.0, -3.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0,
        sim_time=45.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None,
        detect_min_size=1e9, radar=True),
    # Detector false positives at a random bearing and size, as observed in AirSim. The radar
    # consistency gate must reject them: fusing one is enough to walk the monocular width off.
    'radar_falsepos': dict(
        target_init=[120.0, 95.0, 25.0],
        vel_fn=lambda t: [-4.0, -3.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0,
        sim_time=45.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None,
        detect_min_size=25.0, radar=True, false_positive_prob=0.25),

    # Re-intercept: radar mid-course -> vision handover (terminal) -> a forced bbox dropout LONGER
    # than radar.vision_lost_timeout (1.5 s) while the radar keeps publishing -> the node must judge a
    # terminal miss and re-engage the radar relay for a second pass. Needs radar_reintercept_enable
    # (run_all_scenarios.sh passes radar_reintercept_enable:=true). Tune the dropout window so it
    # lands AFTER the [HANDOVER] log and BEFORE the first pass is within kill_range; verify in the node
    # log that [REINTERCEPT] fires and a second handover + intercept follows.
    'radar_reintercept': dict(
        target_init=[110.0, 55.0, 20.0],
        vel_fn=lambda t: [-4.0, -2.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0,
        sim_time=55.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=(5.5, 10.5),
        detect_min_size=25.0, radar=True),

    # Over-bounding tracker vs the width lock + the W re-anchor safety net. The bbox is inflated
    # 1.6x until t=8s (covering the handover, which locks ~0.4s after first vision), so W locks at
    # ~1.6x truth and the monocular range is over-estimated by ~1.6x — the exact "thinks it's far,
    # actually close, dither stays armed" field failure. When the factor reverts at 8s, the
    # still-tracking radar disagrees with r_vis = focal*W/bbox by a sustained ~1.6x and
    # [W-REANCHOR] must repair the lock mid-flight (window: 8s -> ~9.6s, before the ~10.5s
    # intercept). The bbox-px + radar dither gates independently stop the serpentine throughout.
    'radar_overbound': dict(
        target_init=[120.0, 95.0, 25.0],
        vel_fn=lambda t: [-4.0, -3.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0,
        sim_time=45.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None,
        detect_min_size=25.0, radar=True,
        bbox_overbound=1.6, bbox_overbound_until=8.0),

    # ---- Terminal entry-speed tests ----
    # Isolate the terminal vision phase from the mid-course: the interceptor starts already at
    # cruise speed V aimed straight at the target, ~45 m out (the AirSim radar->vision handover
    # condition). uav_vel_init is consumed in SimNode.__init__. Sweep with --v_entry.
    'terminal_v10': dict(
        target_init=[45.0, 0.0, 10.0], vel_fn=lambda t: [0.0, 0.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0, uav_vel_init=[10.0, 0.0, 0.0],
        sim_time=12.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None, detect_min_size=15.0),
    'terminal_v15': dict(
        target_init=[45.0, 0.0, 10.0], vel_fn=lambda t: [0.0, 0.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0, uav_vel_init=[15.0, 0.0, 0.0],
        sim_time=12.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None, detect_min_size=15.0),
    'terminal_v20': dict(
        target_init=[45.0, 0.0, 10.0], vel_fn=lambda t: [0.0, 0.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0, uav_vel_init=[20.0, 0.0, 0.0],
        sim_time=12.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None, detect_min_size=15.0),
    # Lateral entry component: 15 m/s closing with ~5 m/s cross — exercises the lateral PN +
    # close-range lateral_decay that the high-entry-speed trace flagged as a contributor.
    'terminal_v15_cross': dict(
        target_init=[45.0, 12.0, 10.0], vel_fn=lambda t: [-1.0, 1.0, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0, uav_vel_init=[14.0, 5.0, 0.0],
        sim_time=12.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None, detect_min_size=15.0),
    # Both radar + vision available from startup (close target). Tests the guidance-ownership logic:
    # vision should own the cache from the first consistent bbox (skip the radar mid-course), with
    # the radar only anchoring W. The target is close enough that the camera sees it from t=0.
    'radar_close': dict(
        target_init=[45.0, 5.0, 10.0], vel_fn=lambda t: [-1.5, 0.5, 0.0],
        uav_init=[0.0, 0.0, 10.0], uav_yaw=0.0,
        sim_time=15.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None, detect_min_size=15.0, radar=True),

    # ---- High-pitch / bench-rig scenarios (quaternion attitude layer) ----
    # Missile-mode steep dive: target below-and-ahead inside the initial FOV. The axial channel
    # rails and the thrust vector pitches past the old 65/68 deg Euler envelope (the node must
    # run with max_tilt 95 + a_z_limit 12 via run_scenario.sh dive). Pass = intercept with the
    # two-channel attitude layer holding the camera on target through the dive.
    'dive': dict(
        target_init=[30.0, 0.0, -6.0], vel_fn=lambda t: [0.0, 0.0, 0.0],
        uav_init=[0.0, 0.0, 8.0], uav_yaw=0.0,
        sim_time=12.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None, detect_min_size=15.0),

    # Bench rig, pitch sweep: translation LOCKED, rotation free; target descends slowly so the
    # commanded pitch sweeps from ~30 deg through the 90 deg operating point. Validates the
    # attitude layer's continuity and pointing across the old Euler singularity (run with
    # max_tilt 95 via run_scenario.sh rig_pitch_sweep). Success: boresight error stays ~0 deg.
    'rig_pitch_sweep': dict(
        target_init=[10.0, 0.0, 0.0], vel_fn=lambda t: [0.0, 0.0, -0.3],
        uav_init=[0.0, 0.0, 0.0], uav_yaw=0.0, uav_pitch_deg=30.0,
        sim_time=16.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None, detect_min_size=15.0,
        rig=True, rig_vib=0.0),

    # Bench rig at the 90 deg operating point: rig pre-positioned at 60 deg, target 10 m ahead
    # and 4 m BELOW the rotation center (the descent channel drives the thrust tilt past 90
    # deg) sweeping laterally — pointing must stay authoritative where the old Euler interface
    # was singular and its lateral PN destabilized the boresight.
    'rig_90_sweep': dict(
        target_init=[10.0, 0.0, -7.0],
        # Static for the first 6 s (let the attitude settle at the operating point — the realistic
        # bench procedure), then a gentle lateral sweep for the tracking metrics.
        vel_fn=lambda t: [0.0, 0.0, 0.0] if t < 6.0
            else [0.0, 2.5 * math.sin(2.0 * math.pi * 0.07 * (t - 6.0)), 0.0],
        uav_init=[0.0, 0.0, 0.0], uav_yaw=0.0, uav_pitch_deg=80.0,
        sim_time=24.0, noise_pos=1.5, noise_size=2.0,
        target_width=1.5, dropout=None, detect_min_size=15.0,
        rig=True, rig_vib=0.0),
}


# ============================================================================
# FOV-position sweep: place the target initially at a named image position
# (normalized image coords: nx = right, ny = down, in [-1, 1]; 0.67 ~ near the
# edge, 0.33 ~ between center and edge). Used to test robustness across the FOV.
# ============================================================================
FOV_POSITIONS = {
    'topleft':     (-0.67, -0.67),
    'top':         ( 0.00, -0.67),
    'topright':    ( 0.67, -0.67),
    'left':        (-0.67,  0.00),
    'midleft':     (-0.33,  0.00),
    'midright':    ( 0.33,  0.00),
    'right':       ( 0.67,  0.00),
    'bottomleft':  (-0.67,  0.67),
    'bottom':      ( 0.00,  0.67),
    'bottomright': ( 0.67,  0.67),
}


def target_init_for_fov(scenario, fov_name):
    """Initial target position that places the target at the named FOV image position, at the
    scenario's nominal range. Camera at start = body yaw + gimbal bias down-gaze (gimbal_pitch
    initializes to GIMBAL_BIAS). The world-frame velocity profile is unchanged."""
    nx, ny = FOV_POSITIONS[fov_name]
    uav_pos = np.array(scenario['uav_init'], dtype=float)
    R_b2n = euler_to_R(0.0, 0.0, scenario['uav_yaw'])
    R_b2c = Rx(GIMBAL_BIAS) @ R_B2C_BASE                       # gimbal starts at the bias
    ray_cam = np.array([nx * math.tan(FOV_HALF_H), ny * math.tan(FOV_HALF_V), 1.0])
    ray_world = R_b2n @ R_b2c.T @ ray_cam
    ray_world /= np.linalg.norm(ray_world)
    nominal_range = float(np.linalg.norm(np.array(scenario['target_init'], dtype=float) - uav_pos))
    return (uav_pos + nominal_range * ray_world).tolist()


# ============================================================================
# Sim node
# ============================================================================
class SimNode(Node):
    def __init__(self, scenario_name, scenario):
        super().__init__('imm_ukf_sim_node')
        self.scenario_name = scenario_name
        self.scenario = scenario
        self.get_logger().info(f"Scenario: {scenario_name}")

        # QoS: odom sub in node uses SensorDataQoS (BEST_EFFORT); bbox sub uses default
        # (RELIABLE, depth 10). Match each.
        sensor_qos = rclpy.qos.QoSProfile(
            depth=10, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        reliable_qos = rclpy.qos.QoSProfile(
            depth=10, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/fpv/odom', sensor_qos)
        self.imu_pub = self.create_publisher(Imu, '/mavros/imu/data', sensor_qos)
        self.bbox_pub = self.create_publisher(Detection2D, '/fpv/target_bbox', reliable_qos)
        self.trigger_pub = self.create_publisher(Header, '/fpv/tracker_trigger', 10)
        self.gimbal_pub = self.create_publisher(Imu, '/fpv/gimbal', sensor_qos)
        self.radar_enabled = bool(scenario.get('radar', False))
        # Bench-scale radar noise override: the module defaults are tuned for 50-100 m intercept
        # standoffs (3 m position noise); at a 10 m rig distance that injects +-17 deg of
        # bearing wobble into the fused estimate. Bench scenarios set a mocap-class 0.3 m.
        self.radar_pos_std = float(scenario.get('radar_pos_std', RADAR_POS_NOISE))
        if self.radar_enabled:
            self.radar_target_pub = self.create_publisher(Odometry, '/fpv/radar/target', reliable_qos)
            self.radar_own_pub = self.create_publisher(Odometry, '/fpv/radar/interceptor', reliable_qos)
            self.radar_queue = []   # (release_time, target_msg, own_msg)

        # Subscribers
        self.control_sub = self.create_subscription(
            ControlCommand, '/fpv/control_command', self.control_callback, 10)
        self.mission_state_sub = self.create_subscription(
            Bool, '/fpv/mission_state', self.mission_state_callback, 10)

        # State
        self.uav = QuadState()
        self.uav.pos = np.array(scenario['uav_init'], dtype=float)
        # uav_pitch_deg: bench-rig pre-positioning (nose-down positive), e.g. the 60 deg rig
        # start point; 0 for normal flight scenarios.
        self.uav.R_b2n = euler_to_R(0.0, np.deg2rad(scenario.get('uav_pitch_deg', 0.0)), scenario['uav_yaw'])
        self.uav.vel = np.array(scenario.get('uav_vel_init', np.zeros(3)), dtype=float)
        self.rig_mode = bool(scenario.get('rig', False))
        self.r_imu = np.array(scenario.get('r_imu', [0.08, 0.0, -0.06]), dtype=float)
        self.own_vel_int = np.zeros(3)      # strapdown own_vel mirror (bench drift metric)
        self.target_pos = np.array(scenario['target_init'], dtype=float)
        self.target_vel = np.zeros(3)
        self.latest_cmd = None  # (throttle, [wx,wy,wz], stamp)
        self.mission_active = False
        self.mission_launched = False

        # Logging
        self.log = {'t': [], 'range': [], 'los_rate': [], 'uav_pos': [], 'tgt_pos': [],
                    'cmd_throttle': [], 'cmd_rate': [], 'nan_count': 0,
                    'pitch': [], 'roll': [], 'az_err': [], 'own_vel': []}
        self.start_time = None
        self.intercepted = False
        self.intercept_t = -1.0
        self.min_range = 1e9
        # Radar-relay handover metrics
        self.first_vision_t = -1.0
        self.first_vision_range = -1.0
        self.min_range_before_vision = 1e9
        self.vision_frames = 0
        self.false_pos_emitted = 0

        # Timers
        self.dt_ctrl = 1.0 / CONTROL_RATE
        self.dt_cam = 1.0 / CAM_RATE
        self.control_timer = self.create_timer(self.dt_ctrl, self.control_loop)
        self.cam_timer = self.create_timer(self.dt_cam, self.cam_loop)
        self.launch_timer = self.create_timer(1.0, self.launch_once)
        if self.radar_enabled:
            self.radar_timer = self.create_timer(1.0 / RADAR_RATE, self.radar_sample)
            self.radar_dispatch_timer = self.create_timer(0.005, self.radar_dispatch)

        self.get_logger().info(
            f"Sim node ready. focal={FOCAL_LENGTH:.1f} h_fov={CAM_FOV_DEG} "
            f"v_fov={2*math.degrees(FOV_HALF_V):.1f} res={CAM_WIDTH_PX}x{CAM_HEIGHT_PX} "
            f"gimbal(tau={GIMBAL_TAU}s bias={math.degrees(GIMBAL_BIAS):.1f}deg)")
        if self.radar_enabled:
            det_range = (FOCAL_LENGTH * scenario['target_width'] /
                         max(scenario.get('detect_min_size', 0.0), 1e-9))
            self.get_logger().info(
                f"Radar relay ON: {RADAR_RATE:.0f}Hz pos_noise={RADAR_POS_NOISE:.1f}m"
                f"(+{RADAR_RANGE_NOISE_FRAC*100:.1f}%/range) vel_noise={RADAR_VEL_NOISE:.1f}m/s "
                f"delay={RADAR_DELAY_MIN*1e3:.0f}-{RADAR_DELAY_MAX*1e3:.0f}ms | "
                f"detector arms below ~{det_range:.0f}m")

    # ---------------- radar relay ----------------
    def radar_sample(self):
        """Sample ground truth for both vehicles, corrupt it, and queue it for delayed release."""
        if self.start_time is None:
            return
        rng = float(np.linalg.norm(self.target_pos - self.uav.pos))
        pos_std = min(self.radar_pos_std + RADAR_RANGE_NOISE_FRAC * rng, RADAR_POS_NOISE_MAX)
        stamp = self.get_clock().now()
        tgt_msg = self.build_radar_msg(stamp, self.target_pos, self.target_vel, pos_std)
        own_msg = self.build_radar_msg(stamp, self.uav.pos, self.uav.vel, pos_std)
        delay = float(np.random.uniform(RADAR_DELAY_MIN, RADAR_DELAY_MAX))
        self.radar_queue.append((stamp.nanoseconds * 1e-9 + delay, tgt_msg, own_msg))

    def build_radar_msg(self, stamp, pos, vel, pos_std):
        msg = Odometry()
        msg.header.stamp = stamp.to_msg()     # MEASUREMENT epoch, not the publish time
        msg.header.frame_id = "map"
        msg.child_frame_id = "map"            # twist is world-frame (radar measures ground velocity)
        p = pos + np.random.normal(0.0, pos_std, 3)
        v = vel + np.random.normal(0.0, RADAR_VEL_NOISE, 3)
        msg.pose.pose.position.x = float(p[0])
        msg.pose.pose.position.y = float(p[1])
        msg.pose.pose.position.z = float(p[2])
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.linear.x = float(v[0])
        msg.twist.twist.linear.y = float(v[1])
        msg.twist.twist.linear.z = float(v[2])
        return msg

    def radar_dispatch(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        while self.radar_queue and self.radar_queue[0][0] <= now:
            _, tgt_msg, own_msg = self.radar_queue.pop(0)
            self.radar_target_pub.publish(tgt_msg)
            self.radar_own_pub.publish(own_msg)

    def launch_once(self):
        if self.mission_launched:
            return
        self.mission_launched = True
        self.launch_timer.cancel()
        msg = Header()
        msg.stamp = self.get_clock().now().to_msg()
        msg.frame_id = "MISSION"
        self.trigger_pub.publish(msg)
        self.start_time = time.monotonic()
        self.get_logger().info("Published MISSION trigger")

    def mission_state_callback(self, msg):
        if msg.data and not self.mission_active:
            self.mission_active = True
            self.get_logger().info("Mission active (node entered TRACKING)")
        elif not msg.data and self.mission_active:
            self.mission_active = False
            self.get_logger().info("Mission ended")

    def control_callback(self, msg):
        if msg.control_mode != ControlCommand.BODY_RATES:
            return
        throttle = float(msg.collective_thrust)
        rates = np.array([float(msg.bodyrates.x), float(msg.bodyrates.y), float(msg.bodyrates.z)])
        if not (math.isfinite(throttle) and np.all(np.isfinite(rates))):
            self.log['nan_count'] += 1
            return
        self.latest_cmd = (throttle, rates)

    def control_loop(self):
        if self.start_time is None:
            return
        t = time.monotonic() - self.start_time
        if t > self.scenario['sim_time']:
            self.finish()
            return

        # target dynamics
        self.target_vel = np.array(self.scenario['vel_fn'](t), dtype=float)
        self.target_pos = self.target_pos + self.target_vel * self.dt_ctrl

        # apply control if available
        a_world = np.zeros(3)
        if self.latest_cmd is not None:
            throttle, rates = self.latest_cmd
            if self.rig_mode:
                # Bench rig: translation locked — only attitude integrates, and the IMU sees the
                # rig rotation artifacts (the strapdown's only input; true own accel is 0).
                a_world = step_rig(self.uav, rates, self.dt_ctrl, self.r_imu)
                _vib = float(self.scenario.get('rig_vib', 0.0))
                if _vib > 0.0:
                    a_world = a_world + np.random.normal(0.0, _vib, 3)
            else:
                a_world = step_quadrotor(self.uav, throttle, rates, self.dt_ctrl)
            self.own_vel_int += a_world * self.dt_ctrl

        # Semi-strapdown gimbal: servo the cam-x pitch every control step (before the camera
        # projection reads module R_B2C), then publish the ACTUAL camera-to-world orientation on
        # /fpv/gimbal (sensor_msgs/Imu) — q_C_W in the FLU camera frame (x=fwd, y=left, z=up), the
        # SAME convention as the C++ interceptor node, the hardware gimbal driver, and AirSim.
        # FLU = RDF right-multiplied by R_B2C_BASE (FLU->RDF): (FLU->ENU) = (RDF->ENU)·(FLU->RDF),
        # where RDF->ENU = R_b2n @ R_B2C.T (inverse of the sim projection's R_n2c = R_B2C @ R_b2n.T).
        gimbal_total = update_gimbal(self.uav, self.dt_ctrl)
        R_B2C[:, :] = Rx(gimbal_total) @ R_B2C_BASE
        R_c2n = (self.uav.R_b2n @ R_B2C.T) @ R_B2C_BASE
        w, x, y, z = R_to_quat(R_c2n)
        gmsg = Imu()
        gmsg.header.stamp = self.get_clock().now().to_msg()
        gmsg.header.frame_id = "gimbal"
        gmsg.orientation.w = float(w)
        gmsg.orientation.x = float(x)
        gmsg.orientation.y = float(y)
        gmsg.orientation.z = float(z)
        # angular_velocity / orientation_covariance are NOT consumed by the IMM-UKF node (it runs its
        # own IMM-UKF + body-IMU strapdown). Populated best-effort for contract consistency with
        # AirSim, which reads them: body rate as the camera-rate approximation, and the
        # camera world euler (r,p,y) packed into the covariance[0:3] slots.
        gmsg.angular_velocity.x = float(self.uav.omega[0])
        gmsg.angular_velocity.y = float(self.uav.omega[1])
        gmsg.angular_velocity.z = float(self.uav.omega[2])
        g_roll, g_pitch, g_yaw = R_to_euler(R_c2n)
        gmsg.orientation_covariance = [g_roll, g_pitch, g_yaw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.gimbal_pub.publish(gmsg)

        # range + LOS rate
        rel = self.target_pos - self.uav.pos
        rng = float(np.linalg.norm(rel))
        self.min_range = min(self.min_range, rng)
        if rng < INTERCEPT_RANGE and not self.intercepted and not self.rig_mode:
            self.intercepted = True
            self.intercept_t = t
            self.get_logger().info(f"*** INTERCEPT at t={t:.2f}s range={rng:.2f}m ***")
        if rng > 0.1 and np.linalg.norm(self.uav.vel) > 0.1:
            los_rate = float(np.linalg.norm(np.cross(rel, self.uav.vel)) / (rng * rng))
        else:
            los_rate = 0.0

        # log
        self.log['t'].append(t)
        self.log['range'].append(rng)
        self.log['los_rate'].append(los_rate)
        _roll, _pitch, _ = R_to_euler(self.uav.R_b2n)
        self.log['pitch'].append(math.degrees(_pitch))
        self.log['roll'].append(math.degrees(_roll))
        _b = R_c2n[:, 2]                      # camera +Z (boresight) in world
        if np.linalg.norm(_b[0:2]) > 0.1 and np.linalg.norm(rel[0:2]) > 1e-6:
            _az_b = math.atan2(_b[1], _b[0])
            _az_t = math.atan2(rel[1], rel[0])
            self.log['az_err'].append(math.degrees(math.atan2(
                math.sin(_az_t - _az_b), math.cos(_az_t - _az_b))))
        else:
            self.log['az_err'].append(float('nan'))
        self.log['own_vel'].append(float(np.linalg.norm(self.own_vel_int)))
        self.log['uav_pos'].append(self.uav.pos.copy())
        self.log['tgt_pos'].append(self.target_pos.copy())
        if self.latest_cmd is not None:
            self.log['cmd_throttle'].append(self.latest_cmd[0])
            self.log['cmd_rate'].append(self.latest_cmd[1].copy())
        else:
            self.log['cmd_throttle'].append(0.0)
            self.log['cmd_rate'].append(np.zeros(3))

        # publish odom (legacy/debug — the ODOMETRY-FREE node does not subscribe) + IMU
        self.publish_odom(t)
        self.publish_imu(a_world)

        if t % 2.0 < self.dt_ctrl:
            _az = self.log['az_err'][-1] if self.log['az_err'] else float('nan')
            self.get_logger().info(
                f"t={t:.1f} range={rng:.2f} min={self.min_range:.2f} "
                f"pitch={self.log['pitch'][-1]:.1f} roll={self.log['roll'][-1]:.1f} "
                f"az_err={_az:.1f} vel={self.uav.vel}")

    def publish_odom(self, t):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.pose.pose.position.x = float(self.uav.pos[0])
        msg.pose.pose.position.y = float(self.uav.pos[1])
        msg.pose.pose.position.z = float(self.uav.pos[2])
        w, x, y, z = R_to_quat(self.uav.R_b2n)
        msg.pose.pose.orientation.w = float(w)
        msg.pose.pose.orientation.x = float(x)
        msg.pose.pose.orientation.y = float(y)
        msg.pose.pose.orientation.z = float(z)
        # mavros local_position/odom: pose in frame_id (map=ENU/world), twist in
        # child_frame_id (base_link=body FLU). Mirror that here so the C++ node's
        # body->world rotation in odom_callback is exercised correctly. Without
        # this, sim publishes world-frame twist and the node double-rotates.
        msg.child_frame_id = "base_link"
        vel_body = self.uav.R_b2n.T @ self.uav.vel
        msg.twist.twist.linear.x = float(vel_body[0])
        msg.twist.twist.linear.y = float(vel_body[1])
        msg.twist.twist.linear.z = float(vel_body[2])
        self.odom_pub.publish(msg)

    def publish_imu(self, a_world):
        """Bias-free IMU at the control rate: orientation (b2n quaternion), body angular velocity,
        and body-frame specific force f_b = R^T (a_world + g·e_z). The node's strapdown
        preintegration reconstructs dv_own = ∫(R·f_b − g)dt from exactly this."""
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        w, x, y, z = R_to_quat(self.uav.R_b2n)
        msg.orientation.w = float(w)
        msg.orientation.x = float(x)
        msg.orientation.y = float(y)
        msg.orientation.z = float(z)
        msg.angular_velocity.x = float(self.uav.omega[0])
        msg.angular_velocity.y = float(self.uav.omega[1])
        msg.angular_velocity.z = float(self.uav.omega[2])
        f_body = self.uav.R_b2n.T @ (a_world + np.array([0.0, 0.0, G]))
        msg.linear_acceleration.x = float(f_body[0])
        msg.linear_acceleration.y = float(f_body[1])
        msg.linear_acceleration.z = float(f_body[2])
        self.imu_pub.publish(msg)

    def cam_loop(self):
        if self.start_time is None:
            return
        t = time.monotonic() - self.start_time
        u, v, s, in_fov = project_target(
            self.uav.pos, self.uav.R_b2n, self.target_pos,
            self.scenario['target_width'], FOCAL_LENGTH, FOV_HALF_H, FOV_HALF_V)
        has_bbox = in_fov
        # Detector resolution limit: a real detector cannot fire on a target smaller than a few
        # tens of pixels, which is exactly the regime the radar relay exists to cover.
        if has_bbox and s < self.scenario.get('detect_min_size', 0.0):
            has_bbox = False
        # dropout
        if has_bbox and self.scenario['dropout'] is not None:
            a, b = self.scenario['dropout']
            if a <= t < b:
                has_bbox = False
        # False positives: the AirSim detector fires on clouds/structures at long range, at a random
        # bearing and a random (usually small) apparent size. Exercises the radar consistency gate.
        phantom = False
        fp_prob = self.scenario.get('false_positive_prob', 0.0)
        if not has_bbox and fp_prob > 0.0 and np.random.random() < fp_prob:
            u = float(np.random.uniform(-0.45, 0.45) * CAM_WIDTH_PX)
            v = float(np.random.uniform(-0.45, 0.45) * CAM_HEIGHT_PX)
            s = float(np.random.uniform(5.0, 35.0))
            has_bbox = True
            phantom = True
            self.false_pos_emitted += 1
        if has_bbox and not phantom:
            self.vision_frames += 1
            if self.first_vision_t < 0.0:
                self.first_vision_t = t
                self.first_vision_range = float(np.linalg.norm(self.target_pos - self.uav.pos))
                self.get_logger().info(
                    f"*** FIRST VISION at t={t:.2f}s range={self.first_vision_range:.1f}m "
                    f"bbox={s:.1f}px ***")
        elif not has_bbox and self.first_vision_t < 0.0:
            self.min_range_before_vision = min(
                self.min_range_before_vision,
                float(np.linalg.norm(self.target_pos - self.uav.pos)))
        msg = Detection2D()
        msg.header.stamp = self.get_clock().now().to_msg()
        if has_bbox:
            u += np.random.normal(0, self.scenario['noise_pos'])
            v += np.random.normal(0, self.scenario['noise_pos'])
            s += np.random.normal(0, self.scenario['noise_size'])
            # Tracker over-bounding injection (the locked-W-inflated failure class): a real visual
            # tracker over-bounds the bbox by a multiplicative factor, so the radar->vision handover
            # locks an inflated W and the monocular range is over-estimated by the same factor.
            # Applied AFTER detect_min_size (detection fires on the TRUE size) and to real
            # detections only. bbox_overbound_until reverts the factor to 1.0 after t (s) so the
            # node's W-vs-radar consistency check sees a sustained disagreement (tests re-anchor).
            ob = float(self.scenario.get('bbox_overbound', 1.0))
            if ob != 1.0 and t < float(self.scenario.get('bbox_overbound_until', float('inf'))):
                s *= ob
            s = max(1.0, s)
            msg.header.frame_id = "camera_frame"
            msg.bbox.center.position.x = float(u + CAM_WIDTH_PX / 2.0)
            msg.bbox.center.position.y = float(v + CAM_HEIGHT_PX / 2.0)
            msg.bbox.center.theta = 0.0
            msg.bbox.size_x = float(s)
            msg.bbox.size_y = float(s)
        else:
            msg.header.frame_id = ""  # empty = target lost
        self.bbox_pub.publish(msg)

    def finish(self):
        self.get_logger().info("Sim finished, saving results...")
        self.control_timer.cancel()
        self.cam_timer.cancel()
        self.save_results()
        raise SystemExit(0)

    def _dither_close_metrics(self):
        """Dither activity inside true range < 15 m: (mean |v_lat| m/s, 0.2-0.4 Hz power fraction).

        v_lat is the ownship velocity perpendicular to the TRUE LOS (xy plane) — the dither drives
        exactly this. A dither that stays armed at close range because the monocular range is
        over-estimated shows up as a high value; with the close gates working it collapses.
        Returns (nan, nan) when the segment is too short to FFT (< 17 samples).
        """
        try:
            t = np.asarray(self.log['t'])
            pos = np.asarray(self.log['uav_pos'])
            tgt = np.asarray(self.log['tgt_pos'])
            if t.size < 17:
                return float('nan'), float('nan')
            dt = float(np.median(np.diff(t)))
            if dt <= 0:
                return float('nan'), float('nan')
            vel = np.gradient(pos, dt, axis=0)
            r_xy = tgt[:, :2] - pos[:, :2]
            r_xy_norm = np.linalg.norm(r_xy, axis=1)
            e_los = np.divide(r_xy, r_xy_norm[:, None], out=np.zeros_like(r_xy),
                              where=r_xy_norm[:, None] > 1e-6)
            e_perp = np.stack([-e_los[:, 1], e_los[:, 0]], axis=1)
            v_lat = np.abs(np.einsum("ij,ij->i", vel[:, :2], e_perp))
            true_range = np.linalg.norm(tgt - pos, axis=1)
            mask = true_range < 15.0
            if np.count_nonzero(mask) < 17:
                return float('nan'), float('nan')
            seg = v_lat[mask] - v_lat[mask].mean()
            freqs = np.fft.rfftfreq(seg.size, d=dt)
            spec = np.abs(np.fft.rfft(seg)) ** 2
            band = (freqs > 0.20) & (freqs < 0.40)
            total = float(spec.sum())
            frac = float(spec[band].sum()) / total if (band.any() and total > 0) else float('nan')
            return float(v_lat[mask].mean()), frac
        except Exception:
            return float('nan'), float('nan')

    def save_results(self):
        out_dir = os.path.dirname(os.path.abspath(__file__))
        base = f"ros2_sim_{self.scenario_name}"
        # CSV log
        csv_path = os.path.join(out_dir, base + ".csv")
        with open(csv_path, 'w') as f:
            f.write("t,range,los_rate,uav_x,uav_y,uav_z,tgt_x,tgt_y,tgt_z,throttle,wx,wy,wz\n")
            for i in range(len(self.log['t'])):
                f.write(f"{self.log['t'][i]:.4f},{self.log['range'][i]:.4f},"
                        f"{self.log['los_rate'][i]:.6f},"
                        f"{self.log['uav_pos'][i][0]:.4f},{self.log['uav_pos'][i][1]:.4f},"
                        f"{self.log['uav_pos'][i][2]:.4f},"
                        f"{self.log['tgt_pos'][i][0]:.4f},{self.log['tgt_pos'][i][1]:.4f},"
                        f"{self.log['tgt_pos'][i][2]:.4f},"
                        f"{self.log['cmd_throttle'][i]:.4f},"
                        f"{self.log['cmd_rate'][i][0]:.4f},{self.log['cmd_rate'][i][1]:.4f},"
                        f"{self.log['cmd_rate'][i][2]:.4f}\n")
        # summary
        sum_path = os.path.join(out_dir, base + "_summary.txt")
        final_range = self.log['range'][-1] if self.log['range'] else 0.0
        max_los = max(self.log['los_rate']) if self.log['los_rate'] else 0.0
        r0 = self.log['range'][0] if self.log['range'] else 0.0
        single_pass = not detect_reopen(self.log['range'], r0)
        # Dither activity in the close-range terminal window (true range < 15 m): mean lateral
        # velocity perpendicular to the LOS + its 0.2-0.4 Hz spectral fraction (the dither band,
        # same math as analyze_radar_scenario.py). A dither that stays armed at close range
        # (over-estimated monocular range) shows up as a high value here.
        dither_vlat, dither_frac = self._dither_close_metrics()
        with open(sum_path, 'w') as f:
            f.write(f"Scenario: {self.scenario_name}\n")
            f.write(f"Intercepted: {self.intercepted}\n")
            if self.intercepted:
                f.write(f"Intercept time: {self.intercept_t:.2f}s\n")
            f.write(f"Single pass (no range reopen): {single_pass}\n")
            f.write(f"Min range: {self.min_range:.2f}m\n")
            f.write(f"Final range: {final_range:.2f}m\n")
            f.write(f"Max LOS rate: {max_los:.4f}rad/s\n")
            f.write(f"NaN commands: {self.log['nan_count']}\n")
            if np.isfinite(dither_vlat):
                f.write(f"Dither |v_lat| mean (true r<15m): {dither_vlat:.2f}m/s\n")
                if np.isfinite(dither_frac):
                    f.write(f"Dither 0.2-0.4Hz frac (true r<15m): {dither_frac:.3f}\n")
            if self.radar_enabled:
                f.write(f"Radar relay: enabled ({RADAR_RATE:.0f}Hz, "
                        f"{RADAR_POS_NOISE:.1f}m+{RADAR_RANGE_NOISE_FRAC*100:.1f}%/range noise, "
                        f"{RADAR_DELAY_MIN*1e3:.0f}-{RADAR_DELAY_MAX*1e3:.0f}ms delay)\n")
                f.write(f"Initial range: {r0:.2f}m\n")
                f.write(f"First vision t: {self.first_vision_t:.2f}s\n")
                f.write(f"First vision range: {self.first_vision_range:.2f}m\n")
                f.write(f"Radar-only closure: {r0:.1f}m -> "
                        f"{self.min_range_before_vision:.1f}m\n")
                f.write(f"Vision frames: {self.vision_frames}\n")
                f.write(f"False positives emitted: {self.false_pos_emitted}\n")
            if self.rig_mode:
                t_arr = np.asarray(self.log['t'])
                settled = t_arr > 3.0
                az_abs = np.abs(np.asarray(self.log['az_err'])[settled])
                az_abs = az_abs[np.isfinite(az_abs)]
                f.write("Bench (rig) metrics:\n")
                f.write(f"  pitch min/max: {min(self.log['pitch']):.1f}/{max(self.log['pitch']):.1f} deg\n")
                f.write(f"  |roll| max: {max(abs(r) for r in self.log['roll']):.1f} deg\n")
                if az_abs.size:
                    f.write(f"  az_err mean/p95/max (after 3s): "
                            f"{az_abs.mean():.2f}/{np.percentile(az_abs, 95):.2f}/{az_abs.max():.2f} deg\n")
                f.write(f"  strapdown own_vel drift (rig artifacts): "
                        f"{self.log['own_vel'][-1]:.3f} m/s\n")
            f.write(f"Sim time: {self.log['t'][-1]:.2f}s\n" if self.log['t'] else "Sim time: 0\n")
        self.get_logger().info(
            f"Results: {base}.csv + {base}_summary.txt | "
            f"intercept={self.intercepted} min_range={self.min_range:.2f} "
            f"single_pass={single_pass} nan={self.log['nan_count']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='nominal', choices=list(SCENARIOS.keys()))
    parser.add_argument('--fov_pos', default=None, choices=list(FOV_POSITIONS.keys()),
                        help='override target_init: place the target at this FOV image position, at '
                             'the scenario nominal range (world-frame velocity unchanged)')
    parser.add_argument('--sim_time', type=float, default=None,
                        help='override the scenario sim_time (s)')
    parser.add_argument('--radar', action='store_true',
                        help='force-enable the simulated radar relay feed for this scenario')
    parser.add_argument('--detect_min_size', type=float, default=None,
                        help='override the detector minimum bbox size (px); 0 = always detect')
    parser.add_argument('--bbox_overbound', type=float, default=None,
                        help='multiply the published bbox size by this factor (simulate an '
                             'over-bounding tracker -> inflated W lock -> over-estimated range)')
    parser.add_argument('--bbox_overbound_until', type=float, default=None,
                        help='revert the bbox_overbound factor to 1.0 after this sim time (s)')
    parser.add_argument('--v_entry', type=float, default=None,
                        help='override the interceptor entry speed (m/s): set uav_vel_init along the '
                             '(target-init) direction at this magnitude. Sweep the terminal entry speed.')
    args = parser.parse_args()

    scen = dict(SCENARIOS[args.scenario])          # copy so overrides don't mutate SCENARIOS
    name = args.scenario
    if args.fov_pos is not None:
        scen['target_init'] = target_init_for_fov(scen, args.fov_pos)
        name = f"{args.scenario}-{args.fov_pos}"
    if args.sim_time is not None:
        scen['sim_time'] = args.sim_time
    if args.radar:
        scen['radar'] = True
    if args.detect_min_size is not None:
        scen['detect_min_size'] = args.detect_min_size
    if args.bbox_overbound is not None:
        scen['bbox_overbound'] = float(args.bbox_overbound)
    if args.bbox_overbound_until is not None:
        scen['bbox_overbound_until'] = float(args.bbox_overbound_until)
    if args.v_entry is not None:
        _dir = np.array(scen['target_init'], dtype=float) - np.array(scen['uav_init'], dtype=float)
        _dir[2] = 0.0   # keep the entry speed horizontal (aim at the target's azimuth)
        _n = np.linalg.norm(_dir)
        if _n > 1e-3:
            scen['uav_vel_init'] = (_dir / _n) * float(args.v_entry)
            name = f"{args.scenario}-v{args.v_entry}"

    rclpy.init()
    node = SimNode(name, scen)
    np.random.seed(42)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
