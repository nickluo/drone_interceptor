"""
Deep diagnostic #2: Instrument the EKF state (especially W estimate and position error)
to understand WHY the range estimate is so wrong in failure cases.
"""
import numpy as np
import sys
import os

# Import the modules directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_filter import EMAFilter
from camera_model import CameraModel, PerceptionSim
from target_ekf import TargetEKF
from drone_model_dq import (
    DroneModelDQ,
    pose_to_dq_np, dq_to_pose_np, normalize_dq_np,
)

# Copy constants from solver_flat_dq.py
MASS = 4.0; G = 9.8065; A_MAX = 20.0; A_MIN = 1.0; DRAG_COEFF = 0.1625
ATT_LIMIT = np.pi / 5; IXX = 0.03; IYY = 0.03; IZZ = 0.06; TAU_MAX = 2.0
CTRL_FREQ_HZ = 50; TF = 0.8; N_HORIZON = int(TF * CTRL_FREQ_HZ); V_DESIRED = 18.0
SIM_TIME = 15.0; INTERCEPTION_DISTANCE = 2.0
PERCEPTION_DELAY = 0.08; BBOX_POS_STD_DEV = 3.0; BBOX_SIZE_STD_DEV = 3.0
TARGET_PHYSICAL_SIZE_TRUTH = 0.5; TARGET_WIDTH_INITIAL_EST = 1.0
CAM_FOV_DEG = 30.0; CAM_FOCAL_LENGTH_PHYSICAL = 16.0; CAM_PIXEL_SIZE = 3.45
CAM_FOCAL_LENGTH = CAM_FOCAL_LENGTH_PHYSICAL * 1000 / CAM_PIXEL_SIZE
GIMBAL_RP_COEFF = 12.0; GIMBAL_Y_COEFF = 10.0; GIMBAL_DISTURBANCE_COEFF = 0.5

W_LOS = 40.0; W_ZEM = 35.0; W_CLOSING = 1.0; W_ALIGN = 5.0
W_CONTROL_F = 0.0005; W_CONTROL_TAU = 0.005; W_TILT = 0.0005; W_OMEGA = 0.003; W_FOV = 120.0
W_COST = np.array([W_LOS, W_ZEM, W_CLOSING, W_ALIGN, W_CONTROL_F, W_CONTROL_TAU, W_TILT, W_OMEGA, W_FOV])

DRONE_INITIAL_STATE = np.array([0,0,10, 0,0,0, 0,0,0, 0,0,0, 0,0,0])
TARGET_INITIAL_POS = np.array([100.0, 20.0, 15.0])
TARGET_VELOCITY = np.array([-5.0, 5.0, 0.5])

MODEL_CONSTANTS = {
    'G': G, 'MASS': MASS, 'IXX': IXX, 'IYY': IYY, 'IZZ': IZZ,
    'TAU_MAX': TAU_MAX, 'A_MIN': A_MIN, 'A_MAX': A_MAX,
    'ATT_LIMIT': ATT_LIMIT, 'N_HORIZON': N_HORIZON, 'TF': TF,
    'DRONE_INITIAL_STATE': DRONE_INITIAL_STATE, 'CAM_FOV_DEG': CAM_FOV_DEG,
}

def dq_state_to_euler_state(dq_state):
    dq = dq_state[0:8]; omega = dq_state[8:11]; v_body = dq_state[11:14]; gimbal = dq_state[14:17]
    q, pos = dq_to_pose_np(dq)
    euler = DroneModelDQ.quaternion_to_euler(q)
    R_mat = DroneModelDQ.quaternion_to_rotation_matrix(q)
    v_world = R_mat @ v_body
    return np.hstack([pos, v_world, euler, omega, gimbal])

def euler_state_to_dq_state(euler_state):
    pos = euler_state[0:3]; v_world = euler_state[3:6]; euler = euler_state[6:9]
    omega = euler_state[9:12]; gimbal = euler_state[12:15]
    q = DroneModelDQ.euler_to_quaternion(euler[0], euler[1], euler[2])
    dq = pose_to_dq_np(q, pos)
    R_mat = DroneModelDQ.quaternion_to_rotation_matrix(q)
    v_body = R_mat.T @ v_world
    return np.hstack([dq, omega, v_body, gimbal])


def run_single_instrumented(seed=None):
    """Run one simulation with full EKF instrumentation."""
    if seed is not None:
        np.random.seed(seed)
    
    drone_model = DroneModelDQ(MODEL_CONSTANTS)
    acados_solver = drone_model.generate_solver()

    dt = TF / N_HORIZON
    N_sim = int(SIM_TIME / dt)
    x_current_dq = euler_state_to_dq_state(DRONE_INITIAL_STATE)
    target_pos_true = TARGET_INITIAL_POS.copy()
    target_vel_true = TARGET_VELOCITY.copy()

    camera = CameraModel(CAM_FOCAL_LENGTH, TARGET_PHYSICAL_SIZE_TRUTH, CAM_FOV_DEG,
                         BBOX_POS_STD_DEV, BBOX_SIZE_STD_DEV)
    estimator = TargetEKF(dt, CAM_FOCAL_LENGTH, BBOX_POS_STD_DEV, BBOX_SIZE_STD_DEV,
                           initial_width_est=TARGET_WIDTH_INITIAL_EST,
                           range_init_bias=0.6, q_accel=3.0, q_W=0.0002)
    perceptor = PerceptionSim(camera, PERCEPTION_DELAY, dt)

    target_lost_counter = 0
    x_current_euler = dq_state_to_euler_state(x_current_dq)
    history = {'x': [x_current_euler], 'u': [], 'target_true': [target_pos_true.copy()],
               'target_vel_true': [target_vel_true.copy()], 'target_est_ekf': [], 'target_est_raw': [], 'distance': []}

    u_hover = np.array([MASS * G, 0, 0, 0])
    for k in range(N_HORIZON):
        acados_solver.set(k, 'x', x_current_dq); acados_solver.set(k, 'u', u_hover)
    acados_solver.set(N_HORIZON, 'x', x_current_dq)

    last_good_pos_est = None; last_good_vel_est = None; last_good_time = 0.0
    terminal_phase = False

    # === Instrumentation ===
    log = {
        'step': [], 'true_dist': [], 'est_dist': [], 
        'ekf_W': [], 'pos_error': [], 'vel_error': [],
        'visible': [], 'ekf_pos': [], 'true_pos': [],
        'drone_pos': [], 'drone_vel_world': [],
    }

    for i in range(N_sim):
        x_euler_for_perception = dq_state_to_euler_state(x_current_dq)
        full, target_pos_true = perceptor.update(x_euler_for_perception, target_pos_true, target_vel_true, history)
        if not full:
            continue
        drone_state_at_meas, meas_bbox, meas_visible = perceptor.get_measure()

        if estimator.is_initialized:
            estimator.predict(dt)

        if meas_visible:
            target_lost_counter = 0
            drone_pos = drone_state_at_meas[0:3]
            psi = drone_state_at_meas[8]
            gr, gp, gy = drone_state_at_meas[12], drone_state_at_meas[13], drone_state_at_meas[14]
            if not estimator.is_initialized:
                estimator.initialize(meas_bbox, drone_pos, psi, gr, gp, gy)
            else:
                estimator.update(meas_bbox, drone_pos, psi, gr, gp, gy)
        else:
            target_lost_counter += 1

        MAX_BLIND_STEPS = int(N_HORIZON * 2.0)
        if target_lost_counter > MAX_BLIND_STEPS:
            break

        if not estimator.is_initialized:
            u_optimal_raw = np.array([MASS * G, 0, 0, 0])
        else:
            pos_est = estimator.x[0:3]
            vel_est = estimator.x[3:6]

            if meas_visible or target_lost_counter == 0:
                last_good_pos_est = pos_est.copy()
                last_good_vel_est = vel_est.copy()
                last_good_time = i * dt

            q_cur, pos_cur = dq_to_pose_np(x_current_dq[0:8])
            R_cur = DroneModelDQ.quaternion_to_rotation_matrix(q_cur)
            v_world_cur = R_cur @ x_current_dq[11:14]
            
            if target_lost_counter > 0 and last_good_pos_est is not None:
                dt_since_good = i * dt - last_good_time
                pos_est_dr = last_good_pos_est + last_good_vel_est * dt_since_good
                vel_est_dr = last_good_vel_est.copy()
                dist_to_dr = np.linalg.norm(pos_est_dr - pos_cur)
                if dist_to_dr < 15.0:
                    pos_est = pos_est_dr
                    vel_est = vel_est_dr
                    if not terminal_phase and dist_to_dr < 10.0:
                        terminal_phase = True

            dist_to_target = np.linalg.norm(pos_est - pos_cur)

            # ===== LOG =====
            true_dist = np.linalg.norm(target_pos_true - pos_cur)
            pos_error = np.linalg.norm(pos_est - target_pos_true)
            vel_error_val = np.linalg.norm(vel_est - target_vel_true)
            log['step'].append(i)
            log['true_dist'].append(true_dist)
            log['est_dist'].append(dist_to_target)
            log['ekf_W'].append(estimator.x[6])
            log['pos_error'].append(pos_error)
            log['vel_error'].append(vel_error_val)
            log['visible'].append(meas_visible)
            log['ekf_pos'].append(pos_est.copy())
            log['true_pos'].append(target_pos_true.copy())
            log['drone_pos'].append(pos_cur.copy())
            log['drone_vel_world'].append(v_world_cur.copy())

            # ======== DQ-NMPC ========
            acados_solver.set(0, "lbx", x_current_dq)
            acados_solver.set(0, "ubx", x_current_dq)
            dist_current = dist_to_target
            v_desired_adaptive = V_DESIRED * np.clip(dist_current / 6.0, 0.7, 1.0)

            for k in range(N_HORIZON + 1):
                t_k = k * dt
                p_k = np.hstack([pos_est + t_k * vel_est, vel_est, v_desired_adaptive, DRAG_COEFF, W_COST, GIMBAL_RP_COEFF, GIMBAL_Y_COEFF])
                acados_solver.set(k, "p", p_k)

            acados_solver.solve()
            u_optimal_raw = acados_solver.get(0, "u")
            x_current_dq = acados_solver.get(1, "x")

            if np.any(np.isnan(x_current_dq)):
                break

            x_current_dq[0:8] = normalize_dq_np(x_current_dq[0:8])
            omega_cur = x_current_dq[8:11]
            x_current_dq[14] += GIMBAL_DISTURBANCE_COEFF * omega_cur[0] * dt
            x_current_dq[15] += GIMBAL_DISTURBANCE_COEFF * omega_cur[1] * dt
            x_current_dq[16] += GIMBAL_DISTURBANCE_COEFF * omega_cur[2] * dt

            for k in range(N_HORIZON - 1):
                x_next = acados_solver.get(k + 1, "x")
                x_next[0:8] = normalize_dq_np(x_next[0:8])
                u_next = acados_solver.get(k + 1, "u")
                acados_solver.set(k, "x", x_next); acados_solver.set(k, "u", u_next)
            acados_solver.set(N_HORIZON - 1, "u", u_next)
            x_last = acados_solver.get(N_HORIZON, "x")
            x_last[0:8] = normalize_dq_np(x_last[0:8])
            acados_solver.set(N_HORIZON, "x", x_last)

        target_pos_true += target_vel_true * dt
        x_current_euler = dq_state_to_euler_state(x_current_dq)
        history['x'].append(x_current_euler.copy())
        history['u'].append(u_optimal_raw.copy())
        history['target_true'].append(target_pos_true.copy())
        history['target_vel_true'].append(target_vel_true.copy())

        q_cur, pos_cur = dq_to_pose_np(x_current_dq[0:8])
        distance = np.linalg.norm(target_pos_true - pos_cur)
        history['distance'].append(distance)

        if distance < INTERCEPTION_DISTANCE:
            return True, log

    return False, log


def main():
    # Run until we get a few failures and successes
    n_fail = 0
    n_succ = 0
    fail_logs = []
    succ_logs = []
    
    seed = 0
    while n_fail < 3 or n_succ < 3:
        success, log = run_single_instrumented(seed)
        if success:
            n_succ += 1
            if n_succ <= 3:
                succ_logs.append((seed, log))
        else:
            n_fail += 1
            fail_logs.append((seed, log))
        seed += 1
        if seed > 200:
            break
    
    print(f"Collected {len(fail_logs)} failures, {len(succ_logs)} successes")
    print(f"Seeds tested: {seed}")
    
    # ===== Print detailed comparison =====
    for label, logs in [("FAILURE", fail_logs), ("SUCCESS", succ_logs)]:
        print(f"\n{'='*70}")
        print(f"{label} CASES")
        print(f"{'='*70}")
        
        for seed_val, log in logs:
            steps = np.array(log['step'])
            true_d = np.array(log['true_dist'])
            est_d = np.array(log['est_dist'])
            W = np.array(log['ekf_W'])
            pos_err = np.array(log['pos_error'])
            vel_err = np.array(log['vel_error'])
            vis = np.array(log['visible'])
            
            # Find key moments
            # Last visible step
            vis_indices = np.where(vis)[0]
            if len(vis_indices) > 0:
                last_vis_idx = vis_indices[-1]
            else:
                last_vis_idx = 0
            
            # Min true distance
            min_true_idx = np.argmin(true_d)
            
            print(f"\n  Seed {seed_val}:")
            print(f"    Total steps logged: {len(steps)}")
            print(f"    Min true distance: {true_d[min_true_idx]:.2f}m at step {steps[min_true_idx]}")
            print(f"    Last visible step: {steps[last_vis_idx]} (idx {last_vis_idx})")
            
            # EKF state at key moments: mid-flight, last visible, min distance
            for desc, idx in [("mid-flight", len(steps)//2), ("last visible", last_vis_idx), ("closest approach", min_true_idx)]:
                if idx >= len(steps):
                    continue
                print(f"\n    At {desc} (step {steps[idx]}):")
                print(f"      true_dist={true_d[idx]:.1f}m, est_dist={est_d[idx]:.1f}m")
                print(f"      pos_error={pos_err[idx]:.1f}m, vel_error={vel_err[idx]:.1f}m") 
                print(f"      EKF W={W[idx]:.3f}m (truth={TARGET_PHYSICAL_SIZE_TRUTH}m, ratio={W[idx]/TARGET_PHYSICAL_SIZE_TRUTH:.2f}x)")
                print(f"      visible={vis[idx]}")
            
            # EKF W convergence
            print(f"\n    W convergence: initial={W[0]:.3f}m → mid={W[len(W)//2]:.3f}m → last_vis={W[last_vis_idx]:.3f}m → end={W[-1]:.3f}m")
            print(f"    Pos error convergence: initial={pos_err[0]:.1f}m → mid={pos_err[len(pos_err)//2]:.1f}m → last_vis={pos_err[last_vis_idx]:.1f}m")
            
            # What direction is the drone flying relative to the target at closest approach?
            if min_true_idx > 0 and min_true_idx < len(steps):
                drone_vel = log['drone_vel_world'][min_true_idx]
                drone_pos = log['drone_pos'][min_true_idx]
                true_pos = log['true_pos'][min_true_idx]
                los = true_pos - drone_pos
                los_norm = los / (np.linalg.norm(los) + 1e-6)
                v_along_los = np.dot(drone_vel, los_norm)
                v_speed = np.linalg.norm(drone_vel)
                v_perp = np.sqrt(max(v_speed**2 - v_along_los**2, 0))
                
                ekf_pos = log['ekf_pos'][min_true_idx]
                ekf_los = ekf_pos - drone_pos
                ekf_los_norm = ekf_los / (np.linalg.norm(ekf_los) + 1e-6)
                angle_between = np.degrees(np.arccos(np.clip(np.dot(los_norm, ekf_los_norm), -1, 1)))
                
                print(f"\n    At closest approach:")
                print(f"      Drone speed: {v_speed:.1f}m/s (along LOS: {v_along_los:.1f}m/s, perp: {v_perp:.1f}m/s)")
                print(f"      Angle(true LOS, est LOS) = {angle_between:.1f}°")
                print(f"      est target is {np.linalg.norm(ekf_pos - true_pos):.1f}m from true target")
                

if __name__ == '__main__':
    main()
