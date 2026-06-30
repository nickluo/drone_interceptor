"""
AC-MPC + TensorRT INT8 闭环原型 (DQ-NMPC 变体, 与 AC-MPC 训练目标一致)

之前的 solver_acmpc_trt_quaternion.py 把 AC-MPC actor 预测的 8D √W
喂给了 prototype 的 *四元数* NMPC 求解器 — 这是错的:
AC-MPC 训练的 actor 假定下游求解器是
ACMPC.models.drone_model_dq.DroneModelDQ 所定义的 *DQ-NMPC*
(17D 状态, 李代数对数代价残差, 8D √W 参数),
两者的代价残差语义并不相同, 故权重无法跨求解器复用。

本文件直接调用 AC-MPC 仓库自带的 DQ-NMPC 求解器
(/home/nick/workspace/AC-MPC/c_generated_code +
 acados_ocp_drone_interceptor_mpc_dq.json),
其余仿真骨架 (CameraModel + PerceptionSim + bearing-angle TargetEKF + plot)
继承自 prototype 的 solver_flat_dq.py。

数据流:
    perception (euler 15D) ↔ drone_state (quat 16D, AC-MPC 策略口径)
                            ↔ solver (DQ 17D)

obs 构造与 AC-MPC InterceptorVecEnv._get_obs 完全一致:
    obs(30) = [drone_state_quat(16) | target_obs(6) | rel_features(8)]

使用:
    python3 solver_acmpc_trt_dq.py                       # 默认 INT8 + 虚拟相机 acmpc
    python3 solver_acmpc_trt_dq.py --precision fp32      # 浮点对照
    python3 solver_acmpc_trt_dq.py --no-trt              # AC-MPC w_init 静态 (见下)
    python3 solver_acmpc_trt_dq.py --baseline cmaes      # CMA-ES 离线优化的可部署静态权重
    python3 solver_acmpc_trt_dq.py --perception acmpc    # 跳过虚拟相机, 直接用训练域参数
    python3 solver_acmpc_trt_dq.py --virtual-camera none # 禁用重投影 (会裸跑, --perception prototype 会 OOD)

⚠️  感知解耦 (--perception / --virtual-camera) 说明:
    actor 的 obs (30D) 里 6 项来自 EKF (target_pos/vel + LOS/range 派生),
    而 EKF 的输出依赖相机内参 (focal_length) + 像素噪声 + q_accel.
    actor 训练时用的是虚拟相机 (focal=400px, fov=90°, std=1.0px, q=4.0);
    原型默认用的是接近硬件的相机 (focal=4638px, fov=30°, std=3.0px, q=3.0).
    两者 bearing/scale 差 3-12 倍, actor obs 会 OOD → INT8 量化再放大.

    --virtual-camera acmpc (默认) 的做法:
        保留真实相机几何 (可见性 + FOV 裁剪),
        但在 bbox 进 EKF 前乘以 (f_virt / f_real) = 400/4638 ≈ 0.0862;
        EKF 也用虚拟参数重建.
        结果: EKF 输出同分布 → obs 同分布 → actor 拦截能力 ≈ 训练时.
        量化线性 + 无额外训练, ROS2 端可复用同一段适配器.

⚠️  --no-trt (w_init) 说明:
    AC-MPC 的 w_init = [10, 3.16, 0.316, 0.0316, 0.01, 0.316, 0.01, 1.0]
    是给 RL/CMA-ES 训练用的 *随机种子*, 不是可部署的控制器。
    其中 W_los (=10) 远大于 W_close / W_zem, W_u_thrust (=0.01) 几乎不约束推力 →
    NMPC 会选择 "大姿态偏转对准 LOS + 低油门" 这种局部最优解, 飞机丢高度 →
    本原型里拦截必然失败 (min_dist ~ 30m, 飞机会下坠到 z<0)。
    这是 *预期* 表现, 仅作为 TRT 是否有效的对比基线。
    对比: solver_flat_dq.py 用的是 *手调* 9D 权重 (W_LOS=100, W_FOV=10,
    带 v_desired_adaptive), 针对该固定场景调出来的, 自然能裸跑拦截。
    如需可部署的静态控制器, 用 --baseline cmaes 加载 CMA-ES 权重。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---- prototype 感知 (与 solver_flat_dq.py 一致) ----
from exp_filter import EMAFilter
from camera_model import CameraModel, PerceptionSim
from target_ekf import TargetEKF

# ---- AC-MPC 仓库 ----
ACMPC_ROOT = Path("/home/nick/workspace/AC-MPC")
if str(ACMPC_ROOT) not in sys.path:
    sys.path.insert(0, str(ACMPC_ROOT))

from ACMPC.models.drone_model_dq import (        # noqa: E402
    DroneModelDQ,
    quat_state_to_dq_state,
    dq_state_to_quat_state,
    normalize_dq_np,
    dq_to_pose_np,
)
from drone_model_quaternion import DroneModelQuaternion  # noqa: E402  (prototype)

DEFAULT_ACMPC_ARTIFACTS = ACMPC_ROOT / "artifacts" / "trt"
SHOW_RESULTS = True

# ==============================================================================
# 与 AC-MPC interceptor_acados_dq.yaml::model_constants 保持一致
# ==============================================================================
G = 9.8065
MASS = 4.0
A_MAX = 20.0
A_MIN = 1.0
DRAG_COEFF = 0.1625
ROLL_PITCH_RATE_MAX = 2.0 * np.pi
YAW_RATE_MAX = np.pi
ATT_LIMIT = np.pi / 5
RATE_TC = 0.02
GIMBAL_RP_COEFF = 12.0
GIMBAL_Y_COEFF = 10.0

CTRL_FREQ_HZ = 50
TF = 0.8
N_HORIZON = int(TF * CTRL_FREQ_HZ)        # = 40
V_DESIRED = 20.0

# AC-MPC config 中的 8D w_init / w_min / w_max
# 顺序 = [W_los, W_align, W_close, W_rates, W_tilt, W_u_alpha, W_u_thrust, W_zem]
W_INIT_ACMPC = np.array([10.0, 3.16, 0.316, 0.0316, 0.01, 0.316, 0.01, 1.0],
                        dtype=np.float32)
W_MIN_ACMPC = np.array([3.0, 0.001, 0.01, 0.001, 1.0e-8, 0.005, 1.0e-7, 0.01],
                       dtype=np.float32)
W_MAX_ACMPC = np.array([50.0, 30.0, 20.0, 1.0, 0.5, 5.0, 0.5, 50.0],
                       dtype=np.float32)

WEIGHT_NAMES = ["W_los", "W_align", "W_close", "W_rates",
                "W_tilt", "W_u_alpha", "W_u_thrust", "W_zem"]

# ==============================================================================
# 仿真 / 感知参数 (沿用 solver_flat_dq.py)
# ==============================================================================
SIM_TIME = 15.0
INTERCEPTION_DISTANCE = 0.5
TRACKING_DISTANCE = 0.0

PERCEPTION_DELAY = 0.08
TARGET_PHYSICAL_SIZE_TRUTH = 0.5
TARGET_WIDTH_INITIAL_EST = 1.0

# 两套真实相机 + EKF 预设。选 actor 训练域还是原型历史值。
PERCEPTION_PRESETS = {
    "acmpc": dict(
        cam_focal_length=400.0,
        cam_fov_deg=90.0,
        bbox_pos_std=1.0,
        bbox_size_std=1.0,
        ekf_q_accel=4.0,
    ),
    "prototype": dict(
        cam_focal_length=16.0 * 1000.0 / 3.45,   # 4638 px (16mm / 3.45µm)
        cam_fov_deg=30.0,
        bbox_pos_std=3.0,
        bbox_size_std=3.0,
        ekf_q_accel=3.0,
    ),
}

# 与 actor 训练 / INT8 标定严格一致的"虚拟相机". 启用后:
#   1) CameraModel 仍按真实相机几何产生 bbox (保留真实可见性/FOV 裁剪);
#   2) bbox 的 (u, v, size) 被重投影到虚拟相机 (乘以 f_virt / f_real);
#   3) EKF 用虚拟相机参数构造 (focal=400, pos/size_std=1.0),
#      其输出均值+协方差与训练域同分布 -> actor obs 回到 in-distribution。
# 零训练 / 零标定的感知-actor 解耦, 代价只是一个线性重投影。
VIRTUAL_CAMERA_PRESETS = {
    "acmpc": dict(
        focal_length=400.0,
        bbox_pos_std=1.0,
        bbox_size_std=1.0,
        ekf_q_accel=4.0,
    ),
}

GIMBAL_DISTURBANCE_COEFF = 0.5

DRONE_INITIAL_STATE_EULER = np.array([
    0.0, 0.0, 10.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0])
TARGET_INITIAL_POS = np.array([100.0, 20.0, 15.0])
TARGET_VELOCITY = np.array([-5.0, 5.0, 0.5])

# AC-MPC DroneModelDQ 期望的 model_constants
ACMPC_MODEL_CONSTANTS = {
    'G': G, 'MASS': MASS, 'A_MAX': A_MAX, 'A_MIN': A_MIN,
    'DRAG_COEFF': DRAG_COEFF,
    'ROLL_PITCH_RATE_MAX': ROLL_PITCH_RATE_MAX,
    'YAW_RATE_MAX': YAW_RATE_MAX,
    'ATT_LIMIT': ATT_LIMIT,
    'RATE_TC': RATE_TC,
    'GIMBAL_RP_COEFF': GIMBAL_RP_COEFF,
    'GIMBAL_Y_COEFF': GIMBAL_Y_COEFF,
    'CTRL_FREQ_HZ': CTRL_FREQ_HZ,
    'TF': TF, 'N_HORIZON': N_HORIZON,
    'V_DESIRED': V_DESIRED,
    'DRONE_INITIAL_STATE': DRONE_INITIAL_STATE_EULER,
}


# ==============================================================================
# 状态转换工具 (15D euler ↔ 16D quat; 17D DQ 由 AC-MPC 提供)
# ==============================================================================
def euler_state_to_quat_state(euler_state: np.ndarray) -> np.ndarray:
    pos_vel = euler_state[0:6]
    euler = euler_state[6:9]
    omega = euler_state[9:12]
    gimbal = euler_state[12:15]
    quat = DroneModelQuaternion.euler_to_quaternion(euler[0], euler[1], euler[2])
    return np.hstack([pos_vel, quat, omega, gimbal])


def quat_state_to_euler_state(quat_state: np.ndarray) -> np.ndarray:
    pos_vel = quat_state[0:6]
    quat = quat_state[6:10]
    omega = quat_state[10:13]
    gimbal = quat_state[13:16]
    euler = DroneModelQuaternion.quaternion_to_euler(quat)
    return np.hstack([pos_vel, euler, omega, gimbal])


def normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


# ==============================================================================
# Cost weight providers
# ==============================================================================
class StaticWeightProvider:
    """AC-MPC w_init 作为 √W 静态基线 (canonical baseline)。"""
    def __init__(self, sqrt_w: np.ndarray = np.sqrt(W_INIT_ACMPC)):
        self.weights = sqrt_w.astype(np.float32)
        self.name = "static_w_init"
        self.last_latency_us = 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.weights.copy()


class TRTWeightProvider:
    """TensorRT actor 推理: obs(30) → √W(8). 输入归一化已 baked 进 ONNX 图。"""

    def __init__(self, engine_path: Path, max_batch: int = 1):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "需要 tensorrt + pycuda: "
                "pip install --user --break-system-packages pycuda tensorrt"
            ) from e

        self.cuda = cuda
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"反序列化引擎失败: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.input_name = n
            else:
                self.output_name = n

        self.obs_dim = self.engine.get_tensor_shape(self.input_name)[1]
        self.out_dim = self.engine.get_tensor_shape(self.output_name)[1]
        self.max_batch = max_batch
        self.d_in = cuda.mem_alloc(max_batch * self.obs_dim * 4)
        self.d_out = cuda.mem_alloc(max_batch * self.out_dim * 4)
        self.stream = cuda.Stream()
        self.name = engine_path.stem
        self.last_latency_us = 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs[None, :]
        b = obs.shape[0]
        obs = np.ascontiguousarray(obs.astype(np.float32))
        out = np.empty((b, self.out_dim), dtype=np.float32)

        t0 = time.perf_counter()
        self.context.set_input_shape(self.input_name, (b, self.obs_dim))
        self.cuda.memcpy_htod_async(self.d_in, obs, self.stream)
        self.context.set_tensor_address(self.input_name, int(self.d_in))
        self.context.set_tensor_address(self.output_name, int(self.d_out))
        self.context.execute_async_v3(self.stream.handle)
        self.cuda.memcpy_dtoh_async(out, self.d_out, self.stream)
        self.stream.synchronize()
        self.last_latency_us = (time.perf_counter() - t0) * 1e6

        return out[0] if b == 1 else out


# ==============================================================================
# AC-MPC 30D obs 构造 (与 InterceptorVecEnv._get_obs 完全一致)
# ==============================================================================
def build_acmpc_obs(
    quat_state: np.ndarray,         # [16]
    ekf_pos: np.ndarray,            # [3]
    ekf_vel: np.ndarray,            # [3]
    steps_remaining_norm: float,
    difficulty: float = 0.5,
) -> np.ndarray:
    drone = quat_state.astype(np.float32)
    target_obs = np.concatenate([ekf_pos, ekf_vel]).astype(np.float32)

    rel_pos = ekf_pos - drone[0:3]
    dist = float(np.linalg.norm(rel_pos)) + 1e-8
    los = rel_pos / dist

    rel_vel = ekf_vel - drone[3:6]
    closing_speed = -float(np.dot(rel_vel, los))
    tta = dist / max(closing_speed, 1.0)

    drone_speed = float(np.linalg.norm(drone[3:6])) + 1e-8
    vel_norm = drone[3:6] / drone_speed
    alignment = float(np.dot(vel_norm, los))

    dxy = float(np.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2)) + 1e-8
    los_az = float(np.arctan2(rel_pos[1], rel_pos[0]) / np.pi)
    los_el = float(np.clip(np.arctan2(rel_pos[2], dxy) / (np.pi / 2.0),
                           -1.0, 1.0))

    rel_feat = np.array([
        dist / 100.0,
        closing_speed / 20.0,
        float(np.clip(tta / 10.0, 0.0, 1.0)),
        alignment, los_az, los_el,
        float(steps_remaining_norm),
        float(difficulty),
    ], dtype=np.float32)
    return np.concatenate([drone, target_obs, rel_feat]).astype(np.float32)


# ==============================================================================
# AC-MPC DQ 求解器加载
# ==============================================================================
def build_acmpc_dq_solver():
    """加载 (或在缺失时构建) AC-MPC 的 DroneModelDQ acados 求解器。

    必须在 AC-MPC 仓库目录下执行, 因为 acados_template 写入的 JSON 路径与
    c_generated_code 都是相对于 CWD 的。
    """
    cwd_orig = Path.cwd()
    os.chdir(ACMPC_ROOT)
    try:
        drone_model = DroneModelDQ(ACMPC_MODEL_CONSTANTS)
        solver = drone_model.generate_solver()
    finally:
        os.chdir(cwd_orig)
    return solver


# ==============================================================================
# 主仿真
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["int8", "fp16", "fp32"],
                        default="int8")
    parser.add_argument("--engine-dir", default=str(DEFAULT_ACMPC_ARTIFACTS))
    parser.add_argument("--no-trt", action="store_true",
                        help="跳过 TRT, 使用静态 √W (见 --baseline)")
    parser.add_argument("--baseline", choices=["w_init", "cmaes"],
                        default="w_init",
                        help="--no-trt 时的静态 √W 来源: "
                             "w_init (训练种子, 预期失败) 或 "
                             "cmaes (checkpoints/cmaes_dq_v3_best_sqrt_w.npy)")
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perception", choices=list(PERCEPTION_PRESETS),
                        default="prototype",
                        help="真实相机 + EKF 预设: "
                             "acmpc=与训练域一致 (focal=400/fov=90/std=1.0), "
                             "prototype=与 solver_flat_dq.py 一致 "
                             "(focal=4638/fov=30/std=3.0, 接近真实硬件)")
    parser.add_argument("--virtual-camera",
                        choices=["none"] + list(VIRTUAL_CAMERA_PRESETS),
                        default="acmpc",
                        help="在 EKF 前重投影 bbox 到虚拟相机: "
                             "acmpc=强制对齐到训练域 (推荐, 解耦感知与actor), "
                             "none=直接送真实 bbox 给 EKF")
    parser.add_argument("--no-clip", dest="clip_weights", action="store_false",
                        help="禁用 √W 裁剪 (默认开启)")
    parser.set_defaults(clip_weights=True)
    args = parser.parse_args()
    np.random.seed(args.seed)

    # ---- 感知预设 + 虚拟相机 ----
    perc = PERCEPTION_PRESETS[args.perception]
    cam_focal = perc["cam_focal_length"]
    cam_fov_deg = perc["cam_fov_deg"]
    bbox_pos_std = perc["bbox_pos_std"]
    bbox_size_std = perc["bbox_size_std"]
    if args.virtual_camera == "none":
        virt = None
        ekf_focal = cam_focal
        ekf_pos_std = bbox_pos_std
        ekf_size_std = bbox_size_std
        ekf_q = perc["ekf_q_accel"]
        bbox_scale = 1.0
    else:
        virt = VIRTUAL_CAMERA_PRESETS[args.virtual_camera]
        ekf_focal = virt["focal_length"]
        ekf_pos_std = virt["bbox_pos_std"]
        ekf_size_std = virt["bbox_size_std"]
        ekf_q = virt["ekf_q_accel"]
        bbox_scale = ekf_focal / cam_focal
    print(f"[INFO] 真实相机: {args.perception}  focal={cam_focal:.1f}px "
          f"fov={cam_fov_deg:.0f}°  bbox_pos_std={bbox_pos_std}  "
          f"bbox_size_std={bbox_size_std}")
    if virt is None:
        print("[INFO] 虚拟相机: 禁用 (EKF 看真实 bbox)")
    else:
        print(f"[INFO] 虚拟相机: {args.virtual_camera}  focal={ekf_focal:.1f}px "
              f"bbox_scale={bbox_scale:.4f}  ekf_pos_std={ekf_pos_std}  "
              f"ekf_q_accel={ekf_q}")
    if args.perception == "prototype" and virt is None and not args.no_trt:
        print("[⚠️ WARN] --perception prototype + --virtual-camera none + TRT: "
              "obs 会 OOD, 预期拦截退化。")

    sqrt_w_min = np.sqrt(W_MIN_ACMPC)
    sqrt_w_max = np.sqrt(W_MAX_ACMPC)

    # ---- weight provider ----
    if args.no_trt:
        if args.baseline == "cmaes":
            cmaes_path = ACMPC_ROOT / "checkpoints" / "cmaes_dq_v3_best_sqrt_w.npy"
            if not cmaes_path.exists():
                print(f"[ERROR] CMA-ES checkpoint 不存在: {cmaes_path}",
                      file=sys.stderr)
                sys.exit(1)
            sqrt_w = np.load(cmaes_path).astype(np.float32).reshape(-1)
            if sqrt_w.shape[0] != 8:
                print(f"[ERROR] CMA-ES √W 形状不对: {sqrt_w.shape}, 期望 (8,)",
                      file=sys.stderr)
                sys.exit(1)
            weight_provider = StaticWeightProvider(sqrt_w)
            weight_provider.name = "static_cmaes"
            print(f"[INFO] 模式: STATIC-CMAES √W = {sqrt_w.round(3)}")
            print(f"[INFO] CMA-ES 离线优化的可部署权重 ({cmaes_path.name})")
        else:
            weight_provider = StaticWeightProvider()
            print(f"[INFO] 模式: STATIC-w_init √W = {weight_provider.weights.round(3)}")
            print("[⚠️ WARN] w_init 是 RL/CMA-ES 训练随机种子, 不是可部署控制器。")
            print("[⚠️ WARN] 预期拦截失败 (~30m miss, 飞机会丢高度); "
                  "仅作为 TRT 是否有效的对比基线。")
            print("[⚠️ WARN] 如需可部署静态控制器, 用 --baseline cmaes。")
    else:
        engine_path = (Path(args.engine_dir)
                       / f"actor_inference_{args.precision}.engine")
        if not engine_path.exists():
            print(f"[ERROR] 找不到引擎: {engine_path}", file=sys.stderr)
            sys.exit(1)
        weight_provider = TRTWeightProvider(engine_path, max_batch=1)
        print(f"[INFO] 模式: TRT-{args.precision.upper()} ({engine_path.name})")

    # ---- AC-MPC DQ 求解器 ----
    print("[INFO] 加载 AC-MPC DQ-NMPC 求解器 ...")
    acados_solver = build_acmpc_dq_solver()
    print("[INFO] AC-MPC DQ-NMPC 求解器就绪。")

    dt = TF / N_HORIZON
    N_sim = int(SIM_TIME / dt)

    # 主状态: 16D 四元数 (与 AC-MPC 策略口径一致)
    x_quat = euler_state_to_quat_state(DRONE_INITIAL_STATE_EULER)
    target_pos_true = TARGET_INITIAL_POS.copy()
    target_vel_true = TARGET_VELOCITY.copy()

    camera = CameraModel(cam_focal, TARGET_PHYSICAL_SIZE_TRUTH,
                         cam_fov_deg, bbox_pos_std, bbox_size_std)
    estimator = TargetEKF(dt, ekf_focal, ekf_pos_std,
                          ekf_size_std,
                          initial_width_est=TARGET_WIDTH_INITIAL_EST,
                          range_init_bias=0.6,
                          q_accel=ekf_q, q_W=0.01,
                          alpha_R_inflate=1.0)
    perceptor = PerceptionSim(camera, PERCEPTION_DELAY, dt)

    target_lost_counter = 0
    last_good_pos_est = None        # 死算锚点 (最后一次目标可见时的 EKF 估计)
    last_good_vel_est = None
    last_good_step = 0
    terminal_phase = False           # 终端制导 (近距离 + 盲飞)
    min_dist_so_far = float('inf')
    overshoot_streak = 0             # 连续 "距离增大" 步数, 用于检测 fly-by
    MAX_BLIND_STEPS_NORMAL = int(N_HORIZON * 2.0)        # 远距离丢失容忍 (1.6s)
    MAX_BLIND_STEPS_TERMINAL = int(N_HORIZON * 5.0)      # 终端制导容忍 (4.0s)
    history = {
        'x': [quat_state_to_euler_state(x_quat)],
        'u': [],
        'target_true': [target_pos_true.copy()],
        'target_vel_true': [target_vel_true.copy()],
        'target_est_ekf': [],
        'target_est_raw': [],
        'distance': [],
        'cost_weights': [],
        'inference_us': [],
    }

    # 热启动: 17D DQ 状态
    x_dq_init = quat_state_to_dq_state(x_quat)
    u_hover = np.array([0.0, 0.0, 0.0, G])
    for k in range(N_HORIZON):
        acados_solver.set(k, "x", x_dq_init)
        acados_solver.set(k, "u", u_hover)
    acados_solver.set(N_HORIZON, "x", x_dq_init)

    print("\n[INFO] 开始 AC-MPC + TRT 闭环 (DQ-NMPC 变体) ...")
    solve_time_avg = 0.0
    inf_time_avg = 0.0
    distance = float('inf')
    i = 0
    raw_pos_est = None

    for i in range(N_sim):
        # -------- 1. 感知 (15D euler) --------
        x_euler = quat_state_to_euler_state(x_quat)
        full, target_pos_true = perceptor.update(
            x_euler, target_pos_true, target_vel_true, history)
        if not full:
            continue
        drone_state_at_meas, meas_bbox, meas_visible = perceptor.get_measure()
        # 虚拟相机重投影 (bbox_scale != 1.0 表示启用了 virtual-camera)
        if meas_visible and bbox_scale != 1.0:
            meas_bbox = dict(meas_bbox)
            meas_bbox["u"] *= bbox_scale
            meas_bbox["v"] *= bbox_scale
            meas_bbox["size"] *= bbox_scale

        # -------- 2. EKF 预测 --------
        if estimator.is_initialized:
            estimator.predict(dt)

        # -------- 3. EKF 更新 --------
        if meas_visible:
            target_lost_counter = 0
            drone_pos = drone_state_at_meas[0:3]
            psi = drone_state_at_meas[8]
            gr, gp, gy = drone_state_at_meas[12:15]
            if not estimator.is_initialized:
                estimator.initialize(meas_bbox, drone_pos, psi, gr, gp, gy)
            else:
                estimator.update(meas_bbox, drone_pos, psi, gr, gp, gy)
            raw_pos_est = estimator.get_target_position()
        else:
            target_lost_counter += 1
            raw_pos_est = None
            if estimator.is_initialized and target_lost_counter <= 3:
                dist_dbg_lost = float(np.linalg.norm(target_pos_true - x_quat[0:3]))
                print(f"[WARNING] step {i}: 目标丢失 "
                      f"({target_lost_counter}) dist={dist_dbg_lost:.1f}m")

        max_blind = (MAX_BLIND_STEPS_TERMINAL if terminal_phase
                     else MAX_BLIND_STEPS_NORMAL)
        if target_lost_counter > max_blind:
            phase = "terminal" if terminal_phase else "cruise"
            print(f"\n[FAILED] 目标丢失超过 {max_blind*dt:.1f}s ({phase}), 失败。")
            break

        # -------- 4. 规划 --------
        if not estimator.is_initialized:
            u_optimal_raw = np.array([0.0, 0.0, 0.0, G])
            x_quat[3:6] = 0.0
            x_quat[6:10] = normalize_quat(x_quat[6:10])
            x_quat[13:16] = 0.0
        else:
            pos_est = estimator.x[0:3].copy()
            vel_est = estimator.x[3:6].copy()

            # ---- 死算: 目标丢失时, 用最后一次良好估计做恒速外推, 比 EKF predict 更稳 ----
            if meas_visible:
                last_good_pos_est = pos_est.copy()
                last_good_vel_est = vel_est.copy()
                last_good_step = i
            elif last_good_pos_est is not None:
                dt_since = (i - last_good_step) * dt
                pos_est = last_good_pos_est + last_good_vel_est * dt_since
                vel_est = last_good_vel_est.copy()
                # 进入终端制导: 近距离丢失 → 放宽 blind 容忍
                drone_pos_now = x_quat[0:3]
                dist_dr = float(np.linalg.norm(pos_est - drone_pos_now))
                if not terminal_phase and dist_dr < 15.0:
                    terminal_phase = True
                    print(f"[TERMINAL] step {i}: 进入终端制导 dist={dist_dr:.1f}m (死算)")

            # ---- 4a. Actor 推理 (30D obs → √W 8D) ----
            steps_remaining_norm = max(0.0, (N_sim - i) / N_sim)
            obs = build_acmpc_obs(x_quat, pos_est, vel_est,
                                  steps_remaining_norm, args.difficulty)
            w_pred = weight_provider(obs).astype(np.float64)
            inf_us = weight_provider.last_latency_us

            if args.clip_weights:
                w_pred = np.clip(w_pred, sqrt_w_min, sqrt_w_max)

            history['cost_weights'].append(w_pred.copy())
            history['inference_us'].append(inf_us)

            # ---- 4b. AC-MPC DQ-NMPC 求解 ----
            x_dq = quat_state_to_dq_state(x_quat)
            x_dq[0:8] = normalize_dq_np(x_dq[0:8])
            acados_solver.set(0, "lbx", x_dq)
            acados_solver.set(0, "ubx", x_dq)

            # 19D 参数向量布局 (= ACMPC.mpc.acados_head._build_param_vector):
            #   p[0:3]   target_pos
            #   p[3:6]   target_vel
            #   p[6]     drag_coeff
            #   p[7]     v_desired
            #   p[8]     rate_tc
            #   p[9:17]  sqrt(cost_weights) — 8D
            #   p[17]    gimbal_rp_coeff
            #   p[18]    gimbal_y_coeff
            for k in range(N_HORIZON + 1):
                t_k = k * dt
                pred_target_pos = pos_est + t_k * vel_est
                p_k = np.zeros(19)
                p_k[0:3] = pred_target_pos
                p_k[3:6] = vel_est
                p_k[6] = DRAG_COEFF
                p_k[7] = V_DESIRED
                p_k[8] = RATE_TC
                p_k[9:17] = w_pred
                p_k[17] = GIMBAL_RP_COEFF
                p_k[18] = GIMBAL_Y_COEFF
                acados_solver.set(k, "p", p_k)

            t0 = time.time()
            status = acados_solver.solve()
            solve_us = (time.time() - t0) * 1e6
            if status not in (0, 2) and i % 50 == 0:
                print(f"[WARNING] ACADOS status={status} @ step {i} "
                      f"(继续使用次优解)")

            if i % 50 == 0:
                dist_dbg = np.linalg.norm(target_pos_true - x_quat[0:3])
                w_str = ", ".join(f"{w:.2f}" for w in w_pred)
                print(f"[INFO] step {i:4d}  inf={inf_us:6.1f}us  "
                      f"solve={int(solve_us):5d}us  "
                      f"dist={dist_dbg:6.2f}m  √W=[{w_str}]")

            n = max(i, 1)
            solve_time_avg = solve_time_avg * ((n - 1.0) / n) + solve_us / n
            inf_time_avg = inf_time_avg * ((n - 1.0) / n) + inf_us / n

            u_optimal_raw = acados_solver.get(0, "u")
            x_dq_next = acados_solver.get(1, "x")

            if np.any(np.isnan(x_dq_next)):
                print(f"[ERROR] NaN @ step {i}, 终止。")
                break

            x_dq_next[0:8] = normalize_dq_np(x_dq_next[0:8])
            x_quat = dq_state_to_quat_state(x_dq_next)
            x_quat[6:10] = normalize_quat(x_quat[6:10])

            # 云台扰动 (与 prototype 模型一致)
            wx_, wy_, wz_ = x_quat[10:13]
            x_quat[13] += GIMBAL_DISTURBANCE_COEFF * wx_ * dt
            x_quat[14] += GIMBAL_DISTURBANCE_COEFF * wy_ * dt
            x_quat[15] += GIMBAL_DISTURBANCE_COEFF * wz_ * dt

            # 热启动 shift (17D DQ)
            for k in range(N_HORIZON - 1):
                x_n = acados_solver.get(k + 1, "x")
                x_n[0:8] = normalize_dq_np(x_n[0:8])
                u_n = acados_solver.get(k + 1, "u")
                acados_solver.set(k, "x", x_n)
                acados_solver.set(k, "u", u_n)
            acados_solver.set(N_HORIZON - 1, "u", u_n)
            x_last = acados_solver.get(N_HORIZON, "x")
            x_last[0:8] = normalize_dq_np(x_last[0:8])
            acados_solver.set(N_HORIZON - 1, "x", x_last)

        body_rates = u_optimal_raw[0:3]
        thrust_accel = u_optimal_raw[3]
        u_to_log = np.hstack([body_rates, thrust_accel * MASS])

        # -------- 5. 步进环境 + 记录 --------
        target_pos_true += target_vel_true * dt
        history['x'].append(quat_state_to_euler_state(x_quat))
        history['u'].append(u_to_log.copy())
        history['target_true'].append(target_pos_true.copy())
        history['target_vel_true'].append(target_vel_true.copy())
        if estimator.is_initialized:
            history['target_est_ekf'].append(estimator.x.copy())
        if raw_pos_est is not None:
            history['target_est_raw'].append(raw_pos_est)

        distance = float(np.linalg.norm(target_pos_true - x_quat[0:3]))
        history['distance'].append(distance)

        if distance < min_dist_so_far:
            min_dist_so_far = distance
            overshoot_streak = 0
        else:
            overshoot_streak += 1

        if distance < INTERCEPTION_DISTANCE:
            print(f"\n[SUCCESS] step {i} 拦截成功! dist={distance:.2f}m")
            break

        # 终端制导阶段检测 fly-by: 距离连续增大 ≥ 0.4s 且 min_dist 已经很近 → 退出
        if (terminal_phase and overshoot_streak > int(N_HORIZON * 0.5)
                and min_dist_so_far < 10.0):
            print(f"\n[FLY-BY] step {i}: 已掠过目标 (min_dist={min_dist_so_far:.2f}m, "
                  f"当前 dist={distance:.2f}m) — 终止仿真。")
            break

    min_dist = min(history['distance']) if history['distance'] else 999.0
    print(f"\n[INFO] final_dist={distance:.2f}m  min_dist={min_dist:.2f}m")
    if history['distance']:
        argmin = int(np.argmin(history['distance']))
        xs_arr = np.asarray(history['x'])
        ts_arr = np.asarray(history['target_true'])
        # history['x'] has len = len(distance) + 1 (initial state appended pre-loop)
        idx_x = min(argmin + 1, len(xs_arr) - 1)
        idx_t = min(argmin + 1, len(ts_arr) - 1)
        print(f"[INFO] @min_dist idx={argmin}  drone={xs_arr[idx_x, 0:3]}  "
              f"target={ts_arr[idx_t]}")
        print(f"[INFO] @end             drone={xs_arr[-1, 0:3]}  "
              f"target={ts_arr[-1]}")
    print(f"[INFO] 平均求解耗时: {int(solve_time_avg)} us")
    print(f"[INFO] 平均 Actor 推理耗时: {inf_time_avg:.1f} us "
          f"({weight_provider.name})")

    if SHOW_RESULTS and not args.no_plot:
        plot_results(history, dt, weight_provider.name)


# ==============================================================================
# Plot
# ==============================================================================
def plot_results(hist, dt, mode_label: str):
    if not hist['u']:
        print("[ERROR] 无有效数据。")
        return

    hist['x'] = np.array(hist['x'])
    hist['u'] = np.array(hist['u'])
    hist['target_true'] = np.array(hist['target_true'])
    hist['target_est_ekf'] = (np.array(hist['target_est_ekf'])
                              if hist['target_est_ekf'] else np.array([]))
    hist['target_est_raw'] = (np.array(hist['target_est_raw'])
                              if hist['target_est_raw'] else np.array([]))
    hist['distance'] = np.array(hist['distance'])
    hist['cost_weights'] = (np.array(hist['cost_weights'])
                            if hist['cost_weights'] else np.zeros((0, 8)))
    hist['inference_us'] = (np.array(hist['inference_us'])
                            if hist['inference_us'] else np.zeros(0))

    time_x = np.linspace(0, dt * (len(hist['x']) - 1), len(hist['x']))
    time_u = np.linspace(0, dt * (len(hist['u']) - 1), len(hist['u']))

    # ---- (A) Geometry: 3D + top-down + side -----------------------------------
    # 3D matplotlib axes can be visually misleading; show 2D projections too.
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(f'AC-MPC DQ + {mode_label} — trajectory geometry')

    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.plot(hist['x'][:, 0], hist['x'][:, 1], hist['x'][:, 2],
              label='Drone', c='b', lw=2)
    ax3d.plot(hist['target_true'][:, 0], hist['target_true'][:, 1],
              hist['target_true'][:, 2], label='Target True', c='r', ls='--')
    if len(hist['target_est_ekf']) > 0:
        ax3d.plot(hist['target_est_ekf'][:, 0], hist['target_est_ekf'][:, 1],
                  hist['target_est_ekf'][:, 2], label='Target EKF', c='g')
    ax3d.scatter(*hist['x'][0, 0:3], c='b', s=80, marker='o', label='Drone Start')
    ax3d.scatter(*hist['target_true'][0], c='r', s=80, marker='x', label='Target Start')
    ax3d.set_xlabel('East'); ax3d.set_ylabel('North'); ax3d.set_zlabel('Up')
    ax3d.legend(fontsize=7); ax3d.set_title('3D (perspective — misleading)')

    # min-dist marker
    if len(hist['distance']) > 0:
        argmin_d = int(np.argmin(hist['distance']))
        idx_xm = min(argmin_d + 1, len(hist['x']) - 1)
        idx_tm = min(argmin_d + 1, len(hist['target_true']) - 1)
        d_pt = hist['x'][idx_xm, 0:3]
        t_pt = hist['target_true'][idx_tm]
    else:
        d_pt = t_pt = None

    # Top-down East-North
    ax_tn = fig.add_subplot(2, 2, 2)
    ax_tn.plot(hist['x'][:, 0], hist['x'][:, 1], 'b-', lw=2, label='Drone')
    ax_tn.plot(hist['target_true'][:, 0], hist['target_true'][:, 1],
               'r--', label='Target True')
    if len(hist['target_est_ekf']) > 0:
        ax_tn.plot(hist['target_est_ekf'][:, 0], hist['target_est_ekf'][:, 1],
                   'g-', alpha=0.5, label='Target EKF')
    ax_tn.scatter(*hist['x'][0, 0:2], c='b', marker='o', s=80)
    ax_tn.scatter(*hist['target_true'][0, 0:2], c='r', marker='x', s=80)
    if d_pt is not None:
        ax_tn.plot([d_pt[0], t_pt[0]], [d_pt[1], t_pt[1]],
                   'k-', lw=1.5, label=f'min-dist {hist["distance"][argmin_d]:.2f}m')
        ax_tn.scatter([d_pt[0], t_pt[0]], [d_pt[1], t_pt[1]],
                      c=['b', 'r'], s=60, zorder=5)
    ax_tn.set_xlabel('East [m]'); ax_tn.set_ylabel('North [m]')
    ax_tn.set_title('Top-down (East vs North)')
    ax_tn.legend(fontsize=8); ax_tn.grid(True); ax_tn.set_aspect('equal')

    # Side East-Up
    ax_eu = fig.add_subplot(2, 2, 3)
    ax_eu.plot(hist['x'][:, 0], hist['x'][:, 2], 'b-', lw=2, label='Drone')
    ax_eu.plot(hist['target_true'][:, 0], hist['target_true'][:, 2],
               'r--', label='Target True')
    ax_eu.scatter(*hist['x'][0, [0, 2]], c='b', marker='o', s=80)
    ax_eu.scatter(*hist['target_true'][0, [0, 2]], c='r', marker='x', s=80)
    if d_pt is not None:
        ax_eu.plot([d_pt[0], t_pt[0]], [d_pt[2], t_pt[2]],
                   'k-', lw=1.5, label=f'min-dist {hist["distance"][argmin_d]:.2f}m')
        ax_eu.scatter([d_pt[0], t_pt[0]], [d_pt[2], t_pt[2]],
                      c=['b', 'r'], s=60, zorder=5)
    ax_eu.set_xlabel('East [m]'); ax_eu.set_ylabel('Up [m]')
    ax_eu.set_title('Side (East vs Up)')
    ax_eu.legend(fontsize=8); ax_eu.grid(True); ax_eu.set_aspect('equal')

    # Range vs time
    ax_d = fig.add_subplot(2, 2, 4)
    if len(hist['distance']) > 0:
        t_d = np.linspace(0, dt * (len(hist['distance']) - 1),
                          len(hist['distance']))
        ax_d.plot(t_d, hist['distance'], 'b-')
        ax_d.axhline(INTERCEPTION_DISTANCE, color='r', ls='--',
                     label=f'Intercept ({INTERCEPTION_DISTANCE} m)')
        ax_d.axvline(t_d[argmin_d], color='k', ls=':',
                     label=f'min @ t={t_d[argmin_d]:.2f}s '
                           f'({hist["distance"][argmin_d]:.2f}m)')
    ax_d.set_xlabel('Time [s]'); ax_d.set_ylabel('Range [m]')
    ax_d.set_title('Range vs time')
    ax_d.legend(fontsize=8); ax_d.grid(True); ax_d.set_yscale('log')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # ---- (B) Diagnostics: cost weights, latency, control ----------------------
    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'AC-MPC DQ + {mode_label} — controller diagnostics')

    if len(hist['cost_weights']) > 0:
        n_w = hist['cost_weights'].shape[0]
        t_w = np.linspace(0, dt * (n_w - 1), n_w)
        for j in range(hist['cost_weights'].shape[1]):
            axs[0].plot(t_w, hist['cost_weights'][:, j],
                        label=WEIGHT_NAMES[j])
        axs[0].set_yscale('log')
        axs[0].set_ylabel('√W (log)'); axs[0].legend(ncol=4, fontsize=8)
        axs[0].grid(True)

    if len(hist['inference_us']) > 0:
        n_i = len(hist['inference_us'])
        t_i = np.linspace(0, dt * (n_i - 1), n_i)
        axs[1].plot(t_i, hist['inference_us'])
        axs[1].set_ylabel('Actor inference [us]'); axs[1].grid(True)

    if len(hist['u']) > 0:
        axs[2].plot(time_u, hist['u'][:, 0], label='wx')
        axs[2].plot(time_u, hist['u'][:, 1], label='wy')
        axs[2].plot(time_u, hist['u'][:, 2], label='wz')
        axs[2].set_ylabel('Body rate [rad/s]')
        axs[2].set_xlabel('Time [s]')
        axs[2].legend(); axs[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
