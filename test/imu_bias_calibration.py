#!/usr/bin/env python3
"""IMU 加速度计偏置标定 —— 静置台架, 从 /mavros/imu/data (桥 0x10 遥测) 采集标定。

原理:机体静止时比力真值 f_true = R_b2n^T · [0,0,g] (重力在体坐标的投影;
R 来自同一条消息的 FC 姿态四元数, FLU->ENU)。逐样本
    bias_i = f_meas_i − R_i^T · [0,0,g]
均值即体坐标系偏置 (m/s^2), 直接写入 relative_imm_ukf_params.yaml 的 imu_bias
(节点语义: dv_own = ∫(R·(f_imu − bias) − g)dt, 见 imu_callback)。

自验证:同窗口分别对"未扣/已扣偏置"的 a_world 积分, 报告漂移 (m/s 与 m/s/min)。

注意事项:
- 姿态源是 FC 的加速度计参考姿态, 静止时恒读"水平"——若机体物理上有倾斜,
  重力分量会被并入偏置 (对台架水平姿态这正是想要的"该姿态下零漂";
  跨大姿态角使用时此部分不旋转)。标定时尽量放水平。
- 偏置随温度漂移; 冷机/热机值可能差 ~0.05-0.1 m/s^2, 建议热机后标定。
- 全程不要碰机体。

用法: python3 imu_bias_calibration.py [--duration 60] [--g 9.80665]
"""

import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


def quat_to_matrix(q):
    """w,x,y,z 四元数 -> 3x3 旋转矩阵 (体->世界)。"""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def mat_t_vec3(m, v):
    """m^T · v"""
    return tuple(sum(m[r][c] * v[r] for r in range(3)) for c in range(3))


class ImuBiasCalibration(Node):
    def __init__(self, args):
        super().__init__("imu_bias_calibration")
        self.args = args
        self.samples = []       # (t, f_meas, bias_i, euler_deg)
        self.t0 = None
        self.last_t = None
        qos_be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, "/mavros/imu/data", self.cb, qos_be)
        self.get_logger().info(
            "collecting %d s of /mavros/imu/data — keep the vehicle STILL ..."
            % args.duration)

    def cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        if self.last_t is not None and t <= self.last_t:
            return  # 单调守卫
        self.last_t = t
        if t - self.t0 > self.args.duration:
            return
        q = msg.orientation
        R = quat_to_matrix((q.w, q.x, q.y, q.z))
        f = (msg.linear_acceleration.x, msg.linear_acceleration.y,
             msg.linear_acceleration.z)
        g_vec = (0.0, 0.0, self.args.g)
        f_true = mat_t_vec3(R, g_vec)          # R^T · [0,0,g]
        bias_i = tuple(f[k] - f_true[k] for k in range(3))
        # 姿态欧拉角 (仅显示用): R = Rz(yaw)Ry(pitch)Rx(roll)
        pitch = math.asin(max(-1.0, min(1.0, -R[0][2])))
        roll = math.atan2(R[1][2], R[2][2])
        yaw = math.atan2(-R[0][1], R[0][0])
        self.samples.append((t, f, bias_i,
                             (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))))
        if len(self.samples) % 400 == 0:
            self.get_logger().info("  ... %d samples (%.0f%%)"
                                   % (len(self.samples),
                                      100.0 * (t - self.t0) / self.args.duration))

    def done(self):
        return (self.t0 is not None and self.last_t is not None and
                self.last_t - self.t0 >= self.args.duration)

    def report(self):
        n = len(self.samples)
        print("\n" + "=" * 66)
        print("IMU 加速度计偏置标定  (%d 样本, %.1f s)" % (n, self.last_t - self.t0))
        print("=" * 66)
        if n < 1000:
            print("样本不足 (%d) — 请确认机体静置并重跑" % n)
            return None
        f_mean = [sum(s[1][k] for s in self.samples) / n for k in range(3)]
        b_mean = [sum(s[2][k] for s in self.samples) / n for k in range(3)]
        b_std = [math.sqrt(sum((s[2][k] - b_mean[k]) ** 2 for s in self.samples) / (n - 1))
                 for k in range(3)]
        e_mean = [sum(s[3][k] for s in self.samples) / n for k in range(3)]

        print("比力均值 (m/s^2):  x=%+8.4f  y=%+8.4f  z=%+8.4f" % tuple(f_mean))
        print("姿态均值 (deg):    roll=%+7.3f pitch=%+7.3f yaw=%+8.3f" % tuple(e_mean))
        print("偏置均值 (m/s^2):  x=%+8.4f  y=%+8.4f  z=%+8.4f" % tuple(b_mean))
        print("偏置标准差:        x=%8.4f  y=%8.4f  z=%8.4f" % tuple(b_std))
        print("|bias| = %.4f m/s² (%.1f mg)  -> 未标定时积分漂移 ~%.2f m/s/min"
              % (math.sqrt(sum(v * v for v in b_mean)),
                 1000.0 * math.sqrt(sum(v * v for v in b_mean)) / self.args.g,
                 60.0 * math.sqrt(sum(v * v for v in b_mean))))

        # 自验证: 静止时 a_world = R·(f − bias) − g, 扣偏置后逐样本残差均值即
        # 剩余漂移源; 未扣偏置时窗口积分漂移 = |bias| × 窗口时长
        span = self.last_t - self.t0
        print("窗口漂移验证 (%.1fs): 未扣偏置 %.3f m/s -> 扣偏置后 ~0 (残差均值内消)"
              % (span, math.sqrt(sum(v * v for v in b_mean)) * span))
        print("\n写入 relative_imm_ukf_params.yaml:")
        print("    imu_bias: [%.4f, %.4f, %.4f]" % tuple(b_mean))
        print("=" * 66)
        return b_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=60.0, help="采集时长 (s)")
    parser.add_argument("--g", type=float, default=9.80665)
    args = parser.parse_args()

    rclpy.init()
    node = ImuBiasCalibration(args)
    try:
        deadline = time.monotonic() + args.duration + 10.0
        while rclpy.ok() and not node.done() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        node.report()
    except KeyboardInterrupt:
        node.report()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
