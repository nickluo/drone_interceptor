#!/usr/bin/env python3
"""未标定台架 MISSION 的 0x20 流保活节点 (demand-only 模式专用)。

背景: motor_parameters 未标定时, 桥整帧丢弃 /fpv/control_command 力指令,
cmd_valid 永不置位 -> 0x20 控制流在 mode_req 发出后即停 -> FC 看门狗撤销
OFFBOARD -> 桥的"任务中 OFFBOARD 丢失"保护自动发 DISMISS (2026-11-09 实测
MISSION 0.15s 即被杀; 若 OFFBOARD 恰好未授予则任务存活——纯抽签)。

本节点把拦截器的控制命令原样转发到 /fpv/control_command_raw (桥 YAML 注明
未标定期"只允许 raw 油门"的通道, 不经推力模型): 体速率/姿态指令直通,
collective_thrust 固定为 --throttle (默认 0.0 = 集体油门 0 -> OFFBOARD 怠速带地板,
与需求-only 语义一致, 电机不会因需求加速)。副作用: 0x20 流持续新鲜 ->
OFFBOARD 稳定保持 -> 不再触发 auto-DISMISS, 任务由飞行员 DISMISS 脉冲正常终止。

标定完成后请停用本节点 (正常力指令通道恢复, raw 转发反而会顶掉推力模型输出)。
"""

import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from quadrotor_msgs.msg import ControlCommand


class RawKeepalive(Node):
    def __init__(self, args):
        super().__init__("raw_keepalive")
        self.throttle = args.throttle
        self.forwarded = 0
        self.last_log = 0.0
        qos_be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.raw_pub = self.create_publisher(ControlCommand, "/fpv/control_command_raw", 10)
        self.create_subscription(ControlCommand, "/fpv/control_command",
                                 self.cb, qos_be)
        self.get_logger().info(
            "raw keepalive ready: forwarding control_command -> control_command_raw "
            "with fixed collective_thrust=%.2f (0=idle floor)" % self.throttle)

    def cb(self, msg):
        raw = ControlCommand()
        raw.header = msg.header
        raw.control_mode = msg.control_mode
        raw.armed = msg.armed
        raw.is_hover_state = msg.is_hover_state
        raw.bodyrates = msg.bodyrates
        raw.collective_thrust = self.throttle   # 归一化集体油门, 直通 (不经推力模型)
        self.raw_pub.publish(raw)
        self.forwarded += 1
        import time
        now = time.monotonic()
        if now - self.last_log > 5.0:
            self.last_log = now
            self.get_logger().info("forwarded=%d (rates pass-through, throttle=%.2f)"
                                   % (self.forwarded, self.throttle))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--throttle", type=float, default=0.0,
                        help="固定集体油门 0..1 (默认 0 = OFFBOARD 怠速地板)")
    args = parser.parse_args()
    rclpy.init()
    node = RawKeepalive(args)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
