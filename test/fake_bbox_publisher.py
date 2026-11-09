#!/usr/bin/env python3
"""假视觉目标发布器 —— 真机无桨静态 MISSION 测试专用 (替代 airsim_tracker)。

在 /fpv/gimbal (桥的云台融合位姿) 驱动下, 发布一个保持在相机 FOV 中心附近的
假 bbox 检测 + tracker 状态 + 云台瞄准 LOS, 使 drone_interceptor 与
custom_link_bridge 的完整视觉链路在没有任何相机/检测器的情况下运转:

  /fpv/target_bbox   vision_msgs/Detection2D  检测框 (像素, 中心 = 图像中心 + 偏移)
  /fpv/tracker_state std_msgs/Int8            0 检测中 / 1 发现 / 2 跟踪 / 3 丢失
  /fpv/target_los    geometry_msgs/TwistStamped  云台瞄准 LOS (默认指向画面中心)
  (订阅) /fpv/gimbal  sensor_msgs/Imu          发布节拍 + 时间戳来源
  (订阅) /fpv/tracker_trigger std_msgs/Header  ON 启动 / OFF 停止 / DISMISS 忽略
                                               (与真 tracker 的起桨边沿语义一致)

关键约定 (见 interceptor_node_relative_imm_ukf.cpp bbox_callback /
camera_pose_history.h at()):
  * bbox.header.stamp 必须与 /fpv/gimbal 的某帧 stamp 一致 —— 这里逐字复制驱动帧的
    stamp, 保证估计器 gimbal_history_ 精确匹配 (或落入 <=50 ms 的零阶保持窗)。
  * bbox 中心是绝对像素坐标 (估计器自己减主点); 尺寸取 width (size_x) 供单目测距
    r = focal * w_guess / size_x, 默认 80 px @ fx~1057, w_guess 1.0 -> r~13.2 m。
  * frame_id 非空 = 有效检测 (空串 = 丢失)。

用法 (先 source ROS + 工作区):
  python3 fake_bbox_publisher.py                     # 等 arm 边沿 ON 后自动发布
  python3 fake_bbox_publisher.py --always-on         # 无需 ON 立即发布 (预检查用)
  python3 fake_bbox_publisher.py --bbox-width 120 --rate 30
"""

import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Header, Int8
from vision_msgs.msg import BoundingBox2D, Detection2D

STATE_DETECTING, STATE_FOUND, STATE_TRACKING, STATE_LOST = 0, 1, 2, 3


class FakeBboxPublisher(Node):
    def __init__(self, args):
        super().__init__("fake_bbox_publisher")
        self.args = args
        self.active = args.always_on
        self.activated_at = time.monotonic() if self.active else None
        self.state_sent = None
        self.last_pub_mono = 0.0
        self.last_pub_stamp = 0.0
        self.pub_count = 0
        self.gimbal_count = 0
        self.last_gimbal_mono = None
        self.min_period = 1.0 / args.rate if args.rate > 0 else 0.0
        self.next_heartbeat = 0.0
        self.next_report = time.monotonic() + 5.0

        qos_be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.detection_pub = self.create_publisher(Detection2D, "/fpv/target_bbox", 10)
        self.state_pub = self.create_publisher(Int8, "/fpv/tracker_state", 10)
        self.los_pub = self.create_publisher(TwistStamped, "/fpv/target_los", 10)
        self.gimbal_sub = self.create_subscription(Imu, "/fpv/gimbal", self.gimbal_cb, qos_be)
        self.trigger_sub = self.create_subscription(
            Header, "/fpv/tracker_trigger", self.trigger_cb, 10)

        self.cx = args.image_width / 2.0 + args.offset_x
        self.cy = args.image_height / 2.0 + args.offset_y
        self.get_logger().info(
            "fake_bbox_publisher ready: center=(%.1f,%.1f) px size=%.0fx%.0f px "
            "rate=%.0f Hz lock_delay=%.1f s always_on=%d (r_est ~= focal*w_guess/size)"
            % (self.cx, self.cy, args.bbox_width, args.bbox_height, args.rate,
               args.lock_delay, self.active))
        if not self.active:
            self.get_logger().info("waiting for /fpv/tracker_trigger ON (arm edge) ...")

    # ---- /fpv/tracker_trigger: ON 启动 / OFF 停止 / DISMISS 忽略 ----
    def trigger_cb(self, msg):
        frame = msg.frame_id
        if frame == "ON" and not self.active:
            self.active = True
            self.activated_at = time.monotonic()
            self.state_sent = None
            self.publish_state(STATE_FOUND, "target found")
            self.state_sent = STATE_FOUND
            self.get_logger().info("ON received -> publishing fake detections")
        elif frame == "OFF" and self.active:
            self.active = False
            self.publish_state(STATE_LOST, "tracker off")
            self.get_logger().info("OFF received -> stopped")
        elif frame in ("MISSION", "DISMISS"):
            self.get_logger().info("%s frame ignored (tracker keeps running)" % frame)

    def publish_state(self, value, label):
        self.state_pub.publish(Int8(data=value))
        self.get_logger().info("tracker_state -> %d (%s)" % (value, label))

    # ---- /fpv/gimbal: 发布节拍 + 时间戳来源 ----
    def gimbal_cb(self, msg):
        self.gimbal_count += 1
        self.last_gimbal_mono = time.monotonic()
        if not self.active:
            return
        now = time.monotonic()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0 or stamp <= self.last_pub_stamp:
            return
        if now - self.last_pub_mono < self.min_period:
            return

        # --- Detection2D: stamp 逐字复制 gimbal 帧 ---
        det = Detection2D()
        det.header.stamp = msg.header.stamp
        det.header.frame_id = "camera_frame"
        det.bbox = BoundingBox2D()
        det.bbox.center.position.x = float(self.cx)
        det.bbox.center.position.y = float(self.cy)
        det.bbox.size_x = float(self.args.bbox_width)
        det.bbox.size_y = float(self.args.bbox_height)
        self.detection_pub.publish(det)

        # --- 云台瞄准 LOS: 画面中心 (与 bbox 偏移一致时才非零) ---
        los = TwistStamped()
        los.header.stamp = msg.header.stamp
        if self.args.offset_x == 0.0 and self.args.offset_y == 0.0:
            los.header.frame_id = "camera_flu"
        else:
            # 简针孔换算; 云台瞄准仅在有偏移时使用, 默认 0 保持画面中心
            los.header.frame_id = "camera_flu"
            los.twist.linear.z = math.atan2(self.args.offset_x, self.args.focal)
            los.twist.linear.y = math.atan2(-self.args.offset_y, self.args.focal)
        self.los_pub.publish(los)

        self.last_pub_mono = now
        self.last_pub_stamp = stamp
        self.pub_count += 1

        # --- tracker_state: FOUND -> TRACKING (锁存一次 + 心跳) ---
        if self.state_sent != STATE_TRACKING and self.activated_at is not None and \
                now - self.activated_at >= self.args.lock_delay:
            self.publish_state(STATE_TRACKING, "tracking")
            self.state_sent = STATE_TRACKING
            self.next_heartbeat = now + 5.0
        elif self.state_sent == STATE_TRACKING and now >= self.next_heartbeat:
            self.state_pub.publish(Int8(data=STATE_TRACKING))
            self.next_heartbeat = now + 5.0

    def report(self):
        now = time.monotonic()
        if now < self.next_report:
            return
        self.next_report = now + 5.0
        gimbal_age = (now - self.last_gimbal_mono) if self.last_gimbal_mono else None
        self.get_logger().info(
            "active=%d published=%d gimbal_msgs=%d gimbal_age=%s"
            % (self.active, self.pub_count, self.gimbal_count,
               ("%.2fs" % gimbal_age) if gimbal_age is not None else "never"))
        if gimbal_age is not None and gimbal_age > 3.0:
            self.get_logger().warn(
                "/fpv/gimbal stale (>3s): bridge or gimbal port down — detections stalled")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rate", type=float, default=30.0,
                        help="detection publish rate (Hz, 挂靠 gimbal 帧)")
    parser.add_argument("--bbox-width", type=float, default=80.0,
                        help="bbox 宽 (px, 单目测距用; 80px@fx1057,w1.0 -> ~13.2m)")
    parser.add_argument("--bbox-height", type=float, default=80.0, help="bbox 高 (px)")
    parser.add_argument("--offset-x", type=float, default=0.0,
                        help="目标相对图像中心的水平偏移 (px, +右)")
    parser.add_argument("--offset-y", type=float, default=0.0,
                        help="目标相对图像中心的垂直偏移 (px, +下)")
    parser.add_argument("--image-width", type=int, default=1536, help="相机图像宽 (px)")
    parser.add_argument("--image-height", type=int, default=960, help="相机图像高 (px)")
    parser.add_argument("--focal", type=float, default=1057.3,
                        help="焦距 (px, 仅用于偏移时的 LOS 角换算)")
    parser.add_argument("--lock-delay", type=float, default=0.5,
                        help="ON 到 tracker_state=2(Tracking) 的延迟 (s)")
    parser.add_argument("--always-on", action="store_true",
                        help="启动即发布 (不等 arm 边沿 ON; 预检查用)")
    args = parser.parse_args()

    rclpy.init()
    node = FakeBboxPublisher(args)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.report()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
