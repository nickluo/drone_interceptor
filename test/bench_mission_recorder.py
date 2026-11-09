#!/usr/bin/env python3
"""真机无桨静态 MISSION 台架记录器 —— 记录加速度需求与真实油门输出。

订阅拦截器/桥的完整数据面, 按话题落 CSV, 以 MISSION 触发为 t0 锚点, Ctrl-C 后
在终端打印发射画像报告 (首指令 / 峰值 / 首秒均值 / 饱和时长 / 电机转速响应)。

记录通道:
  /fpv/control_command  quadrotor_msgs/ControlCommand  加速度需求 collective_thrust
                                                      (m/s^2, 100Hz) —— 主记录
  /fpv/motor_rpm        Float32MultiArray              0x16 ESC 遥测四电机转速 (真实输出)
  /mavros/state         mavros_msgs/State              armed / mode / guided(=OFFBOARD)
  /mavros/battery       sensor_msgs/BatteryState       电压 (V) / 电流 (A)
  /mavros/rc/in         mavros_msgs/RCIn               RC 镜像 (含触发通道 CH10)
  /fpv/mission_state /fpv/tracker_trigger /fpv/gimbal_mode /mavros/extended_state
  /fpv/failsafe /fpv/gimbal /fpv/static_pressure /fpv/temperature_baro

真实油门 (桥发给 FC 的集体油门 0..1) 有两个口径:
  1) 推力模型镜像 (需要 motor_parameters 已标定): collective = F_req->占空比->集体油门,
     公式与节点 pushControl 完全一致 (scripts/propulsion_model.py), u_idle 经 /fpv/cli
     读 FC motor_idle。未标定 (a3..a1 全零) 时镜像禁用, 记 NaN 并告警 —— 此时桥会
     整帧丢弃力指令 (电机停在怠速带), 油门输出无意义, 只有加速度需求有效。
  2) /fpv/motor_rpm: 固件 OFFBOARD 混控后的真实电机域输出 (任何口径下都有效)。

用法 (先 source ROS + 工作区):
  python3 bench_mission_recorder.py                       # 默认输出 /tmp/bench_mission_<时间>
  python3 bench_mission_recorder.py --out /tmp/myrun --mission-warn-s 6
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "apm_bridge", "scripts"))

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from mavros_msgs.msg import ExtendedState, RCIn, State
from quadrotor_msgs.msg import ControlCommand
from sensor_msgs.msg import BatteryState, FluidPressure, Imu, Temperature
from std_msgs.msg import Bool, Float32MultiArray, Header, String

CTRL_MODES = {0: "NONE", 1: "ATTITUDE", 2: "BODY_RATES", 3: "ANG_ACC", 4: "ROTOR"}
IDLE_RPM_WARN_FLOOR = 100.0   # 低于此值视为停转而不是怠速


class ThrottleMirror:
    """节点 QuadraticThrustModel 反解的 Python 镜像 (未标定自动禁用)。"""

    def __init__(self, yaml_path, lipo_cells, voltage_comp, baro_comp):
        self.enabled = False
        self.reason = ""
        self.voltage_comp = voltage_comp
        self.baro_comp = baro_comp
        self.model = None
        self.u_idle = None
        try:
            import yaml
            import propulsion_model as pm
            with open(yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            node_cfg = cfg.get("custom_link_bridge", {}).get("ros__parameters", cfg)
            mp = node_cfg.get("motor_parameters", {})
            a3, a2, a1 = (float(mp.get(k, 0.0)) for k in ("a3", "a2", "a1"))
            if a3 == 0.0 and a2 == 0.0 and a1 == 0.0:
                self.reason = ("motor_parameters a3..a1 全零 (未标定): 桥正在整帧丢弃力指令, "
                               "油门被钉在 0/怠速 —— 只记录加速度需求, 油门列记 NaN")
                return
            self.pm = pm
            self.model = pm.PropulsionModel.from_parameters(mp, lipo_cells, pm.DEFAULT_U_IDLE)
            self.enabled = True
        except Exception as exc:  # noqa: BLE001 - 镜像是可选增强
            self.reason = f"推力模型镜像加载失败 ({exc}); 油门列记 NaN"

    def set_u_idle(self, u_idle):
        if self.enabled and u_idle is not None and math.isfinite(u_idle):
            self.u_idle = u_idle
            self.model.set_u_idle(u_idle)

    def throttle(self, collective_ms2, mass, env_factor, vbat, current):
        """节点 pushControl 同式: F_req = collective*mass -> duty -> 集体油门 0..1。"""
        if not self.enabled:
            return float("nan")
        env = env_factor if self.baro_comp else 1.0
        v, i = (vbat, current) if self.voltage_comp else (float("nan"),) * 2
        try:
            return self.model.collective(collective_ms2 * mass, env, v, i)
        except Exception:  # noqa: BLE001
            return float("nan")


class BenchMissionRecorder(Node):
    def __init__(self, args):
        super().__init__("bench_mission_recorder")
        self.args = args
        os.makedirs(args.out, exist_ok=True)
        self.t_start = time.time()
        self.t0_mission = None            # MISSION 锚点 (触发帧或 mission_state 边沿)
        self.ctrl_rows = deque()
        self.rpm_rows = deque()
        self.slow_rows = deque()
        self.events = []

        # 最新状态
        self.state = None
        self.battery = None
        self.rc = None
        self.ext_state = None
        self.failsafe = False
        self.pressure = None
        self.temp = None
        self.gimbal_pitch_deg = float("nan")
        self.last_ctrl = None
        self.last_rpm = ()
        self.mission_active = False
        self.gimbal_mode = None
        self.trigger_ch_values = deque(maxlen=200)

        qos_be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_tl = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                            reliability=ReliabilityPolicy.RELIABLE)

        self.create_subscription(ControlCommand, "/fpv/control_command", self.cb_ctrl, 10)
        self.create_subscription(Float32MultiArray, "/fpv/motor_rpm", self.cb_rpm, 10)
        self.create_subscription(State, "/mavros/state", self.cb_state, 10)
        self.create_subscription(BatteryState, "/mavros/battery", self.cb_battery, 10)
        self.create_subscription(RCIn, "/mavros/rc/in", self.cb_rc, qos_be)
        self.create_subscription(ExtendedState, "/mavros/extended_state", self.cb_ext, 10)
        self.create_subscription(Bool, "/fpv/mission_state", self.cb_mission_state, 10)
        self.create_subscription(Header, "/fpv/tracker_trigger", self.cb_trigger, 10)
        self.create_subscription(String, "/fpv/gimbal_mode", self.cb_gimbal_mode, qos_tl)
        self.create_subscription(Bool, "/fpv/failsafe", self.cb_failsafe, 10)
        self.create_subscription(Imu, "/fpv/gimbal", self.cb_gimbal, qos_be)
        self.create_subscription(FluidPressure, "/fpv/static_pressure",
                                 lambda m: setattr(self, "pressure", m.fluid_pressure), qos_be)
        self.create_subscription(Temperature, "/fpv/temperature_baro",
                                 lambda m: setattr(self, "temp", m.temperature), qos_be)

        self.mirror = ThrottleMirror(args.bridge_yaml, args.lipo_cells,
                                     args.voltage_comp, args.baro_comp)
        self.u_idle_requested = False

        self.log("recorder started, out=%s", args.out)
        if self.mirror.enabled:
            self.log("throttle mirror ENABLED (calibrated); waiting for /fpv/cli motor_idle")
        else:
            self.log("throttle mirror DISABLED: %s", self.mirror.reason)
        self.next_status = time.monotonic() + 1.0

    # ---------------- helpers ----------------
    def now_rel(self):
        return time.time() - self.t_start

    def mission_rel(self):
        return None if self.t0_mission is None else time.time() - self.t0_mission

    def log(self, fmt, *a):
        text = fmt % a if a else fmt
        self.events.append((self.now_rel(), "[%8.2fs] %s" % (self.now_rel(), text)))
        self.get_logger().info(text)

    def warn(self, fmt, *a):
        text = fmt % a if a else fmt
        self.events.append((self.now_rel(), "[%8.2fs] [WARN] %s" % (self.now_rel(), text)))
        self.get_logger().warn(text)

    def env_factor(self):
        if not self.args.baro_comp or self.pressure is None or self.temp is None:
            return 1.0
        try:
            return self.mirror.pm.environment_factor(self.pressure, self.temp)
        except Exception:  # noqa: BLE001
            return float("nan")

    # ---------------- callbacks ----------------
    def cb_ctrl(self, msg):
        t_rel = self.now_rel()
        vbat = self.battery.voltage if self.battery is not None else float("nan")
        amps = self.battery.current if self.battery is not None else float("nan")
        thr = self.mirror.throttle(msg.collective_thrust, self.args.mass,
                                   self.env_factor(), vbat, amps)
        self.last_ctrl = msg
        self.ctrl_rows.append((t_rel, self.mission_rel(),
                               msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                               msg.collective_thrust, thr,
                               msg.bodyrates.x, msg.bodyrates.y, msg.bodyrates.z,
                               CTRL_MODES.get(msg.control_mode, str(msg.control_mode)),
                               int(msg.armed), int(msg.is_hover_state),
                               vbat, amps))

    def cb_rpm(self, msg):
        self.last_rpm = tuple(msg.data)
        self.rpm_rows.append((self.now_rel(), self.mission_rel()) + tuple(msg.data))

    def cb_state(self, msg):
        prev = self.state
        self.state = msg
        if prev is None or (prev.armed, prev.mode, prev.guided) != \
                (msg.armed, msg.mode, msg.guided):
            self.log("FC state: mode=%s armed=%d offboard(guided)=%d",
                     msg.mode, msg.armed, msg.guided)

    def cb_battery(self, msg):
        self.battery = msg

    def cb_rc(self, msg):
        self.rc = msg
        if len(msg.channels) > self.args.trigger_channel:
            self.trigger_ch_values.append(
                (self.now_rel(), msg.channels[self.args.trigger_channel]))

    def cb_ext(self, msg):
        prev = self.ext_state.landed_state if self.ext_state is not None else None
        self.ext_state = msg
        if prev != msg.landed_state:
            self.log("landed_state -> %d", msg.landed_state)

    def cb_mission_state(self, msg):
        if msg.data and not self.mission_active:
            if self.t0_mission is None:
                self.t0_mission = time.time()
            self.log("MISSION_STATE -> True (t0 anchored)")
        elif not msg.data and self.mission_active:
            dur = time.time() - self.t0_mission if self.t0_mission else float("nan")
            self.log("MISSION_STATE -> False (duration %.2fs)", dur)
        self.mission_active = msg.data

    def cb_trigger(self, msg):
        self.log("tracker_trigger frame: %s", msg.frame_id or "<empty>")
        if msg.frame_id == "MISSION" and self.t0_mission is None:
            self.t0_mission = time.time()
            self.log("MISSION frame -> t0 anchored")

    def cb_gimbal_mode(self, msg):
        if msg.data != self.gimbal_mode:
            self.gimbal_mode = msg.data
            self.log("gimbal_mode -> %s", msg.data)

    def cb_failsafe(self, msg):
        if msg.data and not self.failsafe:
            self.warn("FC FAILSAFE flag set!")
        self.failsafe = msg.data

    def cb_gimbal(self, msg):
        q = msg.orientation
        # 四元数 -> 俯仰角 (仅上下文显示)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        self.gimbal_pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))

    # ---------------- 周期任务 ----------------
    def fetch_u_idle_once(self):
        """链路建立后经 /fpv/cli 读一次 FC motor_idle (镜像 duty->集体油门换算需要)。"""
        if not self.mirror.enabled or self.u_idle_requested or self.state is None:
            return
        self.u_idle_requested = True
        try:
            # spin=True: 本记录器主循环此刻阻塞在 periodic 内, 必须由 fetch 内部
            # spin_once 驱动服务 future, 否则必然超时
            u_idle = self.mirror.pm.fetch_u_idle(self, timeout=3.0, spin=True)
            self.mirror.set_u_idle(u_idle)
            self.log("FC motor_idle -> u_idle=%.4f (throttle mirror live)", u_idle)
        except Exception as exc:  # noqa: BLE001
            self.warn("motor_idle fetch failed (%s); mirror uses default %.3f",
                      exc, self.mirror.model.u_idle)

    def periodic(self):
        now = time.monotonic()
        if now < self.next_status:
            return
        self.next_status = now + 1.0
        self.fetch_u_idle_once()
        vbat = self.battery.voltage if self.battery is not None else float("nan")
        self.slow_rows.append((self.now_rel(), vbat,
                               self.battery.current if self.battery is not None else float("nan"),
                               self.state.mode if self.state else "?",
                               int(self.state.armed) if self.state else 0,
                               int(self.state.guided) if self.state else 0,
                               self.ext_state.landed_state if self.ext_state is not None else -1,
                               int(self.failsafe), self.gimbal_pitch_deg,
                               self.gimbal_mode or "?"))
        mode = self.state.mode if self.state else "?"
        armed = self.state.armed if self.state else 0
        offb = self.state.guided if self.state else 0
        rpm = [r for r in self.last_rpm if math.isfinite(r)]
        rpm_txt = ("min%.0f/max%.0f" % (min(rpm), max(rpm))) if rpm else "n/a"
        m_rel = ("%.2fs" % self.mission_rel()) if self.t0_mission else "-"
        thr = self.ctrl_rows[-1][4] if self.ctrl_rows else float("nan")
        self.get_logger().info(
            "STATUS mode=%s armed=%d offb=%d mission=%s vbat=%.2fV rpm[%s] "
            "last_collective=%.2f m/s^2 mirror_throttle=%s gimbal_pitch=%.1f"
            % (mode, armed, offb, m_rel, vbat, rpm_txt,
               self.ctrl_rows[-1][3] if self.ctrl_rows else float("nan"),
               ("%.3f" % thr) if math.isfinite(thr) else "n/a", self.gimbal_pitch_deg))

        # ---- 安全提示 ----
        if vbat == vbat and vbat < self.args.lipo_cells * 3.5:
            self.warn("LOW BATTERY %.2fV", vbat)
        if self.mission_active and self.t0_mission is not None:
            dur = time.time() - self.t0_mission
            if dur > self.args.mission_warn_s:
                self.warn("mission running %.1fs (static bench: demand stays saturated — "
                          "DISMISS (trigger pulse) once data is captured)", dur)
        if armed and rpm and max(rpm) > 0 and min(rpm) < IDLE_RPM_WARN_FLOOR <= max(rpm):
            self.warn("motor rpm split while armed (min %.0f vs max %.0f) — check props off",
                      min(rpm), max(rpm))

    # ---------------- 落盘与报告 ----------------
    def save(self):
        def w(path, header, rows):
            with open(path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(header)
                wr.writerows(rows)
        w(os.path.join(self.args.out, "control.csv"),
          ["t_rel_s", "t_mission_s", "cmd_stamp_s", "collective_ms2", "mirror_throttle",
           "wx_rad_s", "wy_rad_s", "wz_rad_s", "mode", "armed", "hover", "vbat_v", "current_a"],
          self.ctrl_rows)
        w(os.path.join(self.args.out, "rpm.csv"),
          ["t_rel_s", "t_mission_s", "m0_rpm", "m1_rpm", "m2_rpm", "m3_rpm"], self.rpm_rows)
        w(os.path.join(self.args.out, "slow.csv"),
          ["t_rel_s", "vbat_v", "current_a", "mode", "armed", "offboard", "landed",
           "failsafe", "gimbal_pitch_deg", "gimbal_mode"],
          self.slow_rows)
        with open(os.path.join(self.args.out, "events.log"), "w", encoding="utf-8") as f:
            for _, line in self.events:
                f.write(line + "\n")
        self.get_logger().info("saved: %s/{control,rpm,slow}.csv + events.log" % self.args.out)

    def report(self):
        """发射画像报告: 加速度需求 + (标定后) 真实油门 + 电机转速响应。"""
        print("\n" + "=" * 72)
        print("MISSION 发射画像报告" + ("  (t0 = MISSION 触发)" if self.t0_mission
                                        else "  (未观测到 MISSION 触发!)"))
        print("=" * 72)
        rows = [r for r in self.ctrl_rows if r[1] is not None]
        if not rows:
            print("没有以 MISSION 为锚点的控制命令 —— 未触发或命令被丢弃。")
            print("若桥日志出现 'Dropping control command: motor_parameters uncalibrated',")
            print("说明推力模型未标定: 需求已记录, 但油门输出为怠速 (无意义)。")
            return
        first = rows[0]
        window = [r for r in rows if 0.0 <= r[1] <= 1.0]
        coll = [r[3] for r in rows]
        thr = [r[4] for r in rows if math.isfinite(r[4])]
        g = 9.8065

        def g_txt(v):
            return "%.2f m/s^2 (%.2f g)" % (v, v / g)

        print("[加速度需求] /fpv/control_command collective_thrust, %d 帧:" % len(rows))
        print("  首指令 (MISSION+%.3fs): %s" % (first[1], g_txt(first[3])))
        print("  +0.10s: %s | +0.50s: %s | +1.00s: %s" % (
            g_txt(next((r[3] for r in rows if r[1] >= 0.10), float("nan"))),
            g_txt(next((r[3] for r in rows if r[1] >= 0.50), float("nan"))),
            g_txt(next((r[3] for r in rows if r[1] >= 1.00), float("nan")))))
        print("  峰值: %s @ t+%.2fs" % (g_txt(max(coll)),
                                         rows[coll.index(max(coll))][1]))
        if window:
            mean1 = sum(r[3] for r in window) / len(window)
            print("  首秒均值: %s | 首秒峰值: %s" % (g_txt(mean1), g_txt(max(r[3] for r in window))))
        print("  首指令体速率: wx=%.2f wy=%.2f wz=%.2f rad/s mode=%s" %
              (first[5], first[6], first[7], first[8]))
        sat_thr = 0.98 * max(coll)
        sat_dur = sum(1 for v in coll if v >= sat_thr) / 100.0
        print("  ≥98%%峰值 持续: ~%.1fs (静态台架无推进, 需求应保持饱和直到 DISMISS)" % sat_dur)

        if thr:
            thr_window = [r[4] for r in rows if 0.0 <= r[1] <= 1.0 and math.isfinite(r[4])]
            print("\n[真实油门·镜像] (桥 ThrustModel 反解, 发往 FC 的集体油门 0..1):")
            print("  首指令: %.3f | 峰值: %.3f | 首秒均值: %.3f" %
                  (thr[0], max(thr),
                   (sum(thr_window) / len(thr_window)) if thr_window else float("nan")))
            print("  饱和(≥0.98): %s" % ("是" if max(thr) >= 0.98 else "否"))
        else:
            print("\n[真实油门·镜像] 不可用: %s" % self.mirror.reason)

        rpm_rows = [r for r in self.rpm_rows if r[1] is not None]
        if rpm_rows:
            def rms(row):
                vals = [v for v in row[2:6] if math.isfinite(v)]
                return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else float("nan")
            pre = [r for r in self.rpm_rows if r[1] is None][-50:]
            base = sum(rms(r) for r in pre) / max(1, len(pre))
            peak = max(rms(r) for r in rpm_rows)
            at_peak = next(r[1] for r in rpm_rows if rms(r) >= 0.98 * peak)
            print("\n[电机转速] 0x16 ESC 遥测 (RMS of 4):")
            print("  触发前基线: %.0f rpm | 峰值: %.0f rpm (t+%.2fs) | 增益 x%.1f" %
                  (base, peak, at_peak, peak / base if base > 1 else float("nan")))
        else:
            print("\n[电机转速] 无 /fpv/motor_rpm 数据 (未解锁或 ESC 遥测未上线)")
        print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = "/tmp/bench_mission_%s" % time.strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", default=default_out, help="输出目录 (CSV + events.log)")
    parser.add_argument("--bridge-yaml",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "..", "..", "apm_bridge", "parameters",
                                             "custom_link_parameters.yaml"),
                        help="桥参数 YAML (推力模型镜像来源)")
    parser.add_argument("--mass", type=float, default=1.067, help="机体质量 kg (力 = a*mass)")
    parser.add_argument("--lipo-cells", type=int, default=6)
    parser.add_argument("--voltage-comp", action="store_true", default=True,
                        help="油门镜像启用电压补偿 (与 YAML voltage_compensation 一致)")
    parser.add_argument("--no-voltage-comp", dest="voltage_comp", action="store_false")
    parser.add_argument("--baro-comp", action="store_true", default=True)
    parser.add_argument("--no-baro-comp", dest="baro_comp", action="store_false")
    parser.add_argument("--trigger-channel", type=int, default=9,
                        help="RC 触发通道下标 (channel_trigger, 0 基; 9 = CH10/AUX6)")
    parser.add_argument("--mission-warn-s", type=float, default=6.0,
                        help="MISSION 持续超此秒数告警提示 DISMISS")
    args = parser.parse_args()

    rclpy.init()
    node = BenchMissionRecorder(args)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.periodic()
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.report()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
