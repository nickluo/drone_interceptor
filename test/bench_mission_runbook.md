# 真机无桨静态 MISSION 台架测试 runbook

目标:办公室静态(无桨叶)验证 MISSION 发射链路——RC 触发 → 假 bbox 目标 → 拦截器
MISSION → OFFBOARD → 电机响应,记录**加速度需求**(/fpv/control_command)与
**真实油门输出**(推力模型镜像 + /fpv/motor_rpm)。

配套脚本(本目录,直接 python3 运行,无需构建):
- `fake_bbox_publisher.py` — 替代 airsim_tracker:FOV 中心假目标(bbox+tracker_state+LOS)
- `bench_mission_recorder.py` — 全数据面记录 + 发射画像报告
- `raw_keepalive.py` — **未标定 demand-only 必开**:转发控制命令到 control_command_raw
  (体速率直通、油门固定 0=怠速), 维持 0x20 流新鲜;否则未标定下 OFFBOARD 授予后
  50ms 看门狗掉线,桥的"任务中 OFFBOARD 丢失"保护(auto-DISMISS, vehicle_node 932 行)
  会在 ~0.15s 杀掉任务(2026-11-09 实测);OFFBOARD 若恰好未授予则任务存活——纯抽签。
  **标定后停用**(正常力指令通道恢复,raw 转发会顶掉推力模型输出)
- `imu_bias_calibration.py` — 静置 60s 标定 imu_bias(已标定 2026-11-09:
  [0.0026, -0.0014, 0.0675],own_vz 漂移 0.33→0.02 m/s²)

## 0. 构建(必须——2026-11-09 源码晚于 install,含 rcTriggerUpdate 修复)

```bash
cd ~/workspace/fpv_ws
colcon build --packages-select apm_bridge drone_interceptor
```

(YAML 参数随包安装;改 `custom_link_parameters.yaml` / `relative_imm_ukf_params.yaml` 后同样要重建)

## 1. 环境(每个终端)

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspace/fpv_ws/install/setup.bash
export ROS_DOMAIN_ID=77          # 避开 domain 0 幽灵发布者;所有终端一致
```

预检:
```bash
pgrep -af "custom_link|interceptor|airsim|tracker"   # 清理残留进程
ls -la /dev/ttyUSB0 /dev/ttyUSB1                     # FC 链路 / C20S 云台
```

## 2. 启动桥(终端 A)

```bash
ros2 launch apm_bridge custom_link.launch.py \
    serial_port:=/dev/ttyUSB0 gimbal_port:=/dev/ttyUSB1 bind_rx:=false
```

等待:`Custom link up`(若无输出先查接线/上电,同 2026-10-21 教训)、云台
`HOMING → AIM`、rssi 恢复(接收机上电稍等,首读 0 是老现象)。

## 3. 启动拦截器(终端 B,正常配置)

```bash
ros2 launch drone_interceptor launch_relative_imm_ukf.py
```

默认 `use_simulation:=False` + 打包的 `relative_imm_ukf_params.yaml`(1536×960/FOV72°,
use_gimbal=true,w_guess=1.0,v_des=20,地面发射 boost 8 m/s²)。

## 4. 启动假 bbox 发布器(终端 C)

```bash
python3 ~/workspace/fpv_ws/src/drone_interceptor/test/fake_bbox_publisher.py
```

- 默认等 arm 边沿 ON 再发布;预检可加 `--always-on`
- 假目标几何:bbox 80 px @ fx≈1057、w_guess 1.0 → r_est≈13.2 m,画面中心(=云台指向);
  调 `--bbox-width`(px)改变估计距离
- 若曾在 fake 启动前已解锁:重新上锁→解锁产生新 ON 边沿,或用 `--always-on`

## 5. 启动记录器(终端 D)+ 未标定 keepalive(终端 E,demand-only 必须)

```bash
python3 ~/workspace/fpv_ws/src/drone_interceptor/test/bench_mission_recorder.py
# 另一终端 (motor_parameters 已标定则跳过):
python3 ~/workspace/fpv_ws/src/drone_interceptor/test/raw_keepalive.py
```

- 输出目录默认 `/tmp/bench_mission_<时间戳>/{control,rpm,slow}.csv + events.log`
- **motor_parameters 未标定时**启动即告警:油门镜像记 NaN,桥会整帧丢弃力指令
  (电机停在怠速带)——此时仅加速度需求有效;标定后重跑即得完整油门链
- 可选并行 rosbag:
  `ros2 bag record -o /tmp/bench_bag /fpv/control_command /fpv/motor_rpm /mavros/state /mavros/battery /mavros/rc/in /fpv/mission_state /fpv/tracker_trigger /fpv/gimbal /fpv/gimbal_mode /mavros/extended_state /fpv/failsafe`

## 6. 解锁前预检

- [ ] `ros2 topic echo /mavros/rc/in --once` 翻转遥控器 Trigger 开关:
      **channels[9](CH10/AUX6,channel_trigger=9)>1900** 才能触发(保持 >0.5s 判定)
- [ ] 记录器 STATUS:armed=0、rpm 全 0、mode=拨杆模式
- [ ] fake 日志:gimbal_msgs 持续增长(云台链路活)
- [ ] 桥日志无 ERROR

## 7. 执行序列(用户操作 RC)

| 步骤 | RC 操作 | 预期 |
|---|---|---|
| 1 | ARM 开关解锁(AUX1 高) | 桥 `Props spinning (armed): tracker trigger ON`;fake 开始发布,~0.5s 后 tracker_state=2;四电机 motor_idle 慢转(~1800rpm,无桨正常) |
| 2 | (确认) | 记录器 STATUS offb=0;`ros2 topic get` 不需要——fake 日志 published 计数增长 |
| 3 | **Trigger 脉冲**(CH10 拨高保持 >0.5s,回中) | 桥 `Tracker Triggered : MISSION` → OFFBOARD 接管 → 拦截器 MISSION:100Hz 集体推力需求(首指令预计 ~5g/满油门量级),云台→MISSION 模式 |
| 4 | 数据足够(3-5s)后**再打一次脉冲** | `DISMISS` → MISSION 结束、OFFBOARD 退出(电机回怠速) |
| 5 | ARM 关闭上锁 | 电机停转;桥 `Props stopped (disarmed): tracker trigger OFF`;fake 停止发布 |
| 6 | 终端 D Ctrl-C | 打印发射画像报告,CSV 落盘 |

静态台架特性(预期,非异常):
- 无推进 → 估计器 a_own≈0 自洽,需求**保持饱和**(AirSim 中 5g→1g 的下降靠机体加速,
  这里不会发生)——首指令≈持续值,直到 DISMISS
- 地面发射 boost ON_GROUND 检测生效:前 3s a_z≥8 m/s² 保底
- 未标定时:OFFBOARD 仍会接合(0x20 以油门 0 发送),电机停留怠速带

## 8. 安全要点

- **全程确认桨叶已拆**
- OFFBOARD 下指令停发/油门为零 = 回怠速慢转,**不是停转**;只有上锁才停转
- MISSION 持续 >6s 记录器告警提醒 DISMISS(静态满需求,电机满档时间别拖长)
- 电压 <21.0 V(6S×3.5)、failsafe 置位 → 记录器告警
- 异常处理顺序:再打一次 Trigger(DISMISS)→ 上锁 → 必要时直接断电
- ACRO↔OFFBOARD 周期抖动 = 接收链路间歇掉帧(已知硬件问题),不影响数据判读

## 9. 报告产物

- 终端发射画像:加速度需求(首指令/+0.1/+0.5/+1.0s/峰值/首秒均值/≥98% 持续)、
  镜像油门(标定后:首指令/峰值/首秒均值/是否饱和)、电机 rpm(触发前基线→峰值)
- CSV 明细(control.csv 含逐帧 collective_thrust + mirror_throttle)
