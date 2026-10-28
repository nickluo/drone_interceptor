#!/usr/bin/env bash
# Run one relative_imm_ukf scenario end-to-end: launch the C++ node, drive it with the ROS2 sim
# harness, then tear both down. Logs land in /tmp/imm_ukf_<scenario>.node.log.
#
#   ./run_scenario.sh nominal
#   ./run_scenario.sh radar_reintercept radar_reintercept_enable:=true   # override a launch arg

SCEN="${1:?usage: run_scenario.sh <scenario> [node_param:=val ...] [extra sim args...]}"
shift || true
# Leading key:=val args are forwarded to `ros2 launch` as node-param overrides (the launch args
# declared in launch_relative_imm_ukf.py, e.g. radar_reintercept_enable). The first arg without '='
# starts the sim args (everything after is passed to ros2_sim_node.py).
LPARAMS=""
while [ $# -gt 0 ]; do case "$1" in *=*) LPARAMS="$LPARAMS $1"; shift;; *) break;; esac; done

WS=/home/nick/workspace/fpv_ws
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"

# Radar scenarios (name starting with "radar"): the launch file OVERRIDES radar.enable with the
# use_radar launch arg (default False), so the node needs use_radar:=true even though
# config/relative_imm_ukf_params.yaml sets radar.enable: true. Non-radar scenarios never publish
# a track, so the node stays pure-vision for them (exercises the no-radar path).
case "$SCEN" in
    radar*) LPARAMS="$LPARAMS use_radar:=true" ;;
    # High-pitch scenarios of the quaternion attitude layer: raise the thrust tilt cap past the
    # flight 65/68 deg envelope to the missile-mode/bench 95 deg (the yaml keeps the flight
    # profile; the real rig/bench brings its own profile). The bench (rig) scenarios additionally
    # FREEZE the monocular width at the w_guess initial value: on the translation-locked bench
    # there is no ownship parallax, so the (W, range) scale is unobservable and would drift —
    # with a known-size bench target the frozen W pins r = focal*W/bbox deterministically.
    # No radar in the bench scenarios (keep the bench chain minimal — pure vision).
    rig_*) LPARAMS="$LPARAMS extra_params:=physics.max_tilt_deg=95.0,physics.max_tilt_terminal_deg=95.0,estimation.freeze_width=true,estimation.w_guess=1.5,controller.visual_speed_gain=0.0,controller.a_lat_tilt_start_deg=50.0" ;;
    dive) LPARAMS="$LPARAMS extra_params:=physics.max_tilt_deg=95.0,physics.max_tilt_terminal_deg=95.0" ;;
esac
NODE_LOG="${TMPDIR:-/tmp}/imm_ukf_${SCEN}.node.log"
SIM_LOG="${TMPDIR:-/tmp}/imm_ukf_${SCEN}.sim.log"
rm -f "$NODE_LOG" "$SIM_LOG"

ros2 launch drone_interceptor launch_relative_imm_ukf.py \
    use_simulation:=true $LPARAMS >"$NODE_LOG" 2>&1 &
LAUNCH_PID=$!
sleep 4

python3 "$WS/src/drone_interceptor/prototype/ros2_sim_node.py" --scenario "$SCEN" "$@" \
    >"$SIM_LOG" 2>&1
SIM_RC=$?

kill -INT $LAUNCH_PID 2>/dev/null
for _ in $(seq 1 20); do kill -0 $LAUNCH_PID 2>/dev/null || break; sleep 0.25; done
kill -KILL $LAUNCH_PID 2>/dev/null
pkill -f interceptor_node_relative_imm_ukf 2>/dev/null
wait $LAUNCH_PID 2>/dev/null

echo "=== sim rc=$SIM_RC | logs: $NODE_LOG $SIM_LOG ==="
cat "$WS/src/drone_interceptor/prototype/ros2_sim_${SCEN}_summary.txt" 2>/dev/null
