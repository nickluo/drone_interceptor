#!/usr/bin/env bash
# Run one relative_imm_ukf scenario end-to-end: launch the C++ node, drive it with the ROS2 sim
# harness, then tear both down. Logs land in /tmp/imm_ukf_<scenario>.node.log.
#
#   ./run_scenario.sh radar_relay --radar
#   ./run_scenario.sh nominal

SCEN="${1:?usage: run_scenario.sh <scenario> [extra sim args...]}"
shift || true

WS=/home/nick/workspace/fpv_ws
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"

# Radar is enabled through the radar block in config/relative_imm_ukf_params.yaml; the launch file
# no longer exposes an override. Non-radar scenarios simply never publish a track.
NODE_LOG="${TMPDIR:-/tmp}/imm_ukf_${SCEN}.node.log"
SIM_LOG="${TMPDIR:-/tmp}/imm_ukf_${SCEN}.sim.log"
rm -f "$NODE_LOG" "$SIM_LOG"

ros2 launch drone_interceptor launch_relative_imm_ukf.py \
    use_simulation:=true >"$NODE_LOG" 2>&1 &
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
