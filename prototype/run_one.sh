#!/usr/bin/env bash
# Verify one interception scenario in the ROS2 sim against the built IMM-UKF node.
# Launches the node, runs the sim, tears the node down, prints the summary.
# Usage: ./run_one.sh [nominal|crossing|maneuvering|dropout|noise|wrong_w]
# NOTE: set -u is deferred until AFTER sourcing ROS setup.bash — jazzy's setup.bash references
# unset vars and aborts the whole script under `set -u` before the sim ever runs.
SCEN="${1:-nominal}"
WS="/media/nick/data/workspace/fpv_ws"
PKG="$WS/src/drone_interceptor"
source /opt/ros/jazzy/setup.bash >/dev/null 2>&1
source "$WS/install/setup.bash" >/dev/null 2>&1
set -u

# Detach the node into its own session/process-group so teardown can't signal this shell.
# use_simulation:=true disables body-rate EMA attenuation (the Python harness has no inner
# attitude loop; AirSim/real HW need it). See the use_simulation flag convention.
setsid ros2 launch drone_interceptor launch_relative_imm_ukf.py use_simulation:=true >/tmp/node_${SCEN}.log 2>&1 &
NODEPG=$!
trap 'kill -TERM -'"$NODEPG"' 2>/dev/null' EXIT

for _ in $(seq 1 60); do
  ros2 topic list 2>/dev/null | grep -q '^/fpv/control_command$' && break
  sleep 0.25
done
sleep 1.0   # let subscriptions/qos settle

python3 "$PKG/prototype/ros2_sim_node.py" --scenario "$SCEN"
echo "===== SUMMARY ($SCEN) ====="
cat "$PKG/prototype/ros2_sim_${SCEN}_summary.txt" 2>/dev/null || echo "(no summary)"
