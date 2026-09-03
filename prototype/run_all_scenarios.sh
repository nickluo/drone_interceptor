#!/usr/bin/env bash
# One-click: run every IMM-UKF sim scenario and print a summary table.
#
# Vision scenarios run on the default ROS_DOMAIN_ID. Radar scenarios run on an isolated domain
# (RADAR_DOMAIN below) so they do not collide with a possibly-running airsim_tracker/radar_launch.py
# (which double-publishes /fpv/radar/* on the default domain). A fresh C++ node is launched per
# scenario (run_scenario.sh launches + drives + tears down each time — the node does not re-arm).
#
#   ./run_all_scenarios.sh                 # vision + radar suites
#   ./run_all_scenarios.sh --vision-only   # skip the radar suite
#
# Requires: `colcon build --packages-select drone_interceptor` already run (run_scenario.sh sources
# install/setup.bash). Exit code: 0 iff every scenario intercepted with 0 NaN commands.

set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

VISION_SCENARIOS=(nominal maneuvering crossing dropout noise wrong_w)
RADAR_SCENARIOS=(radar_relay radar_maneuver radar_falsepos radar_far radar_only radar_close radar_reintercept radar_overbound)
RADAR_DOMAIN=137

VISION_ONLY=0
[ "${1:-}" = "--vision-only" ] && VISION_ONLY=1

PASS=0
FAIL=0
declare -a ROWS

parse_and_record() {
    local sc="$1"
    local f="ros2_sim_${sc}_summary.txt"
    if [ ! -s "$f" ]; then
        ROWS+=("$sc|MISSING_SUMMARY|-|-|-")
        FAIL=$((FAIL + 1))
        return
    fi
    local intercepted nan minr mtime
    intercepted=$(grep -i 'Intercepted:' "$f" | awk '{print $2}')
    minr=$(grep -i 'Min range:' "$f" | awk '{print $3}')
    # Exact label match: other summary lines can legitimately contain the substring "nan"
    # (e.g. "Dither 0.2-0.4Hz frac ...: nan" when the close-range segment is too short to FFT).
    nan=$(grep -i '^NaN commands:' "$f" | awk '{print $3}')
    mtime=$(stat -c '%y' "$f" | cut -d. -f1)
    if [ "$intercepted" = "True" ] && [ "$nan" = "0" ]; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
    ROWS+=("$sc|$intercepted|$minr|$nan|$mtime")
}

run_one() {
    local sc="$1"
    shift || true
    echo ">>> [$sc] $*"
    ./run_scenario.sh "$sc" "$@" >/dev/null 2>&1
    parse_and_record "$sc"
}

echo "== Vision suite (default domain) =="
for sc in "${VISION_SCENARIOS[@]}"; do run_one "$sc"; done

if [ "$VISION_ONLY" -eq 0 ]; then
    echo "== Radar suite (isolated ROS_DOMAIN_ID=$RADAR_DOMAIN) =="
    export ROS_DOMAIN_ID=$RADAR_DOMAIN
    for sc in "${RADAR_SCENARIOS[@]}"; do
        # radar_reintercept only exercises the feature with the launch-arg override enabled.
        if [ "$sc" = "radar_reintercept" ]; then
            run_one "$sc" radar_reintercept_enable:=true
        else
            run_one "$sc"
        fi
    done
    unset ROS_DOMAIN_ID
fi

echo
printf '%-18s %-13s %-10s %-5s %-20s\n' "SCENARIO" "INTERCEPTED" "MIN_RANGE" "NaN" "SUMMARY_MTIME"
printf '%-18s %-13s %-10s %-5s %-20s\n' "--------" "-----------" "---------" "---" "-------------"
for r in "${ROWS[@]}"; do
    IFS='|' read -r a b c d e <<< "$r"
    printf '%-18s %-13s %-10s %-5s %-20s\n' "$a" "$b" "$c" "$d" "$e"
done
echo
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
