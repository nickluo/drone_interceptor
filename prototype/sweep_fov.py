#!/usr/bin/env python3
"""Sweep: each of the 6 ros2_sim scenarios x 10 initial FOV positions (topleft, top, topright,
left, midleft, midright, right, bottomleft, bottom, bottomright). Launches a FRESH
interceptor_node_relative_imm_ukf per run (the node does not re-arm after stop), runs the sim
with --fov_pos --sim_time, parses the summary, and writes a results table.

Usage: python3 prototype/sweep_fov.py
"""
import os, re, signal, subprocess, sys, time

PKG = "/media/nick/data/workspace/fpv_ws/src/drone_interceptor"
EXE = "/media/nick/data/workspace/fpv_ws/install/drone_interceptor/lib/drone_interceptor/interceptor_node_relative_imm_ukf"
PARAMS = "/media/nick/data/workspace/fpv_ws/install/drone_interceptor/share/drone_interceptor/config/relative_imm_ukf_params.yaml"
SETUP = "source /opt/ros/jazzy/setup.bash && source /media/nick/data/workspace/fpv_ws/install/setup.bash"
SIM_TIME = "12"

SCENARIOS = ["nominal", "crossing", "maneuvering", "dropout", "noise", "wrong_w"]
FOV = ["topleft", "top", "topright", "left", "midleft", "midright",
       "right", "bottomleft", "bottom", "bottomright"]

NODE_LOG = "/tmp/sweep_node.log"
SIM_LOG = "/tmp/sweep_sim.log"


def kill_stray():
    subprocess.run("pkill -9 -f '/interceptor_node_relative_imm_ukf --ros-args' || true",
                   shell=True)
    time.sleep(2.0)


def wait_ready(timeout=25):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if "node ready" in open(NODE_LOG).read():
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def parse_summary(name):
    sf = os.path.join(PKG, "prototype", f"ros2_sim_{name}_summary.txt")
    try:
        txt = open(sf).read()
    except OSError:
        return None
    def g(pat):
        m = re.search(pat, txt)
        return m.group(1) if m else "?"
    return {
        "intercepted": g(r"Intercepted:\s*(\w+)"),
        "intercept_t": g(r"Intercept time:\s*([\d.]+)s"),
        "min_range": g(r"Min range:\s*([\d.]+)m"),
        "nan": g(r"NaN commands:\s*(\d+)"),
    }


def run_one(scen, fov):
    name = f"{scen}-{fov}"
    kill_stray()
    # launch node (own process group, line-buffered). log-level INFO so "node ready" is emitted.
    node = subprocess.Popen(
        f"bash -c \"{SETUP} && stdbuf -oL -eL {EXE} --ros-args --params-file {PARAMS} "
        f"-p use_simulation:=true --log-level INFO > {NODE_LOG} 2>&1\"",
        shell=True, start_new_session=True)
    try:
        if not wait_ready():
            return {"intercepted": "NODE_NOT_READY", "intercept_t": "-", "min_range": "-", "nan": "-"}
        subprocess.run(
            f"bash -c \"cd {PKG} && {SETUP} && python3 prototype/ros2_sim_node.py "
            f"--scenario {scen} --fov_pos {fov} --sim_time {SIM_TIME} > {SIM_LOG} 2>&1\"",
            shell=True, check=False)
    finally:
        try:
            os.killpg(os.getpgid(node.pid), signal.SIGINT)
            time.sleep(1.0)
            os.killpg(os.getpgid(node.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    return parse_summary(name) or {"intercepted": "NO_SUMMARY", "intercept_t": "-", "min_range": "-", "nan": "-"}


def main():
    results = {}
    total = len(SCENARIOS) * len(FOV)
    i = 0
    t_start = time.time()
    for scen in SCENARIOS:
        for fov in FOV:
            i += 1
            r = run_one(scen, fov)
            results[(scen, fov)] = r
            print(f"[{i}/{total}] {scen:11s} {fov:11s} -> int={r['intercepted']:5s} "
                  f"t={r['intercept_t']:>5} min={r['min_range']:>5} nan={r['nan']}  "
                  f"({time.time()-t_start:.0f}s)", flush=True)
    kill_stray()
    # write table
    out = os.path.join(PKG, "prototype", "sweep_fov_results.txt")
    with open(out, "w") as f:
        f.write("FOV-position sweep (intercept Y/N, min_range m, intercept_t s, NaN)\n\n")
        hdr = f"{'scenario':<13}" + "".join(f"{fovv:<17}" for fovv in FOV)
        f.write(hdr + "\n")
        for scen in SCENARIOS:
            cells = []
            n_ok = 0
            for fov in FOV:
                r = results[(scen, fov)]
                ok = r["intercepted"] == "True"
                n_ok += ok
                cells.append(f"{'Y' if ok else 'N'}/{r['min_range']:>4}/{r['intercept_t']:>4}")
            f.write(f"{scen:<13}" + "".join(f"{c:<17}" for c in cells) +
                    f"  ({n_ok}/{len(FOV)})\n")
    print(f"\nResults written to {out}")
    # also print pass count
    total_ok = sum(1 for r in results.values() if r["intercepted"] == "True")
    print(f"TOTAL intercepts: {total_ok}/{total}")


if __name__ == "__main__":
    main()
