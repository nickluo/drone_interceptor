#!/usr/bin/env python3
"""
Visualize ROS2 interception-sim verification results.

Reads the CSV/summary logs written by ros2_sim_node.py (one per scenario) and produces:
  ros2_sim_results.png      2x3 grid: range vs time per scenario (intercept radius + intercept point)
  ros2_sim_trajectories.png 2x3 grid: top-down XY pursuit geometry (UAV vs target)

CSV columns: t,range,los_rate,uav_x,uav_y,uav_z,tgt_x,tgt_y,tgt_z,throttle,wx,wy,wz
Usage: python3 plot_results.py
"""
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INTERCEPT_RANGE = 2.0  # m, success radius used by ros2_sim_node.py
ORDER = ["nominal", "crossing", "maneuvering", "dropout", "noise", "wrong_w"]

# scenario -> (label, target-motion hint) for titles
HINT = {
    "nominal": "stationary target",
    "crossing": "crossing, 3.6 m/s",
    "maneuvering": "sinusoidal weave",
    "dropout": "bbox dropout 6-8s",
    "noise": "heavy meas noise",
    "wrong_w": "wrong size prior",
}


def load(scen):
    csv = os.path.join(HERE, f"ros2_sim_{scen}.csv")
    summ = os.path.join(HERE, f"ros2_sim_{scen}_summary.txt")
    if not os.path.exists(csv):
        return None
    data = np.genfromtxt(csv, delimiter=",", names=True)
    # drop the pre-start logging artifact row(s) where range<=0
    mask = data["range"] > 0
    data = data[mask]
    meta = {"intercepted": False, "intercept_t": None, "min_range": float(np.min(data["range"]))}
    if os.path.exists(summ):
        txt = open(summ).read()
        m = re.search(r"Intercepted:\s*(True|False)", txt)
        if m:
            meta["intercepted"] = (m.group(1) == "True")
        m = re.search(r"Intercept time:\s*([0-9.]+)s", txt)
        if m:
            meta["intercept_t"] = float(m.group(1))
    return data, meta


def range_at(data, t):
    """Range value at the sample closest to time t."""
    i = int(np.argmin(np.abs(data["t"] - t)))
    return float(data["range"][i]), float(data["t"][i])


def window_end(data, meta):
    """Last plot index: stop at the FIRST local minimum of range after the intercept crossing
    (the dive bottom immediately following the successful intercept). This shows the approach
    and the intercept pass, then stops — later re-approaches / loiter / fly-away (which may hold
    the GLOBAL min for scenarios like nominal/dropout) are not drawn. Non-intercepted runs plot
    in full."""
    r = data["range"]
    if not meta["intercepted"]:
        return len(r)
    below = np.where(r <= INTERCEPT_RANGE)[0]
    if len(below) == 0:
        return len(r)
    i = int(below[0])  # first crossing of the intercept radius
    while i + 1 < len(r) and r[i + 1] <= r[i]:
        i += 1  # descend to the bottom of this first pass
    return i + 1


# ---------------------------------------------------------------- figure 1: range vs time
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
fig.suptitle("ROS2 Interception Sim — Range vs Time (approach through intercept; post-intercept cropped)", fontsize=14, y=0.98)
for ax, scen in zip(axes.flat, ORDER):
    res = load(scen)
    if res is None:
        ax.text(0.5, 0.5, f"{scen}\n(no data)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        continue
    data, meta = res
    end = window_end(data, meta)
    t, r = data["t"][:end], data["range"][:end]
    ax.plot(t, r, color="#1f77b4", lw=1.6, label="range")
    ax.axhline(INTERCEPT_RANGE, color="red", ls="--", lw=1.2, label=f"intercept radius ({INTERCEPT_RANGE} m)")
    # min-range marker (closest approach = end of plotted window)
    i_min = int(np.argmin(r))
    ax.scatter([t[i_min]], [r[i_min]], color="orange", zorder=5, s=55,
               label=f"min range {r[i_min]:.2f} m")
    # intercept marker (first crossing of the intercept radius)
    if meta["intercepted"] and meta["intercept_t"] is not None:
        ri, ti = range_at(data, meta["intercept_t"])
        ax.scatter([ti], [ri], marker="*", color="green", zorder=6, s=220, edgecolors="black", lw=0.5,
                   label=f"intercept t={meta['intercept_t']:.2f}s")
    status = "PASS" if meta["intercepted"] else "FAIL"
    color = "green" if meta["intercepted"] else "red"
    ax.set_title(f"{scen}  [{status}]  —  {HINT.get(scen,'')}", fontsize=11,
                 color=color, fontweight="bold")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("range to target (m)")
    ax.set_ylim(0, float(np.max(r)) * 1.05)
    ax.set_xlim(0, float(t[-1]))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out1 = os.path.join(HERE, "ros2_sim_results.png")
fig.savefig(out1, dpi=130)
plt.close(fig)
print(f"wrote {out1}")

# ---------------------------------------------------------------- figure 2: XY trajectories
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("ROS2 Interception Sim — Top-down Pursuit Trajectories (XY)", fontsize=15, y=0.98)
for ax, scen in zip(axes.flat, ORDER):
    res = load(scen)
    if res is None:
        ax.set_axis_off()
        continue
    data, meta = res
    end = window_end(data, meta)
    ux, uy = data["uav_x"][:end], data["uav_y"][:end]
    tx, ty = data["tgt_x"][:end], data["tgt_y"][:end]
    ax.plot(ux, uy, color="#1f77b4", lw=1.6, label="interceptor")
    ax.plot(tx, ty, color="#d62728", lw=1.6, ls="--", label="target")
    ax.scatter([ux[0]], [uy[0]], marker="o", color="#1f77b4", s=50, edgecolors="black", zorder=5)
    ax.scatter([tx[0]], [ty[0]], marker="s", color="#d62728", s=50, edgecolors="black", zorder=5)
    # intercept marker on the interceptor path (within the plotted window)
    if meta["intercepted"] and meta["intercept_t"] is not None:
        i = int(np.argmin(np.abs(data["t"] - meta["intercept_t"])))
        ax.scatter([ux[i]], [uy[i]], marker="*", color="green", s=260, edgecolors="black", lw=0.6, zorder=6,
                   label=f"intercept t={meta['intercept_t']:.2f}s")
    status = "PASS" if meta["intercepted"] else "FAIL"
    ax.set_title(f"{scen}  [{status}]  —  {HINT.get(scen,'')}",
                 color="green" if meta["intercepted"] else "red", fontweight="bold", fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out2 = os.path.join(HERE, "ros2_sim_trajectories.png")
fig.savefig(out2, dpi=130)
plt.close(fig)
print(f"wrote {out2}")

# ---------------------------------------------------------------- console summary table
print("\n%-12s %-8s %-12s %-10s" % ("scenario", "result", "intercept_t", "min_range"))
print("-" * 46)
npass = 0
for scen in ORDER:
    res = load(scen)
    if res is None:
        print("%-12s (no data)" % scen)
        continue
    data, meta = res
    npass += int(meta["intercepted"])
    it = f"{meta['intercept_t']:.2f}s" if meta["intercept_t"] is not None else "—"
    # closest approach of the shown (pre-cutoff) window = first-bottom for intercepted runs
    end = window_end(data, meta)
    min_r = float(np.min(data["range"][:end]))
    print("%-12s %-8s %-12s %-10s" % (scen, "PASS" if meta["intercepted"] else "FAIL", it, f"{min_r:.2f}m"))
print("-" * 46)
print("PASS %d/6" % npass)
