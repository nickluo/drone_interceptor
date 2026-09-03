#!/usr/bin/env python3
"""Radar-relay before/after metrics for interceptor_node_relative_imm_ukf.

Reads the ROS2 sim CSV (ground truth) + the C++ node log and prints a one-block
metric summary for speed smoothness, target-width error, and range error. Used to
quantify the C1/C2 (smoothness), C3 (estimator), C4 (width/range) changes.

Usage:
    analyze_radar_scenario.py <scenario> <dir>
        <dir> contains ros2_sim_<scenario>.csv, ros2_sim_<scenario>_summary.txt,
        and <scenario>.node.log  (e.g. prototype/baseline  or  prototype/after)

The CSV columns are: t,range,los_rate,uav_x,uav_y,uav_z,tgt_x,tgt_y,tgt_z,
                      throttle,wx,wy,wz
"""
import sys
import re
from pathlib import Path

import numpy as np

TRUE_WIDTH = 1.5  # all radar sim scenarios use target width 1.5 m


def read_summary(scen, d):
    p = d / f"ros2_sim_{scen}_summary.txt"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        m = re.match(r"\s*([^:]+?):\s*(.+?)\s*$", line)
        if m:
            out[m.group(1).strip().lower()] = m.group(2).strip()
    return out


def _fnum(s):
    """Parse the leading float from a summary value like '5.00s' or '0.11m' or '85.60m'."""
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else float("nan")


def smoothness(csv_path, t_end_radar):
    """Speed smoothness metrics over the radar-only phase (t < t_end_radar).

    Returns dict with std/mean groundspeed, accel RMS, jerk RMS, strong-brake count,
    and the ~0.3 Hz spectral power of the lateral velocity (the dither frequency)."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = data["t"]
    pos = np.stack([data["uav_x"], data["uav_y"], data["uav_z"]], axis=1)
    tgt = np.stack([data["tgt_x"], data["tgt_y"], data["tgt_z"]], axis=1)

    # Intervals (CSV is fixed-rate). dt from median diff (robust to gaps).
    dt = float(np.median(np.diff(t)))
    mask = t < t_end_radar if np.isfinite(t_end_radar) else np.ones_like(t, dtype=bool)
    # Need a few samples after the phase boundary for differencing; keep mask as-is.

    vel = np.gradient(pos, dt, axis=0)            # ownship ground velocity (m/s)
    spd = np.linalg.norm(vel, axis=1)
    accel = np.gradient(spd, dt)                   # forward acceleration (m/s^2)
    jerk = np.gradient(accel, dt)                  # m/s^3

    # Lateral velocity (perpendicular to LOS in the xy plane) — the dither drives this.
    r_xy = tgt[:, :2] - pos[:, :2]
    r_xy_norm = np.linalg.norm(r_xy, axis=1)
    e_los = np.divide(r_xy, r_xy_norm[:, None], out=np.zeros_like(r_xy),
                      where=r_xy_norm[:, None] > 1e-6)
    e_perp = np.stack([-e_los[:, 1], e_los[:, 0]], axis=1)
    v_lat = np.abs(np.einsum("ij,ij->i", vel[:, :2], e_perp))

    # 0.3 Hz power from the lateral-velocity spectrum (dither signature).
    if np.count_nonzero(mask) > 16:
        seg = v_lat[mask] - v_lat[mask].mean()
        if seg.size > 16:
            freqs = np.fft.rfftfreq(seg.size, d=dt)
            spec = np.abs(np.fft.rfft(seg)) ** 2
            band = (freqs > 0.20) & (freqs < 0.40)
            dither_power = float(spec[band].sum()) if band.any() else float("nan")
            total_power = float(spec.sum()) if spec.sum() > 0 else 1.0
            dither_frac = dither_power / total_power
        else:
            dither_power = dither_frac = float("nan")
    else:
        dither_power = dither_frac = float("nan")

    a = accel[mask]
    j = jerk[mask]
    return {
        "phase_s": float(t[mask][-1] - t[mask][0]) if mask.any() else 0.0,
        "n": int(mask.sum()),
        "spd_mean": float(spd[mask].mean()),
        "spd_std": float(spd[mask].std()),
        "accel_rms": float(np.sqrt((a ** 2).mean())),
        "jerk_rms": float(np.sqrt((j ** 2).mean())),
        "brake_events": int((a < -2.0).sum()),          # strong decelerations
        "dither_0.3hz_power": dither_power,
        "dither_0.3hz_frac": dither_frac,
    }


def width_error(log_path):
    """Locked target width from the [HANDOVER] log line; None if no radar width lock fired."""
    if not log_path.exists():
        return None
    txt = log_path.read_text(errors="replace")
    m = re.search(r"\[HANDOVER\].*?W=([\d.]+)m", txt)
    if not m:
        return None
    w = float(m.group(1))
    kept = re.search(r"kept (\d+)/(\d+)", txt)
    # Post-lock W re-anchor events ([W-REANCHOR]): each carries the corrected W and the sustained
    # r_vis/r_radar ratio that triggered it. None if the safety net never fired.
    reanchor = [
        {"w": float(a), "ratio": float(b)}
        for a, b in re.findall(r"\[W-REANCHOR\].*?-> W=([\d.]+)m.*?ratio ([\d.]+)", txt)
    ]
    return {
        "w": w,
        "err": w - TRUE_WIDTH,
        "kept": int(kept.group(1)) if kept else None,
        "total": int(kept.group(2)) if kept else None,
        "reanchor": reanchor,
    }


def range_error(csv_path, log_path, t_end_radar):
    """Radar-phase range accuracy: node's filtered |r| (from [RADAR] log, ~0.5 s) vs CSV truth.

    Restricted to the radar-only phase so the terminal fly-away (where r_est collapses) does
    not dominate. The [RADAR] log is throttle-sampled ~every 0.5 s without per-line timestamps,
    so we slice the first ~t_end_radar/0.5 samples and map them to truth by even cadence."""
    if not log_path.exists() or not np.isfinite(t_end_radar):
        return None
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = data["t"]; rng = data["range"]
    lines = re.findall(r"\[RADAR\].*?\|r\|=([\d.]+)", log_path.read_text(errors="replace"))
    if len(lines) < 3:
        return None
    r_est_all = np.array([float(x) for x in lines])
    # Keep only the radar-phase samples (~0.5 s cadence from t0).
    k = max(3, int(t_end_radar / 0.5))
    r_est = r_est_all[:k]
    t0 = t[0]
    t_log = t0 + 0.5 * np.arange(len(r_est))
    t_log = np.clip(t_log, t[0], min(t_end_radar, t[-1]))
    truth = np.interp(t_log, t, rng)
    err = r_est - truth
    return {
        "n": int(len(r_est)),
        "rms": float(np.sqrt((err ** 2).mean())),
        "maxabs": float(np.abs(err).max()),
        "mean": float(err.mean()),
    }


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    scen, dstr = sys.argv[1], Path(sys.argv[2])
    csv = dstr / f"ros2_sim_{scen}.csv"
    log = dstr / f"{scen}.node.log"
    summ = read_summary(scen, dstr)

    # Radar-only phase ends at first vision (radar_only never acquires -> whole run).
    fvt = _fnum(summ["first vision t"]) if "first vision t" in summ else float("nan")
    t_end_radar = fvt if fvt >= 0 else float("inf")

    print(f"=== {scen}  [{dstr.name}] ===")
    if summ:
        print(f"  intercept     : {summ.get('intercepted','?')}   "
              f"min_range={summ.get('min range','?')}   final={summ.get('final range','?')}   "
              f"single_pass={summ.get('single pass (no range reopen)','?')}")
        print(f"  radar closure : init={summ.get('initial range','?')}  "
              f"first_vision={summ.get('first vision t','?')}s @ {summ.get('first vision range','?')}  "
              f"vision_frames={summ.get('vision frames','?')}  fp={summ.get('false positives emitted','?')}")

    if csv.exists():
        s = smoothness(csv, t_end_radar)
        print(f"  smoothness (radar phase, {s['n']} samples, {s['phase_s']:.1f}s):")
        print(f"    spd mean={s['spd_mean']:.2f} std={s['spd_std']:.2f} m/s | "
              f"accel_rms={s['accel_rms']:.2f} | jerk_rms={s['jerk_rms']:.1f} m/s^3")
        print(f"    brake_events(dv/dt<-2)={s['brake_events']} | "
              f"dither 0.3Hz frac={s['dither_0.3hz_frac']:.3f} (power={s['dither_0.3hz_power']:.3g})")
    else:
        print(f"  [no CSV {csv}]")

    w = width_error(log)
    if w is None:
        print(f"  width         : NO radar HANDOVER lock fired (width stayed at guess/froze)")
    else:
        kt = f"  kept {w['kept']}/{w['total']}" if w["kept"] is not None else ""
        print(f"  width         : W={w['w']:.3f}m  err={w['err']:+.3f}m (truth {TRUE_WIDTH}){kt}")
        if w.get("reanchor"):
            for ra in w["reanchor"]:
                print(f"    [W-REANCHOR] -> W={ra['w']:.3f}m  ratio={ra['ratio']:.2f}")
        else:
            print(f"    reanchor     : none fired")
    print()


if __name__ == "__main__":
    main()
