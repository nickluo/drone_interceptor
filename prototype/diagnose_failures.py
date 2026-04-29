"""
Deep diagnostic: Instrument solver_flat_dq to log full trajectory data
for both success and failure runs, then analyze the failure modes.
"""
import subprocess
import numpy as np
import re
import sys

def run_single_diagnostic():
    """Run solver_flat_dq.py with full logging and return structured data."""
    result = subprocess.run(
        [sys.executable, 'solver_flat_dq.py'],
        capture_output=True, text=True, timeout=60
    )
    output = result.stdout + result.stderr
    
    success = '[SUCCESS]' in output
    failed = '[FAILED]' in output
    
    # Extract key events
    events = []
    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue
        if '[WARNING]' in line and '目标丢失' in line:
            # Parse: [WARNING] 步骤 400: 目标丢失! 盲飞: 3/80, dist=2.3m
            m = re.search(r'步骤 (\d+).*盲飞: (\d+)/(\d+), dist=([0-9.]+)m', line)
            if m:
                events.append({
                    'type': 'target_lost',
                    'step': int(m.group(1)),
                    'blind_count': int(m.group(2)),
                    'blind_max': int(m.group(3)),
                    'true_dist': float(m.group(4))
                })
        elif '[TERMINAL]' in line:
            m = re.search(r'步骤 (\d+).*dist=([0-9.]+)m', line)
            if m:
                events.append({
                    'type': 'terminal',
                    'step': int(m.group(1)),
                    'dist': float(m.group(2))
                })
        elif '[INFO]' in line and 'DQ-NMPC solve' in line:
            m = re.search(r'步骤 (\d+).*dist=([0-9.]+)m.*v_des=([0-9.]+)', line)
            if m:
                events.append({
                    'type': 'mpc_info',
                    'step': int(m.group(1)),
                    'est_dist': float(m.group(2)),
                    'v_des': float(m.group(3))
                })
        elif '[SUCCESS]' in line:
            m = re.search(r'步骤 (\d+).*距离: ([0-9.]+)m', line)
            if m:
                events.append({
                    'type': 'success',
                    'step': int(m.group(1)),
                    'dist': float(m.group(2))
                })
        elif '[FAILED]' in line:
            events.append({'type': 'failed'})
    
    return {
        'success': success,
        'failed': failed,
        'events': events,
        'output': output
    }


def analyze_run(run_data, run_id):
    """Analyze a single run for failure patterns."""
    events = run_data['events']
    
    # Find first target loss event
    first_loss = None
    last_loss = None
    terminal_event = None
    for e in events:
        if e['type'] == 'target_lost':
            if first_loss is None:
                first_loss = e
            last_loss = e
        elif e['type'] == 'terminal':
            terminal_event = e
    
    # Find closest MPC info before first loss
    mpc_before_loss = None
    if first_loss:
        for e in events:
            if e['type'] == 'mpc_info' and e['step'] <= first_loss['step']:
                mpc_before_loss = e
    
    # Find MPC info after first loss (during blind flight)
    mpc_during_blind = []
    if first_loss:
        for e in events:
            if e['type'] == 'mpc_info' and e['step'] > first_loss['step']:
                mpc_during_blind.append(e)
    
    return {
        'run_id': run_id,
        'success': run_data['success'],
        'first_loss': first_loss,
        'last_loss': last_loss,
        'terminal': terminal_event,
        'mpc_before_loss': mpc_before_loss,
        'mpc_during_blind': mpc_during_blind,
    }


def main():
    N_RUNS = 50
    print(f"Running {N_RUNS} diagnostic simulations...\n")
    
    results = []
    successes = 0
    failures = 0
    
    for i in range(N_RUNS):
        try:
            data = run_single_diagnostic()
            analysis = analyze_run(data, i+1)
            results.append(analysis)
            if data['success']:
                successes += 1
            else:
                failures += 1
            status = "OK" if data['success'] else "FAIL"
            print(f"  Run {i+1:2d}: {status}", end="")
            if analysis['first_loss']:
                fl = analysis['first_loss']
                print(f"  (first_loss @ step {fl['step']}, true_dist={fl['true_dist']:.1f}m)", end="")
            print()
        except Exception as e:
            print(f"  Run {i+1}: ERROR - {e}")
            failures += 1
    
    print(f"\n{'='*70}")
    print(f"OVERALL: {successes}/{N_RUNS} success ({100*successes/N_RUNS:.0f}%)")
    print(f"{'='*70}")
    
    # ===== ANALYSIS OF FAILURES =====
    fail_runs = [r for r in results if not r['success']]
    succ_runs = [r for r in results if r['success']]
    
    if not fail_runs:
        print("\nNo failures to analyze!")
        return
    
    print(f"\n{'='*70}")
    print("FAILURE ANALYSIS")
    print(f"{'='*70}")
    
    # 1. At what true distance did target first get lost?
    fail_first_loss_dists = []
    succ_first_loss_dists = []
    for r in fail_runs:
        if r['first_loss']:
            fail_first_loss_dists.append(r['first_loss']['true_dist'])
    for r in succ_runs:
        if r['first_loss']:
            succ_first_loss_dists.append(r['first_loss']['true_dist'])
    
    print(f"\n1. TRUE DISTANCE at first target loss:")
    if fail_first_loss_dists:
        print(f"   FAILURES: mean={np.mean(fail_first_loss_dists):.1f}m, "
              f"std={np.std(fail_first_loss_dists):.1f}m, "
              f"range=[{np.min(fail_first_loss_dists):.1f}, {np.max(fail_first_loss_dists):.1f}]m")
    if succ_first_loss_dists:
        print(f"   SUCCESS:  mean={np.mean(succ_first_loss_dists):.1f}m, "
              f"std={np.std(succ_first_loss_dists):.1f}m, "
              f"range=[{np.min(succ_first_loss_dists):.1f}, {np.max(succ_first_loss_dists):.1f}]m")
    
    # 2. Estimated distance at last MPC before loss vs true distance
    print(f"\n2. ESTIMATED vs TRUE distance at loss (perception error):")
    for r in fail_runs:
        fl = r['first_loss']
        mpc = r['mpc_before_loss']
        if fl and mpc:
            print(f"   Run {r['run_id']:2d}: true_dist={fl['true_dist']:.1f}m, "
                  f"est_dist={mpc['est_dist']:.1f}m (at step {mpc['step']}), "
                  f"error={abs(mpc['est_dist'] - fl['true_dist']):.1f}m")
    
    # 3. What happens during blind flight?
    print(f"\n3. BLIND FLIGHT trajectory (est_dist during MPC after loss):")
    for r in fail_runs:
        fl = r['first_loss']
        if fl and r['mpc_during_blind']:
            trajectory_str = " → ".join([f"{m['est_dist']:.1f}m" for m in r['mpc_during_blind'][:5]])
            print(f"   Run {r['run_id']:2d}: lost at true={fl['true_dist']:.1f}m, "
                  f"then est_dist: {trajectory_str}")
    
    # 4. Terminal guidance analysis
    print(f"\n4. TERMINAL GUIDANCE details:")
    fail_terminal_dists = []
    succ_terminal_dists = []
    fail_no_terminal = 0
    succ_no_terminal = 0
    for r in fail_runs:
        if r['terminal']:
            fail_terminal_dists.append(r['terminal']['dist'])
        else:
            fail_no_terminal += 1
    for r in succ_runs:
        if r['terminal']:
            succ_terminal_dists.append(r['terminal']['dist'])
        else:
            succ_no_terminal += 1
    
    print(f"   FAILURES with terminal guidance: {len(fail_terminal_dists)}/{len(fail_runs)}")
    if fail_terminal_dists:
        print(f"     terminal dist: mean={np.mean(fail_terminal_dists):.1f}m, "
              f"range=[{np.min(fail_terminal_dists):.1f}, {np.max(fail_terminal_dists):.1f}]m")
    print(f"   SUCCESS with terminal guidance:  {len(succ_terminal_dists)}/{len(succ_runs)}")
    if succ_terminal_dists:
        print(f"     terminal dist: mean={np.mean(succ_terminal_dists):.1f}m, "
              f"range=[{np.min(succ_terminal_dists):.1f}, {np.max(succ_terminal_dists):.1f}]m")
    print(f"   Failures WITHOUT terminal ({fail_no_terminal}): target never got close enough for dead-reckoning")
    
    # 5. Step analysis: how many steps between first loss and failure?
    print(f"\n5. BLIND FLIGHT DURATION (steps between loss and end):")
    for r in fail_runs:
        fl = r['first_loss']
        ll = r['last_loss']
        if fl and ll:
            blind_duration_steps = ll['blind_count']
            blind_duration_sec = blind_duration_steps * 0.02  # dt = TF/N
            print(f"   Run {r['run_id']:2d}: blind for {blind_duration_steps} steps ({blind_duration_sec:.2f}s), "
                  f"first @ true_dist={fl['true_dist']:.1f}m, "
                  f"last @ true_dist={ll['true_dist']:.1f}m, "
                  f"Δdist={ll['true_dist'] - fl['true_dist']:+.1f}m")
    
    # 6. Summary diagnosis
    print(f"\n{'='*70}")
    print("SUMMARY DIAGNOSIS")
    print(f"{'='*70}")
    
    # Count patterns
    flythrough = 0  # true distance increased during blind
    stuck = 0       # true distance barely changed
    early_loss = 0  # lost target far away
    
    for r in fail_runs:
        fl = r['first_loss']
        ll = r['last_loss']
        if fl and ll:
            dist_change = ll['true_dist'] - fl['true_dist']
            if dist_change > 3.0:
                flythrough += 1
            elif fl['true_dist'] > 8.0:
                early_loss += 1
            else:
                stuck += 1
    
    print(f"  FLY-THROUGH (target behind drone, dist↑): {flythrough}")
    print(f"  STUCK/ORBITING (dist ~constant, far):     {stuck}")
    print(f"  EARLY LOSS (lost at >8m, never recovered): {early_loss}")


if __name__ == '__main__':
    main()
