#!/usr/bin/env python3
"""
run_stop_grid.py — Stop parameter grid search (Session 11).

Runs 28 experiments across stop types, stop percentages, and min hold days.
After WF grid, runs OOS evaluation on top candidates (beating WF=0.722).
Prints final ranked table. Does NOT commit.

CHAMPION TARGETS:  WF > 0.722,  OOS > 1.548
"""
import io, sys, re, signal, time
import numpy as np

# ── Load data once ────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
import prepare
data = prepare.load_data()
print("Data loaded.\n", flush=True)

import agent
import evaluate as eval_mod

# ── Experiment grid ───────────────────────────────────────────────────────────
EXPERIMENTS = []

# Type C: no stop (2 experiments)
for mhd in [5, 25]:
    EXPERIMENTS.append(dict(
        id=f"C-mhd{mhd:02d}",
        stop_type="none", stop_pct=None, parabolic=None, min_hold=mhd,
    ))

# Type B: fixed stop (7 pcts × 2 mhd = 14 experiments)
for pct in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    for mhd in [5, 25]:
        EXPERIMENTS.append(dict(
            id=f"B{int(pct*100):02d}-mhd{mhd:02d}",
            stop_type="fixed", stop_pct=pct, parabolic=None, min_hold=mhd,
        ))

# Type A: adaptive stop (6 pairs × 2 mhd = 12 experiments)
ADAPTIVE_PAIRS = [
    (0.20, 0.30),   # current champion params
    (0.20, 0.35),
    (0.25, 0.35),
    (0.25, 0.40),
    (0.30, 0.40),
    (0.30, 0.45),
]
for stop_pct, parabolic in ADAPTIVE_PAIRS:
    for mhd in [5, 25]:
        EXPERIMENTS.append(dict(
            id=f"A{int(stop_pct*100):02d}p{int(parabolic*100):02d}-mhd{mhd:02d}",
            stop_type="adaptive", stop_pct=stop_pct, parabolic=parabolic, min_hold=mhd,
        ))

# ── Run WF grid ───────────────────────────────────────────────────────────────
results_table = []

print(f"{'='*72}")
print(f"  STOP GRID — {len(EXPERIMENTS)} experiments  (WF champion target: >0.722)")
print(f"{'='*72}\n")

for idx, exp in enumerate(EXPERIMENTS, 1):
    t0 = time.time()

    # Set agent globals for this experiment
    agent.STOP_TYPE      = exp["stop_type"]
    agent.STOP_LOSS_PCT  = exp["stop_pct"] if exp["stop_pct"] is not None else 0.20
    agent.STOP_PARABOLIC = exp["parabolic"] if exp["parabolic"] is not None else 0.30
    agent.MIN_HOLD_DAYS  = exp["min_hold"]
    agent.HYPOTHESIS     = (
        f"GRID {exp['id']}: type={exp['stop_type']} "
        f"pct={exp['stop_pct']} para={exp['parabolic']} mhd={exp['min_hold']}"
    )

    # Reset the SIGALRM for each experiment (prepare.py sets 1h on import)
    signal.alarm(0)
    signal.alarm(3600)

    print(f"[{idx:02d}/{len(EXPERIMENTS)}] {exp['id']:<24}", end="  ", flush=True)

    captured = io.StringIO()
    try:
        # Capture DIAG output from generate_signals
        old_stdout = sys.stdout
        sys.stdout = captured
        weights = agent.generate_signals(data)
        sys.stdout = old_stdout

        diag_out = captured.getvalue()
        # Print DIAG line
        for line in diag_out.split('\n'):
            if '[DIAG]' in line:
                print(line.strip(), flush=True)

        # Extract stops_per_yr
        m = re.search(r'stops/yr=([\d.]+)', diag_out)
        stops_per_yr = float(m.group(1)) if m else 0.0

        # Walk-forward backtest
        wf = prepare.run_backtest(weights, data)

        wf_sharpe = wf["sharpe"]
        wf_min    = wf["sharpe_min"]
        neg_wins  = sum(1 for w in wf.get("windows", []) if w["sharpe"] < 0)
        trades_yr = wf["trades_per_year"]
        max_dd    = wf["max_drawdown"]

        ok = (wf_sharpe >= 0.3 and wf_min >= -1.2 and neg_wins <= 4 and trades_yr <= 150)
        status = "OK" if ok else "FAIL"

        elapsed = time.time() - t0
        print(f"  WF={wf_sharpe:+.3f}  min={wf_min:+.3f}  neg={neg_wins}  "
              f"t/yr={trades_yr:.1f}  DD={max_dd:.1f}%  stp/yr={stops_per_yr:.1f}  "
              f"[{status}]  ({elapsed:.0f}s)",
              flush=True)

        results_table.append(dict(
            id=exp["id"], stop_type=exp["stop_type"],
            stop_pct=exp["stop_pct"], parabolic=exp["parabolic"], min_hold=exp["min_hold"],
            wf_sharpe=wf_sharpe, wf_min=wf_min, neg_windows=neg_wins,
            trades_yr=trades_yr, max_dd=max_dd, stops_per_yr=stops_per_yr,
            status=status, weights=weights,
        ))

    except Exception as e:
        sys.stdout = old_stdout
        print(f"  ERROR: {e}", flush=True)
        results_table.append(dict(
            id=exp["id"], stop_type=exp["stop_type"],
            stop_pct=exp["stop_pct"], parabolic=exp["parabolic"], min_hold=exp["min_hold"],
            wf_sharpe=-99.0, wf_min=-99.0, neg_windows=99,
            trades_yr=0, max_dd=0, stops_per_yr=0,
            status="ERROR", weights=None,
        ))

# ── WF ranked table ───────────────────────────────────────────────────────────
print(f"\n\n{'='*100}")
print(f"  WF RANKED RESULTS  (champion target: WF > 0.722)")
print(f"{'='*100}")
print(f"  {'ID':<24}  {'type':<8}  {'pct':>5}  {'para':>5}  {'mhd':>4}  "
      f"{'WF':>6}  {'min':>6}  {'neg':>3}  {'t/yr':>5}  {'DD%':>6}  {'stp/y':>5}  status")
print(f"  {'-'*92}")

sorted_results = sorted(results_table, key=lambda x: -x["wf_sharpe"])
CHAMPION_ID = "A20p30-mhd25"  # current committed champion
for r in sorted_results:
    marker = "  ◄ CHAMPION" if r["id"] == CHAMPION_ID else ""
    pct_s  = f"{r['stop_pct']:.2f}" if r["stop_pct"] is not None else "  ---"
    para_s = f"{r['parabolic']:.2f}" if r["parabolic"] is not None else "  ---"
    print(f"  {r['id']:<24}  {r['stop_type']:<8}  {pct_s:>5}  {para_s:>5}  "
          f"{r['min_hold']:>4}  {r['wf_sharpe']:>6.3f}  {r['wf_min']:>6.3f}  "
          f"{r['neg_windows']:>3}  {r['trades_yr']:>5.1f}  {r['max_dd']:>6.1f}  "
          f"{r['stops_per_yr']:>5.1f}  {r['status']}{marker}")

# ── OOS evaluation for candidates beating WF=0.722 ───────────────────────────
WF_THRESHOLD = 0.722
candidates = [
    r for r in sorted_results
    if r["status"] == "OK" and r["wf_sharpe"] > WF_THRESHOLD and r["weights"] is not None
]

print(f"\n  Configs beating WF={WF_THRESHOLD}: {len(candidates)}")

if not candidates:
    print("\n  No config beats WF baseline. No OOS evaluation needed.")
    print("  VERDICT: Current champion S10-005 + STOP-010b remains champion.")
    sys.exit(0)

# Run OOS on top candidates (cap at 5)
MAX_OOS = min(5, len(candidates))
print(f"\n  Running OOS evaluation on top {MAX_OOS} candidates...\n")

oos_results = []

for r in candidates[:MAX_OOS]:
    # Re-configure agent with this experiment's params
    agent.STOP_TYPE      = r["stop_type"]
    agent.STOP_LOSS_PCT  = r["stop_pct"] if r["stop_pct"] is not None else 0.20
    agent.STOP_PARABOLIC = r["parabolic"] if r["parabolic"] is not None else 0.30
    agent.MIN_HOLD_DAYS  = r["min_hold"]
    agent.HYPOTHESIS     = f"OOS-EVAL {r['id']}"

    signal.alarm(0)
    signal.alarm(3600)

    print(f"  OOS: {r['id']}", flush=True)
    try:
        metrics, trade_log, daily_log = eval_mod.run_oos_backtest(r["weights"], data)
        oos_sharpe = metrics["sharpe"]
        oos_maxdd  = metrics["max_drawdown"]
        oos_ret    = metrics["total_return_pct"]
        oos_neg    = metrics.get("neg_windows", 0)  # not in OOS metrics
        stop_trades = metrics["stop_trades"]

        # Check commit criteria
        neg_wins = r["neg_windows"]
        wf_min   = r["wf_min"]
        commit_ok = (
            oos_sharpe > 1.548 and
            r["wf_sharpe"] > 0.722 and
            wf_min > -1.5 and
            neg_wins <= 4
        )

        verdict = "COMMIT" if commit_ok else "no"
        print(f"    OOS sharpe={oos_sharpe:.3f}  maxDD={oos_maxdd:.1f}%  "
              f"ret={oos_ret:.1f}%  stop_trades={stop_trades}  [{verdict}]",
              flush=True)

        oos_results.append(dict(
            id=r["id"], stop_type=r["stop_type"],
            stop_pct=r["stop_pct"], parabolic=r["parabolic"], min_hold=r["min_hold"],
            wf_sharpe=r["wf_sharpe"], wf_min=wf_min, neg_windows=neg_wins,
            oos_sharpe=oos_sharpe, oos_maxdd=oos_maxdd, oos_ret=oos_ret,
            stops_per_yr=r["stops_per_yr"], stop_trades=stop_trades,
            verdict=verdict,
        ))

    except Exception as e:
        print(f"    OOS ERROR: {e}", flush=True)

# ── Final OOS ranked table ────────────────────────────────────────────────────
if oos_results:
    oos_sorted = sorted(oos_results, key=lambda x: -x["oos_sharpe"])

    print(f"\n\n{'='*100}")
    print(f"  FINAL RANKED TABLE — by OOS Sharpe  (champion targets: WF>0.722, OOS>1.548)")
    print(f"{'='*100}")
    print(f"  {'ID':<24}  {'type':<8}  {'pct':>5}  {'para':>5}  {'mhd':>4}  "
          f"{'WF':>6}  {'OOS':>6}  {'OOS_DD':>7}  {'OOS_ret':>8}  {'stp/y':>5}  verdict")
    print(f"  {'-'*92}")

    for r in oos_sorted:
        pct_s  = f"{r['stop_pct']:.2f}" if r["stop_pct"] is not None else "  ---"
        para_s = f"{r['parabolic']:.2f}" if r["parabolic"] is not None else "  ---"
        marker = "  *** COMMIT ***" if r["verdict"] == "COMMIT" else ""
        print(f"  {r['id']:<24}  {r['stop_type']:<8}  {pct_s:>5}  {para_s:>5}  "
              f"{r['min_hold']:>4}  {r['wf_sharpe']:>6.3f}  {r['oos_sharpe']:>6.3f}  "
              f"{r['oos_maxdd']:>7.1f}  {r['oos_ret']:>8.1f}%  "
              f"{r['stops_per_yr']:>5.1f}  {r['verdict']}{marker}")

    commit_list = [r for r in oos_sorted if r["verdict"] == "COMMIT"]
    if commit_list:
        best = commit_list[0]
        print(f"\n  BEST CANDIDATE: {best['id']}")
        print(f"    stop_type={best['stop_type']}  stop_pct={best['stop_pct']}  "
              f"parabolic={best['parabolic']}  min_hold={best['min_hold']}")
        print(f"    WF={best['wf_sharpe']:.3f}  OOS={best['oos_sharpe']:.3f}  "
              f"OOS_MaxDD={best['oos_maxdd']:.1f}%")
        print(f"\n  ACTION: Update agent.py with these params and commit.")
    else:
        print(f"\n  No candidate beats OOS=1.548.")
        print(f"  VERDICT: Current champion remains. No commit.")
else:
    print("\n  No OOS evaluations completed.")
