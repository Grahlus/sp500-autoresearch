#!/usr/bin/env python3
"""
run.py — Walk-forward experiment runner.
Scores strategy across 7 independent 1-year test windows.
The agent commits only if MEAN Sharpe across all windows improves.
"""
from prepare import load_data, run_backtest, print_metrics
from agent import generate_signals, METRIC, HYPOTHESIS

print(f"\n{'='*62}")
print(f"  Hypothesis : {HYPOTHESIS}")
print(f"  Metric     : {METRIC}")
print(f"{'='*62}")

data    = load_data()
weights = generate_signals(data)

results = run_backtest(weights, data)
print_metrics(results)

score = results.get(METRIC, results["sharpe"])

# ── Constraint checks ─────────────────────────────────────────────────────────
mean_sharpe  = results["sharpe"]
min_sharpe   = results["sharpe_min"]
trades_yr    = results["trades_per_year"]
n_windows    = len(results.get("windows", []))
neg_windows  = sum(1 for w in results.get("windows", []) if w["sharpe"] < 0)

violations = []
if trades_yr > 150:
    violations.append(f"TRADES/YEAR {trades_yr} > 150 limit.")
if mean_sharpe < 0.3:
    violations.append(f"MEAN SHARPE {mean_sharpe} < 0.3 — no consistent edge.")
if neg_windows > 2:
    violations.append(
        f"{neg_windows}/{n_windows} windows negative Sharpe — not robust."
    )
if min_sharpe < -0.5:
    violations.append(
        f"WORST WINDOW Sharpe {min_sharpe} < -0.5 — catastrophic failure in one period."
    )

print(f">>> MEAN SHARPE (all windows) : {mean_sharpe}")
print(f">>> SHARPE RANGE              : {results['sharpe_min']} → {results['sharpe_max']}")
print(f">>> BEAT SPY                  : {results['windows_beat_spy']}")
print(f">>> TRADES/YEAR               : {trades_yr}")
print(f">>> TOTAL COST                : {results['total_cost']}")

if violations:
    print(f"\n{'!'*62}")
    print("  !! CONSTRAINT VIOLATIONS — REVERT THIS EXPERIMENT !!")
    for v in violations:
        print(f"  !! {v}")
    print(f"{'!'*62}")
    print("  ACTION: git checkout agent.py\n")
else:
    print(f"\n  [OK] All constraints passed.\n")
