#!/usr/bin/env python3
"""
run.py — Thin experiment runner. Called by Claude Code each iteration.
Prints SCORE line that the agent uses to decide commit vs revert.
"""
from prepare import load_data, run_backtest, print_metrics
from agent import generate_signals, METRIC, HYPOTHESIS

print(f"\n{'='*62}")
print(f"  Hypothesis : {HYPOTHESIS}")
print(f"  Metric     : {METRIC}")
print(f"{'='*62}")

data    = load_data()
weights = generate_signals(data)

val   = run_backtest(weights, data, split="validation")
train = run_backtest(weights, data, split="train")

print("=== VALIDATION ===")
print_metrics(val)
print("=== TRAIN (sanity / overfit check) ===")
print_metrics(train)

score = val.get(METRIC, val["sharpe"])

# ── Constraint checks ─────────────────────────────────────────────────────────
val_sharpe   = val["sharpe"]
train_sharpe = train["sharpe"]
trades_yr    = val["trades_per_year"]

violations = []
if trades_yr > 150:
    violations.append(
        f"TRADES/YEAR {trades_yr} > 150 — commission drag fatal at $20/trade. "
        f"Use rebalance_days >= 20."
    )
if train_sharpe < 0.5:
    violations.append(
        f"TRAIN SHARPE {train_sharpe} < 0.5 — no edge on 9 years of data. Curve-fit."
    )
if train_sharpe < val_sharpe / 1.6:
    violations.append(
        f"OVERFIT: train={train_sharpe} < val={val_sharpe} / 1.5 = {val_sharpe/1.5:.3f}. "
        f"Val-only outperformance = curve-fit to recent period."
    )

print(f"\n>>> SCORE ({METRIC})     : {score}")
print(f">>> FINAL VALUE (val)  : {val['final_value']}")
print(f">>> TOTAL COST (val)   : {val['total_cost']}  ({val['cost_pct_capital']}% of capital)")
print(f">>> TRADES/YEAR (val)  : {val['trades_per_year']}")
print(f">>> TRAIN SHARPE       : {train_sharpe}  |  VAL SHARPE: {val_sharpe}")

if violations:
    print(f"\n{'!'*62}")
    print("  !! CONSTRAINT VIOLATIONS — REVERT THIS EXPERIMENT !!")
    for v in violations:
        print(f"  !! {v}")
    print(f"{'!'*62}")
    print("  ACTION: git checkout agent.py  — do NOT commit, do NOT log as Yes\n")
else:
    print(f"\n  [OK] All constraints passed — eligible to commit if {METRIC} improved.\n")
