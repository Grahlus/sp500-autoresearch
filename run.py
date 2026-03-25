#!/usr/bin/env python3
"""
run.py — Thin experiment runner. Called by Claude Code each iteration.
Prints SCORE line that the agent uses to decide commit vs revert.
"""
from prepare import load_data, run_backtest, print_metrics
from agent import generate_signals, METRIC, HYPOTHESIS

print(f"\n{'='*60}")
print(f"  Hypothesis : {HYPOTHESIS}")
print(f"  Metric     : {METRIC}")
print(f"{'='*60}")

data    = load_data()
weights = generate_signals(data)

val   = run_backtest(weights, data, split="validation")
train = run_backtest(weights, data, split="train")

print("=== VALIDATION ===")
print_metrics(val)
print("=== TRAIN (sanity / overfit check) ===")
print_metrics(train)

score = val.get(METRIC, val["sharpe"])
print(f"\n>>> SCORE ({METRIC}): {score:.4f}\n")
