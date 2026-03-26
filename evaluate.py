#!/usr/bin/env python3
"""
evaluate.py — TRUE OUT-OF-SAMPLE evaluation of the champion strategy.

This script evaluates agent.py on the period AFTER the validation window.
Neither the agent nor the overfit constraint ever touched this data.

Timeline:
  Train        : 2014-01-02 → ~2022-07  (used for training)
  Validation   : ~2022-07   → ~2024-06  (used for experiment selection)
  OUT-OF-SAMPLE: ~2024-07   → today     ← THIS SCRIPT ONLY

Run once. Do NOT use results to tune agent.py — that contaminates the test.
"""
import sys, signal
import numpy as np
import pandas as pd

signal.signal(signal.SIGALRM, signal.SIG_DFL)
signal.alarm(0)

from prepare import load_data
from agent import generate_signals, HYPOTHESIS

# Cost model — keep in sync with prepare.py
STARTING_CAPITAL     = 100_000.0
COMMISSION_PER_TRADE = 20.0
SLIPPAGE_BPS         = 5


def run_oos_backtest(weights: pd.DataFrame, data: dict) -> dict:
    close  = data["close"]
    open_  = data["open"]
    idx    = close.index
    # OOS = everything after last walk-forward window (W7 ends 2024-07-01)
    # Do NOT use train_end — it changed in session 5 walk-forward engine
    oos_candidates = idx[idx >= pd.Timestamp("2024-07-01")]
    if len(oos_candidates) == 0:
        print("[ERROR] No data after 2024-07-01. Run refresh_data.py.")
        sys.exit(1)
    oos_start = oos_candidates[0]
    print(f"  OOS start : {oos_start.date()} | end: {idx[-1].date()} | days: {len(oos_candidates)}")


    close_oos = close.loc[oos_start:]
    open_oos  = open_.loc[oos_start:]
    n_days    = len(close_oos)

    if n_days < 20:
        print(f"\n[ERROR] OOS too short ({n_days} days). Run refresh_data.py first.")
        sys.exit(1)

    weights_oos = (
        weights
        .reindex(close_oos.index)
        .reindex(columns=close_oos.columns)
        .fillna(0.0)
    )

    slip = SLIPPAGE_BPS / 10_000.0
    cash = STARTING_CAPITAL
    prev_w_norm = pd.Series(0.0, index=close_oos.columns)
    shares_held      = pd.Series(0.0, index=close_oos.columns)
    portfolio_values = []
    commission_total = slippage_total = trade_count = 0.0
    rebal_trades     = 0
    stop_trades      = 0

    # Detect rebalance days: days where weights change (agent rebalanced)
    weight_changes = weights_oos.diff().abs().sum(axis=1)

    for i in range(n_days):
        px_open    = open_oos.iloc[i]
        port_value = cash + (shares_held * px_open).sum()
        portfolio_values.append(port_value)

        if i == 0:
            continue

        target_w = weights_oos.iloc[i - 1]
        gross    = target_w.abs().sum()
        w_norm   = target_w / gross if gross > 0 else target_w

        target_shares = ((w_norm * port_value) / px_open).fillna(0.0)
        target_shares = target_shares.where(px_open > 0, 0.0)
        delta         = target_shares - shares_held

        # Only trade if weight actually changed (new entry, exit, or stop).
        # Skip micro-rebalancing from daily portfolio drift — in live trading
        # you do NOT rebalance every day just because prices moved.
        weight_changed = (w_norm - prev_w_norm).abs() > 1e-6
        delta          = delta.where(weight_changed, 0.0)
        traded         = delta[delta.abs() > 0.01].index

        # Classify: rebalance day vs stop-loss day
        is_rebal = weight_changes.iloc[i] > 1e-6

        trade_cash = 0.0
        for tkr in traded:
            d = delta[tkr]; px = px_open[tkr]
            if px <= 0 or np.isnan(px): continue
            fill_px         = px * (1 + slip) if d > 0 else px * (1 - slip)
            trade_cash     -= d * fill_px
            slippage_total += abs(d) * px * slip
            commission_total += COMMISSION_PER_TRADE
            trade_count    += 1
            if is_rebal:
                rebal_trades += 1
            else:
                stop_trades  += 1

        shares_held  = target_shares.copy()
        prev_w_norm  = w_norm.copy()
        cash        += trade_cash - COMMISSION_PER_TRADE * len(traded)

    pv = pd.Series(portfolio_values, index=close_oos.index)
    r  = pv.pct_change().fillna(0.0)

    if "SPY" in open_oos.columns:
        br = (open_oos["SPY"].shift(-1) / open_oos["SPY"] - 1).fillna(0.0)
    else:
        br = (open_oos.shift(-1) / open_oos - 1).fillna(0.0).mean(axis=1)

    bench_pv = STARTING_CAPITAL * (1 + br).cumprod()
    ann      = 252

    def sharpe(r): return (r.mean()/r.std()*np.sqrt(ann)) if r.std()>1e-9 else 0.0
    def calmar(r):
        cum=( 1+r).cumprod(); dd=(cum/cum.cummax()-1).min()
        ar=cum.iloc[-1]**(ann/max(len(r),1))-1
        return ar/abs(dd) if dd<-1e-6 else 0.0
    def maxdd(r): cum=(1+r).cumprod(); return (cum/cum.cummax()-1).min()

    tot  = pv.iloc[-1]/STARTING_CAPITAL - 1
    btot = bench_pv.iloc[-1]/STARTING_CAPITAL - 1
    tpy  = trade_count / (n_days/ann)

    ann_days = n_days / ann
    return dict(
        oos_start        = str(oos_start.date()),
        oos_end          = str(close_oos.index[-1].date()),
        n_days           = n_days,
        sharpe           = round(sharpe(r), 3),
        calmar           = round(calmar(r), 3),
        max_drawdown     = round(maxdd(r)*100, 2),
        win_rate         = round((r>0).mean()*100, 1),
        ann_vol          = round(r.std()*np.sqrt(ann)*100, 2),
        total_return_pct = round(tot*100, 2),
        bench_return_pct = round(btot*100, 2),
        alpha_ann        = round((r-br).mean()*ann*100, 2),
        final_value      = round(pv.iloc[-1], 0),
        bench_final      = round(bench_pv.iloc[-1], 0),
        net_pnl          = round(pv.iloc[-1]-STARTING_CAPITAL, 0),
        total_trades     = int(trade_count),
        trades_per_year  = round(tpy, 1),
        rebal_trades_yr  = round(rebal_trades / ann_days, 1),
        stop_trades_yr   = round(stop_trades  / ann_days, 1),
        commission_paid  = round(commission_total, 0),
        slippage_paid    = round(slippage_total, 0),
        total_cost       = round(commission_total+slippage_total, 0),
        cost_pct_capital = round((commission_total+slippage_total)/STARTING_CAPITAL*100, 2),
    )


def print_report(m, hypothesis):
    W = 66
    print("\n" + "="*W)
    print("  TRUE OUT-OF-SAMPLE EVALUATION")
    print("  This data was NEVER used during training or experiment selection.")
    print("="*W)
    print(f"  Strategy   : {hypothesis}")
    print(f"  OOS period : {m['oos_start']} → {m['oos_end']}  ({m['n_days']} trading days)")
    print(f"  Capital    : ${STARTING_CAPITAL:,.0f}  |  "
          f"Commission: ${COMMISSION_PER_TRADE:.0f}/trade  |  Slippage: {SLIPPAGE_BPS}bps")
    print("-"*W)
    rows = [
        ("",                "── Core ──────────────────────────────"),
        ("Sharpe",          m['sharpe']),
        ("Calmar",          m['calmar']),
        ("Max drawdown",    f"{m['max_drawdown']}%"),
        ("Win rate",        f"{m['win_rate']}%"),
        ("Ann. volatility", f"{m['ann_vol']}%"),
        ("",                "── Returns ───────────────────────────"),
        ("Strategy return", f"+{m['total_return_pct']}%"),
        ("SPY return",      f"+{m['bench_return_pct']}%"),
        ("Alpha (ann.)",    f"{m['alpha_ann']}%"),
        ("",                "── Dollar P&L ────────────────────────"),
        ("Starting capital",f"${STARTING_CAPITAL:,.0f}"),
        ("Final value",     f"${m['final_value']:,.0f}"),
        ("Net P&L",         f"${m['net_pnl']:,.0f}"),
        ("SPY final",       f"${m['bench_final']:,.0f}"),
        ("",                "── Costs ─────────────────────────────"),
        ("Total trades",    m['total_trades']),
        ("Trades/year",     m['trades_per_year']),
        ("  rebal trades/yr", m['rebal_trades_yr']),
        ("  stop  trades/yr", m['stop_trades_yr']),
        ("Commission",      f"${m['commission_paid']:,.0f}"),
        ("Slippage",        f"${m['slippage_paid']:,.0f}"),
        ("Total cost",      f"${m['total_cost']:,.0f}  ({m['cost_pct_capital']}% of capital)"),
    ]
    for label, value in rows:
        if label == "":
            print(f"\n  {value}")
        else:
            print(f"    {label:<22}: {value}")

    s = m['sharpe']
    beat = m['total_return_pct'] > m['bench_return_pct']
    verdict = (
        "STRONG  — Sharpe >= 1.5. Generalised well. Live trading candidate." if s >= 1.5 else
        "GOOD    — Sharpe >= 1.0. Solid out-of-sample performance." if s >= 1.0 else
        "WEAK    — Sharpe >= 0.5. Thin edge. Paper trade first." if s >= 0.5 else
        "FAILED  — Sharpe < 0.5. Did not generalise."
    )
    print(f"\n{'='*W}")
    print(f"  Verdict  : {verdict}")
    print(f"  Beat SPY : {'YES' if beat else 'NO'} "
          f"({m['total_return_pct']}% vs {m['bench_return_pct']}%)")
    print(f"  DIAG     : {m['rebal_trades_yr']} rebal trades/yr  |  "
          f"{m['stop_trades_yr']} stop-exit trades/yr")
    print("="*W + "\n")


if __name__ == "__main__":
    print("\n>>> Loading data and generating signals …")
    data    = load_data()
    weights = generate_signals(data)
    print("\n>>> Running true out-of-sample backtest …")
    m = run_oos_backtest(weights, data)
    print_report(m, HYPOTHESIS)
