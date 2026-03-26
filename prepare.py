#!/usr/bin/env python3
"""
prepare.py — FROZEN. Do not edit.

Walk-forward validation engine (Session 5+).

Window design (3-year train, 1-year test, 1-year step):
  W1: train 2014-2017  test 2018
  W2: train 2015-2018  test 2019
  W3: train 2016-2019  test 2020
  W4: train 2017-2020  test 2021
  W5: train 2018-2021  test 2022
  W6: train 2019-2022  test 2023
  W7: train 2020-2023  test 2024-H1

OOS (never touched): 2024-07 → today  (evaluate.py only)

Scoring: mean Sharpe across all 7 windows.
A strategy must beat the baseline on the AVERAGE, not cherry-pick one window.

Cost model:
  $100,000 starting capital per window (independent resets)
  $20 commission per ticker traded
  5bps slippage one-way
  Trades only on weight changes (no drift rebalancing)
"""
import signal, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
EXPERIMENT_TIMEOUT_SECS = 60 * 60

STARTING_CAPITAL     = 100_000.0
COMMISSION_PER_TRADE = 20.0
SLIPPAGE_BPS         = 5

# Walk-forward windows: 3-year train, 6-month test, 6-month step (overlapping)
# 14 windows vs 7 before — harder to game, more regime coverage
WF_WINDOWS = [
    ("2014-01-01", "2017-01-01", "2017-01-01", "2017-07-01"),
    ("2014-07-01", "2017-07-01", "2017-07-01", "2018-01-01"),
    ("2015-01-01", "2018-01-01", "2018-01-01", "2018-07-01"),
    ("2015-07-01", "2018-07-01", "2018-07-01", "2019-01-01"),
    ("2016-01-01", "2019-01-01", "2019-01-01", "2019-07-01"),
    ("2016-07-01", "2019-07-01", "2019-07-01", "2020-01-01"),
    ("2017-01-01", "2020-01-01", "2020-01-01", "2020-07-01"),  # COVID crash
    ("2017-07-01", "2020-07-01", "2020-07-01", "2021-01-01"),  # COVID recovery
    ("2018-01-01", "2021-01-01", "2021-01-01", "2021-07-01"),  # bull peak
    ("2018-07-01", "2021-07-01", "2021-07-01", "2022-01-01"),  # late bull
    ("2019-01-01", "2022-01-01", "2022-01-01", "2022-07-01"),  # bear start
    ("2019-07-01", "2022-07-01", "2022-07-01", "2023-01-01"),  # bear bottom
    ("2020-01-01", "2023-01-01", "2023-01-01", "2023-07-01"),  # recovery
    ("2020-07-01", "2023-07-01", "2023-07-01", "2024-07-01"),  # AI bull
]

def _timeout_handler(signum, frame):
    print("\n[TIMEOUT] Experiment exceeded time budget.", flush=True)
    sys.exit(1)

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(EXPERIMENT_TIMEOUT_SECS)


def load_data() -> dict:
    prices = pd.read_parquet(DATA_DIR / "sp500_prices.parquet")
    vix    = pd.read_parquet(DATA_DIR / "vix.parquet")
    fg     = pd.read_parquet(DATA_DIR / "fear_greed.parquet")

    close  = prices["close"]
    open_  = prices["open"]
    high   = prices["high"]
    low    = prices["low"]
    volume = prices["volume"]
    idx    = close.index

    vix_s = vix["vix"].reindex(idx).ffill()
    fg_s  = fg["fg_value"].reindex(idx).ffill()

    # Keep train_end for evaluate.py compatibility (last 2 years = OOS guard)
    train_end = idx[-int(252 * 2)]

    print(f"  Universe  : {close.shape[1]} stocks")
    print(f"  Capital   : ${STARTING_CAPITAL:,.0f} per window")
    print(f"  Commission: ${COMMISSION_PER_TRADE:.0f}/trade  |  Slippage: {SLIPPAGE_BPS}bps")
    print(f"  OOS guard : {train_end.date()} → {idx[-1].date()} (never touched in run.py)")

    return dict(
        close=close, open=open_, high=high, low=low, volume=volume,
        vix=vix_s, fear_greed=fg_s,
        train_end=train_end, index=idx,
    )


def _backtest_window(weights: pd.DataFrame, data: dict,
                     test_start: str, test_end: str) -> dict:
    """Run backtest on a single test window."""
    close  = data["close"]
    open_  = data["open"]
    idx    = close.index

    # Slice to test window
    ts = idx[idx >= test_start][0]
    te = idx[idx < test_end][-1] if any(idx < test_end) else idx[-1]

    close_w = close.loc[ts:te]
    open_w  = open_.loc[ts:te]
    n_days  = len(close_w)

    if n_days < 20:
        return None

    weights_w = (
        weights
        .reindex(close_w.index)
        .reindex(columns=close_w.columns)
        .fillna(0.0)
    )

    slip         = SLIPPAGE_BPS / 10_000.0
    cash         = STARTING_CAPITAL
    shares_held  = pd.Series(0.0, index=close_w.columns)
    prev_w_norm  = pd.Series(0.0, index=close_w.columns)
    port_vals    = []
    comm_total   = slip_total = trade_count = 0.0

    for i in range(n_days):
        px_open    = open_w.iloc[i]
        port_value = cash + (shares_held * px_open).sum()
        port_vals.append(port_value)

        if i == 0:
            continue

        target_w = weights_w.iloc[i - 1]
        gross    = target_w.abs().sum()
        w_norm   = target_w / gross if gross > 0 else target_w

        target_shares = ((w_norm * port_value) / px_open).fillna(0.0)
        target_shares = target_shares.where(px_open > 0, 0.0)
        delta         = target_shares - shares_held
        changed       = (w_norm - prev_w_norm).abs() > 1e-6
        delta         = delta.where(changed, 0.0)
        traded        = delta[delta.abs() > 0.01].index

        trade_cash = 0.0
        for tkr in traded:
            d = delta[tkr]; px = px_open[tkr]
            if px <= 0 or np.isnan(px): continue
            fill_px    = px * (1 + slip) if d > 0 else px * (1 - slip)
            trade_cash -= d * fill_px
            slip_total += abs(d) * px * slip
            comm_total += COMMISSION_PER_TRADE
            trade_count += 1

        shares_held = target_shares.copy()
        prev_w_norm = w_norm.copy()
        cash       += trade_cash - COMMISSION_PER_TRADE * len(traded)

    pv  = pd.Series(port_vals, index=close_w.index)
    r   = pv.pct_change().fillna(0.0)
    ann = 252

    if "SPY" in open_w.columns:
        br = (open_w["SPY"].shift(-1) / open_w["SPY"] - 1).fillna(0.0)
    else:
        br = (open_w.shift(-1) / open_w - 1).fillna(0.0).mean(axis=1)

    def sharpe(r): return (r.mean()/r.std()*np.sqrt(ann)) if r.std()>1e-9 else 0.0
    def calmar(r):
        cum=( 1+r).cumprod(); dd=(cum/cum.cummax()-1).min()
        ar=cum.iloc[-1]**(ann/max(len(r),1))-1
        return ar/abs(dd) if dd<-1e-6 else 0.0
    def maxdd(r): cum=(1+r).cumprod(); return (cum/cum.cummax()-1).min()

    bench_pv  = STARTING_CAPITAL * (1 + br).cumprod()
    total_ret = pv.iloc[-1] / STARTING_CAPITAL - 1
    bench_ret = bench_pv.iloc[-1] / STARTING_CAPITAL - 1

    return dict(
        window      = f"{test_start[:7]}→{test_end[:7]}",
        sharpe      = round(sharpe(r), 3),
        calmar      = round(calmar(r), 3),
        max_dd      = round(maxdd(r)*100, 2),
        total_ret   = round(total_ret*100, 2),
        bench_ret   = round(bench_ret*100, 2),
        final_value = round(pv.iloc[-1], 0),
        trades_yr   = round(trade_count / (n_days/ann), 1),
        cost        = round(comm_total + slip_total, 0),
        n_days      = n_days,
    )


def run_backtest(weights: pd.DataFrame, data: dict,
                 split: str = "validation") -> dict:
    """
    Walk-forward backtest across all WF_WINDOWS.
    Returns aggregate metrics + per-window breakdown.
    The 'split' argument is kept for API compatibility but ignored —
    walk-forward always runs all windows.
    """
    results = []
    for tr_s, tr_e, te_s, te_e in WF_WINDOWS:
        # Pass train context to generate_signals via data dict
        w = _backtest_window(weights, data, te_s, te_e)
        if w:
            results.append(w)

    if not results:
        return {"sharpe": 0.0, "error": "no windows"}

    sharpes  = [r["sharpe"] for r in results]
    calmars  = [r["calmar"] for r in results]
    rets     = [r["total_ret"] for r in results]
    max_dds  = [r["max_dd"] for r in results]
    trades   = [r["trades_yr"] for r in results]
    costs    = [r["cost"] for r in results]
    wins     = sum(1 for r in results if r["total_ret"] > r["bench_ret"])

    agg = dict(
        split            = "walk-forward",
        # Aggregate
        sharpe           = round(float(np.mean(sharpes)), 3),
        sharpe_std       = round(float(np.std(sharpes)), 3),
        sharpe_min       = round(float(np.min(sharpes)), 3),
        sharpe_max       = round(float(np.max(sharpes)), 3),
        calmar           = round(float(np.mean(calmars)), 3),
        max_drawdown     = round(float(np.mean(max_dds)), 2),
        total_return_pct = round(float(np.mean(rets)), 2),
        windows_beat_spy = f"{wins}/{len(results)}",
        trades_per_year  = round(float(np.mean(trades)), 1),
        total_cost       = f"${sum(costs):,.0f}",
        cost_pct_capital = round(sum(costs)/STARTING_CAPITAL*100, 2),
        n_days           = sum(r["n_days"] for r in results),
        # Per-window breakdown
        windows          = results,
    )
    return agg


def print_metrics(m: dict):
    print("\n── Walk-Forward Results ─────────────────────────────────────")
    if "error" in m:
        print(f"  ERROR: {m['error']}")
        return

    print(f"  [Aggregate across {len(m['windows'])} windows]")
    fields = [
        ("Mean Sharpe",     f"{m['sharpe']} ± {m['sharpe_std']}"),
        ("Sharpe range",    f"{m['sharpe_min']} → {m['sharpe_max']}"),
        ("Mean Calmar",     m["calmar"]),
        ("Mean MaxDD",      f"{m['max_drawdown']}%"),
        ("Mean Return",     f"{m['total_return_pct']}%"),
        ("Beat SPY",        m["windows_beat_spy"]),
        ("Trades/yr",       m["trades_per_year"]),
        ("Total cost",      m["total_cost"]),
    ]
    for k, v in fields:
        print(f"    {k:<22}: {v}")

    print(f"\n  [Per-window breakdown]")
    print(f"  {'Window':<18} {'Sharpe':>7} {'Calmar':>7} "
          f"{'MaxDD':>7} {'Ret%':>7} {'vs SPY':>7} {'Trades':>7}")
    print(f"  {'-'*62}")
    for w in m["windows"]:
        spy_beat = "+" if w["total_ret"] > w["bench_ret"] else " "
        print(f"  {w['window']:<18} {w['sharpe']:>7.3f} {w['calmar']:>7.3f} "
              f"{w['max_dd']:>6.1f}% {w['total_ret']:>6.1f}% "
              f"{spy_beat}{w['total_ret']-w['bench_ret']:>5.1f}% "
              f"{w['trades_yr']:>7.1f}")
    print("─────────────────────────────────────────────────────────────\n")
