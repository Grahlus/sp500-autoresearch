#!/usr/bin/env python3
"""
ml_run.py — Walk-forward ML signal experiments.

Replaces naive momentum rank in S10-005 with ML-predicted forward return.
Each WF window: train on [train_s, train_e], generate signals on [test_s, test_e].
Strict no-lookahead: model never sees test data during training.

Experiments:
  ML-001: Random Forest
  ML-002: XGBoost
  ML-003: LightGBM
  ML-004: Ensemble (RF + XGBoost average rank)

Usage: uv run python ml_run.py
"""
import sys, signal
import numpy as np
import pandas as pd
import importlib.util as _ilu

sys.path.insert(0, '/home/mrlearn/sp500-autoresearch')

def timeout_handler(signum, frame):
    print("\n[TIMEOUT]", flush=True); sys.exit(1)
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(90 * 60)

from prepare import load_data, _backtest_window, WF_WINDOWS, STARTING_CAPITAL
from ml_signal import (precompute_features, build_training_matrix,
                       predict_rank, FEATURE_COLS)

# Import S10-005 agent for SP500 filter and non-ranking logic
import requests
from io import StringIO

def _get_sp500(close_cols) -> set:
    try:
        url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        tbl  = pd.read_html(StringIO(resp.text), attrs={"id": "constituents"})[0]
        sp500 = set(tbl["Symbol"].str.replace(".", "-", regex=False).tolist())
        sp500.add("SPY")
        return sp500
    except Exception:
        return set(close_cols)

# ── S10-005 parameters (frozen) ───────────────────────────────────────────────
LOOKBACK_WEEKS = 26
SKIP_WEEKS     = 3
REBAL_WEEKS    = 4
TOP_PCT        = 0.025
MA_WEEKS       = 20
STOP_LOSS_PCT  = 0.20
INV_VOL_DAYS   = 15
MIN_HOLD_DAYS  = 5
FG_MIN         = 10.0
EXIT_PCT_RANK  = 0.97

LB_DAYS    = LOOKBACK_WEEKS * 5   # 130
SKIP_DAYS  = SKIP_WEEKS * 5       # 15
MA_DAYS    = MA_WEEKS * 5         # 100
REBAL_DAYS = REBAL_WEEKS * 5      # 20


def generate_signals_ml(close_sp, volume_sp, fg, features, model,
                        train_end: str) -> pd.DataFrame:
    """
    Like S10-005 generate_signals() but replaces:
        combo = mom.rank(pct=True)
    with:
        combo = predict_rank(features, model, date_loc, filtered_tickers)

    train_end: model is valid for dates > train_end (already trained).
    All other logic (stops, rank exit, MA/vol filter, inv-vol weighting) unchanged.
    """
    dates   = close_sp.index
    tickers = close_sp.columns
    n       = len(dates)

    weights     = pd.DataFrame(0.0, index=dates, columns=tickers)
    pos_high    = pd.Series(np.nan, index=tickers)
    entry_day   = pd.Series(-999,   index=tickers)
    current_pos = pd.Series(0.0,    index=tickers)

    for i in range(LB_DAYS, n):
        today  = close_sp.iloc[i]
        fg_val = float(fg[i])

        # Daily momentum rank for rank-exit check
        mom_today  = (close_sp.iloc[i - SKIP_DAYS] / close_sp.iloc[i - LB_DAYS] - 1)
        mom_today  = mom_today.replace([np.inf, -np.inf], np.nan)
        ma_today   = close_sp.iloc[max(0, i - MA_DAYS):i].mean()
        above_ma_now = close_sp.iloc[i] > ma_today
        avg_vol_now  = volume_sp.iloc[max(0, i - 20):i].mean()
        high_vol_now = avg_vol_now >= avg_vol_now.median()
        filt_mask    = above_ma_now & high_vol_now
        mom_filt     = mom_today.where(filt_mask).dropna()
        if len(mom_filt) > 0:
            rank_today = mom_filt.rank(pct=True)
        else:
            rank_today = mom_today.rank(pct=True)

        # Rank-based exit (same as S10-005)
        for tkr in list(current_pos[current_pos > 0].index):
            days_held = i - int(entry_day.get(tkr, i))
            if days_held >= MIN_HOLD_DAYS:
                tkr_rank = float(rank_today.get(tkr, 1.0)) if tkr in rank_today.index else 1.0
                if tkr_rank < (1.0 - EXIT_PCT_RANK):
                    current_pos[tkr] = 0.0
                    pos_high[tkr]    = np.nan
                    entry_day[tkr]   = -999

        # Adaptive trailing stop (same as S10-005)
        for tkr in list(current_pos[current_pos > 0].index):
            ph = pos_high.get(tkr, np.nan)
            if not np.isnan(ph) and ph > 0:
                if today[tkr] > ph:
                    pos_high[tkr] = today[tkr]
                    ph = today[tkr]
                days_held  = i - int(entry_day.get(tkr, i))
                recent_mom = today[tkr] / close_sp.iloc[max(0, i - 20)][tkr] - 1
                stop_pct   = 0.30 if recent_mom > 0.05 else STOP_LOSS_PCT
                if days_held >= MIN_HOLD_DAYS and today[tkr] < ph * (1 - stop_pct):
                    current_pos[tkr] = 0.0
                    pos_high[tkr]    = np.nan
                    entry_day[tkr]   = -999

        # Rebalance (every 20 trading days when F&G >= 10)
        if i % REBAL_DAYS == 0 and fg_val >= FG_MIN:
            mom   = (close_sp.iloc[i - SKIP_DAYS] / close_sp.iloc[i - LB_DAYS] - 1)
            mom   = mom.replace([np.inf, -np.inf], np.nan)
            ma    = close_sp.iloc[max(0, i - MA_DAYS):i].mean()
            above_ma = today > ma
            avg_vol  = volume_sp.iloc[max(0, i - 20):i].mean()
            high_vol = avg_vol >= avg_vol.median()

            ma_200  = close_sp.iloc[max(0, i - 200):i].mean()
            breadth = float((today > ma_200).mean())

            # ── ML RANKING (replaces mom.rank(pct=True)) ─────────────────────
            above_ma_high_vol = above_ma & high_vol
            filtered_tickers  = mom.where(above_ma_high_vol).dropna().index

            if model is not None and len(filtered_tickers) > 0:
                combo = predict_rank(close_sp, features, model, i, filtered_tickers)
                # Fallback: any filtered tickers not covered by ML
                missing = filtered_tickers.difference(combo.index)
                if len(missing) > 0:
                    mom_fb = mom.loc[missing].rank(pct=True)
                    combo  = pd.concat([combo, mom_fb])
            else:
                # Baseline fallback (model=None → pure momentum)
                combo = mom.rank(pct=True)

            filt = combo.where(above_ma_high_vol).dropna()

            if not filt.empty:
                if breadth < 0.40:
                    eff_pct = 0.010
                elif fg_val > 70:
                    eff_pct = 0.015
                else:
                    eff_pct = TOP_PCT

                n_top       = max(1, int(len(filt) * eff_pct))
                top_tickers = filt.nlargest(n_top).index

                vol_ret      = close_sp.iloc[max(0, i - INV_VOL_DAYS):i][top_tickers].pct_change().std()
                inv_vol      = (1.0 / vol_ret.replace(0, np.nan)).fillna(0.0)
                inv_vol_norm = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else inv_vol

                new_pos = pd.Series(0.0, index=tickers)
                for tkr in top_tickers:
                    new_pos[tkr] = inv_vol_norm.get(tkr, 0.0)

                for tkr in top_tickers:
                    if current_pos.get(tkr, 0.0) == 0.0:
                        pos_high[tkr]  = today[tkr]
                        entry_day[tkr] = i
                for tkr in current_pos[current_pos > 0].index:
                    if new_pos[tkr] == 0.0:
                        pos_high[tkr]  = np.nan
                        entry_day[tkr] = -999

                current_pos = new_pos.copy()

        weights.iloc[i] = current_pos

    return weights


def run_ml_experiment(model_name: str, model_factory,
                      close_sp, volume_sp, fg, features,
                      data: dict) -> dict:
    """
    Full walk-forward: for each window, train on [train_s, train_e], test on [test_s, test_e].
    Returns aggregate and per-window results.
    """
    print(f"\n{'='*66}")
    print(f"  {model_name}")
    print(f"{'='*66}")

    all_windows = []
    all_importances = []

    for w_idx, (train_s, train_e, test_s, test_e) in enumerate(WF_WINDOWS):
        print(f"  Window {w_idx+1:2d}/{len(WF_WINDOWS)}: train→{train_e[:7]}  test {test_s[:7]}→{test_e[:7]} ... ",
              end='', flush=True)

        # ── Train ────────────────────────────────────────────────────────────
        X_train, y_train, n_dates = build_training_matrix(
            close_sp, features, train_s, train_e)

        if len(X_train) < 500:
            print(f"SKIP (only {len(X_train)} training rows)")
            continue

        model = model_factory()
        model.fit(X_train, y_train)

        # Train R² check
        y_hat_train = model.predict(X_train)
        ss_res = np.sum((y_train - y_hat_train) ** 2)
        ss_tot = np.sum((y_train - y_train.mean()) ** 2)
        train_r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        # Feature importances (tree models)
        if hasattr(model, 'feature_importances_'):
            imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
            all_importances.append(imp)

        print(f"n={len(X_train):,}  trainR²={train_r2:.4f}", end='  ', flush=True)

        # ── Generate signals for test window ─────────────────────────────────
        # We need weights for the FULL dataset up to test_e for _backtest_window
        # But generate_signals_ml takes the full close/volume — just filter by date later
        # For efficiency: generate weights for the slice [test_s start, test_e]
        # then score with _backtest_window which slices anyway

        # Generate signals: pass model trained on train_end
        weights_full = generate_signals_ml(
            close_sp, volume_sp, fg, features, model, train_e)

        # Score this window
        wm = _backtest_window(weights_full, data, test_s, test_e)
        if wm is None:
            print("NO RESULT")
            continue

        all_windows.append(wm)
        print(f"Sharpe={wm['sharpe']:+.3f}  MaxDD={wm['max_dd']:.1f}%")

    if not all_windows:
        return {}

    sharpes      = [w['sharpe'] for w in all_windows]
    mean_sharpe  = float(np.mean(sharpes))
    neg_windows  = sum(1 for s in sharpes if s < 0)
    worst_window = min(sharpes)
    best_window  = max(sharpes)
    trades_yr    = float(np.mean([w['trades_yr'] for w in all_windows]))
    beat_spy     = sum(1 for w in all_windows if w['total_ret'] > w['bench_ret'])

    print(f"\n  MEAN SHARPE  : {mean_sharpe:.3f}")
    print(f"  RANGE        : {worst_window:.3f} → {best_window:.3f}")
    print(f"  NEG WINDOWS  : {neg_windows}/{len(all_windows)}")
    print(f"  BEAT SPY     : {beat_spy}/{len(all_windows)}")
    print(f"  TRADES/YR    : {trades_yr:.1f}")

    # Aggregate feature importances
    if all_importances:
        mean_imp = pd.concat(all_importances, axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"  TOP FEATURES : {', '.join(f'{k}={v:.3f}' for k,v in mean_imp.head(5).items())}")

    violations = []
    if mean_sharpe  < 0.3:      violations.append(f"MEAN_SHARPE={mean_sharpe:.3f}<0.3")
    if neg_windows  > 3:        violations.append(f"NEG_WIN={neg_windows}>3")
    if worst_window < -1.0:     violations.append(f"WORST={worst_window:.3f}<-1.0")
    if trades_yr    > 150:      violations.append(f"TRADES={trades_yr:.0f}>150")

    if violations:
        print(f"  !! VIOLATIONS: {', '.join(violations)}")
    else:
        print(f"  [OK] All constraints passed")

    return dict(
        model_name=model_name, mean_sharpe=mean_sharpe,
        worst=worst_window, neg_windows=neg_windows,
        trades_yr=trades_yr, beat_spy=beat_spy,
        n_windows=len(all_windows), violations=violations,
        importances=mean_imp if all_importances else None,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*66}")
print("  ML MOMENTUM RANKING EXPERIMENTS")
print(f"{'='*66}")
print("  Baseline (S10-005): WF Sharpe=0.722  OOS Sharpe=1.548")
print("  Goal: WF > 0.722, neg_windows ≤ 3, worst > -1.0")

print("\n>>> Loading data …")
data    = load_data()
close   = data['close']
volume  = data['volume']
vix_s   = data['vix']
fg_s    = data['fear_greed']

# SP500 filter
print(">>> Filtering SP500 universe …")
sp500   = _get_sp500(close.columns)
close_sp  = close[[c for c in close.columns if c in sp500]]
volume_sp = volume[close_sp.columns]

fg = fg_s.reindex(close_sp.index).ffill().fillna(50.0).values

print(f">>> SP500 universe: {close_sp.shape[1]} stocks")
print(">>> Precomputing features …")
features = precompute_features(close_sp, volume_sp, vix_s, fg_s)
print(">>> Features ready. Starting experiments …")

# ── Model factories ───────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def rf_factory():
    return RandomForestRegressor(
        n_estimators=200, max_depth=6,
        min_samples_leaf=20, max_features=0.5,
        random_state=42, n_jobs=-1
    )

def xgb_factory():
    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0,
    )

def lgbm_factory():
    return LGBMRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        num_leaves=31, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )

# ── Run experiments ───────────────────────────────────────────────────────────
all_results = []

r1 = run_ml_experiment("ML-001: Random Forest",    rf_factory,
                       close_sp, volume_sp, fg, features, data)
all_results.append(r1)

r2 = run_ml_experiment("ML-002: XGBoost",          xgb_factory,
                       close_sp, volume_sp, fg, features, data)
all_results.append(r2)

r3 = run_ml_experiment("ML-003: LightGBM",         lgbm_factory,
                       close_sp, volume_sp, fg, features, data)
all_results.append(r3)

# ── ML-004: Ensemble (RF + XGB average rank) ──────────────────────────────────
print(f"\n{'='*66}")
print("  ML-004: Ensemble (RF + XGB average predicted rank)")
print(f"{'='*66}")

all_windows_ens = []
all_importances_ens = []

for w_idx, (train_s, train_e, test_s, test_e) in enumerate(WF_WINDOWS):
    print(f"  Window {w_idx+1:2d}/{len(WF_WINDOWS)}: train→{train_e[:7]}  test {test_s[:7]}→{test_e[:7]} ... ",
          end='', flush=True)

    from ml_signal import build_training_matrix as _btm

    X_train, y_train, n_dates = _btm(close_sp, features, train_s, train_e)
    if len(X_train) < 500:
        print(f"SKIP")
        continue

    rf_model  = rf_factory();  rf_model.fit(X_train, y_train)
    xgb_model = xgb_factory(); xgb_model.fit(X_train, y_train)

    # Ensemble predict: average of rank of predictions
    class EnsembleModel:
        def __init__(self, m1, m2):
            self.m1, self.m2 = m1, m2
        def predict(self, X):
            p1 = pd.Series(self.m1.predict(X)).rank(pct=True).values
            p2 = pd.Series(self.m2.predict(X)).rank(pct=True).values
            return 0.5 * p1 + 0.5 * p2
        @property
        def feature_importances_(self):
            return (self.m1.feature_importances_ + self.m2.feature_importances_) / 2

    ens_model = EnsembleModel(rf_model, xgb_model)

    imp = pd.Series(ens_model.feature_importances_, index=FEATURE_COLS)
    all_importances_ens.append(imp)

    weights_full = generate_signals_ml(
        close_sp, volume_sp, fg, features, ens_model, train_e)

    wm = _backtest_window(weights_full, data, test_s, test_e)
    if wm is None:
        print("NO RESULT")
        continue

    all_windows_ens.append(wm)
    print(f"Sharpe={wm['sharpe']:+.3f}  MaxDD={wm['max_dd']:.1f}%")

if all_windows_ens:
    sharpes_e    = [w['sharpe'] for w in all_windows_ens]
    mean_sharpe_e = float(np.mean(sharpes_e))
    neg_win_e    = sum(1 for s in sharpes_e if s < 0)
    worst_e      = min(sharpes_e)
    beat_spy_e   = sum(1 for w in all_windows_ens if w['total_ret'] > w['bench_ret'])
    trades_yr_e  = float(np.mean([w['trades_yr'] for w in all_windows_ens]))

    print(f"\n  MEAN SHARPE  : {mean_sharpe_e:.3f}")
    print(f"  RANGE        : {worst_e:.3f} → {max(sharpes_e):.3f}")
    print(f"  NEG WINDOWS  : {neg_win_e}/{len(all_windows_ens)}")
    print(f"  BEAT SPY     : {beat_spy_e}/{len(all_windows_ens)}")

    if all_importances_ens:
        mean_imp_e = pd.concat(all_importances_ens, axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"  TOP FEATURES : {', '.join(f'{k}={v:.3f}' for k,v in mean_imp_e.head(5).items())}")

    violations_e = []
    if mean_sharpe_e < 0.3:  violations_e.append(f"MEAN_SHARPE={mean_sharpe_e:.3f}<0.3")
    if neg_win_e     > 3:    violations_e.append(f"NEG_WIN={neg_win_e}>3")
    if worst_e       < -1.0: violations_e.append(f"WORST={worst_e:.3f}<-1.0")
    if trades_yr_e   > 150:  violations_e.append(f"TRADES={trades_yr_e:.0f}>150")
    if violations_e:
        print(f"  !! VIOLATIONS: {', '.join(violations_e)}")
    else:
        print(f"  [OK] All constraints passed")

    all_results.append(dict(
        model_name="ML-004: Ensemble (RF+XGB)", mean_sharpe=mean_sharpe_e,
        worst=worst_e, neg_windows=neg_win_e,
        trades_yr=trades_yr_e, beat_spy=beat_spy_e,
        n_windows=len(all_windows_ens), violations=violations_e,
        importances=mean_imp_e if all_importances_ens else None,
    ))

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*66}")
print("  FINAL SUMMARY — ML vs Baseline (S10-005 WF=0.722)")
print(f"{'='*66}")
print(f"  {'Model':<30} {'WF Sharpe':>10} {'Worst':>7} {'NegWin':>7} {'BeatSPY':>8}")
print("  " + "-"*64)
print(f"  {'S10-005 baseline (momentum rank)':<30} {'0.722':>10} {'-0.610':>7} {'2':>7} {'?':>8}")
for r in all_results:
    if not r:
        continue
    viol_str = " !!" if r['violations'] else " OK"
    bspy = f"{r['beat_spy']}/{r['n_windows']}"
    print(f"  {r['model_name']:<30} {r['mean_sharpe']:>10.3f} {r['worst']:>7.3f} "
          f"{r['neg_windows']:>7} {bspy:>8}{viol_str}")

best_valid = [r for r in all_results if r and not r['violations']]
if best_valid:
    champion = max(best_valid, key=lambda r: r['mean_sharpe'])
    print(f"\n  >>> BEST: {champion['model_name']}  WF={champion['mean_sharpe']:.3f}")
    if champion['mean_sharpe'] > 0.722:
        print(f"  >>> BEATS BASELINE — run evaluate.py to check OOS")
    else:
        print(f"  >>> Does not beat baseline WF=0.722 — no commit")
else:
    print("\n  >>> No model passed all constraints")

signal.alarm(0)
