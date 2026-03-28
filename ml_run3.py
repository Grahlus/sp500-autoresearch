#!/usr/bin/env python3
"""
ml_run3.py — Regime-conditional ML ranking experiments.

Hypothesis: ML improves stock selection in bull regimes only.
In bear regimes (breadth ≤ threshold), fall back to naive momentum rank.
Also tests 1-year training window for faster regime adaptation.

Experiments:
  ML-010: RF         + regime switch breadth=0.50
  ML-011: XGB        + regime switch breadth=0.50
  ML-012: XGB        + regime switch breadth=0.40 (tighter bear)
  ML-013: XGB        + regime switch breadth=0.60 (looser bear)
  ML-014: RF+XGB ens + regime switch breadth=0.50
  ML-015: XGB        + 1-year training window (recency bias)

Success: WF > 0.722 AND W11 (2022H1) Sharpe > -1.5
"""
import sys, signal, warnings
import numpy as np
import pandas as pd
import importlib.util as _ilu

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/mrlearn/sp500-autoresearch')

def timeout_handler(signum, frame):
    print("\n[TIMEOUT]", flush=True); sys.exit(1)
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(90 * 60)

from prepare import load_data, _backtest_window, WF_WINDOWS
from ml_signal import precompute_features, build_training_matrix, FEATURE_COLS
from ml_run import (_get_sp500,
                    LB_DAYS, SKIP_DAYS, MA_DAYS, REBAL_DAYS,
                    MIN_HOLD_DAYS, FG_MIN, EXIT_PCT_RANK,
                    STOP_LOSS_PCT, TOP_PCT, INV_VOL_DAYS)
from ml_signal import predict_rank

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# ── Regime-conditional signal generator ──────────────────────────────────────

def generate_signals_regime(close_sp, volume_sp, fg, features,
                             model, breadth_threshold: float) -> pd.DataFrame:
    """
    Like generate_signals_ml() but adds regime-conditional ranking:
      breadth > threshold → ML rank
      breadth ≤ threshold → momentum rank (fallback)

    Breadth is computed fresh at each rebalance date (no lookahead).
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

        # Daily momentum rank for rank-exit (always momentum, never ML)
        mom_today    = (close_sp.iloc[i - SKIP_DAYS] / close_sp.iloc[i - LB_DAYS] - 1)
        mom_today    = mom_today.replace([np.inf, -np.inf], np.nan)
        ma_today     = close_sp.iloc[max(0, i - MA_DAYS):i].mean()
        above_ma_now = today > ma_today
        avg_vol_now  = volume_sp.iloc[max(0, i - 20):i].mean()
        high_vol_now = avg_vol_now >= avg_vol_now.median()
        filt_mask    = above_ma_now & high_vol_now
        mom_filt     = mom_today.where(filt_mask).dropna()
        rank_today   = mom_filt.rank(pct=True) if len(mom_filt) > 0 else mom_today.rank(pct=True)

        # Rank-based exit (momentum rank)
        for tkr in list(current_pos[current_pos > 0].index):
            days_held = i - int(entry_day.get(tkr, i))
            if days_held >= MIN_HOLD_DAYS:
                tkr_rank = float(rank_today.get(tkr, 1.0)) if tkr in rank_today.index else 1.0
                if tkr_rank < (1.0 - EXIT_PCT_RANK):
                    current_pos[tkr] = 0.0; pos_high[tkr] = np.nan; entry_day[tkr] = -999

        # Adaptive trailing stop
        for tkr in list(current_pos[current_pos > 0].index):
            ph = pos_high.get(tkr, np.nan)
            if not np.isnan(ph) and ph > 0:
                if today[tkr] > ph: pos_high[tkr] = today[tkr]; ph = today[tkr]
                days_held  = i - int(entry_day.get(tkr, i))
                recent_mom = today[tkr] / close_sp.iloc[max(0, i - 20)][tkr] - 1
                stop_pct   = 0.30 if recent_mom > 0.05 else STOP_LOSS_PCT
                if days_held >= MIN_HOLD_DAYS and today[tkr] < ph * (1 - stop_pct):
                    current_pos[tkr] = 0.0; pos_high[tkr] = np.nan; entry_day[tkr] = -999

        # Rebalance
        if i % REBAL_DAYS == 0 and fg_val >= FG_MIN:
            mom      = (close_sp.iloc[i - SKIP_DAYS] / close_sp.iloc[i - LB_DAYS] - 1)
            mom      = mom.replace([np.inf, -np.inf], np.nan)
            ma       = close_sp.iloc[max(0, i - MA_DAYS):i].mean()
            above_ma = today > ma
            avg_vol  = volume_sp.iloc[max(0, i - 20):i].mean()
            high_vol = avg_vol >= avg_vol.median()
            ma_200   = close_sp.iloc[max(0, i - 200):i].mean()
            breadth  = float((today > ma_200).mean())

            filtered = mom.where(above_ma & high_vol).dropna().index

            # ── REGIME SWITCH ─────────────────────────────────────────────────
            if breadth > breadth_threshold and model is not None and len(filtered) > 0:
                # Bull regime: use ML rank
                combo = predict_rank(close_sp, features, model, i, filtered)
                miss  = filtered.difference(combo.index)
                if len(miss) > 0:
                    combo = pd.concat([combo, mom.loc[miss].rank(pct=True)])
            else:
                # Bear regime: naive momentum rank
                combo = mom.rank(pct=True)

            filt = combo.where(above_ma & high_vol).dropna()

            if not filt.empty:
                if breadth < 0.40:      eff_pct = 0.010
                elif fg_val > 70:       eff_pct = 0.015
                else:                   eff_pct = TOP_PCT

                n_top        = max(1, int(len(filt) * eff_pct))
                top_tickers  = filt.nlargest(n_top).index
                vol_ret      = close_sp.iloc[max(0, i - INV_VOL_DAYS):i][top_tickers].pct_change().std()
                inv_vol      = (1.0 / vol_ret.replace(0, np.nan)).fillna(0.0)
                inv_vol_norm = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else inv_vol

                new_pos = pd.Series(0.0, index=tickers)
                for tkr in top_tickers: new_pos[tkr] = inv_vol_norm.get(tkr, 0.0)
                for tkr in top_tickers:
                    if current_pos.get(tkr, 0.0) == 0.0:
                        pos_high[tkr] = today[tkr]; entry_day[tkr] = i
                for tkr in current_pos[current_pos > 0].index:
                    if new_pos[tkr] == 0.0: pos_high[tkr] = np.nan; entry_day[tkr] = -999
                current_pos = new_pos.copy()

        weights.iloc[i] = current_pos

    return weights


# ── Walk-forward runner ───────────────────────────────────────────────────────

def run_exp(label, model_factory, breadth_threshold,
            close_sp, volume_sp, fg, features, data,
            train_days=None):
    """
    train_days: if set, use only last N trading days for training (recency window).
                None = use full 3-year window (default).
    """
    print(f"\n{'='*66}")
    print(f"  {label}")
    print(f"  breadth_thr={breadth_threshold}  train_days={'3yr' if train_days is None else train_days}")
    print(f"{'='*66}")

    all_windows, all_imps = [], []
    # Track W11 (index 10, 0-based) specifically
    w11_sharpe = None

    for w_idx, (train_s, train_e, test_s, test_e) in enumerate(WF_WINDOWS):
        print(f"  W{w_idx+1:02d}: {test_s[:7]}→{test_e[:7]} ... ", end='', flush=True)

        # Adjust train_start for recency window
        if train_days is not None:
            idx = close_sp.index
            te_loc = (idx <= train_e).sum() - 1
            ts_loc = max(0, te_loc - train_days)
            effective_train_s = str(idx[ts_loc].date())
        else:
            effective_train_s = train_s

        X_tr, y_tr, nd = build_training_matrix(
            close_sp, features, effective_train_s, train_e)

        if len(X_tr) < 200:   # lower floor for 1yr window
            print(f"SKIP (n={len(X_tr)})"); continue

        model = model_factory()
        model.fit(X_tr, y_tr)

        yh = model.predict(X_tr)
        ss_res = np.sum((y_tr - yh)**2); ss_tot = np.sum((y_tr - y_tr.mean())**2)
        train_r2 = 1 - ss_res/ss_tot if ss_tot > 1e-12 else 0.0

        if hasattr(model, 'feature_importances_'):
            all_imps.append(pd.Series(model.feature_importances_, index=FEATURE_COLS))

        weights = generate_signals_regime(
            close_sp, volume_sp, fg, features, model, breadth_threshold)
        wm = _backtest_window(weights, data, test_s, test_e)
        if wm is None:
            print("NO RESULT"); continue

        all_windows.append(wm)
        tag = " ← W11" if w_idx == 10 else ""
        print(f"n={len(X_tr):,}  R²={train_r2:.3f}  Sharpe={wm['sharpe']:+.3f}  MaxDD={wm['max_dd']:.1f}%{tag}")

        if w_idx == 10:
            w11_sharpe = wm['sharpe']

    if not all_windows:
        print("  NO RESULTS"); return {}

    sharpes     = [w['sharpe'] for w in all_windows]
    mean_sharpe = float(np.mean(sharpes))
    neg_win     = sum(1 for s in sharpes if s < 0)
    worst       = min(sharpes)
    trades_yr   = float(np.mean([w['trades_yr'] for w in all_windows]))
    beat_spy    = sum(1 for w in all_windows if w['total_ret'] > w['bench_ret'])

    print(f"\n  MEAN SHARPE   : {mean_sharpe:.3f}")
    print(f"  RANGE         : {worst:.3f} → {max(sharpes):.3f}")
    print(f"  NEG WINDOWS   : {neg_win}/{len(all_windows)}")
    print(f"  BEAT SPY      : {beat_spy}/{len(all_windows)}")
    print(f"  W11 (2022H1)  : {w11_sharpe if w11_sharpe is not None else 'N/A'}")

    if all_imps:
        mean_imp = pd.concat(all_imps, axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"  TOP FEATURES  : {', '.join(f'{k}={v:.3f}' for k,v in mean_imp.head(5).items())}")
    else:
        mean_imp = None

    violations = []
    if mean_sharpe < 0.3:   violations.append(f"MEAN={mean_sharpe:.3f}<0.3")
    if neg_win     > 3:     violations.append(f"NEG_WIN={neg_win}>3")
    if worst       < -1.0:  violations.append(f"WORST={worst:.3f}<-1.0")
    if trades_yr   > 150:   violations.append(f"TRADES={trades_yr:.0f}>150")
    if violations: print(f"  !! VIOLATIONS: {', '.join(violations)}")
    else:          print(f"  [OK] All constraints passed")

    # Regime-switch effectiveness check
    if w11_sharpe is not None:
        if w11_sharpe > -1.5:
            print(f"  [OK] W11 regime switch working: {w11_sharpe:.3f} > -1.5")
        else:
            print(f"  [!!] W11 still bad: {w11_sharpe:.3f} ≤ -1.5 (regime switch not helping)")

    return dict(label=label, mean_sharpe=mean_sharpe, worst=worst,
                neg_windows=neg_win, trades_yr=trades_yr, beat_spy=beat_spy,
                n_windows=len(all_windows), violations=violations,
                w11_sharpe=w11_sharpe,
                importances=mean_imp)


# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\n{'='*66}")
print("  SESSION ML-2: Regime-conditional ML ranking")
print(f"{'='*66}")
print("  Prior results: ML-001..009 all failed WF=0.722")
print("  Fix: bear-regime fallback to momentum rank")

data      = load_data()
close     = data['close']
volume    = data['volume']
vix_s     = data['vix']
fg_s      = data['fear_greed']

sp500     = _get_sp500(close.columns)
close_sp  = close[[c for c in close.columns if c in sp500]]
volume_sp = volume[close_sp.columns]
fg        = fg_s.reindex(close_sp.index).ffill().fillna(50.0).values

print(f">>> SP500: {close_sp.shape[1]} stocks — precomputing features …")
features  = precompute_features(close_sp, volume_sp, vix_s, fg_s)
print(">>> Ready.\n")

# Baseline W11 check from prior runs: ~-2.0 to -2.9 for pure ML
# Target: W11 > -1.5 to confirm regime switch is working

def rf_factory():
    return RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        max_features=0.5, random_state=42, n_jobs=-1)

def xgb_factory():
    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        n_jobs=-1, verbosity=0)

all_results = []

# ML-010: RF + breadth=0.50 switch
r10 = run_exp("ML-010: RF + regime-switch breadth=0.50",
              rf_factory, 0.50, close_sp, volume_sp, fg, features, data)
all_results.append(r10)

# ML-011: XGB + breadth=0.50 switch
r11 = run_exp("ML-011: XGB + regime-switch breadth=0.50",
              xgb_factory, 0.50, close_sp, volume_sp, fg, features, data)
all_results.append(r11)

# ML-012: XGB + breadth=0.40 (tighter bear guard)
r12 = run_exp("ML-012: XGB + regime-switch breadth=0.40",
              xgb_factory, 0.40, close_sp, volume_sp, fg, features, data)
all_results.append(r12)

# ML-013: XGB + breadth=0.60 (looser bear guard)
r13 = run_exp("ML-013: XGB + regime-switch breadth=0.60",
              xgb_factory, 0.60, close_sp, volume_sp, fg, features, data)
all_results.append(r13)

# ML-014: RF+XGB ensemble + breadth=0.50
print(f"\n{'='*66}")
print("  ML-014: RF+XGB ensemble + regime-switch breadth=0.50")
print(f"{'='*66}")

all_w14, all_i14 = [], []
w11_sharpe_14 = None

for w_idx, (train_s, train_e, test_s, test_e) in enumerate(WF_WINDOWS):
    print(f"  W{w_idx+1:02d}: {test_s[:7]}→{test_e[:7]} ... ", end='', flush=True)

    X_tr, y_tr, nd = build_training_matrix(close_sp, features, train_s, train_e)
    if len(X_tr) < 500:
        print(f"SKIP"); continue

    rf  = rf_factory();  rf.fit(X_tr, y_tr)
    xgb = xgb_factory(); xgb.fit(X_tr, y_tr)

    class EnsModel:
        def __init__(self, m1, m2): self.m1, self.m2 = m1, m2
        def predict(self, X):
            p1 = pd.Series(self.m1.predict(X)).rank(pct=True).values
            p2 = pd.Series(self.m2.predict(X)).rank(pct=True).values
            return 0.5 * p1 + 0.5 * p2
        @property
        def feature_importances_(self):
            return (self.m1.feature_importances_ + self.m2.feature_importances_) / 2

    ens = EnsModel(rf, xgb)
    all_i14.append(pd.Series(ens.feature_importances_, index=FEATURE_COLS))

    weights = generate_signals_regime(close_sp, volume_sp, fg, features, ens, 0.50)
    wm = _backtest_window(weights, data, test_s, test_e)
    if wm is None: print("NO RESULT"); continue
    all_w14.append(wm)
    tag = " ← W11" if w_idx == 10 else ""
    print(f"Sharpe={wm['sharpe']:+.3f}  MaxDD={wm['max_dd']:.1f}%{tag}")
    if w_idx == 10: w11_sharpe_14 = wm['sharpe']

if all_w14:
    sh14 = [w['sharpe'] for w in all_w14]
    ms14 = float(np.mean(sh14)); wr14 = min(sh14)
    nw14 = sum(1 for s in sh14 if s < 0)
    ty14 = float(np.mean([w['trades_yr'] for w in all_w14]))
    bs14 = sum(1 for w in all_w14 if w['total_ret'] > w['bench_ret'])
    print(f"\n  MEAN SHARPE   : {ms14:.3f}")
    print(f"  RANGE         : {wr14:.3f} → {max(sh14):.3f}")
    print(f"  NEG WINDOWS   : {nw14}/{len(all_w14)}")
    print(f"  W11 (2022H1)  : {w11_sharpe_14}")
    if all_i14:
        mi14 = pd.concat(all_i14, axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"  TOP FEATURES  : {', '.join(f'{k}={v:.3f}' for k,v in mi14.head(5).items())}")
    v14 = []
    if ms14 < 0.3: v14.append(f"MEAN={ms14:.3f}<0.3")
    if nw14  > 3:  v14.append(f"NEG={nw14}>3")
    if wr14 < -1.0: v14.append(f"WORST={wr14:.3f}<-1.0")
    if ty14 > 150: v14.append(f"TRADES={ty14:.0f}>150")
    if v14: print(f"  !! VIOLATIONS: {', '.join(v14)}")
    else:   print(f"  [OK] All constraints passed")
    if w11_sharpe_14 is not None:
        flag = "[OK]" if w11_sharpe_14 > -1.5 else "[!!]"
        print(f"  {flag} W11={w11_sharpe_14:.3f}")

    all_results.append(dict(label="ML-014: Ensemble+regime 0.50",
                            mean_sharpe=ms14, worst=wr14, neg_windows=nw14,
                            trades_yr=ty14, beat_spy=bs14, n_windows=len(all_w14),
                            violations=v14, w11_sharpe=w11_sharpe_14,
                            importances=mi14 if all_i14 else None))

# ML-015: XGB + 1-year training window
r15 = run_exp("ML-015: XGB + 1yr training window (recency)",
              xgb_factory, 0.50, close_sp, volume_sp, fg, features, data,
              train_days=252)
all_results.append(r15)

# Also try: ML-016 XGB + 1yr window + full breadth fallback (breadth=0.50)
# (already in r15), plus XGB 1yr window with no breadth filter (breadth=0.0 → always ML)
r16 = run_exp("ML-016: XGB + 1yr window + no regime filter",
              xgb_factory, 0.0, close_sp, volume_sp, fg, features, data,
              train_days=252)
all_results.append(r16)

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*66}")
print("  SESSION ML-2 SUMMARY")
print(f"{'='*66}")
print(f"  {'Label':<40} {'WF':>6} {'Worst':>7} {'Neg':>5} {'W11':>7}")
print("  " + "-"*70)
print(f"  {'S10-005 baseline':40} {'0.722':>6} {'-0.610':>7} {'2':>5} {'?':>7}")
print(f"  {'ML-002 (best prior, no regime)':40} {'0.573':>6} {'-2.109':>7} {'3':>5} {'~-2.5':>7}")
for r in all_results:
    if not r: continue
    tag  = " !!" if r.get('violations') else " OK"
    w11s = f"{r['w11_sharpe']:.3f}" if r.get('w11_sharpe') is not None else "N/A"
    print(f"  {r['label']:<40} {r['mean_sharpe']:>6.3f} {r['worst']:>7.3f} "
          f"{r['neg_windows']:>5} {w11s:>7}{tag}")

valid = [r for r in all_results if r and not r.get('violations')]
if valid:
    best = max(valid, key=lambda r: r['mean_sharpe'])
    print(f"\n  >>> BEST VALID: {best['label']}  WF={best['mean_sharpe']:.3f}")
    if best['mean_sharpe'] > 0.722:
        print("  >>> BEATS BASELINE 0.722 — update agent.py and run evaluate.py")
    else:
        print("  >>> Does not beat baseline 0.722")
else:
    print("\n  >>> No model passed all constraints")

signal.alarm(0)
