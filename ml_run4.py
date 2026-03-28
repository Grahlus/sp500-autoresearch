#!/usr/bin/env python3
"""
ml_run4.py — Targeted experiments to fix W04 (2018H2) worst window.

ML-013 (XGB breadth=0.60): WF=0.683, WORST=-1.320 (W04). Fails worst<-1.2.
Strategy: push breadth threshold higher to reduce ML exposure in moderate-bear.

Also try:
  - Soft blend (0.7*mom + 0.3*ml) instead of hard switch
  - 1-year window + breadth=0.60 combination
  - breadth=0.65, breadth=0.70

Note: actual run.py worst constraint is < -1.2, not -1.0.
Success: WF > 0.722 AND worst > -1.2 AND neg_win ≤ 4.
"""
import sys, signal, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/mrlearn/sp500-autoresearch')

def timeout_handler(signum, frame):
    print("\n[TIMEOUT]", flush=True); sys.exit(1)
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(90 * 60)

from prepare import load_data, _backtest_window, WF_WINDOWS
from ml_signal import precompute_features, build_training_matrix, predict_rank, FEATURE_COLS
from ml_run import (_get_sp500,
                    LB_DAYS, SKIP_DAYS, MA_DAYS, REBAL_DAYS,
                    MIN_HOLD_DAYS, FG_MIN, EXIT_PCT_RANK,
                    STOP_LOSS_PCT, TOP_PCT, INV_VOL_DAYS)
from xgboost import XGBRegressor


def generate_signals_regime_blend(close_sp, volume_sp, fg, features,
                                   model, breadth_threshold, blend_alpha=1.0):
    """
    breadth_threshold: if breadth ≤ thr → use 0% ML (pure momentum)
                       if breadth > thr → use blend_alpha*ML + (1-blend_alpha)*momentum
    blend_alpha=1.0 → hard switch (ML only in bull)
    blend_alpha=0.5 → soft blend (50% ML in bull)
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

        mom_today    = (close_sp.iloc[i - SKIP_DAYS] / close_sp.iloc[i - LB_DAYS] - 1)
        mom_today    = mom_today.replace([np.inf, -np.inf], np.nan)
        ma_today     = close_sp.iloc[max(0, i - MA_DAYS):i].mean()
        above_ma_now = today > ma_today
        avg_vol_now  = volume_sp.iloc[max(0, i - 20):i].mean()
        high_vol_now = avg_vol_now >= avg_vol_now.median()
        filt_mask    = above_ma_now & high_vol_now
        mom_filt     = mom_today.where(filt_mask).dropna()
        rank_today   = mom_filt.rank(pct=True) if len(mom_filt) > 0 else mom_today.rank(pct=True)

        for tkr in list(current_pos[current_pos > 0].index):
            days_held = i - int(entry_day.get(tkr, i))
            if days_held >= MIN_HOLD_DAYS:
                tkr_rank = float(rank_today.get(tkr, 1.0)) if tkr in rank_today.index else 1.0
                if tkr_rank < (1.0 - EXIT_PCT_RANK):
                    current_pos[tkr] = 0.0; pos_high[tkr] = np.nan; entry_day[tkr] = -999

        for tkr in list(current_pos[current_pos > 0].index):
            ph = pos_high.get(tkr, np.nan)
            if not np.isnan(ph) and ph > 0:
                if today[tkr] > ph: pos_high[tkr] = today[tkr]; ph = today[tkr]
                days_held  = i - int(entry_day.get(tkr, i))
                recent_mom = today[tkr] / close_sp.iloc[max(0, i - 20)][tkr] - 1
                stop_pct   = 0.30 if recent_mom > 0.05 else STOP_LOSS_PCT
                if days_held >= MIN_HOLD_DAYS and today[tkr] < ph * (1 - stop_pct):
                    current_pos[tkr] = 0.0; pos_high[tkr] = np.nan; entry_day[tkr] = -999

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
            mom_rank = mom.rank(pct=True)

            if breadth > breadth_threshold and model is not None and len(filtered) > 0:
                ml_rank = predict_rank(close_sp, features, model, i, filtered)
                miss    = filtered.difference(ml_rank.index)
                if len(miss) > 0:
                    ml_rank = pd.concat([ml_rank, mom.loc[miss].rank(pct=True)])
                # Blend: alpha*ML + (1-alpha)*momentum
                if blend_alpha < 1.0:
                    both = ml_rank.index.intersection(mom_rank.index)
                    blended = blend_alpha * ml_rank.loc[both] + (1 - blend_alpha) * mom_rank.loc[both]
                    combo = blended.rank(pct=True)
                else:
                    combo = ml_rank
            else:
                combo = mom_rank

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


def run_exp(label, model_factory, breadth_thr, blend_alpha,
            close_sp, volume_sp, fg, features, data, train_days=None):
    print(f"\n{'='*66}")
    print(f"  {label}")
    print(f"  breadth_thr={breadth_thr}  blend_alpha={blend_alpha}  train={'3yr' if train_days is None else f'{train_days}d'}")
    print(f"{'='*66}")

    all_windows, all_imps = [], []
    w11_sharpe = None

    for w_idx, (train_s, train_e, test_s, test_e) in enumerate(WF_WINDOWS):
        print(f"  W{w_idx+1:02d}: {test_s[:7]}→{test_e[:7]} ... ", end='', flush=True)

        if train_days is not None:
            idx    = close_sp.index
            te_loc = (idx <= train_e).sum() - 1
            ts_loc = max(0, te_loc - train_days)
            eff_ts = str(idx[ts_loc].date())
        else:
            eff_ts = train_s

        X_tr, y_tr, _ = build_training_matrix(close_sp, features, eff_ts, train_e)
        if len(X_tr) < 200:
            print(f"SKIP (n={len(X_tr)})"); continue

        model = model_factory()
        model.fit(X_tr, y_tr)

        if hasattr(model, 'feature_importances_'):
            all_imps.append(pd.Series(model.feature_importances_, index=FEATURE_COLS))

        weights = generate_signals_regime_blend(
            close_sp, volume_sp, fg, features, model, breadth_thr, blend_alpha)
        wm = _backtest_window(weights, data, test_s, test_e)
        if wm is None: print("NO RESULT"); continue

        all_windows.append(wm)
        tag = " ← W04" if w_idx == 3 else (" ← W11" if w_idx == 10 else "")
        print(f"Sharpe={wm['sharpe']:+.3f}  MaxDD={wm['max_dd']:.1f}%{tag}")
        if w_idx == 10: w11_sharpe = wm['sharpe']

    if not all_windows:
        print("  NO RESULTS"); return {}

    sharpes     = [w['sharpe'] for w in all_windows]
    mean_sharpe = float(np.mean(sharpes))
    neg_win     = sum(1 for s in sharpes if s < 0)
    worst       = min(sharpes)
    trades_yr   = float(np.mean([w['trades_yr'] for w in all_windows]))

    print(f"\n  MEAN SHARPE  : {mean_sharpe:.3f}")
    print(f"  RANGE        : {worst:.3f} → {max(sharpes):.3f}")
    print(f"  NEG WINDOWS  : {neg_win}/{len(all_windows)}")
    print(f"  W11 (2022H1) : {w11_sharpe}")
    print(f"  WORST WINDOW : {worst:.3f}")

    if all_imps:
        mean_imp = pd.concat(all_imps, axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"  TOP FEATURES : {', '.join(f'{k}={v:.3f}' for k,v in mean_imp.head(4).items())}")
    else:
        mean_imp = None

    # Actual run.py constraints
    violations = []
    if mean_sharpe < 0.3:  violations.append(f"MEAN={mean_sharpe:.3f}<0.3")
    if neg_win     > 4:    violations.append(f"NEG_WIN={neg_win}>4")
    if worst       < -1.2: violations.append(f"WORST={worst:.3f}<-1.2")
    if trades_yr   > 150:  violations.append(f"TRADES={trades_yr:.0f}>150")
    if violations: print(f"  !! VIOLATIONS: {', '.join(violations)}")
    else:
        print(f"  [OK] All constraints passed")
        if mean_sharpe > 0.722: print(f"  >>> BEATS BASELINE 0.722!")
        else: print(f"  >>> Passes constraints but WF={mean_sharpe:.3f} < 0.722")

    return dict(label=label, mean_sharpe=mean_sharpe, worst=worst,
                neg_windows=neg_win, trades_yr=trades_yr,
                n_windows=len(all_windows), violations=violations,
                w11_sharpe=w11_sharpe, importances=mean_imp)


# ── Setup ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*66}")
print("  ML-4: Targeted experiments — fix W04 worst window")
print(f"{'='*66}")
print("  ML-013 best so far: WF=0.683, WORST=-1.320 (W04)")
print("  Actual constraint: worst > -1.2  neg_windows ≤ 4")

data      = load_data()
close     = data['close']
volume    = data['volume']

sp500     = _get_sp500(close.columns)
close_sp  = close[[c for c in close.columns if c in sp500]]
volume_sp = volume[close_sp.columns]
fg        = data['fear_greed'].reindex(close_sp.index).ffill().fillna(50.0).values

print(f">>> SP500: {close_sp.shape[1]} stocks — precomputing features …")
features  = precompute_features(close_sp, volume_sp, data['vix'], data['fear_greed'])
print(">>> Ready.\n")

def xgb_factory():
    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        n_jobs=-1, verbosity=0)

all_results = []

# Breadth scan: tighten toward 1.0 to reduce ML exposure
for exp_id, thr in [("ML-017", 0.65), ("ML-018", 0.70), ("ML-019", 0.75)]:
    r = run_exp(f"{exp_id}: XGB breadth={thr}", xgb_factory,
                thr, 1.0, close_sp, volume_sp, fg, features, data)
    all_results.append(r)

# Soft blend: use ML partially in bull regime
for exp_id, thr, alpha in [
    ("ML-020", 0.60, 0.5),    # 50% ML blend at breadth > 0.60
    ("ML-021", 0.60, 0.3),    # 30% ML blend at breadth > 0.60
    ("ML-022", 0.50, 0.5),    # 50% ML blend at breadth > 0.50
]:
    r = run_exp(f"{exp_id}: XGB blend={alpha} breadth>{thr}", xgb_factory,
                thr, alpha, close_sp, volume_sp, fg, features, data)
    all_results.append(r)

# 1yr training + breadth=0.60 combo
r_combo = run_exp("ML-023: XGB 1yr+breadth=0.60", xgb_factory,
                  0.60, 1.0, close_sp, volume_sp, fg, features, data, train_days=252)
all_results.append(r_combo)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n\n{'='*66}")
print("  FINAL SUMMARY")
print(f"{'='*66}")
print(f"  {'Label':<38} {'WF':>6} {'Worst':>7} {'Neg':>5} {'W11':>7}")
print("  " + "-"*65)
print(f"  {'S10-005 baseline':38} {'0.722':>6} {'-0.610':>7} {'2':>5} {'?':>7}")
print(f"  {'ML-013 (best prior)':38} {'0.683':>6} {'-1.320':>7} {'3':>5} {'-0.405':>7}")
for r in all_results:
    if not r: continue
    tag  = " !!" if r.get('violations') else " OK"
    w11s = f"{r['w11_sharpe']:.3f}" if r.get('w11_sharpe') is not None else "N/A"
    print(f"  {r['label']:<38} {r['mean_sharpe']:>6.3f} {r['worst']:>7.3f} "
          f"{r['neg_windows']:>5} {w11s:>7}{tag}")

valid = [r for r in all_results if r and not r.get('violations')]
if valid:
    best = max(valid, key=lambda r: r['mean_sharpe'])
    print(f"\n  >>> BEST VALID: {best['label']}  WF={best['mean_sharpe']:.3f}")
    if best['mean_sharpe'] > 0.722:
        print("  >>> BEATS BASELINE — update agent.py")
    else:
        print("  >>> Passes constraints but WF < 0.722 baseline")
else:
    print(f"\n  >>> No model passed all constraints")
    best_close = min(all_results, key=lambda r: len(r.get('violations', ['x'])) if r else 10)
    if best_close:
        print(f"  >>> Closest: {best_close['label']}  violations: {best_close.get('violations')}")

signal.alarm(0)
