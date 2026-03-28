#!/usr/bin/env python3
"""
ml_run2.py — ML signal experiments with cross-sectional neutralization.

Root cause of ML-001..004 failure:
  - Target y = raw forward return → model learns market regime, not stock selection
  - VIX/breadth dominate feature importance → market-timing, not stock picking
  - In bear windows: all returns negative, model ranks randomly among negative stocks

Fixes tested:
  ML-005: XGBoost + RANK target (cross-sectional rank of fwd_ret per date)
  ML-006: XGBoost + EXCESS target (fwd_ret - cross-sectional mean per date)
  ML-007: XGBoost + EXCESS target + stock-only features (remove vix/fg/breadth)
  ML-008: XGBoost + RANK target  + stock-only features
  ML-009: Best XGB config with increased regularization
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
from ml_signal import precompute_features, predict_rank, FEATURE_COLS
from ml_run import generate_signals_ml, _get_sp500
from xgboost import XGBRegressor
import requests
from io import StringIO

STOCK_FEATURE_COLS = [c for c in FEATURE_COLS if c not in ('vix', 'fg', 'breadth')]

MIN_HISTORY = 260
SAMPLE_EVERY = 10
FWD_DAYS = 20


def build_training_matrix_v2(close, features, train_start, train_end,
                               target_type='rank', use_market_features=True):
    """
    Enhanced training matrix with cross-sectional neutralization.

    target_type:
      'raw'    — raw forward return (original, bad)
      'rank'   — rank(fwd_ret) per date (cross-sectional, market-neutral)
      'excess' — fwd_ret - mean(fwd_ret) per date (cross-sectional excess)

    use_market_features: if False, drop vix/fg/breadth columns
    """
    idx = close.index
    ts  = idx[idx >= train_start][0]
    te  = idx[idx <= train_end][-1]

    date_locs = [i for i, d in enumerate(idx)
                 if ts <= d <= te
                 and i >= MIN_HISTORY
                 and i % SAMPLE_EVERY == 0
                 and i + FWD_DAYS < len(idx)]

    if not date_locs:
        return np.empty((0, len(FEATURE_COLS))), np.empty(0), 0

    stock_cols = ['mom_26w', 'mom_13w', 'mom_4w', 'mom_52w',
                  'ma_ratio', 'vol_ratio', 'realized_vol', 'rsi_14', 'above_ma']
    feat_cols_use = STOCK_FEATURE_COLS if not use_market_features else FEATURE_COLS

    all_X, all_y = [], []

    for d_loc in date_locs:
        fwd_loc = d_loc + FWD_DAYS
        fwd_ret = (close.iloc[fwd_loc] / close.iloc[d_loc] - 1)

        feat_df = pd.DataFrame({
            col: features[col].iloc[d_loc]
            for col in stock_cols
        })
        feat_df['vix']     = float(features['vix'].iloc[d_loc])
        feat_df['fg']      = float(features['fg'].iloc[d_loc])
        feat_df['breadth'] = float(features['breadth'].iloc[d_loc])
        feat_df['_target'] = fwd_ret

        feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(feat_df) < 5:
            continue

        # Cross-sectional target transformation
        if target_type == 'rank':
            feat_df['_target'] = feat_df['_target'].rank(pct=True)
        elif target_type == 'excess':
            feat_df['_target'] = feat_df['_target'] - feat_df['_target'].mean()

        all_X.append(feat_df[feat_cols_use].values)
        all_y.append(feat_df['_target'].values)

    if not all_X:
        return np.empty((0, len(feat_cols_use))), np.empty(0), 0

    X = np.vstack(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.float32)
    return X, y, len(date_locs)


def predict_rank_v2(close, features, model, date_loc, tickers_filter, use_market_features=True):
    """Like predict_rank() but respects use_market_features flag."""
    stock_cols = ['mom_26w', 'mom_13w', 'mom_4w', 'mom_52w',
                  'ma_ratio', 'vol_ratio', 'realized_vol', 'rsi_14', 'above_ma']
    feat_cols_use = STOCK_FEATURE_COLS if not use_market_features else FEATURE_COLS

    feat_df = pd.DataFrame({
        col: features[col].iloc[date_loc]
        for col in stock_cols
    }, index=close.columns)
    feat_df['vix']     = float(features['vix'].iloc[date_loc])
    feat_df['fg']      = float(features['fg'].iloc[date_loc])
    feat_df['breadth'] = float(features['breadth'].iloc[date_loc])

    feat_df = feat_df.loc[feat_df.index.intersection(tickers_filter)]
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

    missing = feat_df[feat_df.isnull().any(axis=1)].index
    valid   = feat_df.dropna()

    if valid.empty:
        mom = features['mom_26w'].iloc[date_loc].loc[tickers_filter]
        return mom.rank(pct=True)

    X_pred = valid[feat_cols_use].values.astype(np.float32)
    preds  = model.predict(X_pred)
    pred_s = pd.Series(preds, index=valid.index)

    if len(missing) > 0:
        mom_fb = features['mom_26w'].iloc[date_loc].loc[missing]
        fb_scaled = mom_fb.rank(pct=True) * pred_s.std() + pred_s.mean()
        pred_s = pd.concat([pred_s, fb_scaled])

    return pred_s.rank(pct=True)


def generate_signals_ml_v2(close_sp, volume_sp, fg, features, model,
                            train_end, use_market_features=True):
    """Same as generate_signals_ml but uses predict_rank_v2."""
    from ml_run import (LB_DAYS, SKIP_DAYS, MA_DAYS, REBAL_DAYS,
                        MIN_HOLD_DAYS, FG_MIN, EXIT_PCT_RANK,
                        STOP_LOSS_PCT, TOP_PCT, INV_VOL_DAYS)

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
        above_ma_now = close_sp.iloc[i] > ma_today
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

            if model is not None and len(filtered) > 0:
                combo = predict_rank_v2(close_sp, features, model, i,
                                        filtered, use_market_features)
                miss  = filtered.difference(combo.index)
                if len(miss) > 0:
                    combo = pd.concat([combo, mom.loc[miss].rank(pct=True)])
            else:
                combo = mom.rank(pct=True)

            filt = combo.where(above_ma & high_vol).dropna()

            if not filt.empty:
                if breadth < 0.40:      eff_pct = 0.010
                elif fg_val > 70:       eff_pct = 0.015
                else:                   eff_pct = TOP_PCT

                n_top       = max(1, int(len(filt) * eff_pct))
                top_tickers = filt.nlargest(n_top).index
                vol_ret     = close_sp.iloc[max(0, i - INV_VOL_DAYS):i][top_tickers].pct_change().std()
                inv_vol     = (1.0 / vol_ret.replace(0, np.nan)).fillna(0.0)
                inv_vol_norm = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else inv_vol
                new_pos = pd.Series(0.0, index=tickers)
                for tkr in top_tickers:
                    new_pos[tkr] = inv_vol_norm.get(tkr, 0.0)
                for tkr in top_tickers:
                    if current_pos.get(tkr, 0.0) == 0.0:
                        pos_high[tkr] = today[tkr]; entry_day[tkr] = i
                for tkr in current_pos[current_pos > 0].index:
                    if new_pos[tkr] == 0.0:
                        pos_high[tkr] = np.nan; entry_day[tkr] = -999
                current_pos = new_pos.copy()

        weights.iloc[i] = current_pos

    return weights


def run_exp(label, model_factory, target_type, use_mkt_features,
            close_sp, volume_sp, fg, features, data):
    feat_cols_use = STOCK_FEATURE_COLS if not use_mkt_features else FEATURE_COLS
    n_feats = len(feat_cols_use)

    print(f"\n{'='*66}")
    print(f"  {label}")
    print(f"  target={target_type}  mkt_features={use_mkt_features}  n_features={n_feats}")
    print(f"{'='*66}")

    all_windows, all_imps, train_r2s = [], [], []

    for w_idx, (train_s, train_e, test_s, test_e) in enumerate(WF_WINDOWS):
        print(f"  W{w_idx+1:02d}: {test_s[:7]}→{test_e[:7]} ... ", end='', flush=True)

        X_tr, y_tr, nd = build_training_matrix_v2(
            close_sp, features, train_s, train_e,
            target_type=target_type, use_market_features=use_mkt_features)

        if len(X_tr) < 500:
            print(f"SKIP (n={len(X_tr)})"); continue

        model = model_factory(n_feats)
        model.fit(X_tr, y_tr)

        yh = model.predict(X_tr)
        ss_res = np.sum((y_tr - yh)**2); ss_tot = np.sum((y_tr - y_tr.mean())**2)
        train_r2 = 1 - ss_res/ss_tot if ss_tot > 1e-12 else 0.0
        train_r2s.append(train_r2)

        if hasattr(model, 'feature_importances_'):
            all_imps.append(pd.Series(model.feature_importances_, index=feat_cols_use))

        weights = generate_signals_ml_v2(
            close_sp, volume_sp, fg, features, model, train_e,
            use_market_features=use_mkt_features)
        wm = _backtest_window(weights, data, test_s, test_e)
        if wm is None:
            print("NO RESULT"); continue

        all_windows.append(wm)
        print(f"R²={train_r2:.3f}  Sharpe={wm['sharpe']:+.3f}  MaxDD={wm['max_dd']:.1f}%")

    if not all_windows:
        print("  NO RESULTS"); return {}

    sharpes     = [w['sharpe'] for w in all_windows]
    mean_sharpe = float(np.mean(sharpes))
    neg_win     = sum(1 for s in sharpes if s < 0)
    worst       = min(sharpes)
    trades_yr   = float(np.mean([w['trades_yr'] for w in all_windows]))
    beat_spy    = sum(1 for w in all_windows if w['total_ret'] > w['bench_ret'])
    avg_r2      = float(np.mean(train_r2s)) if train_r2s else 0.0

    print(f"\n  MEAN SHARPE   : {mean_sharpe:.3f}")
    print(f"  RANGE         : {worst:.3f} → {max(sharpes):.3f}")
    print(f"  NEG WINDOWS   : {neg_win}/{len(all_windows)}")
    print(f"  BEAT SPY      : {beat_spy}/{len(all_windows)}")
    print(f"  AVG TRAIN R²  : {avg_r2:.4f}")

    if all_imps:
        mean_imp = pd.concat(all_imps, axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"  TOP FEATURES  : {', '.join(f'{k}={v:.3f}' for k,v in mean_imp.head(5).items())}")

    violations = []
    if mean_sharpe < 0.3:   violations.append(f"MEAN={mean_sharpe:.3f}<0.3")
    if neg_win     > 3:     violations.append(f"NEG_WIN={neg_win}>3")
    if worst       < -1.0:  violations.append(f"WORST={worst:.3f}<-1.0")
    if trades_yr   > 150:   violations.append(f"TRADES={trades_yr:.0f}>150")
    if violations: print(f"  !! VIOLATIONS: {', '.join(violations)}")
    else:          print(f"  [OK] All constraints passed")

    return dict(label=label, mean_sharpe=mean_sharpe, worst=worst,
                neg_windows=neg_win, trades_yr=trades_yr, beat_spy=beat_spy,
                n_windows=len(all_windows), violations=violations,
                avg_r2=avg_r2,
                importances=mean_imp if all_imps else None)


# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\n{'='*66}")
print("  ML MOMENTUM v2 — Cross-sectional neutralization")
print(f"{'='*66}")
print("  Baseline WF=0.722  |  ML-001..004 all failed")
print("  Fix: rank target + optional remove market features")

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

# Model factories (accept n_features parameter)
def xgb_factory(n_features):
    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0,
    )

def xgb_heavy_reg(n_features):
    return XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.03,
        subsample=0.6, colsample_bytree=0.6,
        min_child_weight=30, reg_alpha=1.0, reg_lambda=5.0,
        random_state=42, n_jobs=-1, verbosity=0,
    )

all_results = []

# ML-005: XGB + rank target (market-neutral target, all features)
r5 = run_exp("ML-005: XGB rank-target (all features)", xgb_factory,
             'rank', True, close_sp, volume_sp, fg, features, data)
all_results.append(r5)

# ML-006: XGB + excess-return target (all features)
r6 = run_exp("ML-006: XGB excess-return target (all features)", xgb_factory,
             'excess', True, close_sp, volume_sp, fg, features, data)
all_results.append(r6)

# ML-007: XGB + excess target + stock-only features
r7 = run_exp("ML-007: XGB excess-return, stock features only", xgb_factory,
             'excess', False, close_sp, volume_sp, fg, features, data)
all_results.append(r7)

# ML-008: XGB + rank target + stock-only features
r8 = run_exp("ML-008: XGB rank-target, stock features only", xgb_factory,
             'rank', False, close_sp, volume_sp, fg, features, data)
all_results.append(r8)

# ML-009: XGB heavy regularization + rank target + stock features only
r9 = run_exp("ML-009: XGB heavy-reg rank-target stock-only", xgb_heavy_reg,
             'rank', False, close_sp, volume_sp, fg, features, data)
all_results.append(r9)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n\n{'='*66}")
print("  SUMMARY ML v2")
print(f"{'='*66}")
print(f"  {'Label':<40} {'WF':>6} {'Worst':>7} {'Neg':>5} {'R²':>7}")
print("  " + "-"*70)
print(f"  {'S10-005 baseline':40} {'0.722':>6} {'-0.610':>7} {'2':>5} {'n/a':>7}")
for r in all_results:
    if not r: continue
    tag = " !!" if r['violations'] else " OK"
    print(f"  {r['label']:<40} {r['mean_sharpe']:>6.3f} {r['worst']:>7.3f} "
          f"{r['neg_windows']:>5} {r.get('avg_r2',0):>7.4f}{tag}")

valid = [r for r in all_results if r and not r['violations']]
if valid:
    best = max(valid, key=lambda r: r['mean_sharpe'])
    print(f"\n  >>> BEST VALID: {best['label']}  WF={best['mean_sharpe']:.3f}")
    if best['mean_sharpe'] > 0.722:
        print("  >>> BEATS BASELINE — run evaluate.py")
    else:
        print("  >>> Does not beat baseline 0.722")
else:
    print("\n  >>> No model passed all constraints")

signal.alarm(0)
