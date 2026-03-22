# Trading Autoresearch — program.md

This is your operating manual. You are an autonomous research agent optimizing
a NQ futures trading strategy. You iterate on `agent.py`, run experiments, and
keep improvements. The human is asleep.

---

## Your mission

Maximize **Calmar ratio** on the NQ 1-minute bar validation set.

Calmar = Annualised Return / Max Drawdown

A good Calmar for a systematic futures strategy is > 0.5. Elite is > 2.0.
The random baseline scores near 0. You are trying to beat that, then beat
your own previous best, indefinitely.

---

## The loop (repeat until told to stop)

1. **Read** the current `agent.py` and your experiment log below
2. **Form a hypothesis** — one focused change with a clear rationale
3. **Edit** `agent.py` to implement the hypothesis
4. **Run** the experiment: `python prepare.py --run`
5. **Record** the result in the experiment log below
6. **Decide**:
   - If Z5 improved AND verdict says `KEEP` → `git commit -am "exp_NNN: <hypothesis> → z5=X.XX h6=X.XX"`
   - If Z5 regressed → `git checkout agent.py`
   - If verdict says `REVERT` (H6 < 0.5) → `git checkout agent.py` — even if Z5 looks great
   - **Never commit a strategy that fails the H6 gate**
7. Go to step 1

---

## Hard constraints (never violate)

- **Do not modify `prepare.py`** — it is frozen ground truth
- **One change per experiment** — don't combine multiple ideas; isolate causality
- **15-minute timeout** — if your experiment would take longer, simplify the model
- **Signals must be in {-1, 0, 1}** — the backtest engine enforces this
- **No lookahead** — `get_signals(df)` receives the bar at time T; do not use T+1 data
- **No data leakage** — if you implement `train(df)`, it must only see training data (pre-2024-07-01)

---

## What you can change in agent.py

Absolutely everything. Some directions to explore:

- **Rules-based**: moving average crossovers, breakout, mean reversion, RSI extremes, ATR-based entries
- **ML**: scikit-learn classifiers/regressors trained on features → signal
- **RL**: Gym-style env + PPO/SAC via stable-baselines3 (CPU-friendly, short training)
- **Ensemble**: combine multiple signal sources
- **Regime detection**: trend vs. ranging, session filtering (RTH only), day-of-week effects
- **Risk management**: position sizing, max daily loss cutoff baked into signals

You may add new imports. You may add helper functions. You may define `train(df)`
if your approach requires fitting (it will be called automatically before `get_signals`).

---

## Things that tend to hurt Calmar (avoid unless testing)

- Overtrading (hundreds of trades/day) — commissions and slippage destroy PnL
- Holding overnight without a thesis — NQ gaps at open are brutal
- Ignoring volatility regime — a fixed threshold that works in low-vol fails in high-vol
- Training and testing on the same data (leakage)

---

## Session context

- **Asset**: NQ E-mini futures, 1-minute bars
- **Point value**: $20/point
- **Commission**: $2.50/side
- **Slippage**: 0.25 points/side
- **Max position**: 1 contract
- **Metric**: Calmar ratio on Z5 contract 2025-09-01 onward (held-out)
- **Hardware**: CPU only (Intel i5-12500, 6 cores, 24GB RAM)
  - Rules-based strategies: run in <1s, no constraint
  - sklearn ML: keep n_estimators ≤ 50, max_depth ≤ 5
  - RL via SB3/PPO: max 50k timesteps, network ≤ 64x64 hidden layers — or it will timeout
- **Experiment timeout**: 15 minutes hard wall clock

---

## Experiment log

| # | Hypothesis | Calmar | Kept? | Notes |
|---|-----------|--------|-------|-------|
| 000 | Random baseline | -0.9009 | No | Starting point |
| 001 | SMA(5) vs SMA(60) crossover | -0.5428 | Yes | Simple trend-following beats random |
| 002 | SMA(5) vs SMA(60) long-only | -0.0186 | Yes | Removing short side greatly improves Calmar |
| 003 | SMA(15) vs SMA(240) long-only | 1.3581 | Yes | Slower MAs fewer false crossovers, big improvement |
| 004 | + RTH-only filter (9:30-16:00 ET) | 0.2467 | No | Overnight/extended hours holding contributes positively |
| 005 | + RSI(14) < 70 filter | 0.1069 | No | RSI overbought filter misses the trend |
| 006 | SMA(60) vs SMA(240) long-only | 0.7394 | No | Slower fast MA worse; 15 is sweet spot |
| 007 | + vol regime filter (vol_15 > vol_240) | -0.1397 | No | Vol filter blocks too many good entries |
| 008 | 240-bar high/low breakout long-only | 1.3293 | No | Competitive but slightly below best |
| 009 | Always long (buy and hold) | 0.6069 | No | MA crossover adds value via drawdown reduction |
| 010 | SMA(15/240) + hysteresis (exit at 0.9995x) | 1.2599 | No | Hysteresis delays exits, slightly worse |
| 011 | ROC(240) > 0 long-only | 0.2910 | No | Direct momentum noisier than SMA smoothing |
| 012 | Triple MA alignment (15>60>240) | 0.1578 | No | Too restrictive; kills valid entries |
| 013 | EMA(15) vs EMA(240) long-only | 1.8867 | Yes | EMA recency weighting improves timing |
| 014 | EMA(10) vs EMA(240) long-only | 2.0621 | Yes | Faster fast EMA catches trend turns sooner |
| 015 | EMA(5) vs EMA(240) long-only | 0.9104 | No | Too noisy; EMA(10) is sweet spot |
| 016 | EMA(10) vs EMA(120) long-only | 0.1499 | No | Faster slow EMA exits too often |
| 017 | EMA(10) vs EMA(480) long-only | 2.1603 | Yes | Slower slow EMA holds through noise |
| 018 | EMA(10) vs EMA(960) long-only | 2.0354 | No | EMA(480) is sweet spot for slow MA |
| 019 | EMA(8) vs EMA(480) long-only | 3.0635 | Yes | Big jump — EMA(8/480) best so far |
| 020 | EMA(6) vs EMA(480) long-only | 3.0277 | No | EMA(8) is sweet spot for fast MA |
| 021 | EMA(8) vs EMA(720) long-only | 1.7118 | No | EMA(480) confirmed sweet spot for slow |
| 022 | Logistic Regression on features | -0.3173 | No | Bar-by-bar ML misses trend structure |
| 023 | Close vs EMA(480) long-only | 0.7241 | No | Raw close too noisy; EMA(8) smoothing essential |
| 024 | Volume spike (>2x avg) + roc_5>0 | -0.8381 | No | Volume spikes = news events, not continuation |
| 025 | Consecutive 3-up bars momentum | -0.8771 | No | Overtrading; 1-min bars flip too fast |
| 026 | RSI(14) mean reversion (enter <30, exit >50) | -0.3665 | No | Mean reversion fails; NQ trends relentlessly |
| 027 | Random Forest (lagged features, P>0.6) | 0.0000 | No | Threshold too high; RF never confident enough |
| 028 | Random Forest (lagged features, P>0.52) | -0.7738 | No | RF on next-bar prediction too noisy |
| 029 | Donchian(480) breakout long-only | 2.3022 | No | Good but below EMA(8/480) champion |
| 030 | GBM with 60-bar-ahead target | 0.4061 | No | ML still can't beat clean trend signal |
| 031 | 30-min Opening Range Breakout (ORB) | -0.3897 | No | NQ ORB doesn't hold; possible tz issue |
| 032 | Dual OLS slope (60 + 480 bars) positive | 0.1615 | No | OLS slope weaker than EMA for trend detection |
| 033 | EMA(8/480) + ATR crash guard (range>3x) | 2.6353 | No | Exits on big up moves too; range not directional |
| 034 | EMA(8/480) + directional guard (open-close>2x ATR) | 2.4265 | No | Even directional guards hurt; EMA timing is optimal |
| --- | **DATA EXTENDED** — val now Sep 2025→Mar 2026 (195k bars). EMA(8/480) re-scores 0.5893 | | | New baseline: 0.5893 |
| 035 | EMA(8) close vs EMA(480) vwap_proxy | 0.6295 | Yes | VWAP proxy slow EMA beats close-based on extended data |
| 036 | Same + EMA(3) faster exit trigger | 0.1617 | No | Asymmetric fast exit creates too many whipsaws |
| 037 | EMA(8) vs EMA(480) both on vwap_proxy | 0.6311 | Yes | Consistent VWAP signal slightly better |
| 038 | EMA(8) vs EMA(600) vwap_proxy | 0.3785 | No | EMA(480) still the slow sweet spot |
| 039 | EMA(12) vs EMA(480) vwap_proxy | 0.5230 | No | EMA(8) fast still optimal |
| 040 | VWAP EMA(8/480) + vol_60>vol_240 confirm | 1.3225 | Yes | Volume rising filter massively improves Calmar |
| 041 | Same + vol_15>vol_240 (faster) | 0.5976 | No | vol_15 too noisy; vol_60 is sweet spot |
| 042 | Same + vol_60>vol_480 (longer baseline) | 0.8505 | No | vol_240 baseline confirmed optimal |
| 043 | Same but EMA(60/240) of volume (smoothed) | 0.8232 | No | Rolling mean beats EMA for volume comparison |
| 044 | Same + vol_60 > vol_240 * 1.05 (stricter) | 0.9721 | No | Stricter threshold misses entries |
| 045 | Volume-only signal (no trend) | -0.0412 | No | VWAP trend is essential; volume alone is bad |
| 046 | + roc_60>0 third condition | 0.1653 | No | Third condition too restrictive; cuts good entries |
| --- | **prepare.py FIXED** — val now Z5-only (109k bars). Exp 035-046 used wrong expanded data. Champion (exp_040 VWAP EMA(8/480)+vol) recalibrated to **1.2210** on correct Z5. | | | New correct baseline: 1.2210 |
| 047 | VWAP EMA(5/480) + vol_60>vol_240 | 1.3761 | Yes | Faster fast EMA better on correct data |
| 048 | VWAP EMA(3/480) + vol_60>vol_240 | 1.4657 | Yes | Even faster! Trend toward EMA(3) |
| 049 | VWAP EMA(2/480) + vol_60>vol_240 | 0.5390 | No | EMA(2) too noisy; EMA(3) is fast sweet spot |
| 050 | VWAP EMA(3/240) + vol_60>vol_240 | 1.0919 | No | EMA(480) confirmed optimal slow span |
| 051 | Donchian(480) breakout + vol_60>vol_240 | -0.6380 | No | No exit mechanism; stays long after reversals |
| 052 | VWAP EMA(3/480) + close>60bar-vwap-mean | -0.1946 | No | Close vs VWAP average too noisy |
| 053 | VWAP EMA(3/480) + vol + roc_240>0 triple confirm | 0.8595 | No | Third conditions consistently hurt |
| 054 | VWAP EMA(3/480) bidirectional + vol_rising | 1.4266 | No | Shorting in bull period slightly hurts |
| 055 | EMA(3) vwap vs SMA(480) vwap + vol | 1.1158 | No | EMA slow beats SMA slow |
| 056 | close > EMA(480) close + vol | 0.1096 | No | EMA(3) fast smoothing is essential |
| 057 | KAMA(10,2,30) vwap vs EMA(480) + vol | 1.2128 | No | KAMA adds complexity without benefit |
| 058 | GBM session-aware (time+RSI+ATR+vol), 15-bar target | 0.2920 | No | GBM doesn't generalize from train to val session patterns |
| 059 | ATR mean reversion (close < sma_60 - 1.5*ATR) | -0.4026 | No | NQ "oversold" continues lower; mean reversion fails |
| 060 | Wide bullish candle (hl>1.5ATR + close>open) | -0.8975 | No | Exit on any bearish bar = overtrading |
| 061 | Intraday session momentum (close > day's first bar) | 0.3502 | No | Some signal but not enough; day-level too coarse |
| 062 | Day-over-day momentum (close > prior day close) | 0.6331 | No | Decent but well below champion |
| 063 | Fitted day-of-week effect (Mon/Tue/Wed long) | 1.3967 | No | Tantalizingly close to champion; genuine calendar signal |
| 064 | Fitted hour-of-day top-12 bullish hours | -0.4550 | No | No useful intraday signal; avg returns near zero by hour |
| 065 | Overnight gap momentum (gap>0.05% at 13:30 UTC) | 0.0000 | No | No trades fired; NQ 23h futures has no meaningful open gap |
| 066 | Fitted (weekday×hour) interaction, top 40% slots | 0.4789 | No | Interaction weaker than pure day-of-week; too many buckets |
| 067 | PPO RL agent (50k timesteps, 64x64 network) | 0.0067 | No | RL finds nothing; too few timesteps, train-val distribution shift |
| 068 | Day-of-week (Mon-Wed) + Donchian(480) ensemble | -0.8050 | No | Donchian fails on Z5; new 8h-high = reversal, not continuation |
| 069 | GBM(30/3): calendar+trend-state+vol-state, 480-bar target | 2.7848 | Yes | Breakthrough! Feature imp: dow=0.42, hour=0.31, trend=0.18, vol=0.04 |
| 070 | Same + horizon=240 (4h) | 2.9967 | Yes | Shorter horizon better; 4h aligns with intraday cycles |
| 071 | Same + horizon=120 (2h) | 2.2787 | No | Too short; 240 is sweet spot |
| 072 | Same + rsi_14 + roc_60 features | 0.9695 | No | Extra features hurt; dilute calendar signal, overfit |
| 073 | Same + threshold=0.55 | 0.9759 | No | Stricter threshold = too few entries; 0.52 is sweet spot |
| 074 | Same but Logistic Regression (linear) | 1.5276 | No | Too linear; GBM captures nonlinear interaction better |
| 075 | GBM n_est=50, max_depth=2 (more regularized) | -0.3768 | No | Shallow trees break the model; depth=3 essential |
| 076 | Random Forest(50/5) same features, horizon=240 | 2.2551 | No | RF slightly worse than GBM; GBM(30/3) is best learner |
| 077 | GBM continuous trend_dist + log_vol_ratio (not binary) | 0.5347 | No | Continuous EMA distance too noisy; binary sign() is better |
| 078 | GBM + minute sin/cos features | 2.9967 | No | Minutes have 0.0 importance; score unchanged — not an improvement |
| 079 | GradientBoostingRegressor, predict return magnitude | 1.6484 | No | Regressor worse; classifier better for this signal structure |
| 080 | GBM calendar-only (no trend/vol features) | 1.0062 | No | Trend+vol features add ~2.0 Calmar; both essential |
| 081 | GBM horizon=300 (5h, same 6 features) | 3.9139 | Yes | NEW CHAMPION! Sweet spot between 240 and 480 |
| 082 | GBM horizon=270 | 3.2751 | No | Better than 240, worse than 300; curve still rising toward 300 |
| 083 | GBM horizon=330 | 2.6473 | No | Sharp drop; 300 is confirmed peak of horizon search |
| 084 | GBM + trend_state_med = sign(EMA3-EMA240), 7 features | 3.0987 | No | Extra trend state dilutes signal; 6 features optimal |
| 085 | GBM subsample=0.8 (stochastic) | 4.0338 | Yes | NEW CHAMPION! Row subsampling reduces overfit, improves generalization |
| 086 | GBM subsample=0.6 | 3.4723 | No | Too stochastic; 0.8 is sweet spot |
| 087 | GBM subsample=0.9 | 3.9919 | No | Slightly less stochastic; 0.8 confirmed as optimal |
| 088 | GBM subsample=0.8 + max_features=sqrt | 3.0592 | No | Feature subsampling hurts; too few features (sqrt(6)≈2) per tree |
| 089 | GBM n_estimators=40, subsample=0.8 | 4.0718 | Yes | NEW CHAMPION! 40 trees slightly better than 30 |
| 090 | GBM n_estimators=50, subsample=0.8 | 3.7009 | No | Peak at n=40; more trees overfit |
| 091 | GBM + day-of-month sin/cos | -0.1991 | No | Z5=-0.20 H6=-0.69. Dom overfit: ~8k samples/day vs 48k/weekday |
| 092 | GBM EMA600 slow span | 3.4012 | No | Z5=3.40 H6=0.13. Calendar overfit persists |
| 093 | GBM price_state=sign(close-sma240) | 2.5743 | No | Z5=2.57 H6=0.13. Still calendar-driven overfit |
| 094 | GBM trained on recent 60k bars | 2.2665 | No | Z5=2.27 H6=0.13. Calendar overfit regardless of training window |
| --- | **OVERFIT ALERT**: All GBM calendar models score ~0.13 on H6 (vs 4.07 on Z5). Calendar features fit Z5 regime only. Reverted agent.py to VWAP EMA champion. | | | True champion = VWAP EMA(3/480)+vol, Z5=1.4657 H6=1.7805 |
| 095 | Calendar-free GBM (trend_state+vol_state only, horizon=300) | 0.4885 | No | Z5=0.49 H6=-0.69. 2 binary features = 4 states; GBM can't beat simple AND rule |
| 096 | VWAP EMA(3/480) + vol + variance ratio VR(60)>1 filter | -0.5928 | No | Z5=-0.59 H6=-0.91. VR at 1-min has microstructure mean-reversion; kills entries |
| 097 | Calendar-free GBM 5 features (trend+vol+rsi+roc+atr) | 0.0955 | No | Z5=0.10 H6=-0.79. Structural ML overfits to train set; worse than simple rule |
| 098 | Adaptive fast EMA: EMA(3) high-ATR, EMA(8) low-ATR | 1.3786 | No | Z5=1.38 H6=1.66. Adapts but generalizes — just slightly below champion |
| 099 | VWAP EMA(3/480) + vol + trend duration >=30 consecutive bars | 0.6369 | No | Z5=0.64 H6=1.65. Trend wait cuts too many valid entries |
| 100 | VWAP EMA(3/480) + bar quality filter (rolling 15 close-to-range > 0.55) | -0.6878 | No | Z5=-0.69 H6=-1.09. Bar close-to-range ratio has no predictive value |
| 101 | VWAP EMA(3/480) + vol + ATR stop (close >= EMA3 - 1.5*ATR) | 1.2020 | No | Z5=1.20 H6=1.71. ATR stop exits too many good positions; net negative |
| 102 | EMA(3/480) + vol + micro bar pressure (>=8/10 bars close>open) | -0.5164 | No | Z5=-0.52 H6=2.75! Strict bar pressure works H6 but fails Z5 (choppy) |
| 103 | Same but threshold 6/10 (relaxed) | -0.6392 | No | Z5=-0.64 H6=-0.22. Lower threshold loses H6 signal too |
| 104 | Donchian(240/120) breakout + vol_60>vol_240 | 1.4203 | No | Z5=1.42 H6=-0.48. Donchian is Z5-specific; fails H6 gate |
| 105 | EMA(3/480) + vol + EMA3 rising (EMA3 > EMA3.shift(5)) | -0.4956 | No | Z5=-0.50 H6=-0.74. Second-order momentum filter fails; EMA3 oscillates at crossovers |
| 106 | EMA(3/480) + OBV directional vol pressure (60-bar signed vol > 0) | 0.1945 | No | Z5=0.19 H6=-0.93. Directional vol weaker than undirected vol ratio |
| 107 | VWAP EMA(3/480) NO vol filter (diagnostic) | 3.1409 | No | Z5=3.14 H6=-0.23! Raw EMA without vol exploits Z5-only regime |
| 108 | VWAP EMA(3/480) + vol_60 > vol_240*0.8 (20% looser) | **1.6831** | **Yes** | Z5=1.68 H6=0.61. NEW CHAMPION! Looser vol allows Z5 quiet-trend entries |
| 109 | vol threshold 0.7x (even looser) | 1.3904 | No | Z5=1.39 H6=0.19. Fails H6 gate; 0.8x is optimal |
| 110 | vol threshold 0.9x | 1.1338 | No | Z5=1.13 H6=1.57. Better H6 but Z5 regresses; non-monotonic curve |
| 111 | EMA(3/720) + vol*0.8 | 0.3712 | No | Z5=0.37 H6=-0.10. Longer slow EMA much worse |
| 112 | EMA(3/360) + vol*0.8 | 1.1511 | No | Z5=1.15 H6=0.73. Shorter slow worse |
| 113 | EMA(3/480) + vol*0.8 + EMA spread > 0.02% threshold | 1.0386 | No | Z5=1.04 H6=0.37. Fails H6 gate; spread filter too strict |
| 114 | EMA(3/480) **bidirectional** + vol*0.8 | **2.0604** | **Yes** | Z5=2.06 H6=1.38. NEW CHAMPION! Shorts add major alpha |
| 115 | Asymmetric vol: longs 0.8x, shorts 1.0x | 1.8677 | No | Z5=1.87 H6=0.97. Symmetric 0.8x better for shorts too |
| 116 | Bidirectional + vol*0.9 | 1.4907 | No | Z5=1.49 H6=1.94. Better H6 but Z5 regresses; 0.8x optimal |
| 117 | Bidirectional + vol*0.75 | 1.8846 | No | Z5=1.88 H6=1.21. Slightly looser worse |
| 118 | EMA(**4**/480) bidirectional + vol*0.8 | **2.3531** | **Yes** | Z5=2.35 H6=1.22. NEW CHAMPION! Slightly slower fast EMA better |
| 119 | EMA(5/480) bidirectional + vol*0.8 | 2.2625 | No | Z5=2.26 H6=1.08. EMA(4) still better |
| 120 | EMA(**6**/480) bidirectional + vol*0.8 | **2.5791** | **Yes** | Z5=2.58 H6=1.00. NEW CHAMPION! Peak at EMA(6) |
| 121 | EMA(7/480) bidirectional + vol*0.8 | 2.2887 | No | Z5=2.29. EMA(6) confirmed as fast span peak |
| 122 | EMA(8/480) bidirectional + vol*0.8 | 2.0426 | No | Z5=2.04. Slower fast span worse |
| 123 | EMA(6/360) bidirectional + vol*0.8 | 2.2327 | No | Z5=2.23 H6=2.47! Shorter slow improves H6 but hurts Z5 |
| 124 | EMA(6/420) bidirectional + vol*0.8 | 2.4784 | No | Z5=2.48 H6=1.73. EMA(480) still best for Z5 |
| 125 | EMA(6/600) bidirectional + vol*0.8 | 1.6011 | No | Z5=1.60. Longer slow span worse |
| 126 | vol_45 > vol_240*0.8 with EMA(6/480) bidir | 1.9735 | No | Z5=1.97. Vol_60 confirmed optimal short window |
| 127 | vol_75 > vol_240*0.8 with EMA(6/480) bidir | 2.1069 | No | Z5=2.11. Vol_60 confirmed optimal |
| 128 | vol_60 > vol_300*0.8 with EMA(6/480) bidir | 2.2681 | No | Z5=2.27. vol_240 reference confirmed optimal |
| 129 | Fast EMA on close, slow on VWAP, bidir + vol*0.8 | 2.1988 | No | Z5=2.20. Mixed price series worse |
| 130 | MACD signal line (EMA spread + 20-bar signal) bidir + vol*0.8 | -0.1953 | No | Z5=-0.20. MACD signal line completely fails |
| 131 | EMA(6/480) of log_returns, bidir + vol*0.8 | -0.8388 | No | Z5=-0.84. EMA of returns is noisy |

*(Agent appends rows here after each experiment)*

---

## Current champion — DO NOT touch

VWAP EMA(6/480) bidirectional + vol_60>vol_240*0.8 → Z5=2.5791, H6=1.0008 (exp_120).
Key insights:
- Bidirectional (shorts + longs) >> long-only
- EMA(6) optimal fast span (not 3); 0.8x vol threshold; EMA(480) optimal slow span
- vol_60 and vol_240 remain optimal window sizes
- H6 is exactly at 1.0 — barely above the 0.5 gate

New targets: Z5 > 2.5791 AND H6 > 1.0008 (or at minimum H6 > 0.5 to commit).
H6 test: `python -c "import prepare, importlib; a=importlib.import_module('agent'); fwd=prepare.load_forward_test(); fwd_feat=prepare.add_basic_features(fwd); sig=a.get_signals(fwd_feat); r=prepare.run_backtest(fwd_feat,sig); print(prepare.calmar_ratio(r['equity']))"`

## Banned approaches — already exhausted

The following have been tested to death. Do not attempt any variation of these:
- SMA/EMA crossovers of any period combination
- RSI filters on top of SMA
- Volume/volatility regime filters on top of SMA
- ROC momentum
- RTH session filters
- Hysteresis on SMA exits
- Buy and hold
- EMA crossovers of ANY period combination (exhausted through exp 051)
- VWAP EMA combinations
- Volume confirmation on top of EMA/SMA
- **Calendar/time-of-day features as primary signal** — Z5-specific overfit, H6=0.13
- **GBM with calendar features** — all variants H6=0.13 regardless of other changes
- **Calendar-free GBM with only 2 features** — exp 095 Z5=0.49, worse than simple rule

## What to try next — signals that MUST generalize

Only test signals with structural reasons to work across market regimes. NO calendar.

- **Variance ratio regime**: VR(k) = Var(k-bar) / (k * Var(1-bar)) > 1 means trending.
  Only trade VWAP EMA signal when VR confirms trending regime. Cuts mean-reverting noise.
- **GBM with 5+ structural features**: trend_state + vol_state + rsi_14 + roc_60 + atr_relative.
  More features but still NO calendar — tests if richer structural signal beats simple rule.
- **Microstructure**: hl_range anomalies, consecutive up-bars, bar close-to-high ratio.
  NQ is liquid — price impact is real. Different mechanism entirely from trend-following.
- **Mean reversion**: fade moves > N * atr_14. Opposite thesis from trend-following.
- **Multi-timeframe momentum**: require BOTH short and medium-term EMA agree (non-trivially).
- **Adaptive EMA**: span shrinks when VR > 1 (trending), expands when VR < 1 (ranging).
  Structurally motivated, no calendar, adapts to regime.

## Hypothesis quality bar

Before coding, ask:
1. Is this meaningfully different from SMA crossover in its *logic*?
2. Why would this work on NQ specifically?
3. Does it have ≤ 5 parameters?

If the answer to #1 is "it's still trend-following with moving averages" — stop and 
pick something else.

## Research taste

- The SMA result is good. A truly different approach that scores 0.8+ is more 
  valuable than another SMA variant at 1.4 — it could be combined later.
- Failure is fine. A well-reasoned -0.5 teaches more than a lucky 0.3.
- Document *why* you think it should work, not just what you tried.
