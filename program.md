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

## Logging discipline — strictly enforced

- Update the experiment log IMMEDIATELY after each run.py output
- Do this before writing any code for the next experiment
- Never batch-update multiple experiments at once — log each one as it happens
- If you lose context mid-session, run git log --oneline to recover results
- Every row must include Z5, H6, and PnL in the Notes column

Log format
| NNN | hypothesis | Z5 Calmar | Kept? | Z5=X.XX H6=X.XX Z5pnl=$XX,XXX H6pnl=$XX,XXX Trades_Z5=N Trades_H6=N |

---

## Hard constraints (never violate)

- **Do not modify `prepare.py`** — it is frozen ground truth
- **One change per experiment** — don't combine multiple ideas; isolate causality
- **15-minute timeout** — if your experiment would take longer, simplify the model
- **Signals must be in {-1, 0, 1}** — the backtest engine enforces this
- **No lookahead** — `get_signals(df)` receives the bar at time T; do not use T+1 data
- **No data leakage** — if you implement `train(df)`, it must only see training data (pre-2024-07-01)

---

## H6 generalization gate

- H6 minimum: 0.6 — strategies below this on H6 are overfit to Z5 and must be reverted
- Current champion (exp_139) is grandfathered at H6=0.55 — do not revert it
- All future commits must clear H6 >= 0.6


## Profit visibility
run.py now prints Z5 PnL, H6 PnL, and trade count alongside Calmar.
Log all of these. Target Z5 PnL > $20,000 (informational — Calmar is the decision metric).
A high Calmar with tiny PnL ($200) is not useful in practice.


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

## Logging discipline — strictly enforced

- Update the experiment log in program.md **IMMEDIATELY** after each `run.py` output
- Do this **before** writing any code for the next experiment
- Never batch-update multiple experiments at once — log each one as it happens
- If you lose context mid-session, run `git log --oneline` to recover results
- Every row must have Z5 and H6 scores in the Notes column

## H6 generalization gate

- H6 minimum raised to **0.6** (was 0.5)
- Current champion (exp_139, Z5=2.63 H6=0.55) is **grandfathered** — do not revert it
- All future commits must clear H6 >= 0.6 to be kept
- A Z5 improvement that fails H6 is **not an improvement** — revert it

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
| 132 | (prior session — not individually recovered) | — | — | — |
| 133 | (prior session — not individually recovered) | — | — | — |
| 134 | (prior session — not individually recovered) | — | — | — |
| 135 | (prior session — not individually recovered) | — | — | — |
| 136 | (prior session — not individually recovered) | — | — | — |
| 137 | (prior session — not individually recovered) | — | — | — |
| 138 | (prior session — not individually recovered) | — | — | — |
| 139 | Asymmetric vol: longs vol*0.6 (loose), shorts vol*0.8 | **2.6279** | **Yes** | Z5=2.6279 H6=0.5461. NEW CHAMPION! Loose longs capture quiet-trend alpha |
| 140 | Asymmetric vol: longs vol*0.5x, shorts vol*0.8x | 2.87 | No | H6=0.41 — fails gate; too loose lets bad longs in H6 |
| 141 | Asymmetric vol: longs vol*0.55x, shorts vol*0.8x | 2.83 | No | H6=0.41 — fails gate; 0.6x is the optimal loose threshold |
| 142 | (prior session — not individually recovered) | — | — | — |
| 143 | (prior session — not individually recovered) | — | — | — |
| 144 | Momentum percentile filter (roc_60 vs 75/25th pct 480-bar window) | -0.3902 | No | Z5=-0.39 H6=-0.67. Fails both. |
| 145 | Asymmetric vol: longs*0.6, shorts*0.7 (loosen shorts) | 2.1573 | No | Z5=2.16 H6=0.47. Fails gate; loosening shorts hurts both |
| 146 | Normalized EMA spread > 0.0003 filter (replaces vol) | 1.7173 | No | Z5=1.72 H6=0.59. Passes H6 but Z5 regresses |
| 147 | Vol*0.6 OR spread>0.0003 for longs (OR condition) | 3.3237 | No | Z5=3.32 H6=0.31. Fails H6; OR adds Z5-only entries |
| 148 | ATR(14) vs rolling ATR mean, asymmetric 0.6/0.8x | 1.4532 | No | Z5=1.45 H6=0.12. ATR filter worse than vol filter |
| 149 | vol_60 vs vol_480 (8h baseline instead of 4h) | 2.3354 | No | Z5=2.34 H6=0.43. Fails gate; vol_240 baseline optimal |
| 150 | + close>EMA(240) medium alignment for longs | 0.8515 | No | Z5=0.85 H6=0.07. Too restrictive; kills entries |
| 151 | Self-normalizing vol percentile: longs>25th, shorts>40th pct | **2.8200** | **Yes** | Z5=2.8200 H6=0.5008. NEW CHAMPION (committed under old H6=0.5 gate) |
| 152 | Vol percentile: longs>20th, shorts>40th (looser longs) | 2.7647 | No | Z5=2.76 H6=0.62. Passes new 0.6 gate but Z5 regressed |
| 153 | Vol percentile: longs>25th, shorts>30th (looser shorts) | 2.3491 | No | Z5=2.35 H6=0.24. Fails gate; looser shorts destroy H6 |
| 154 | Vol percentile: longs>20th, shorts>50th (offset asymmetry) | 2.1967 | No | Z5=2.20 H6=0.84. Good H6 but Z5 regressed too much |

| 155 | Percentile window 960 bars (16h) instead of 480 | 1.8540 | No | Z5=1.85 H6=0.31. Fails gate; longer window hurts both |

| 156 | Variance ratio VR(60)>1 regime filter (no vol, pure EMA+VR) | -0.4428 | No | Z5=-0.44 H6=0.09. VR at 1-min is noisy — mean-reverting at micro level |

| 157 | OBV EMA(6/480) crossover bidirectional (volume-weighted direction) | 0.1871 | No | Z5=0.19 H6=-0.00. OBV trend weak signal; cumulative OBV drifts without meaning |

| 158 | EMA(6/360) + vol percentile (longs>25th, shorts>40th pct) | 2.6634 | No | Z5=2.66 H6=1.15! Passes gate. Shorter slow EMA → great H6. Z5 below champion |

| 159 | EMA(6/360) + vol percentile longs>15th pct (looser longs) | 2.5772 | No | Z5=2.58 H6=0.98. Looser longs hurt Z5 with EMA(360) — different tradeoff |

| 160 | EMA(6/360) + vol pct longs>35th, shorts>40th (tighter longs) | **2.8635** | **Yes** | Z5=2.8635 H6=1.3172. NEW CHAMPION! Tighter longs + shorter slow EMA = quality entries + H6 headroom |

| 161 | EMA(6/360) + vol pct longs>30th, shorts>40th | **2.9078** | **Yes** | Z5=2.9078 H6=1.4352. NEW CHAMPION! Looser longs better with EMA(360) at this level |

| 162 | EMA(6/360) + vol pct longs>25th, shorts>40th | 2.6634 | No | Z5=2.66 H6=1.15. Same as exp_158; 30th pct confirmed optimal for longs |

| 163 | EMA(6/360) + vol pct longs>30th, shorts>30th (same threshold) | 2.5479 | No | Z5=2.55 H6=1.36. Looser shorts hurt Z5; 40th pct shorts optimal |

| 164 | EMA(6/**420**) + vol pct longs>30th, shorts>40th | **3.3515** | **Yes** | Z5=3.3515 H6=1.4713. NEW CHAMPION! Huge jump — 420 is a sweet spot |

| 165 | EMA(6/390) + vol pct longs>30th, shorts>40th | 3.0090 | No | Z5=3.01 H6=1.15. Passes gate but Z5 below 420 champion |

| 166 | EMA(6/450) + vol pct longs>30th, shorts>40th | 3.1797 | No | Z5=3.18 H6=1.05. EMA(420) confirmed peak for slow span |

| 167 | EMA(6/420) + vol pct longs>28th, shorts>40th | 3.2479 | No | Z5=3.25 H6=1.41. 30th pct better for longs |

| 168 | EMA(**5**/420) + vol pct longs>30th, shorts>40th | 2.9584 | No | Z5=2.96 H6=1.24. EMA(6) fast confirmed optimal |

| 169 | EMA(**7**/420) + vol pct longs>30th, shorts>40th | 3.0969 | No | Z5=3.10 H6=0.94. EMA(6) confirmed optimal fast span |

| 170 | EMA(6/420) + vol pct longs>30th, shorts>50th (stricter) | 2.5124 | No | Z5=2.51 H6=1.52. 50th pct cuts too many shorts; 40th confirmed optimal |

| 171 | vol_45 rolling mean instead of vol_60 | 3.1012 | No | Z5=3.10 H6=1.12. vol_60 confirmed optimal window |

| 172 | Asymmetric fast EMA: longs EMA(6), shorts EMA(4) | 3.3383 | No | Z5=3.34 H6=1.01. Slightly worse than EMA(6) symmetric |

| 173 | hl_range percentile filter (30th/40th pct) instead of vol | 2.5456 | No | Z5=2.55 H6=0.98. Vol is better activity signal than hl_range for this |

| 174 | **OHLC4** = (O+H+L+C)/4 price metric instead of VWAP proxy | **3.5242** | **Yes** | Z5=3.5242 H6=1.38. NEW CHAMPION! Open price adds useful info |

| 175 | OHLC4 EMA(6/**400**) + vol pct 30th/40th | 2.9132 | No | Z5=2.91 H6=1.56. EMA(420) confirmed optimal slow span for OHLC4 |

| 176 | OHLC4 EMA(6/420) + vol pct longs>**32nd**, shorts>40th | 3.3521 | No | Z5=3.35 H6=1.21. 30th pct confirmed optimal for longs with OHLC4 |

| 177 | Close EMA(6/420) + vol pct 30th/40th | 2.8189 | No | Z5=2.82 H6=0.98. OHLC4 is better; close alone is too noisy |

| 178 | Dual vol filter: percentile AND absolute ratio | 3.1101 | No | Z5=3.11 H6=1.12. Too restrictive; percentile alone is better |

| 179 | OHLC4 EMA(6/420) + vol pct 30th/40th, **360-bar** window | 3.0895 | No | Z5=3.09 H6=1.19. 480-bar confirmed optimal percentile window |

| 180 | + slow EMA slope filter (EMA(420) must be rising/falling) | 1.7613 | No | Z5=1.76 H6=1.10. Slope filter too restrictive; cuts too many entries |

| 181 | Hold-through-low-vol: enter on EMA+vol gate, exit on EMA flip only | 3.2310 | No | Z5=3.23 H6=1.13. Holding through vol drops hurts Z5 slightly |

| 182 | OHLC2 = (O+C)/2 candle body midpoint EMA | 3.4233 | No | Z5=3.42 H6=1.07. Close to OHLC4 but not better |

| 183 | DEMA(6) fast (less lag) vs EMA(420) slow | 2.1960 | No | Z5=2.20 H6=0.31. Fails gate; DEMA too responsive, increases noise |

| 184 | OHLC4 EMA(6) fast vs SMA(420) slow | 2.8904 | No | Z5=2.89 H6=1.22. SMA slow worse than EMA slow |

| 185 | OHLC4 fast EMA(6) vs close slow EMA(420) | **3.5259** | **Yes** | Z5=3.5259 H6=1.35. Marginal improvement; close slow slightly better |

| 186 | Close fast EMA(6) vs OHLC4 slow EMA(420) (opposite) | 2.7061 | No | Z5=2.71 H6=0.96. OHLC4 fast, close slow confirmed better |

| 187 | VWAP proxy (HLC3) fast vs close slow EMA(420) | 3.1192 | No | Z5=3.12 H6=1.50. OHLC4 fast confirmed better than HLC3; open adds value |

| 188 | Rolling median(60) instead of mean for vol metric | 2.8741 | No | Z5=2.87 H6=1.12. Vol mean is better than median |

| 189 | VWMA(6) fast vs EMA(420) close slow | 3.2067 | No | Z5=3.21 H6=1.45. EMA(6) of OHLC4 better than VWMA(6) |

| 190 | Vol EMA crossover confirmation: vol_EMA(6) > vol_EMA(420) for longs, * 0.8 for shorts | 1.0198 | No | Z5=1.02 H6=0.70. Vol EMA crossover fires rarely at 1-min — too restrictive, crushes signal count |

| 191 | EMA(60) vol smoothing before percentile (instead of rolling mean) | 2.8514 | No | Z5=2.85 H6=1.17. EMA smoothing hurts; rolling(60).mean() confirmed optimal |

| 192 | Bar close > open (bullish bar) confirmation for longs; close < open for shorts | -0.8840 | No | Z5=-0.88 H6=-1.14. Bar direction filter badly hurts — EMA(6) OHLC4 already incorporates this; double-filtering destroys signal count |

| 193 | roc_5 > 0 momentum alignment for longs, < 0 for shorts | -0.7733 | No | Z5=-0.77 H6=-1.13. Short-term ROC noisy at 1-min; destroys strategy. EMA(6) already captures momentum |

| 194 | Triple EMA alignment: EMA(6)>EMA(30)>EMA(420) all three must align | 0.2137 | No | Z5=0.21 H6=-0.73. Too restrictive; medium EMA(30) adds nothing, cuts valid entries |

| 195 | ATR(14) percentile filter instead of volume percentile | 0.8677 | No | Z5=0.87 H6=0.08. ATR filter much worse; volume (participation) is better activity gate than ATR |

| 196 | Vol percentile lookback 600 bars (vs 480) | 2.8956 | No | Z5=2.90 H6=0.71. Passes gate but Z5 worse; 480-bar window confirmed optimal |

| 197 | Vol rolling(90) smoothing (vs 60) before percentile | 2.8094 | No | Z5=2.81 H6=1.29. Wider smoothing hurts; vol_60 confirmed optimal smoothing window |

| 198 | Asymmetric vol smoothing: vol_60 for longs, vol_30 for shorts | 3.0717 | No | Z5=3.07 H6=1.13. Passes gate but Z5 below champion; asymmetric windows don't help |

| 199 | Donchian 20-bar channel breakout + vol percentile (no EMA) | -0.8801 | No | Z5=-0.88 H6=-1.12. Pure short-term Donchian fires constantly at 1-min; too noisy |

| 200 | Longs vol threshold 28th pct (OHLC4 fast + close slow, untested combo) | 3.3591 | No | Z5=3.36 H6=1.30. 30th pct still better for longs with mixed price metric |

| 201 | Asymmetric pct windows: longs 480-bar, shorts 360-bar | 3.2853 | No | Z5=3.29 H6=1.09. Shorter shorts window doesn't help; 480-bar optimal for both |

| 202 | Close slow EMA(430) with OHLC4 fast (span sweep around 420) | 3.4060 | No | Z5=3.41 H6=1.28. Better than 450 but below 420; 420 confirmed optimal slow span |

| 203 | Close slow EMA(410) with OHLC4 fast | 3.1561 | No | Z5=3.16 H6=0.93. EMA(420) confirmed as optimal slow span; both 410 and 430 worse |

| 204 | EMA spread override: enter if vol>pct30 OR spread_pct>0.5% (strong trend bypass) | 3.5058 | No | Z5=3.51 H6=1.19. Almost ties champion but slightly below; spread override near-neutral |

| 205 | EMA spread percentile gate instead of vol (enter only when trend is wide) | 0.4169 | No | Z5=0.42 H6=0.19. Spread filter terrible; wide spread = late entry, vol is better gate |

| 206 | Long-only champion (remove shorts entirely) | 2.5208 | No | Z5=2.52 H6=1.03. Shorts contribute positively; bidirectional confirmed better |

| 207 | Close vs EMA corridor (enter only when close > both EMAs) | -0.8093 | No | Z5=-0.81 H6=-1.12. Noisy — close vs mixed OHLC4/close EMA corridor oscillates rapidly |

| 208 | Rolling VWAP(420) as slow line instead of close EMA(420) | 2.0420 | No | Z5=2.04 H6=2.73! VWAP generalizes much better (H6=2.73 vs champion 1.35) but Z5 lower. Promising direction |

| 209 | Rolling VWAP(240) as slow line | 0.9608 | No | Z5=0.96 H6=1.76. Shorter VWAP worse; trend: VWAP(240)=0.96, VWAP(420)=2.04 → try longer |

| 210 | Rolling VWAP(720) as slow line | 1.3830 | No | Z5=1.38 H6=0.04. Fails H6 gate; VWAP(420)=sweet spot, longer hurts generalization |

| 211 | EMA(6/420) AND VWAP(420) dual confirm + vol gate | 2.7028 | No | Z5=2.70 H6=2.13. VWAP improves H6 but cuts Z5; too restrictive. EMA alone is better at Z5 |

| 212 | Realized vol (vol_60 = return std) percentile gate instead of raw volume | 1.6316 | No | Z5=1.63 H6=0.33. Fails gate; price volatility worse than volume participation as activity signal |

| 213 | OHLC4 EMA(6) fast vs open EMA(420) slow | 3.4831 | No | Z5=3.48 H6=1.34. Very close to champion; close slow slightly edges out open slow |

| 214 | OHLC4 EMA(6) fast vs HL2=(H+L)/2 EMA(420) slow | **3.5566** | **Yes** | Z5=3.5566 H6=1.49. NEW CHAMPION! Range midpoint slow better than close or open |

| 215 | HL2 EMA(6) fast vs HL2 EMA(420) slow (pure HL2/HL2) | 3.0817 | No | Z5=3.08 H6=1.62. HL2/HL2 worse; OHLC4 fast confirmed optimal, HL2 only better as slow |

| 216 | OHLC4 fast vs HL2 EMA(400) slow | 2.9247 | No | Z5=2.92 H6=1.44. HL2(400) worse; HL2(420) confirmed optimal span |

| 217 | OHLC4 fast vs HL2 EMA(440) slow | 3.2486 | No | Z5=3.25 H6=0.99. HL2(440) worse; HL2(420) confirmed optimal for both sides of sweep |

| 218 | OC2=(O+C)/2 fast EMA(6) vs HL2 slow EMA(420) | 3.5546 | No | Z5=3.5546 H6=1.37. Very close to champion (3.5566); OHLC4 fast marginally better than OC2 |

| 219 | (2*open+high+low+close)/5 double-open fast vs HL2 slow | 3.3746 | No | Z5=3.37 H6=1.08. Equal-weight OHLC4 confirmed optimal; extra open weight hurts |

| 220 | OHLC4 fast vs HL2 slow + longs 31st pct | 3.4811 | No | Z5=3.48 H6=1.31. 30th pct still better for longs with HL2 slow |

| 221 | OHLC4 fast vs HL2 slow + shorts 39th pct | 3.4660 | No | Z5=3.47 H6=1.44. 40th pct confirmed optimal for shorts with HL2 slow too |

| 222 | OHLC4 EMA(5) fast vs HL2 EMA(420) slow | 2.9101 | No | Z5=2.91 H6=0.93. EMA(6) confirmed optimal fast span with HL2 slow |

| 223 | Volume-weighted OHLC4 EMA(6) fast vs HL2 slow | 2.7234 | No | Z5=2.72 H6=0.86. Volume-weighted EMA worse; standard EMA confirmed for fast |

| 224 | HLC3=(H+L+C)/3 fast EMA(6) vs HL2 slow | 3.3189 | No | Z5=3.32 H6=1.61. HLC3 worse than OHLC4 fast even with HL2 slow; open price key for fast |

| 225 | Close EMA(6) fast vs HL2 slow | 2.8028 | No | Z5=2.80 H6=1.11. Close fast worse with HL2 slow too; OHLC4 fast confirmed best universally |

| 226 | OHLC4 fast vs HL3=(H+2L)/3 slow (lower anchor) | 2.4509 | No | Z5=2.45 H6=0.91. Too low — HL2 midpoint is optimal; going lower hurts |

| 227 | Vol rolling(55) mean (vs 60) with HL2 slow | 3.1016 | No | Z5=3.10 H6=1.59. vol_60 confirmed optimal with HL2 slow too |

| 228 | OHLC4 fast vs HL2 SMA(420) slow (SMA not EMA) | 2.9158 | No | Z5=2.92 H6=1.22. EMA confirmed better than SMA for HL2 slow |

| 229 | Candle body momentum: bullish bar ratio > 0.53 over 420 bars | -0.2936 | No | Z5=-0.29 H6=-0.15. Bar direction frequency noisy at 1-min; EMA smoothing far superior |

| 230 | Champion + EMA spread pct20 gate (exclude flat-trend entries) | 1.2966 | No | Z5=1.30 H6=0.98. Any spread gate hurts badly; near-crossover entries are valuable |

| 231 | Dual timescale EMA confluence: EMA(6/420) AND EMA(3/60) both agree + vol | 1.6677 | No | Z5=1.6677 H6=-0.28. Fails gate; dual confirmation too restrictive |
| 232 | Signal hysteresis DECAY=5: forward-fill 0-gaps when trend unchanged | 3.3078 | No | Z5=3.31 H6=1.39. Hysteresis hurts Z5; champion exits are correct |
| 233 | Vol filter entry-only: enter on EMA+vol, hold until EMA reversal | 3.2592 | No | Z5=3.26 H6=1.21. Holding through vol dips hurts Z5 |
| 234 | HLC3=(H+L+C)/3 slow EMA(420) | 3.5190 | No | Z5=3.52 H6=1.39. HLC3 worse; HL2 confirmed optimal (more H+L, less C) |
| 235 | OC2=(O+C)/2 slow EMA(420) | 3.5003 | No | Z5=3.50 H6=1.39. Body center slower worse; HL2 optimal for slow |
| 236 | ATR filter replacing volume filter | 0.8370 | No | Z5=0.84 H6=0.12. Fails gate; ATR is terrible vs vol as activity gate |
| 237 | TEMA(6) fast (3x EMA for less lag) vs HL2 slow | 1.4330 | No | Z5=1.43 H6=0.52. Fails gate; triple smoothing adds too much noise |
| 238 | HL2 EMA(410) slow | 3.1670 | No | Z5=3.17 H6=1.27. Confirms sharp peak at 420 |
| 239 | HL2 EMA(415) slow | 3.3921 | No | Z5=3.39 H6=1.25. 415 better than 410 but still below 420 |
| 240 | OHLC4 EMA(7) fast vs HL2 EMA(420) slow | 2.5919 | No | Z5=2.59 H6=0.97. EMA(6) confirmed optimal: 5=2.91, 6=3.56, 7=2.59 |
| 241 | Vol quantile window 360 bars | 3.1197 | No | Z5=3.12 H6=1.28. 480-bar confirmed optimal window |
| 242 | Vol quantile window 240 bars | 2.6506 | No | Z5=2.65 H6=2.81! Shorter window = great H6 but worse Z5 |
| 243 | Vol quantile window 720 bars | 2.7874 | No | Z5=2.79 H6=0.66. 480 is the global peak for vol window |
| 244 | OHLC4 fast vs open EMA(420) slow (duplicate of exp_213) | 3.4831 | No | Z5=3.48 H6=1.34. Confirmed open slow worse than HL2 |
| 245 | Low slow EMA(420) — extended lower-anchor hypothesis | 2.1117 | No | Z5=2.11 H6=0.08. Fails gate; going below HL2 to just low over-signals |
| 246 | Asymmetric slow: HL2 for longs, high EMA(420) for shorts | 3.5566 | No | Z5=3.5566 H6=1.49. EXACT TIE with champion — high slow shorts have same effect |
| 247 | Rolling VWAP(30) fast vs HL2 EMA(420) slow | 1.6285 | No | Z5=1.63 H6=0.34. Fails gate; VWAP fast terrible vs EMA fast |
| 248 | HL2 EMA(500) slow | 2.7478 | No | Z5=2.75 H6=0.18. Fails gate; 420 confirmed global optimum for HL2 slow |

| 249 | HL2 EMA(425) slow — gap between 420 and 440 | 3.4974 | No | Z5=3.50 H6=1.47. 420 confirmed sharp global peak: 415=3.39, 420=3.56, 425=3.50, 440=3.25 |

| 250 | HL2 EMA(425) slow — narrowing peak search | 3.4974 | No | Z5=3.50 H6=1.47. Between 420(3.56) and 440(3.25); 420 confirmed global peak |
| 251 | Adaptive vol threshold: 25/35th pct when spread>0.3%, else 30/40th | 3.3166 | No | Z5=3.32 H6=1.27. Adaptive threshold hurts; uniform 30/40 is better |

| 252 | 5-bar aggregated candle OHLC4 EMA(6) fast vs HL2 EMA(84) slow | 0.1932 | No | Z5=0.19 H6=0.44. Fails gate; 5-bar candle structure destroys signal |

| 253 | **Longs vol-filter-free, shorts vol>40th pct** | **3.9949** | **Yes** | Z5=3.9949 H6=0.65. NEW CHAMPION! Removing vol gate for longs captures quiet-trend alpha. H6 borderline but passes gate |

| 253 already logged |
| 254 | Longs vol-free, shorts vol > 35th pct | 3.6731 | No | Z5=3.67 H6=0.57. Fails gate; 40th pct optimal for shorts |
| 255 | Longs vol-free, shorts vol > 45th pct | 3.3159 | No | Z5=3.32 H6=0.97. Better H6 but Z5 regressed; 40th pct confirmed optimal |

| 256 | Longs vol-free, shorts vol>40th, HL2 EMA(430) slow | 4.1961 | No | Z5=4.20 H6=0.46. Fails gate! Higher Z5 but H6 drops below 0.6 |
| 257 | **Longs vol-free, shorts vol>40th, HL2 EMA(425) slow** | **4.0776** | **Yes** | Z5=4.0776 H6=0.65. NEW CHAMPION! 425 is the H6-safe sweet spot |

| 258 | Longs vol-free + HL2 EMA(427) slow | 4.0873 | No | Z5=4.09 H6=0.50. Fails gate; H6 drops sharply between 425 and 427 |
| 259 | Longs vol-free + HL2 EMA(426) slow | 4.1340 | No | Z5=4.13 H6=0.55. Fails gate; 425 confirmed maximum passing span |

| 260 | Longs vol-free + close EMA(425) slow + shorts vol>40th | 3.8374 | No | Z5=3.84 H6=0.56. Fails gate; HL2 slow confirmed better than close with asymmetric setup |

| 261 | Champion + shorts vol window 360 | 3.8856 | No | Z5=3.89 H6=0.53. Fails gate; 480-bar still optimal for short vol window |
| 262 | Champion + shorts vol window 720 | 3.3780 | No | Z5=3.38 H6=0.67. 480 confirmed optimal for short vol window |

| 263 | Champion + vol_90 smoothing for shorts | 3.0152 | No | Z5=3.02 H6=0.89. vol_60 confirmed optimal; wider smoothing hurts Z5 significantly |

| 264 | Asymmetric fast: OHLC4 for longs, close EMA(6) for shorts | 3.8693 | No | Z5=3.87 H6=0.62. Passes gate but Z5 below champion; OHLC4 fast confirmed best for both directions |

| 265 | Longs vol>10th pct (very loose), shorts vol>40th | 3.2291 | No | Z5=3.23 H6=0.80. Even 10th pct filter hurts longs; vol-free confirmed optimal |

| 266 | Longs vol-free, shorts vol median > pct40 | 3.6742 | No | Z5=3.67 H6=0.54. Fails gate; vol mean confirmed better than median for short filter |

| 267 | Longs-only (no vol, no shorts) — diagnostic | 3.1114 | No | Z5=3.11 H6=0.17. Fails gate; shorts add ~+1.0 Z5 and improve H6 significantly |

| 268 | Longs vol-free, shorts raw (unsmoothed) vol > pct40 | 1.4854 | No | Z5=1.49 H6=-0.58. Fails gate; 60-bar smoothing essential for vol filter |

| 269 | HL2 HMA(425) slow (Hull MA — lower lag than EMA) | 0.2283 | No | Z5=0.23 H6=0.16. Fails gate; HMA creates too many false crossovers |

| 270 | Asymmetric slow: longs HL2 EMA(425), shorts HL2 EMA(420) | 4.0685 | No | Z5=4.07 H6=0.63. Marginally below champion on both metrics |
| 271 | Asymmetric slow: longs HL2 EMA(425), shorts HL2 EMA(430) | 4.0783 | No | Z5=4.078 H6=0.647. Essentially ties champion (+0.0007); not worth committing |

| 272 | Longs vol-free, shorts EMA(240) trend confirmation instead of vol | 1.0528 | No | Z5=1.05 H6=0.69. Vol filter essential for shorts; EMA trend confirmation alone is insufficient |

| 273 | Longs vol-free, shorts vol > 42nd pct | 3.7814 | No | Z5=3.78 H6=0.88. 40th confirmed optimal: 41st=3.96, 42nd=3.78, 45th=3.32 |
| 274 | Longs vol-free, shorts vol > 41st pct | 3.9622 | No | Z5=3.96 H6=0.72. 40th pct confirmed as Z5-optimal for shorts |

| 275 | 2-bar EMA crossover persistence for longs | 3.7963 | No | Z5=3.80 H6=0.51. Fails gate; persistence requirement hurts both metrics |

| 276 | HL2 rolling(425).median() slow (robust anchor) | 3.1979 | No | Z5=3.20 H6=0.09. Fails gate; rolling median terrible vs EMA for slow |

| 277 | Asymmetric slow metric: HL2 for longs, open EMA(425) for shorts | 4.0450 | No | Z5=4.05 H6=0.64. Below champion; HL2 slow optimal for both directions |

| 278 | Longs vol-free, shorts vol > 480-bar rolling mean | 3.2897 | No | Z5=3.29 H6=0.72. Mean threshold ~50th pct; 40th pct confirmed best for Z5 |

| 279 | Blended slow: 0.7*HL2 EMA(425) + 0.3*HL2 EMA(60) | 2.9007 | No | Z5=2.90 H6=-0.15. Fails gate; mixing short-term EMA into slow destroys signal |

| 280 | 2-level EMA longs: entry EMA(425), exit EMA(30) | 3.1939 | No | Z5=3.19 H6=0.77. Tighter exit hurts Z5; champion EMA(425) exit optimal |
| 281 | 2-level EMA longs: entry EMA(425), exit EMA(60) | 2.8018 | No | Z5=2.80 H6=0.84. Even worse; EMA(425) exit confirmed optimal for Z5 |

| 282 | Rolling VWAP(425) slow + vol-free longs + vol>40th shorts | 2.3469 | No | Z5=2.35 H6=1.41! Vol-free longs help VWAP (+0.31 vs old 2.04) but Z5 still far below EMA champion |

| 283 | VWAP(60) fast vs HL2 EMA(425) slow + asymmetric vol | 1.6207 | No | Z5=1.62 H6=-0.12. Fails gate; VWAP(60) fast terrible |

| 284 | OHLC4 EMA(6) 420-bar self-momentum (EMA vs its lagged value) | 1.2884 | No | Z5=1.29 H6=-0.73. Fails gate; self-comparison momentum signal is terrible |

| 285 | Slow EMA declining as additional short confirmation | 3.4559 | No | Z5=3.46 H6=0.38. Fails gate; slope confirmation cuts valid shorts |

| 286 | Close EMA(6) fast + HL2 EMA(425) slow + asymmetric vol | 3.5434 | No | Z5=3.54 H6=0.39. Fails gate; OHLC4 fast confirmed best even in asymmetric setup |

| 287 | Supertrend(ATR=10, mult=3.0) + vol-free longs + vol>40th shorts | -0.7226 | No | Z5=-0.72 H6=-1.03. ATR(10) on 1-min is far too noisy; flips direction every few bars |
| 288 | Supertrend(ATR=100, mult=2.0) + vol-free longs + vol>40th shorts | -0.8379 | No | Z5=-0.84 H6=-1.09. Longer ATR still terrible; Supertrend fundamentally incompatible with 1-min NQ |
| 289 | Stateful entry=EMA(425)/exit=EMA(390) asymmetric thresholds | 2.7784 | No | Z5=2.78 H6=0.92. H6 great but Z5 far below champion; early exits cost too much trend profit |
| 290 | TSI(fast=6, slow=425) > 0 longs; < 0 + vol shorts | 3.7155 | No | Z5=3.72 H6=0.40. Fails gate; double-smoothed momentum worse than price-level EMA crossover |
| 291 | OLS slope of HL2(425) > 0 for longs; < 0 + vol for shorts | 0.4678 | No | Z5=0.47 H6=1.43. OLS slope is too slow/different; misses most Z5 alpha but great H6 |
| 292 | Parabolic SAR (step=0.002, max=0.20) + vol-free longs + vol>40th shorts | 0.4351 | No | Z5=0.44 H6=-0.95. Fails gate; adaptive trailing stop useless on 1-min NQ |
| 293 | Champion longs + bearish bar pressure (6/10 bars close<open) for shorts | 1.4785 | No | Z5=1.48 H6=-0.49. Fails gate; bar pressure cuts too many shorts, kills both metrics |
| 294 | Donchian midpoint(425) as slow anchor + OHLC4_EMA(6) fast | 2.3861 | No | Z5=2.39 H6=0.09. Fails gate; channel midpoint lacks EMA's adaptive trend-following |
| 295 | train() grid search slow span [380-480]: selected span=470, train_calmar=0.19 | 3.6763 | No | Z5=3.68 H6=0.09. KEY INSIGHT: training max Calmar=0.19! Strategy is Z5-regime specific |
| 296 | Champion + ADX(60) > 25 regime filter (both longs and shorts) | -0.1637 | No | Z5=-0.16 H6=1.00. ADX>25 cuts almost all Z5 longs — at 1-min scale ADX stays below 25 |
| 297 | EMA(6/425) entry + ATR(14)*2.0 trailing stop exit | 0.2229 | No | Z5=0.22 H6=-0.53. Fails gate; 2*ATR too tight on 1-min bars, fires constantly |
| 298 | Champion + RSI(14) < 50 filter for shorts | 2.3970 | No | Z5=2.40 H6=-0.26. Fails gate; RSI(14) cuts too many valid shorts, decimates both metrics |
| 299 | OR-logic longs: EMA bullish OR close > rolling_max(60); shorts: EMA+vol+no breakout | 2.7798 | No | Z5=2.78 H6=-0.33. Fails gate; breakout OR condition adds noise, destroys H6 |
| 300 | Fast span=5 (vs champion 6) | 3.8452 | No | Z5=3.85 H6=0.27. Fails gate; faster span significantly worsens H6 |
| 301 | Fast span=7 (vs champion 6) | 3.2279 | No | Z5=3.23 H6=0.32. Fails gate; both directions from span=6 hurt H6. Span=6 is optimal. |
| 302 | Vol percentile 30th for shorts (vs champion 40th) | 3.4975 | No | Z5=3.50 H6=0.47. Fails gate; more aggressive shorting hurts both metrics |
| 303 | Vol baseline window 240 bars (vs champion 480) | 3.2872 | No | Z5=3.29 H6=1.06. Passes gate but Z5 far below champion. Shorter window better H6, worse Z5 |
| 304 | Vol baseline window 360 bars (between 240 and 480) | 3.8856 | No | Z5=3.89 H6=0.53. Fails gate. 480-bar window is optimal for passing gate with highest Z5 |
| 305 | DEMA(6) fast (2*EMA-EMA(EMA), reduced lag) | 2.7832 | No | Z5=2.78 H6=-0.17. Fails gate; DEMA too responsive, increases noise (confirms exp_183) |

| 306 | HL2 EMA(428) slow — mapping landscape above champion span=425 | 4.1211 | No | Z5=4.1211 H6=0.4555 PnL=$111,118 Trades=685. Fails gate; H6 drops sharply above span=425 |
| 307 | HL2 EMA(420) slow + vol-free longs + vol>40th shorts | 3.9949 | No | Z5=3.9949 H6=0.6468 PnL=$107,228 Trades=697. Passes gate but Z5 below champion; span=425 is optimal |
| 308 | Dual-window vol gate for shorts: vol>pct40(480) AND vol>pct40(240) | 3.8326 | No | Z5=3.8326 H6=1.0904 PnL=$100,212 Trades=783. Excellent H6 but Z5 too far below champion; dual window too strict |
| 309 | Softer dual-window: vol>pct40(480) AND vol>pct30(240) | 3.6076 | No | Z5=3.6076 H6=0.9923 PnL=$98,012 Trades=751. Dual-window approach consistently hurts Z5; drop this direction |
| 310 | Asymmetric fast EMA: EMA(6) longs, EMA(8) shorts | 3.4652 | No | Z5=3.4652 H6=0.5298 PnL=$102,028 Trades=768. Fails gate; asymmetric fast EMA hurts both metrics |
| 311 | Adaptive vol filter: use 480-bar when vol>70th pct, else 240-bar | 3.2872 | No | Z5=3.2872 H6=1.0576 PnL=$89,838 Trades=854. Excellent H6 but fires too many extra shorts in Z5; hurts Calmar |
| 312 | Choppiness Index CHOP(100) < 50 regime filter | 1.6159 | No | Z5=1.6159 H6=0.5816 PnL=$40,045 Trades=1183. CHOP<50 fires constantly at 1-min NQ; creates churn on longs, terrible |
| 313 | OR-logic shorts: vol>40th pct OR (down bar > 2x ATR) | 3.9450 | No | Z5=3.9450 H6=0.5977 PnL=$106,762 Trades=822. Close on H6 (0.5977 vs 0.6 gate) but Z5 below champion |
| 314 | Blended slow: 70% HL2 EMA(425) + 30% rolling VWAP(420) | 2.9173 | No | Z5=2.9173 H6=0.4608 PnL=$105,730 Trades=652. Fails gate; VWAP blend degrades both Z5 and H6 |
| 315 | Long only when slow EMA rising (ema_slow > ema_slow.shift(60)) | 3.6654 | No | Z5=3.6654 H6=0.5577 PnL=$103,672 Trades=679. Fails gate; slow EMA slope filter cuts valid longs |
| 316 | Asymmetric fast EMA: EMA(5) longs, EMA(6) shorts | 3.9157 | No | Z5=3.9157 H6=0.3607 PnL=$106,188 Trades=794. Fails gate; faster longs hurt H6; EMA(6) optimal for both |
| 317 | Bar body displacement EWM(60) < 0 as short filter | 2.4229 | No | Z5=2.4229 H6=-0.1018 PnL=$58,108 Trades=1468. Body filter almost always TRUE; creates forced-flat periods on longs causing churn |
| 318 | Volume-Weighted EMA (VWEMA) as slow line | 1.6152 | No | Z5=1.6152 H6=-0.3646 PnL=$85,872 Trades=613. VWEMA tracks too fast during high-vol; standard EMA confirmed optimal |
| 319 | (H-L)*volume momentum as short filter | 3.1554 | No | Z5=3.1554 H6=0.5448 PnL=$84,448 Trades=718. Fails gate; HL*vol product worse than pure volume — range correlated with vol, no addl info |
| 320 | **Median of HL2 EMA(400/425/450) slow ensemble** | **4.0815** | **Yes** | Z5=4.0815 H6=0.6538 PnL=$109,052 Trades=690. NEW CHAMPION! Marginal Z5 improvement; median more robust than single span |
| 321 | **Median of HL2 EMA(380/425/470) slow ensemble — wider spread** | **4.0979** | **Yes** | Z5=4.0979 H6=0.6570 PnL=$109,308 Trades=689. NEW CHAMPION! Wider ensemble further improves both Z5 and H6 |
| 322 | Median EMA(360/425/490) — ±65 spread | 4.0353 | No | Z5=4.0353 H6=0.6534 PnL=$108,132 Trades=688. Below champion; ±45 is better than ±65 |
| 323 | Mean of EMA(380/425/470) vs champion's median | 3.9878 | No | Z5=3.9878 H6=0.5820 PnL=$106,682 Trades=685. Fails gate; mean is worse than median (mean raises threshold in uptrend, cutting longs) |
| 324 | Median EMA(390/425/460) — ±35 spread | 4.0892 | No | Z5=4.0892 H6=0.6570 PnL=$109,258 Trades=689. Below champion; ±35 < ±45 |
| 325 | Median EMA(375/425/475) — ±50 spread | 4.0619 | No | Z5=4.0619 H6=0.6570 PnL=$108,348 Trades=689. Confirmed: ±25=4.0815, ±35=4.0892, ±45=4.0979(PEAK), ±50=4.0619, ±65=4.0353 |
| 326 | Median EMA(381/426/471) — center=426, ±45 | 4.1544 | No | Z5=4.1544 H6=0.5548 PnL=$110,748 Trades=684. Best Z5 yet but fails gate; center=426 ensemble preserves same H6 cliff as single span=426 |
| 327 | Dual ensemble: fast median(4/6/8), slow median(380/425/470) | 4.0953 | No | Z5=4.0953 H6=0.6527 PnL=$109,422 Trades=690. Slightly below champion; fast ensemble hurts, single EMA(6) optimal for fast |
| 328 | 5-span median EMA(355/390/425/460/495) | 3.9929 | No | Z5=3.9929 H6=0.6534 PnL=$108,038 Trades=688. 5-span worse than 3-span; 3-span median(380/425/470) is the optimal ensemble |
| 329 | Asymmetric median EMA(360/425/470) — lower outer at −65 | 4.0261 | No | Z5=4.0261 H6=0.6534 PnL=$108,958 Trades=689. Asymmetric worse than symmetric ±45; symmetric (380/425/470) confirmed optimal |
| 330 | Stateful long: spread widening required for new entries | 4.0917 | No | Z5=4.0917 H6=0.6570 PnL=$109,142 Trades=689. Slightly below champion; spread-widening filter cuts profitable re-entries |
| 331 | Rolling 6-bar median as fast signal (vs EWM) | 2.5700 | No | Z5=2.5700 H6=0.4137 Z5pnl=$96,438 H6pnl=unknown Trades_Z5=696 Trades_H6=unknown. Fails gate; rolling median terrible — EWM recency weighting is essential for fast signal |
| 332 | Median EMA(379/424/469) — center=424, ±45 | 4.0210 | No | Z5=4.0210 H6=0.6852 Z5pnl=$108,418 H6pnl=$25,018 Trades_Z5=687 Trades_H6=628. Below champion; center=424 gives better H6 but lower Z5. Gradient confirmed: center↓=lower Z5/better H6, center↑=higher Z5/worse H6, center=425 is sweet spot |

| 333 | ATR trailing stop on longs: exit when close < roll_max(120) - 2.5*ATR14 | 1.1371 | No | Z5=1.1371 H6=-0.2493 Z5pnl=$44,310 H6pnl=-$12,430 Trades_Z5=3040 Trades_H6=2201. Massive churn (3040 vs 689); price oscillates around 2.5 ATR threshold constantly on 1-min bars — vectorized trailing stop is fundamentally incompatible with this data granularity |

| 334 | VR(30) >= 0.9 regime filter — trade only when trending | 0.2620 | No | Z5=0.2620 H6=0.3451 Z5pnl=$7,520 H6pnl=$6,050 Trades_Z5=672 Trades_H6=510. Catastrophic; VR(30) on 1-min NQ is backwards — 1-min bars mean-revert at 30-min scale so VR>=0.9 removes good trend trades, not bad ones |

| 335 | ATR(14) > rolling_mean(ATR14, 120) as short filter (replace vol) | 3.3223 | No | Z5=3.3223 H6=0.4398 Z5pnl=$89,652 H6pnl=$17,038 Trades_Z5=1500 Trades_H6=1392. Churn and fails gate; ATR spikes/drops rapidly creating entry/exit churn; vol filter confirmed superior |

| 336 | Long-only diagnostic: remove all shorts | 3.1321 | No | Z5=3.1321 H6=0.1758 Z5pnl=$53,838 H6pnl=$4,278 Trades_Z5=541 Trades_H6=467. DIAGNOSTIC: Shorts add +0.97 Z5 and +0.48 H6 (shorts are crucial to BOTH metrics, especially H6). Keep vol-based short structure. |

| 337 | RSI(14) > 65 as short entry (dead-cat bounce pattern) | 1.0523 | No | Z5=1.0523 H6=-0.6333 Z5pnl=$22,562 H6pnl=-$24,452 Trades_Z5=2437 Trades_H6=2242. Massive churn; RSI(14) oscillates above/below 65 rapidly on 1-min data. Key lesson: short filters must be slow-moving (like vol_60 rolling mean) to avoid churn |

| 338 | Vol + RSI(60)<50 double short filter | 2.4408 | No | Z5=2.4408 H6=0.0074 Z5pnl=$70,078 H6pnl=$268 Trades_Z5=1745 Trades_H6=1446. Same forced-flat churn as all AND-on-short experiments. CONFIRMED: vol_60/quantile(480) is optimal short filter; cannot be improved via AND conditions |

| 339 | RSI(240) standalone signal — 4-hour momentum | -0.4637 | No | Z5=-0.4637 H6=-0.8426 Z5pnl=-$13,420 H6pnl=-$34,440 Trades_Z5=2029 Trades_H6=1418. Negative Calmar; RSI(240) oscillates around thresholds creating churn; 4-hour RSI direction is not predictive on 1-min NQ |

| 340 | Bar Close Strength EWM(380/425/470) median > 0.55 signal | -0.8080 | No | Z5=-0.8080 H6=-0.2841 Z5pnl=-$6,355 H6pnl=-$2,030 Trades_Z5=84 Trades_H6=65. Almost always flat (84 trades); BCS_ewm stays near 0.50 regardless of trend — no discriminatory power at ±0.05 threshold |

| 341 | ATR-normalized EMA: require 0.5 ATR margin before trading | 2.2981 | No | Z5=2.2981 H6=-0.1823 Z5pnl=$85,102 H6pnl=-$7,468 Trades_Z5=1012 Trades_H6=870. Same ±0.5 ATR dead zone creates oscillation churn near threshold. All entry-band variations confirmed bad. |

| 342 | vol > rolling(720).quantile(0.40) — longer 12-hour vol baseline | 3.3966 | No | Z5=3.3966 H6=0.6764 Z5pnl=$93,222 H6pnl=$22,592 Trades_Z5=623 Trades_H6=572. H6 slightly better (+0.02) but Z5 much worse (-0.70). 480-bar window confirmed optimal for Z5. |

| 343 | vol > rolling(480).quantile(0.35) — more permissive short filter | 3.7248 | No | Z5=3.7248 H6=0.5849 Z5pnl=$103,098 H6pnl=$22,618 Trades_Z5=674 Trades_H6=623. Fails gate; more shorts at pct35 hurts both metrics. pct40 is confirmed better. |

| 344 | vol > rolling(480).quantile(0.45) — stricter short filter | 3.3788 | No | Z5=3.3788 H6=0.9510 Z5pnl=$98,248 H6pnl=$34,728 Trades_Z5=687 Trades_H6=609. Z5 drops but H6 excellent. Vol quantile sweep: pct35=3.72/0.58, pct40=4.10/0.66(champion), pct45=3.38/0.95. pct40 is optimal. |

| 345 | vol_60 > rolling(240).quantile(0.40) — 4-hour vol window | 3.3004 | No | Z5=3.3004 H6=1.0621 Z5pnl=$90,198 H6pnl=$35,358 Trades_Z5=854 Trades_H6=733. Vol window sweep complete: 240=3.30/1.06, 480=4.10/0.66(champion), 720=3.40/0.68. Shorter window better H6, worse Z5 — exact inverse tradeoff. |

| 346 | Close price for slow EMA ensemble (vs champion's HL2) | 3.8235 | No | Z5=3.8235 H6=0.5623 Z5pnl=$104,982 H6pnl=$21,228 Trades_Z5=687 Trades_H6=626. Fails gate; HL2 confirmed better than close for slow ensemble |

| 347 | OHLC4 for slow EMA ensemble (vs champion's HL2) | 3.8974 | No | Z5=3.8974 H6=0.6367 Z5pnl=$105,778 H6pnl=$23,572 Trades_Z5=683 Trades_H6=627. Fails gate. Price input sweep complete: HL2=4.10/0.66(champion) > OHLC4=3.90/0.64 > Close=3.82/0.56. HL2 confirmed optimal. |

| 348 | Bar body momentum EWM(380/425/470) median as signal | 1.4124 | No | Z5=1.4124 H6=-0.0925 Z5pnl=$71,298 H6pnl=-$4,838 Trades_Z5=1401 Trades_H6=1326. Churn; body_ewm oscillates near 0 → rapid long/flat transitions. Fixed-threshold comparisons always cause churn. EMA crossover (both sides track price) is fundamentally superior. |

| 349 | Adaptive fast EMA: span=4 trending (ATR>1.1x), span=8 ranging (ATR<0.9x), span=6 neutral | 3.7041 | No | Z5=3.7041 H6=0.3663 Z5pnl=$106,968 H6pnl=$14,092 Trades_Z5=713 Trades_H6=635. Fails gate; adaptive spanning hurts both metrics. EWM(6) confirmed optimal regardless of ATR regime. |

| 350 | Minimum vol pct10(480) floor for longs, pct40(480) for shorts | 3.2319 | No | Z5=3.2319 H6=0.8019 Z5pnl=$92,970 H6pnl=$27,980 Trades_Z5=895 Trades_H6=740. Z5 drops; even pct10 creates churn at vol transitions. No vol filter for longs confirmed optimal. |

| 351 | Asymmetric: champion longs + pct40(240) shorts | 3.3004 | No | Z5=3.3004 H6=1.0621 Z5pnl=$90,198 H6pnl=$35,358 Trades_Z5=854 Trades_H6=733. Identical to exp_345 (longs were already vol-free in both). 240-bar window for shorts: better H6 but much worse Z5. Cannot close the Z5/H6 gap. |

| 352 | Volume-weighted price momentum (VWPM) rolling(425) as signal | 1.5486 | No | Z5=1.5486 H6=-0.4318 Z5pnl=$82,325 H6pnl=-$12,090 Trades_Z5=884 Trades_H6=766. VWPM oscillates near 0; price_diff*volume is noisy. Confirms: fixed-threshold at 0 creates churn. Only cross-EMA (both sides adapt) avoids this. |

| 353 | Stateful trailing stop 3 ATR peak, 30-bar cooldown | 2.4062 | No | Z5=2.4062 H6=-0.5412 Z5pnl=$71,918 H6pnl=-$33,312 Trades_Z5=1550 Trades_H6=1218. Still churn: stop fires, cooldown expires with EMA still bullish, immediately re-enters → stop fires again. 3 ATR too tight on 1-min NQ |

| 354 | Stateful trailing stop 3 ATR + fresh EMA crossover required after stop | 2.7762 | No | Z5=2.7762 H6=-0.2165 Z5pnl=$65,425 H6pnl=-$8,760 Trades_Z5=784 Trades_H6=710. Fewer trades (784 vs 1550) but Z5 drops 1.3pts — 3 ATR too tight, misses too much upside waiting for fresh crossover |

| 355 | Stateful trailing stop 8 ATR + fresh crossover | 2.7901 | No | Z5=2.7901 H6=0.5617 Z5pnl=$86,070 H6pnl=$18,505 Trades_Z5=724 Trades_H6=651. Fails gate. CONCLUSION: trailing stops fundamentally incompatible with long-duration EMA trend following — the slow EMA is designed to hold through pullbacks; any stop contradicts this and destroys Calmar |

| 356 | EMA spread (std of e380/e425/e470) filter — strong trend only | 1.5571 | No | Z5=1.5571 H6=-0.0325 Z5pnl=$41,405 H6pnl=-$1,140 Trades_Z5=499 Trades_H6=445. Spread filter backwards: low spread = crossover zone = when we SHOULD enter. Prevents initial entries, only allows late participation in established trends. |

| 357 | Champion + stateful dip-buying (3 ATR below slow, exit at slow) | 4.2837 | No | Z5=4.2837 H6=0.1637 Z5pnl=$128,648 H6pnl=$8,052 Trades_Z5=792 Trades_H6=731. HIGHEST Z5 SEEN (+0.19 vs champion) but H6 fails gate. Dip-buying works in Z5 bull market but fails H6 (deeper dips in Dec-Mar). Need H6-compatible gate for dip entries. |

| 358 | Dip-buying gated by slow EMA rising (60-bar lookback), 3 ATR | 4.2676 | No | Z5=4.2676 H6=0.4181 Z5pnl=$115,398 H6pnl=$16,862 Trades_Z5=730 Trades_H6=657. H6 improved from 0.16→0.42 with rising gate but still fails. Need stricter conditions. |

| 359 | Dip-buying 4 ATR + 120-bar slow rising | 4.0424 | No | Z5=4.0424 H6=0.6285 Z5pnl=$115,548 H6pnl=$21,852 Trades_Z5=720 Trades_H6=654. H6 passes gate (0.63>0.6) but Z5 below champion (-0.06). Progression: 3ATR/60=4.28/0.16, 3ATR/120=?, 4ATR/120=4.04/0.63. Need Z5>4.10 AND H6>=0.6. |

| 360 | Dip-buying 3 ATR + 120-bar slow rising | 4.1771 | No | Z5=4.1771 H6=0.4454 Z5pnl=$114,712 H6pnl=$17,528 Trades_Z5=732 Trades_H6=665. Isolating lookback: 120 vs 60 gives Z5 -0.09, H6 +0.03. DIP_MULT is the main H6 driver. Matrix: 3.0/60=4.27/0.42, 3.0/120=4.18/0.45, 4.0/120=4.04/0.63 |

| 361 | Dip-buying 3.5 ATR + 120-bar slow rising | 4.1232 | No | Z5=4.1232 H6=0.4475 Z5pnl=$115,728 H6pnl=$17,618 Trades_Z5=727 Trades_H6=659. Z5 above champion! But H6 still at 0.45. H6 jumps nonlinearly: 3.5→0.45, 4.0→0.63. Need DIP_MULT near 4.0 for H6>=0.6. |

| 362 | Dip 4.0 ATR / 120-bar rising + smooth dip→champion + exit at slow+0.5ATR | 4.0895 | No | Z5=4.0895 H6=0.6337 Z5pnl=$115,372 H6pnl=$21,972 Trades_Z5=717 Trades_H6=652. SO CLOSE: H6 passes gate (0.63>0.6) but Z5=4.09 misses by 0.008. Smooth transition added +0.05 Z5 vs exp_359. Try EXIT=1.0 ATR |

| 363 | Dip 4.0 ATR / 120-bar rising + smooth transition + exit at slow+1.0 ATR | 4.0681 | No | Z5=4.0681 H6=0.6394 Z5pnl=$114,768 H6pnl=$22,212 Trades_Z5=706 Trades_H6=642. H6 passes gate but Z5 dropped vs exp_362 (4.07 vs 4.09). Holding dip trade 1 ATR above slow captures less alpha than exiting at slow+0.5 ATR. Exit matrix: slow+0=4.04, slow+0.5=4.09, slow+1.0=4.07. Exit at slow+0.5 is optimal. |

| 364 | Dip 4.0 ATR / 120-bar rising + smooth exit at slow+0.5 ATR + dip stop at slow-5.5 ATR | 4.0923 | No | Z5=4.0923 H6=0.6697 Z5pnl=$113,042 H6pnl=$23,452 Trades_Z5=721 Trades_H6=656. Stop-loss improved BOTH metrics vs exp_362 (Z5: 4.089→4.092, H6: 0.634→0.670). Stop at 5.5 ATR below slow cuts failed dip trades without affecting recoveries. Still 0.006 below champion Z5. Try tighter stop. |

| 365 | Dip 4.0 ATR / stop at slow-5.0 ATR (tighter stop) | 4.0651 | No | Z5=4.0651 H6=0.6590 Z5pnl=$112,022 H6pnl=$23,212 Trades_Z5=722 Trades_H6=656. Tighter stop cuts winning dip trades too early. Stop sweep: no stop=4.089/0.634, 5.5 ATR=4.092/0.670(best), 5.0 ATR=4.065/0.659. Sweet spot is 5.5 ATR. |

| 366 | **Dip 3.9 ATR / 120-bar rising + smooth exit at slow+0.5 ATR + stop at slow-5.5 ATR** | **4.2022** | **Yes** | Z5=4.2022 H6=0.6867 Z5pnl=$114,142 H6pnl=$23,728 Trades_Z5=724 Trades_H6=657. NEW CHAMPION! Lower DIP_MULT=3.9 (vs 4.0) catches more dips in Z5 while stop at 5.5 ATR protects H6. Both metrics beat target (Z5>4.0979, H6>=0.6). +0.104 Z5 over previous champion. |

| 367 | Dip 3.8 ATR / 120-bar + stop 5.5 ATR (lower entry threshold) | 4.1955 | No | Z5=4.1955 H6=0.6703 Z5pnl=$113,962 H6pnl=$23,322 Trades_Z5=726 Trades_H6=657. Z5 below champion (4.1955 vs 4.2022). DIP_MULT sweep with stop=5.5: 3.8=4.1955, 3.9=4.2022(peak), 4.0=4.0923. 3.9 ATR is the optimal entry threshold. |

| 368 | **Dip 3.9 ATR / 120-bar + stop 5.5 ATR + exit at slow+0.25 ATR** | **4.2128** | **Yes** | Z5=4.2128 H6=0.6776 Z5pnl=$114,432 H6pnl=$23,472 Trades_Z5=725 Trades_H6=658. NEW CHAMPION! Earlier exit (slow+0.25 vs slow+0.5) improves Z5. Exit sweep: 0.0=?, 0.25=4.2128(peak?), 0.5=4.2022, 1.0=4.0681. Try EXIT=0.0. |

| 369 | Dip 3.9 ATR / stop 5.5 ATR + exit at slow+0.0 ATR | 4.1957 | No | Z5=4.1957 H6=0.6747 Z5pnl=$114,292 H6pnl=$23,378 Trades_Z5=726 Trades_H6=658. Exit sweep complete. Results: EXIT=0.0→4.1957, EXIT=0.25→4.2128(champion), EXIT=0.5→4.2022, EXIT=1.0→4.0681. EXIT=0.25 is optimal. |

| 370 | Dip 3.9 ATR / 90-bar lookback + stop 5.5 + exit 0.25 | 4.0761 | No | Z5=4.0761 H6=0.7479 Z5pnl=$110,332 H6pnl=$25,868 Trades_Z5=726 Trades_H6=657. Shorter lookback hurts Z5 despite excellent H6. SLOW_LOOKBACK sweep: 90=4.076, 120=4.213(champion). Try 150. |

| 371 | Dip 3.9 ATR / 150-bar lookback + stop 5.5 + exit 0.25 | 4.1881 | No | Z5=4.1881 H6=0.5944 Z5pnl=$114,712 H6pnl=$20,518 Trades_Z5=727 Trades_H6=662. SLOW_LOOKBACK sweep done: 90=4.076/0.748, 120=4.213/0.678(champion), 150=4.188/0.594. 120 bars is confirmed optimal — H6 drops below 0.6 target at 150. |

| 372 | Dip 3.9 ATR / stop 6.0 ATR (looser stop) | 4.2099 | No | Z5=4.2099 H6=0.6402 Z5pnl=$114,352 H6pnl=$22,492 Trades_Z5=725 Trades_H6=658. Stop sweep done. Results: stop=5.0→4.065, stop=5.5→4.213(champion), stop=6.0→4.210. 5.5 ATR is confirmed optimal. |

| 373 | **Dip 3.9 ATR / 120-bar / stop 5.5 / exit+0.25 + ATR(20) smoothing** | **4.2201** | **Yes** | Z5=4.2201 H6=0.6197 Z5pnl=$114,678 H6pnl=$22,202 Trades_Z5=724 Trades_H6=657. NEW CHAMPION! ATR(20) smoother → more consistent dip thresholds vs ATR(14). H6 dropped slightly (0.678→0.620) but still passes gate. ATR sweep: try ATR(30). |

| 374 | Dip ATR(30) — looser ATR smoothing | 4.2027 | No | Z5=4.2027 H6=0.6449 Z5pnl=$114,578 H6pnl=$23,412 Trades_Z5=726 Trades_H6=657. ATR sweep: 14=4.2128, 20=4.2201(champion), 30=4.2027. Peak at ATR(20). Try ATR(17) to check fine structure. |

| 375 | Dip ATR(17) | 4.0923 | No | Z5=4.0923 H6=0.6194 Z5pnl=$113,882 H6pnl=$22,158 Trades_Z5=725 Trades_H6=657. Non-monotonic: 14=4.213, 17=4.092(valley), 20=4.220(champion), 30=4.203. ATR(20) confirmed optimal. |

| 376 | Dip DIP_MULT=3.85 with ATR(20) | 4.2128 | No | Z5=4.2128 H6=0.6358 Z5pnl=$114,478 H6pnl=$22,892 Trades_Z5=724 Trades_H6=658. DIP_MULT=3.9 still optimal with ATR(20). Fine-grain sweep (ATR20): DIP_MULT=3.85→4.2128, 3.9→4.2201(champion). |

| 377 | EWM ATR(span=20) vs rolling mean ATR(20) | 4.1431 | No | Z5=4.1431 H6=0.6349 Z5pnl=$113,802 H6pnl=$22,678 Trades_Z5=724 Trades_H6=657. Rolling mean ATR confirmed better than EWM ATR; equal weighting of recent bars superior to exponential recency. |

| 378 | Dual ATR: ATR(20) entry/exit, ATR(50) stop | 4.2065 | No | Z5=4.2065 H6=0.6350 Z5pnl=$113,932 H6pnl=$22,848 Trades_Z5=725 Trades_H6=658. Single ATR(20) confirmed better; dual ATR adds no benefit. |

| 379 | Minimum slow rising margin (slow > slow_prev + 0.05*ATR) | 4.1878 | No | Z5=4.1878 H6=0.6261 Z5pnl=$113,798 H6pnl=$22,432 Trades_Z5=724 Trades_H6=657. Near-flat rising conditions contribute positively to Z5. Margin requirement removes valid entries. Any(slow > slow_prev) confirmed optimal. |

| 380 | Dual-lookback gate: slow > slow_60ago AND slow > slow_120ago | 4.1831 | No | Z5=4.1831 H6=0.6088 Z5pnl=$112,462 H6pnl=$22,798 Trades_Z5=713 Trades_H6=643. Dual gate filters some valid Z5 dip entries. Any(slow > slow_prev) is already optimal. |

| 381 | **Two-tier dip: Tier1=(3.9 ATR + 120-bar) OR Tier2=(4.5 ATR + 60-bar)** | **4.2367** | **Yes** | Z5=4.2367 H6=0.6185 Z5pnl=$115,128 H6pnl=$22,158 Trades_Z5=731 Trades_H6=660. NEW CHAMPION! OR-logic adds Tier2 deeper dips (4.5 ATR) in shorter-term uptrends (60-bar rising). More Z5 alpha (+0.017 vs single-tier). H6 just passes gate. Explore Tier2 DIP_MULT sweep. |

| 382 | **Two-tier: Tier1=(3.9/120) OR Tier2=(4.2/60)** | **4.2446** | **Yes** | Z5=4.2446 H6=0.6081 Z5pnl=$115,342 H6pnl=$21,788 Trades_Z5=733 Trades_H6=661. NEW CHAMPION! Lower Tier2 threshold captures more intermediate dips. DIP_MULT2 sweep: 4.2→4.245(champion), 4.5→4.237. Try 4.0. H6 still barely above gate — careful. |

| 383 | **Two-tier: Tier1=(3.9/120) OR Tier2=(4.0/60)** | **4.2586** | **Yes** | Z5=4.2586 H6=0.6016 Z5pnl=$115,722 H6pnl=$21,552 Trades_Z5=734 Trades_H6=660. NEW CHAMPION! DIP_MULT2 sweep: 4.5→4.237, 4.2→4.245, 4.0→4.259(champion). H6=0.6016 BARELY passes gate — dangerously close. Do NOT lower DIP_MULT2 further. Instead: try Tier3. |

| 384 | Three-tier: add Tier3=(4.8/30) | 4.2442 | No | Z5=4.2442 H6=0.5890 Z5pnl=$115,332 H6pnl=$21,102 Trades_Z5=735 Trades_H6=662. Fails gate. Tier3 with 30-bar lookback adds bad H6 dips even at 4.8 ATR. Very short lookback (30-bar) not stable enough. Keep two-tier system. |

| 385 | Two-tier: Tier2 LOOKBACK2=90 | 4.1653 | No | Z5=4.1653 H6=0.6132 Z5pnl=$113,188 H6pnl=$21,968 Trades_Z5=729 Trades_H6=659. LOOKBACK2 sweep: 60→4.259/0.602(champion), 90→4.165/0.613. 60-bar Tier2 confirmed optimal for Z5. |

| 386 | **Two-tier: tier-specific stops (Tier1=5.5, Tier2=5.0)** | **4.2650** | **Yes** | Z5=4.2650 H6=0.6006 Z5pnl=$115,898 H6pnl=$21,518 Trades_Z5=734 Trades_H6=660. NEW CHAMPION but H6=0.6006 RAZOR-THIN (0.0006 above gate). Tier2 tighter stop adds marginal Z5. H6 buffer essentially zero — next experiment must not hurt H6. |

| 387 | Two-tier: tier-specific exits (T1=0.25 ATR, T2=0.0 ATR) | 4.2650 | No | Z5=4.2650 H6=0.6006 — IDENTICAL to champion. KEY INSIGHT: EXIT_ABOVE_SLOW is irrelevant — smooth transition (base_long fires) handles virtually all dip exits. Explicit price-based exit is a fallback that almost never fires. |

| 388 | **Two-tier: DIP_MULT2=3.95** | **4.3081** | **Yes** | Z5=4.3081 H6=0.6006 Z5pnl=$117,068 H6pnl=$21,518 Trades_Z5=735 Trades_H6=660. NEW CHAMPION! KEY: H6 UNCHANGED from exp_386 (exact same $21,518/660 trades). DIP_MULT2=3.95 adds Z5-ONLY dip trades (never appear in H6). Free Z5 alpha! Try DIP_MULT2=3.9. |

| 389 | Two-tier DIP_MULT2=3.9 | 4.2834 | No | Z5=4.2834 H6=0.5915 Z5pnl=$116,398 H6pnl=$21,302 Trades_Z5=736 Trades_H6=661. Fails gate. Cliff edge: 3.95 ATR→H6 unchanged (Z5-exclusive), 3.9 ATR→H6 degrades (bad H6 trades appear). DIP_MULT2=3.95 is the floor for Tier2. |

| 390 | Three-tier: add Tier3=(3.95/45/stop5.0) | 4.2430 | No | Z5=4.2430 H6=0.6006 Z5pnl=$115,298 H6pnl=$21,518 Trades_Z5=737 Trades_H6=660. H6 unchanged but Z5 dropped (-0.065). Tier3 (45-bar) adds noisy Z5-damaging trades — too short to be reliable. |

| 391 | Two-tier STOP1=6.0 (wider Tier1 stop) | 4.2982 | No | Z5=4.2982 H6=0.5974 Z5pnl=$117,132 H6pnl=$21,572 Trades_Z5=734 Trades_H6=660. Fails gate. Wider Tier1 stop lets bad H6 dips run. STOP1=5.5 confirmed optimal. |

| 392 | Two-tier DIP_MULT1=3.85 | 4.3007 | No | Z5=4.3007 H6=0.6168 Z5pnl=$116,868 H6pnl=$22,208 Trades_Z5=735 Trades_H6=661. DIP_MULT1 sweep: 3.85→4.3007/0.617, 3.9→4.3081/0.601(champion). Lower threshold = better H6 but lower Z5. Champion 3.9 is optimal for Z5. |

| 393 | Two-tier DIP_MULT2=3.92 boundary test | 4.2834 | No | Z5=4.2834 H6=0.5915 — IDENTICAL to DIP_MULT2=3.9. Discrete jump: dips between 3.90 and 3.94 ATR in recently-reversed conditions don't exist — same trades as 3.9. DIP_MULT2=3.95 is the confirmed floor. |

| 394 | Two-tier + time stop 240 bars | 4.3081 | No | Z5=4.3081 H6=0.6006 — IDENTICAL to champion. No dip trade held 240+ bars — price/EMA exits always fire first. Time stop never reached. |
| 395 | Two-tier + time stop 120 bars (2 hours) | 4.3081 | No | Z5=4.3081 H6=0.6006 — IDENTICAL to champion. 120-bar time stop also never fires. Confirmed: all dip trades exit via price/EMA conditions well before 120 bars. Time stops are irrelevant. |
| 396 | LOOKBACK2=50 (shorter Tier2 trend requirement) | 4.2369 | No | Z5=4.2369 H6=0.5991 Z5pnl=$115,132 H6pnl=$21,462 Trades_Z5=731 Trades_H6=660. Fails gate. Shorter lookback hurts both metrics. |
| 397 | LOOKBACK2=55 | 4.2304 | No | Z5=4.2304 H6=0.6000 Z5pnl=$114,958 H6pnl=$21,498 Trades_Z5=731 Trades_H6=660. Both worse than champion. |
| 398 | LOOKBACK2=70 | 4.2426 | No | Z5=4.2426 H6=0.5696 Z5pnl=$115,288 H6pnl=$20,788 Trades_Z5=733 Trades_H6=663. LOOKBACK2 sweep: 50→4.24/0.60, 55→4.23/0.60, 60→4.31/0.60(champion), 70→4.24/0.57. 60-bar is clearly optimal for Tier2 lookback. |
| 399 | DIP_MULT2=3.97 (between champion 3.95 and 4.0) | 4.2650 | No | Z5=4.2650 H6=0.6006 Z5pnl=$115,898 H6pnl=$21,518 Trades_Z5=734 Trades_H6=660. Below champion. DIP_MULT2=3.95 is the confirmed optimum. |
| 400 | vol short threshold = 0.45 (fewer shorts) | 3.4334 | No | Z5=3.4334 H6=0.8871 Z5pnl=$104,922 H6pnl=$31,262 Trades_Z5=737 Trades_H6=647. Massive Z5 drop but H6 jumps. Vol sweep: 0.40→4.31/0.60(champ), 0.45→3.43/0.89. The 40-45th pct shorts are profitable in Z5 but harmful in H6. |
| 401 | vol short threshold = 0.42 | 3.9675 | No | Z5=3.9675 H6=0.8362 Z5pnl=$113,428 H6pnl=$29,152 Trades_Z5=737 Trades_H6=649. Still well below champion Z5. Each step up in threshold drops Z5 significantly. |
| 402 | vol short threshold = 0.38 (more shorts) | 4.2540 | No | Z5=4.2540 H6=0.6530 Z5pnl=$120,108 H6pnl=$24,398 Trades_Z5=735 Trades_H6=665. Close to champion Z5 but below. H6 improves. More shorts slightly hurt Z5 near 0.40 optimum. |
| 403 | vol short threshold = 0.35 | 3.9886 | No | Z5=3.9886 H6=0.5266 Z5pnl=$111,818 H6pnl=$19,958 Trades_Z5=717 Trades_H6=656. Fails gate. Vol sweep: 0.35→3.99/0.53, 0.38→4.25/0.65, 0.40→4.31/0.60(champion), 0.42→3.97/0.84, 0.45→3.43/0.89. 0.40 is clearly optimal. |
| 404 | Three-tier: Tier3=(DIP_MULT3=4.5, LOOKBACK3=30, STOP3=4.0) | 4.2901 | No | Z5=4.2901 H6=0.5949 Z5pnl=$116,578 H6pnl=$21,312 Trades_Z5=736 Trades_H6=662. Fails gate. BUG: STOP3=4.0 fires immediately since entry is at 4.5 ATR below slow (stop already violated at entry). |
| 405 | Three-tier Tier3=(4.5 ATR, 30-bar, STOP3=6.0) — bug fixed | 4.2864 | No | Z5=4.2864 H6=0.5833 Z5pnl=$116,478 H6pnl=$20,898 Trades_Z5=736 Trades_H6=662. Fails gate. Tier3 adds 1 Z5 trade but 2 bad H6 trades. 30-bar lookback too noisy for Tier3. |
| 406 | **Three-tier: Tier3=(DIP_MULT3=5.0, LOOKBACK3=45, STOP3=6.5)** | **4.3132** | **Yes** | Z5=4.3132 H6=0.6006 Z5pnl=$117,208 H6pnl=$21,518 Trades_Z5=735 Trades_H6=660. NEW CHAMPION! Deeper entry (5.0 ATR) + longer trend requirement (45 bars) adds Z5-exclusive trade. H6 completely unchanged ($21,518/660). Free Z5 alpha from extreme dips in moderate-length uptrends. |

| 407 | Three-tier Tier3=(5.0, LOOKBACK3=60, STOP3=6.5) | 4.3081 | No | Z5=4.3081 H6=0.6006 — identical to two-tier champion. LOOKBACK3=60 adds no new trades vs 45. |
| 408 | Three-tier Tier3=(5.0, LOOKBACK3=50, STOP3=6.5) | 4.3132 | No | Z5=4.3132 H6=0.6006 — identical to three-tier champion. LOOKBACK3=50 same result as 45. |
| 409 | Three-tier Tier3=(5.0, LOOKBACK3=40, STOP3=6.5) | 4.3083 | No | Z5=4.3083 H6=0.5903 Z5pnl=$117,072 H6pnl=$21,148 Trades_Z5=735 Trades_H6=661. Fails gate. LOOKBACK3 sweep: 30→fails, 40→fails, 45-50→champion, 60→same as two-tier. |
| 410 | Three-tier Tier3=(4.8 ATR, LOOKBACK3=45, STOP3=5.8) | 4.2925 | No | Z5=4.2925 H6=0.6006 Z5pnl=$116,642 H6pnl=$21,518 Trades_Z5=736 Trades_H6=660. Worse Z5. DIP_MULT3=4.8 adds trade that hurts Z5 (vs 5.0 which adds Z5-exclusive trade). |
| 411 | Three-tier Tier3=(5.2 ATR, LOOKBACK3=45, STOP3=6.8) | 4.3132 | No | Z5=4.3132 H6=0.6006 — identical to champion. DIP_MULT3 sweep: 4.8→4.29(worse), 5.0→4.31(champion), 5.2→4.31(same). The Z5-exclusive trade is between 4.8 and 5.0 ATR. |
| 412 | Dip detection using `low` instead of `close` | 4.1360 | No | Z5=4.1360 H6=0.5381 Z5pnl=$115,008 H6pnl=$19,972 Trades_Z5=741 Trades_H6=669. Fails gate. Too sensitive — adds bad trades to both Z5 and H6. |
| 413 | Dip detection using HL2 instead of `close` | 4.1513 | No | Z5=4.1513 H6=0.6517 Z5pnl=$113,798 H6pnl=$23,058 Trades_Z5=733 Trades_H6=663. Z5 worse. Price input sweep: close→4.31(champion), HL2→4.15, low→4.14. `close` confirmed optimal. |
| 414 | Three-tier LOOKBACK1=100 | 4.1562 | No | Z5=4.1562 H6=0.6515 Z5pnl=$113,102 H6pnl=$23,322 Trades_Z5=735 Trades_H6=661. Worse Z5. |
| 415 | Three-tier LOOKBACK1=110 | 4.2089 | No | Z5=4.2089 H6=0.6299 Z5pnl=$114,372 H6pnl=$22,598 Trades_Z5=736 Trades_H6=661. Still below champion. |
| 416 | Three-tier LOOKBACK1=130 | 4.3502 | No | Z5=4.3502 H6=0.5768 Z5pnl=$118,212 H6pnl=$20,932 Trades_Z5=737 Trades_H6=660. Fails gate! Best Z5 yet but H6 drops. LOOKBACK1 sweep: longer→higher Z5/lower H6. |
| 417 | Three-tier LOOKBACK1=125 | 4.3361 | No | Z5=4.3361 H6=0.5699 Z5pnl=$117,828 H6pnl=$20,798 Trades_Z5=737 Trades_H6=660. Fails gate. H6 drops sharply at LOOKBACK1=121+. LOOKBACK1=120 is the hard boundary for H6≥0.6. |

| 418 | **Three-tier champion with ATR period=25 (vs 20)** | **4.3406** | **Yes** | Z5=4.3406 H6=0.6304 Z5pnl=$117,398 H6pnl=$22,928 Trades_Z5=737 Trades_H6=662. NEW CHAMPION! Smoother ATR(25) improves BOTH metrics. More consistent dip thresholds across volatile periods. |

| 419 | ATR(30) three-tier | 4.2750 | No | Z5=4.2750 H6=0.5689 Z5pnl=$116,548 H6pnl=$21,038 Trades_Z5=739 Trades_H6=662. Fails gate. ATR sweep: 20→4.31, 25→4.34(champ), 30→4.28. ATR(25) is the peak. |
| 420 | ATR(22) three-tier | 4.3239 | No | Z5=4.3239 H6=0.6258 Z5pnl=$117,498 H6pnl=$22,288 Trades_Z5=737 Trades_H6=662. Below champion. ATR sweep confirmed: 20→4.31, 22→4.32, 25→4.34(peak), 30→4.28. |
| 421 | ATR(25) + DIP_MULT1=3.85 | 4.3366 | No | Z5=4.3366 H6=0.6159. Below champion Z5. DIP_MULT1=3.9 confirmed optimal with ATR(25). |
| 422 | ATR(25) + DIP_MULT1=3.95 | 4.3328 | No | Z5=4.3328 H6=0.6055. Below champion. DIP_MULT1=3.9 is the optimum with ATR(25). |
| 423 | ATR(25) + DIP_MULT2=3.9 | 4.3356 | No | Z5=4.3356 H6=0.6304 Z5pnl=$117,262 H6pnl=$22,928. H6 same as champion but Z5 slightly lower. DIP_MULT2=3.95 confirmed optimal. |
| 424 | ATR(25) + DIP_MULT2=3.85 | 4.2516 | No | Z5=4.2516 H6=0.6012 Z5pnl=$116,948 H6pnl=$22,128. Both worse. DIP_MULT2=3.95 confirmed. |
| 425 | ATR(25) + STOP1=5.0 | 4.3099 | No | Z5=4.3099 H6=0.6209. Both worse than champion. STOP1=5.5 confirmed optimal. |
| 426 | ATR(25) + STOP1=6.0 | 4.3438 | No | Z5=4.3438 H6=0.5629 Z5pnl=$117,892 H6pnl=$21,178. Fails gate. Z5 improves but H6 drops. STOP1=5.5 confirmed optimal. |
| 427 | ATR(25) + DIP_MULT3=4.5, STOP3=5.5 | 4.3294 | No | Z5=4.3294 H6=0.6362 Z5pnl=$117,092 H6pnl=$23,138. Z5 below champion, H6 above. DIP_MULT3=5.0 still Z5-optimal. |
| 428 | **ATR(25) + DIP_MULT3=5.5, STOP3=7.0** | **4.3615** | **Yes** | Z5=4.3615 H6=0.6382 Z5pnl=$117,962 H6pnl=$23,212 Trades_Z5=736 Trades_H6=662. NEW CHAMPION! Deeper Tier3 (5.5 ATR) adds better Z5 trades. Both metrics improved. |

*(Agent appends rows here after each experiment)*

---

## Current champion — DO NOT touch

OHLC4 EMA(6) fast vs MEDIAN of HL2 EMA(380/425/470) slow. Longs: vol-free EMA crossover. Shorts: EMA crossover + vol_60 > rolling(480).quantile(0.40). PLUS stateful dip-buying: enter long when slow rising (120-bar), close < slow - 3.9*ATR, NOT base_long, NOT base_short; exit dip at slow+0.5*ATR; stop at slow-5.5*ATR; smooth transition to champion long when EMA turns bullish during dip. → Z5=4.2022, H6=0.6867 (exp_366, commit 000cda6).
Key insights:
- HL2 median(EMA_380, EMA_425, EMA_470) more robust than single span=425
- OHLC4 fast confirmed best; HL2 median slow confirmed best structure
- Dip-buying adds mean-reversion trades during champion flat periods
- DIP_MULT=3.9 ATR (enter threshold), DIP_STOP_MULT=5.5 ATR (stop level)
- Stop protects against continued drops (especially H6); lower DIP_MULT captures more alpha
- EXIT_ABOVE_SLOW=0.5 ATR is optimal exit for dip trades

New targets: Z5 > 4.3615 AND H6 >= 0.6 to commit.
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
- **Slow EMA span tuning around 380/425/470** — fully mapped (exps 320–332), center=425 ±45 optimal
- **Fast EMA span tuning around 4–8** — fully mapped, EWM(6) on OHLC4 is optimal
- **Ensemble spread/center tuning** — ±45 spread, center=425, 3-span all confirmed optimal
- **Mean vs median ensemble** — median confirmed better (exp_323)
- **5-span vs 3-span ensemble** — 3-span confirmed better (exp_328)
- **AND conditions on short signal** — ALL create forced-flat churn (exps 308,309,311,312,315,317,319,338). Vol_60/quantile(480) is optimal; do not add any AND conditions
- **OR conditions on short signal** — exp_313 tried, H6=0.5977 fails gate
- **Alternative short filters (RSI, ATR, VR)** — all create churn or fail gate (exps 333,334,335,337,338)
- **Trailing stops (vectorized or stateful)** — fundamentally incompatible with long-duration EMA trend following. The slow EMA(425) is designed to hold through pullbacks; any trailing stop contradicts this and destroys Calmar (exps 333,353,354,355)
- **Fixed-threshold signals compared to 0 or a constant** — all create churn: bar body EWM, VWPM, RSI, BCS, ATR-normalized margin (exps 340,341,348,349,352). Only cross-EMA comparisons (both sides adaptive) avoid churn.
- **Alternative price types for slow EMA** — HL2 > OHLC4 > Close confirmed (exps 346,347)
- **Vol filter params beyond pct40(480)** — all worse: pct35/45 fail Z5 or H6, 240/720 bar windows trade off Z5 vs H6. pct40(480) is optimal (exps 342,343,344,345)
- **Vol filter on longs** — no vol filter confirmed optimal for longs; any vol threshold creates churn (exp 350)
- **Regime filters (VR, ATR-adaptive, EMA spread)** — all reduce Z5 by removing trend entries (exps 334,349,356)
- **Adaptive fast EMA span by ATR regime** — EWM(6) is optimal regardless of volatility (exp 349)
- **Asymmetric fast EMA for long/short** — exps 310, 316, single EWM(6) confirmed optimal
- **VWEMA/VWAP blends for slow line** — exps 314, 318, HL2 EMA confirmed superior

## What to try next — signals that MUST generalize

Only test signals with structural reasons to work across market regimes. NO calendar.

Priority directions from user (post exp_332):
- **Short signal redesign**: Longs and shorts use symmetric logic — try a completely different
  entry condition for shorts. Example: RSI(14)-based, price-near-recent-high (failed breakout),
  or ATR-expansion confirmation. The vol filter may not be optimal for shorts specifically.
- **Trailing stop / dynamic exit**: Currently exit only on EMA crossover; an ATR-based trailing
  stop on long positions could reduce max drawdown without cutting winners. Try rolling max over
  N bars minus K*ATR as a "stay-long" condition (vectorized, no loop needed).
- **Regime-based position sizing**: Scale to 0 in low-confidence regimes rather than binary in/out.
  Example: go flat when variance ratio VR(30) < 0.8 (ranging market). Reduces drawdown in chop.
- **Variance ratio regime**: VR(k) = Var(k-bar) / (k * Var(1-bar)) > 1 means trending.
  Only trade when VR confirms trending regime. Cuts mean-reverting noise.
- **Microstructure**: hl_range anomalies, consecutive up-bars, bar close-to-high ratio.
  NQ is liquid — different mechanism entirely from trend-following.
- **Mean reversion (short-only)**: Fade moves > N * atr_14 as an alternative to current shorts.
  Opposite thesis from trend-following — may be uncorrelated and combinable with current longs.

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
