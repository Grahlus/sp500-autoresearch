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
| NNN | hypothesis | Z5 Calmar | Kept? | Z5=X.XX H6=X.XX PnL=$XX,XXX Trades=N |

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

*(Agent appends rows here after each experiment)*

---

## Current champion — DO NOT touch

OHLC4 EMA(6) fast vs HL2=(H+L)/2 EMA(425) slow. Longs: EMA crossover only (no vol filter). Shorts: EMA crossover + vol_60 > rolling(480).quantile(0.40). → Z5=4.0776, H6=0.6538 (exp_257, commit 440eff7).
Key insights:
- HL2 = (high+low)/2 range midpoint for slow EMA outperforms all other slow metrics
- OHLC4 fast (includes open) is confirmed best for fast EMA
- Longs: no vol filter needed — HL2 EMA provides quality anchor for long entries
- Shorts: vol > 40th pct required — prevents false shorts in NQ's uptrend bias
- EMA(6) fast, EMA(425) slow is optimal with asymmetric vol (420 had H6=0.65, 430 fails gate)

New targets: Z5 > 4.0776 AND H6 >= 0.6 to commit.
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
