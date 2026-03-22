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
6. **Decide**: if Calmar improved → `git commit -am "exp_NNN: <hypothesis> → calmar=X.XX"` and keep
            if Calmar regressed → `git checkout agent.py` and revert
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
- **Hardware**: CPU only — keep models lightweight

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
| 035 | EMA(8) close vs EMA(480) vwap_proxy | 0.6295 | ? | Tested on new extended dataset; beats 0.5893 |

*(Agent appends rows here after each experiment)*

---

## Current champion — DO NOT touch

The best strategy found so far is EMA(8) vs EMA(480) long-only → Calmar 3.0635.
This is your baseline to beat. It is already committed. Do not re-run EMA/SMA variants.

## Banned approaches — already exhausted

The following have been tested to death. Do not attempt any variation of these:
- SMA/EMA crossovers of any period combination
- RSI filters on top of SMA
- Volume/volatility regime filters on top of SMA
- ROC momentum
- RTH session filters
- Hysteresis on SMA exits
- Buy and hold

## What to try next — genuinely new territory

Pick ONE of these families and go deep:

- **Microstructure**: hl_range anomalies, volume spikes, consecutive up/down bars, 
  tick direction runs. NQ is a liquid market — price impact is real.
- **Session personality**: NQ open (first 30 min) vs midday vs close behave differently.
  Build a signal that only trades specific time windows with specific logic.
- **Mean reversion**: fade moves > N * atr_14 within a session, exit at VWAP.
  Different thesis entirely from trend-following.
- **ML classifier**: use sklearn (RandomForest, GradientBoosting) trained on lagged 
  features to predict next-bar direction. Keep n_estimators ≤ 50 for CPU speed.
- **Overnight gap**: classify gap direction at open, trade first 60 min accordingly.
- **Volume profile**: bars where volume >> rolling average signal conviction.

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
