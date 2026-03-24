# Trading Autoresearch — program.md

---

## CURRENT STATE

```
best_exp:        873
best_z5_calmar:  10.1448
best_h6_calmar:  1.1662
best_z5_pnl:     $143,185
best_h6_pnl:     $39,665
trades_z5:       998
trades_h6:       1014
next_exp:        874
run_command:     uv run python run.py
editable_files:  [agent.py, program.md]
frozen_files:    [prepare.py]
```

Champion strategy (exp_698):
Three-tier dip-buying on EMA(6) fast / median(EMA380/425/470 of HL2) slow,
bidirectional with vol_60>pct40(480) short gate, momentum hold gates
(roc_240>0.0005 OR roc_60>0.0002 OR RSI(14)<40), stop loss at 3.6*ATR,
entry dead band 0.05*ATR, fast_declining filter (5-bar, 0.02*ATR threshold),
60-bar dip timeout.

---

## OBJECTIVE

Maximize the **composite score** on Z5 validation:

```
composite = Z5_calmar + Z5_pnl / 25000
```

Current champion composite: 10.1448 + 143185/25000 = **15.872**

Both components matter equally. A strategy with Calmar 6.0 and PnL $50k
scores 6.0 + 2.0 = 8.0 — worse than champion. Do not sacrifice PnL for Calmar
or vice versa.

Secondary gate: H6 >= 0.6 on every kept strategy (generalization check).

---

## DECISION POLICY

After each `uv run python run.py`:

```
z5_composite = z5_calmar + z5_pnl / 25000

if h6_calmar < 0.6:
    → REVERT  (git checkout agent.py)
elif z5_composite <= 15.872:
    → REVERT  (git checkout agent.py)
else:
    → KEEP    (git commit -am "exp_NNN: <hypothesis> → z5=X.XX h6=X.XX pnl=$XX,XXX composite=XX.XX")
    → UPDATE  best_exp, best_z5_calmar, best_h6_calmar, best_z5_pnl, best_h6_pnl in CURRENT STATE above
```

Never commit a strategy that fails the H6 gate, even if composite is high.

---

## EXPERIMENT LOOP

1. Read `agent.py` and CURRENT STATE above
2. Form one hypothesis — clear rationale, one change only
3. Edit `agent.py`
4. Run: `uv run python run.py`
5. Log result immediately — before writing any new code
6. Apply decision policy above — keep or revert
7. Go to 1

---

## LOGGING FORMAT

Append one row per experiment to the log at the bottom of this file:

```
| NNN | hypothesis | Z5 calmar | Kept? | Z5=X.XX H6=X.XX Z5pnl=$XX,XXX H6pnl=$XX,XXX composite=XX.XX |
```

Log IMMEDIATELY after seeing run.py output. Never batch-update.
If context is lost mid-session: `git log --oneline` to recover.

---

## HARD CONSTRAINTS

- Do NOT modify `prepare.py` — frozen ground truth
- One change per experiment — isolate causality
- Timeout is 15 minutes — if model would exceed this, simplify
- Signals must be in {-1, 0, 1}
- No lookahead — `get_signals(df)` sees bar T only, never T+1
- No leakage — `train(df)` must use only training data (pre 2025-09-01)

---

## ALLOWED ACTIONS PER CYCLE

- Read: `agent.py`, `program.md`, `prepare.py`
- Edit: `agent.py`, `program.md` (log rows and CURRENT STATE only)
- Run: `uv run python run.py`
- Git: `git checkout agent.py` (revert) or `git commit -am "..."` (keep)

---

## SESSION CONTEXT

- Asset: NQ E-mini futures, 1-min bars
- Point value: $20/point — Commission: $2.50/side — Slippage: 0.25 pts/side
- Max position: 1 contract
- Train: H5+M5+U5 bars (Dec 2024 → Aug 2025)
- Val (Z5): Sep 2025 → Dec 2025 — this is the optimization target
- Forward (H6): Dec 2025 → Mar 2026 — generalization gate only, never optimize for it
- Hardware: CPU only, Intel i5-12500, 6 cores, 24GB RAM
- sklearn ML: n_estimators ≤ 50, max_depth ≤ 5
- RL (SB3/PPO): max 50k timesteps, network ≤ 64x64

---

## WHAT TO TRY NEXT

The current architecture is deeply converged — all parameter sweeps exhausted through exp 708.
Next improvement must come from a mechanistically different signal. Do NOT vary existing params.

Priority directions:
- Independent short signal — current shorts mirror long logic with vol gate; try a different
  entry condition entirely: failed breakout, RSI extreme on longer TF, mean-reversion fade
- Ensemble with uncorrelated signal — a second strategy scoring Calmar 1.5+ on both Z5/H6
  that can be combined with current longs is more valuable than marginal EMA improvements
- Order flow / microstructure — hl_range anomalies, volume imbalance, consecutive bar patterns;
  completely different mechanism from trend-following
- Mean reversion shorts — fade moves > N*ATR; opposite thesis, potentially uncorrelated

Hypothesis quality bar — before coding, ask:
- Is this mechanistically different from EMA crossover + dip-buying?
- Why would this work on NQ specifically — what is the structural reason?
- Does it have ≤ 5 parameters?

---

## BANNED APPROACHES — EXHAUSTED

### Basic trend-following (exps 001-051)
- SMA/EMA crossovers of any period combination
- RSI filters on top of SMA, ROC momentum, RTH filters, hysteresis, buy and hold
- VWAP EMA combinations, volume confirmation on top of EMA/SMA

### ML calendar models (exps 069-097)
- Calendar/time-of-day features as primary signal — Z5-specific overfit, H6=0.13
- GBM with calendar features — all variants H6=0.13
- Calendar-free GBM with only 2 features — worse than simple rule

### Slow EMA architecture (exps 320-356, 434-436, 445-446, 468-469, 476)
- Slow EMA span center — center=425 is hard optimum; center=426+ collapses H6
- Slow EMA spread — ±45 optimal; ±40 and ±50 both worse
- Ensemble size — 3-span median confirmed; 5-span worse; mean worse than median
- Alternative price types — HL2 > OHLC4 > Close confirmed
- VWEMA and VWAP blends for slow line — HL2 EMA confirmed superior
- Asymmetric fast EMA — single EWM(6) optimal; DEMA extremely bad
- Fast EMA span != 6 — span 5/7/8 all dramatically worse

### Short signal filters (exps 308-356, 400-403, 438-439, 466-467, 519-520)
- AND conditions on short signal — ALL create forced-flat churn
- OR conditions on short signal — fails H6 gate
- Alternative short filters (RSI, ATR, VR, BCS) — churn or fail gate
- Vol quantile threshold != pct40 — sweep complete: pct35/38/42/45 all worse
- Vol rolling window != 480 — sweep complete: 40/80/240/720 all worse
- Vol filter on longs — no filter optimal for longs
- Short entry dead band — delays profitable early-short entries

### Exits and stops (exps 333-355, 461-465, 505-506, 533-534)
- Trailing stops — incompatible with slow EMA trend following
- Symmetric exit dead band — exit hysteresis dominant bad effect
- Tier-specific timeouts — uniform 60-bar confirmed better
- Time stops beyond 60 bars — never fire
- Still-declining exit condition — creates churn
- 2-bar exit confirmation — catastrophic Z5
- EWM-based ATR — rolling mean ATR confirmed optimal

### Dip-buying parameters — fully mapped (exps 357-584)
- LOOKBACK1 != 130 — hard boundary; 131+ fails H6, <125 hurts Z5
- DIP_MULT1 != 3.9, LOOKBACK2 != 60, DIP_MULT2 != 3.95
- LOOKBACK3 range 45-50, DIP_MULT3 != 5.5
- STOP1 != 5.5, STOP2 != 5.0, STOP3 != 8.0
- ATR period != 25 — sweep complete: 14/17/20/22/24/26/27/30 all worse
- Tier4 additions — consistently hurt or identical
- Entry dead band != 0.05*ATR, fast_declining window != 5
- fast_declining magnitude threshold != 0.02*ATR

### Momentum hold gate params — fully mapped (exps 652-708)
- roc_240 threshold != 0.0005 — sweep complete
- roc_60 threshold != 0.0002 — sweep complete
- RSI threshold != 40 — sweep complete (35-45 mapped, 40 is peak)
- Triple-momentum gate adding roc_15 — too reactive, hurts both metrics
- Stop loss != 3.6*ATR — sweep complete (2.5/3.0/3.3/3.6/3.7/3.8/4.0 mapped)
- Dynamic stop by RSI regime — fixed stop=3.6 confirmed optimal

---

## EXPERIMENT LOG

| # | Hypothesis | Z5 Calmar | Kept? | Notes |
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
| 429 | DIP_MULT3=6.0, STOP3=7.5 | 4.3615 | No | Z5=4.3615 H6=0.6382 — identical to champion. No new trades. DIP_MULT3=5.5 → 6.0 adds nothing. |
| 430 | DIP_MULT3=5.5, LOOKBACK3=30, STOP3=7.0 | 4.3497 | No | Z5=4.3497 H6=0.6297 Z5pnl=$117,642 H6pnl=$22,902. Below champion. LOOKBACK3=45 confirmed optimal. |
| 431 | Four-tier: add Tier4=(3.9 ATR, 45-bar, STOP=5.5) | 4.3061 | No | Z5=4.3061 H6=0.6362. Z5 drops — Tier4 3.9/45 adds bad Z5 trade. |
| 432 | Four-tier: Tier4=(4.0 ATR, 45-bar) | 4.3061 | No | Z5=4.3061 H6=0.6362 — identical to 431. Same bad trade fires at 3.9-4.0 ATR/45-bar. |
| 433 | Four-tier: Tier4=(4.5 ATR, 45-bar, STOP=5.5) | 4.3445 | No | Z5=4.3445 H6=0.6362 Z5pnl=$117,502 H6pnl=$23,138 Trades_Z5=737 Trades_H6=662. Adds 1 Z5 trade but nets slightly below champion. |
| 434 | Slow EMA MEDIAN(385/430/475) center=430 | 4.4275 | No | Z5=4.4275 H6=0.4406. Fails H6 gate. Same H6 cliff at center>425 as before. |
| 435 | Slow EMA MEDIAN(382/427/472) center=427 | 4.3289 | No | Z5=4.3289 H6=0.4815. Fails gate. H6 collapses sharply above center=425. |
| 436 | Slow EMA MEDIAN(381/426/471) center=426 | 4.4175 | No | Z5=4.4175 H6=0.5328 Z5pnl=$119,402 H6pnl=$19,928 Trades_Z5=731 Trades_H6=661. Fails gate. Center=425 confirmed sharp optimum: 425→0.64, 426→0.53. |
| 437 | DIP_MULT3=5.3, STOP3=6.8 | 4.3421 | No | Z5=4.3421 H6=0.6307. DIP_MULT3 sweep: 4.5→4.33, 5.0→4.34, 5.3→4.34, 5.5→4.36(champ). 5.5 confirmed optimal. |
| 438 | vol rolling window=40 | 4.3118 | No | Z5=4.3118 H6=0.6698 Trades_Z5=785 H6pnl=$26,528. Z5 drops badly. vol_60 confirmed. |
| 439 | vol rolling window=80 | 3.3406 | No | Z5=3.3406 H6=0.4944. Fails gate. Terrible — wider window destroys short timing. vol_60 confirmed. |
| 440 | STOP3=6.0 (tighter Tier3 stop) | 4.3464 | No | Z5=4.3464 H6=0.6382 — slightly below champion. |
| 441 | **STOP3=8.0 (wider Tier3 stop, 2.5 ATR buffer)** | **4.3650** | **Yes** | Z5=4.3650 H6=0.6382 Z5pnl=$118,058 H6pnl=$23,212 Trades_Z5=736 Trades_H6=662. NEW CHAMPION! Wider stop for extreme dips gives more breathing room. STOP3 sweep: 6.0→4.35, 7.0→4.36, 8.0→4.37(peak?). |
| 442 | STOP3=9.0 | 4.3526 | No | Z5=4.3526 H6=0.6382. Below champion. STOP3 sweep: 6.0→4.35, 7.0→4.36, 8.0→4.37(champ), 9.0→4.35. STOP3=8.0 confirmed optimal. |
| 443 | LOOKBACK3=55 | 4.3650 | No | Z5=4.3650 H6=0.6382 — identical to champion. Same trade at LOOKBACK3=45 or 55. |
| 444 | Quiet-dip vol filter (vol < pct40 for dip entry) | 4.3214 | No | Z5=4.3214 H6=0.6382 Z5pnl=$116,878. Z5 drops — vol filter removes good dip entries. |
| 445 | Slow EMA spread ±50: MEDIAN(375/425/475) | 4.3295 | No | Z5=4.3295 H6=0.6382. Below champion. Spread sweep: ±40→4.36, ±45→4.37(champ), ±50→4.33. |
| 446 | Slow EMA spread ±40: MEDIAN(385/425/465) | 4.3560 | No | Z5=4.3560 H6=0.6382. ±45 confirmed as optimal spread. |
| 447 | STOP2=5.5 | 4.3358 | No | Z5=4.3358 H6=0.6428. Z5 below champion. STOP2=5.0 confirmed. |
| 448 | STOP2=4.5 | 4.3569 | No | Z5=4.3569 H6=0.6247. STOP2 sweep: 4.5→4.36, 5.0→4.37(champ), 5.5→4.34. STOP2=5.0 confirmed optimal. |
| 449 | LOOKBACK1=130 + MIN_RISE1=0.1 | 4.4035 | No | Z5=4.4035 H6=0.5930. H6 improved vs bare LOOKBACK1=130 (+0.016) but still fails gate. |
| 450 | LOOKBACK1=130 + MIN_RISE1=0.5 | 4.3750 | No | Z5=4.3750 H6=0.5945. Z5 drops, H6 barely improves. Bad H6 trade survives stronger filter. |
| 451 | LOOKBACK1=130 + MIN_RISE1=1.0 | 4.3988 | No | Z5=4.3988 H6=0.5904. MIN_RISE filter direction exhausted — bad H6 trade robust to min-rise. |
| 452 | **LOOKBACK1=125 with ATR(25) (no MIN_RISE)** | **4.3883** | **Yes** | Z5=4.3883 H6=0.6016 Z5pnl=$118,688 H6pnl=$22,282 Trades_Z5=737 Trades_H6=662. NEW CHAMPION! ATR(25) shifts the H6 boundary — LOOKBACK1=125 was failing H6 with ATR(20) but passes with ATR(25). More Z5 alpha unlocked. |
| 453 | LOOKBACK1=127 with ATR(25) | 4.3915 | No | Z5=4.3915 H6=0.5989. Fails H6 gate. Better Z5 but H6 still below 0.6. |
| 454 | LOOKBACK1=126 with ATR(25) | 4.3893 | No | Z5=4.3893 H6=0.5966. Fails H6 gate. Boundary confirmed at 125/126. |
| 455 | ATR(27) + LOOKBACK1=125 | 4.3045 | No | Z5=4.3045 H6=0.6415. ATR(27) improves H6 but kills Z5. ATR(25) confirmed optimal. |
| 456 | Fast EMA span=7 | 3.4383 | No | REVERT H6=0.2730. Span=7 dramatically worse both metrics. Span=6 confirmed optimal. |
| 457 | vol_pct80 filter on dip entry | 4.3458 | No | Z5=4.3458 H6=0.6016 Trades_Z5=736. Blocks exactly 1 high-vol Z5 trade (good trade). Vol filter useless. |
| 458 | vol_pct90 filter on dip entry | 4.3458 | No | Z5=4.3458 H6=0.6016 Trades_Z5=736. Same as vol_pct80 — same 1 trade blocked. Vol filter confirmed dead end. |
| 459 | ATR(26) + LOOKBACK1=126 | 4.3155 | No | Z5=4.3155 H6=0.5780. H6 fails gate. ATR(26) not enough to unlock LOOKBACK1=126. |
| 460 | ATR(26) + LOOKBACK1=125 | 4.3146 | No | Z5=4.3146 H6=0.5830. Worse than ATR(25) champion on both metrics. |
| 461 | Trailing stop TRAIL_MULT=2.0 on dip trades | 4.3147 | No | Z5=4.3147 H6=0.5860. Trailing stop fails — cuts good trades in both periods. |
| 462 | Trailing stop TRAIL_MULT=3.5 | 4.3691 | No | Z5=4.3691 H6=0.6009. Closer but still below champion. Trail still cuts winners. |
| 463 | Trailing stop TRAIL_MULT=5.0 | 4.3852 | No | Z5=4.3852 H6=0.5872. Fails gate. Trailing stop consistently hurts H6. |
| 464 | STOP1=5.7 | 4.3732 | No | Z5=4.3732 H6=0.6002. STOP1 fine-tuning: 5.5=champion, 5.6 and 5.7 give same worse result. |
| 465 | STOP1=5.6 | 4.3732 | No | Z5=4.3732 H6=0.6002. Identical to STOP1=5.7 — discrete jump at 5.5 boundary. |
| 466 | Vol quantile window 720 (vs 480) | 3.6876 | No | Z5=3.6876 H6=0.6043. Longer window kills Z5. 480 confirmed optimal. |
| 467 | Vol quantile window 240 | 3.5143 | No | Z5=3.5143 H6=0.9072. More trades (903) but terrible Z5. 480 is optimal. |
| 468 | Slow EMA source = close | 4.0141 | No | Z5=4.0141 H6=0.5149. Significantly worse. HL2 confirmed optimal for slow EMA. |
| 469 | Slow EMA source = OHLC4 | 4.0961 | No | Z5=4.0961 H6=0.5810. Between close and HL2 but fails gate. HL2 is best. |
| 470 | True Range ATR (period=25) LOOKBACK1=125 | 4.2248 | No | Z5=4.2248 H6=0.6142. TR ATR improves H6 but kills Z5. H-L ATR confirmed optimal. |
| 471 | True Range ATR + LOOKBACK1=126 | 4.2257 | No | Z5=4.2257 H6=0.6092. Passes gate but Z5 far below champion. |
| 472 | Recovery-bar entry confirmation (close > prev_close) | 4.1733 | No | Z5=4.1733 H6=0.5879. More trades (762) but worse — creates re-entries. |
| 473 | Tier4: LOOKBACK4=200, DIP_MULT4=3.7, STOP4=5.5 | 4.3434 | No | REVERT H6=0.4416. Shallow dips in 200-bar trend bad in H6. |
| 474 | Tier4: LOOKBACK4=300, DIP_MULT4=3.7, STOP4=5.5 | 4.2372 | No | Z5=4.2372 H6=0.5635. Still fails gate. Tier4 shallow dip approach exhausted. |
| 475 | DEMA(6) for fast signal | 3.0318 | No | REVERT H6=-0.2149. DEMA extremely bad — too noisy. Plain EWM(6) is optimal. |
| 476 | Slow EMA center=420 (spans 375,420,465) | 4.1493 | No | Z5=4.1493 H6=0.5590. Lower center worse. 380/425/470 confirmed as global optimum. |
| 477 | Fast EMA span=5 | 4.2060 | No | REVERT H6=0.2233. Span=5 also terrible. Span=6 is the only viable value. |
| 478 | LOOKBACK1=120 with ATR(25) | 4.3650 | No | Z5=4.3650 H6=0.6382. Pattern: LOOKBACK monotonically improves Z5 until H6 gate fails at 126. Optimum is LOOKBACK1=125. |
| 479 | vol_pct60 filter on dip entry | 4.3458 | No | Z5=4.3458 H6=0.6016 Trades_Z5=736. Same result as vol_pct80/90 — same 1 extreme-vol dip trade blocked. Vol filter approach fully exhausted. |
| 480 | Close-to-high ratio > 0.5 filter on dip entry | 4.1681 | No | Z5=4.1681 H6=0.6083 Trades_Z5=762. More trades, worse Z5. CTH filter creates re-entries like recovery-bar. |
| 481 | 2-bar short confirmation (fast < slow for 2 consecutive bars) | 3.8104 | No | Z5=3.8104 H6=0.6509. More trades (886), terrible Z5. AND conditions on short = banned. |
| 482 | EWM-based ATR (span=25) instead of rolling mean | 4.3056 | No | Z5=4.3056 H6=0.5968. Fails gate. Rolling mean ATR confirmed optimal. |
| 483 | Smoothed LOOKBACK1=126: mean(slow[i-124:i-129]) | 4.3893 | No | Z5=4.3893 H6=0.5966. Same as plain LOOKBACK1=126 — smoothing doesn't change trades. |
| 484 | Smoothed LOOKBACK1=125: mean(slow[i-123:i-128]) | 4.3883 | No | Z5=4.3883 H6=0.6016. Identical to champion — smoothing at 125 changes nothing. |
| 485 | fast_declining filter (fast[i] < fast[i-5]) + LOOKBACK1=125 | 4.3828 | No | Z5=4.3828 H6=0.6555. H6 significantly improved (+0.054)! Z5 slightly lower. Filter is promising. |
| 486 | fast_declining + LOOKBACK1=126 | 4.3813 | No | Z5=4.3813 H6=0.6524. Passes gate but Z5 below champion. Continue sweeping. |
| 487 | fast_declining + LOOKBACK1=127 | 4.3828 | No | Z5=4.3828 H6=0.6556. Same as LOOKBACK1=125. Pattern non-monotonic. |
| 489 | fast_declining + LOOKBACK1=135 | 4.3894 | No | Z5=4.3894 H6=0.5755. Fails gate. Upper boundary around 132-134. |
| 490 | fast_declining + LOOKBACK1=132 | 4.3896 | No | Z5=4.3896 H6=0.6388. Passes gate but Z5 below 130 champion. |
| 491 | fast_declining + LOOKBACK1=131 | 4.3885 | No | Z5=4.3885 H6=0.6586. Passes gate. LOOKBACK1=130 confirmed as optimal peak. |
| 492 | fast_declining window=3 (fast[i] < fast[i-3]) | 4.3872 | No | Z5=4.3872. Below champion. Window=5 confirmed optimal. |
| 493 | fast_declining window=7 (fast[i] < fast[i-7]) | 4.3774 | No | Z5=4.3774. Worse. Window=5 confirmed optimal. |
| 494 | fast_declining on tier1 only (tier2/3 no filter) | 4.3905 | No | Z5=4.3905. Slightly worse than all-tier champion. All-tier filter is better. |
| 495 | DIP_MULT1=3.85 | 4.3203 | No | Z5=4.3203. Much worse. DIP_MULT1=3.9 confirmed optimal. |
| 496 | DIP_MULT1=3.95 | 4.3906 | No | Z5=4.3906. Slightly worse. DIP_MULT1=3.9 confirmed optimal. |
| 497 | STOP1=5.7 | 4.3673 | No | Z5=4.3673. Worse. STOP1=5.5 confirmed optimal. |
| 498 | STOP1=6.0 | 4.3869 | No | Z5=4.3869. Passes gate but below Z5 champion. STOP1=5.5 confirmed. |
| 499 | ATR(27) with fast_declining | 4.2994 | No | Z5=4.2994. Worse. ATR(25) remains optimal even with fast_declining. |
| 500 | LOOKBACK2=65 | 4.3726 | No | Z5=4.3726. Worse. LOOKBACK2=60 confirmed optimal. |
| 501 | LOOKBACK3=40 | 4.3874 | No | Z5=4.3874 H6=0.6535. Passes gate but Z5 below champion. LOOKBACK3=45 confirmed optimal. |
| 502 | DIP_MULT3=5.7 | 4.3924 | No | Z5=4.3924 H6=0.6621. Identical to champion — all existing tier3 entries deeper than 5.7*ATR. |
| 503 | DIP_MULT3=6.0 | 4.3924 | No | Z5=4.3924 H6=0.6621. Identical to champion. All tier3 entries deeper than 6.0*ATR. |
| 504 | DIP_MULT3=5.3 | 4.3783 | No | Z5=4.3783 H6=0.6544. Trades=742. More tier3 entries hurt. DIP_MULT3=5.5 confirmed optimal. |
| 505 | STOP3=7.5 | 4.3889 | No | Z5=4.3889 H6=0.6621. Slightly worse. STOP3=8.0 confirmed optimal. |
| 506 | STOP3=9.0 | 4.3800 | No | Z5=4.3800 H6=0.6621. Worse than 7.5 and 8.0. STOP3=8.0 confirmed. |
| 507 | STOP2=5.5 | 4.3632 | No | Z5=4.3632 H6=0.6669. H6 slightly better but Z5 worse. STOP2=5.0 confirmed optimal. |
| 508 | DIP_MULT2=4.0 | 4.3787 | No | Z5=4.3787 H6=0.6621. Trades=740. Worse. DIP_MULT2=3.95 confirmed optimal. |
| 509 | DIP_MULT2=3.9 | 4.3874 | No | Z5=4.3874 H6=0.6621. Below champion. DIP_MULT2=3.95 confirmed. |
| 510 | ATR(24) | 4.3706 | No | Z5=4.3706 H6=0.6183. Worse on both. ATR(25) confirmed. |
| 511 | Reversed tier priority (tier3 > tier2 > tier1) | 4.3591 | No | Z5=4.3591 H6=0.6365. Both worse. Current priority (tier1 first) is correct. |
| 512 | LOOKBACK1=128 | 4.3828 | No | Z5=4.3828 H6=0.6718. H6 best seen, but Z5 below champion. |
| 513 | LOOKBACK1=129 | 4.3891 | No | Z5=4.3891 H6=0.6660. Close to champion but still below. LB1=130 confirmed peak. |
| 514 | EXIT_ABOVE_SLOW=0.0 | 4.3850 | No | Z5=4.3850 H6=0.6574. Slightly worse. EAS=0.25 confirmed. |
| 515 | EXIT_ABOVE_SLOW=0.5 | 4.3684 | No | Z5=4.3684 H6=0.6444. Worse. EAS=0.25 confirmed optimal. |
| 516 | slow_rising_fast filter (slow[i] > slow[i-5]) | 4.0979 | No | Z5=4.0979 H6=0.6258. Blocks too many entries (689 vs 741). Dead end. |
| 517 | slow_rising_fast filter (slow[i] > slow[i-1]) | 4.0979 | No | Z5=4.0979 H6=0.6570. Identical to 5-bar — slow EMA dip/flat events are same set. Dead end. |
| 518 | LOOKBACK3=50 | 4.3924 | No | Z5=4.3924 H6=0.6621. Identical to champion. Tier3 range 45-50 captures same entries. |
| 519 | vol_pct35 (more shorts) | 4.0642 | No | Z5=4.0642 H6=0.5970. Fails gate. More shorts destroy Z5. |
| 520 | vol_pct45 (fewer shorts) | 3.5856 | No | Z5=3.5856 H6=0.9655. H6 dramatically better but Z5 crashes. vol_pct40 confirmed optimal. |
| 521 | Tier-specific fast_declining windows (t1=3, t2=5, t3=7) | 4.3835 | No | Z5=4.3835 H6=0.6383. Worse on both. Uniform window=5 is optimal. |
| 522 | slow_stalled exit (slow[i] <= slow[i-1]) | 4.0457 | No | Z5=4.0457 H6=0.6213. Trades=785 — churn. Slow EMA flattens often during dips. Dead end. |
| 523 | Time-based dip exit, 90 bars | 4.3924 | No | Z5=4.3924 H6=0.6621. Identical to champion — all dips resolve within 90 bars. |
| 525 | Dip timeout=45 bars | 4.3839 | No | Z5=4.3839 H6=0.6781. Worse Z5, better H6. Too aggressive cuts. |
| 526 | Dip timeout=50 bars | 4.3931 | No | Z5=4.3931 H6=0.6668. Closer but below champion. |
| 527 | Dip timeout=55 bars | 4.3963 | No | Z5=4.3963 H6=0.6622. Still below champion. |
| 528 | Dip timeout=65 bars | 4.4020 | No | Z5=4.4020 H6=0.6821. Just below champion Z5. Better H6. |
| 529 | Dip timeout=70 bars | 4.3931 | No | Z5=4.3931 H6=0.6862. Curve: 45→4.3839, 50→4.3931, 55→4.3963, 60→4.4026(champ), 65→4.4020, 70→4.3931. 60 confirmed peak. |
| 530 | Tier-specific timeouts (t1=75, t2=60, t3=45) | 4.3920 | No | Z5=4.3920 H6=0.6803. Uniform 60 beats tier-specific. |
| 531 | LOOKBACK1=132 + 60-bar timeout | 4.3998 | No | Z5=4.3998 H6=0.6322. Passes gate but Z5 below champion. LB1=130 still optimal. |
| 532 | LOOKBACK1=131 + 60-bar timeout | 4.3987 | No | Z5=4.3987 H6=0.6521. Below champion. LB1=130 confirmed optimal with timeout. |
| 533 | STOP1=6.0 + 60-bar timeout | 4.3821 | No | Z5=4.3821 H6=0.6244. Worse. STOP1=5.5 confirmed. |
| 534 | STOP3=7.5 + 60-bar timeout | 4.3987 | No | Z5=4.3987 H6=0.6556. STOP3=8.0 still optimal. |
| 535 | Trend-only short exit (fast>=slow only) | 3.6785 | No | Z5=3.6785 H6=0.9863. H6 great but Z5 crashes. vol-based exit is essential. |
| 536 | LOOKBACK3=35 + 60-bar timeout | 4.3976 | No | Z5=4.3976 H6=0.6469. LOOKBACK3=45 confirmed. |
| 537 | Entry-price stop 2*ATR + timeout | 4.3859 | No | Z5=4.3859. Worse. |
| 538 | Entry-price stop 3*ATR + timeout | 4.3893 | No | Z5=4.3893. Still below champion. |
| 539 | LOOKBACK2=55 + 60-bar timeout | 4.3035 | No | Z5=4.3035. Much worse. LOOKBACK2=60 confirmed. |
| 540 | Still-declining exit at 30 bars + timeout | 4.3526 | No | Z5=4.3526 H6=0.7128. H6 best ever but Z5 terrible. |
| 541 | Still-declining exit at 45 bars + timeout | 4.3830 | No | Z5=4.3830 H6=0.6744. Both worse than champion. |
| 542 | Still-declining exit at 45 bars alone (no timeout) | 4.3830 | No | Z5=4.3830 H6=0.6744. Same as 541 — still-declining supersedes timeout. |
| 543 | Tier4 (LB4=25, MULT4=9.0, STOP4=13.0) + timeout | 4.3965 | No | Z5=4.3965 H6=0.6556. Worse. Tier4 consistently fails. |
| 544 | Tier4 MULT4=7.0, STOP4=11.0 | 4.3957 | No | Z5=4.3957 H6=0.6556. Tier4 at any parameters is bad. |
| 545 | Sustained fast_declining (fast[i-5] < fast[i-10] also) | 4.3442 | No | Z5=4.3442 H6=0.6160. Too restrictive. Single 5-bar is optimal. |
| 546 | 2-bar long exit confirmation | 3.4560 | No | Z5=3.4560 H6=0.7680. Catastrophic Z5 — holds longs too long, misses shorts. |
| 547 | Entry dead band fast > slow + 0.1*ATR | 4.4836 | No | Z5=4.4836 H6=0.5540. Massive Z5 jump (787 trades) but fails H6 gate. |
| 549 | Entry dead band 0.07*ATR | 4.3928 | No | Z5=4.3928 H6=0.5807. Fails gate. |
| 550 | Entry dead band 0.08*ATR | 4.4703 | No | Z5=4.4703 H6=0.5806. Fails gate. |
| 551 | Entry dead band 0.06*ATR | 4.3678 | No | Z5=4.3678 H6=0.6175. Passes gate but Z5 below champion. 0.05 confirmed optimal. |
| 552 | Symmetric dead band (entry+exit 0.05*ATR) | 3.9782 | No | Z5=3.9782 H6=0.5385. Terrible. Exit dead band destroys performance. Entry-only is correct. |
| 553 | LOOKBACK1=131 with dead band | 4.39xx | No | Below champion. LB1=130 confirmed optimal with dead band. |
| 554 | LOOKBACK1=132 with dead band | 4.39xx | No | Below champion. LB1=130 confirmed. |
| 555 | LOOKBACK1=133 with dead band | 4.39xx | No | Below champion. LB1=130 confirmed. |
| 556 | DIP_MULT1=3.85 with dead band | ~4.39 | No | Worse. DIP_MULT1=3.9 confirmed with dead band. |
| 557 | STOP1=5.3 with dead band | ~4.39 | No | Worse. STOP1=5.5 confirmed with dead band. |
| 558 | Dip entry dead band (apply price threshold to dip detection) | 4.4395 | No | Identical to champion. Dip entries already far below slow; dead band at entry price irrelevant. |
| 559 | Stricter dip-to-base_long transition (require base_long_enter to clear dip_tier) | 4.4395 | No | Identical to champion. base_long_enter already implicitly handles this via entry dead band. |
| 560 | fast_declining window=4 | 4.4399 | No | Z5=4.4399 H6=0.6146. Trivial Z5 gain (+0.0004), H6 worse. Window=5 confirmed optimal. |
| 561 | fast_declining window=6 | 4.4330 | No | Z5=4.4330 H6=0.6003. Both worse. Full window sweep done: 3→4.39, 4→4.44(~tie), 5→4.44(peak), 6→4.43, 7→4.38. Window=5 confirmed global peak. |
| 562 | Short entry dead band (fast < slow - 0.05*ATR required to enter short) | 4.1249 | No | Z5=4.1249 H6=0.6069 Trades=785. Significant Z5 drop — short dead band delays profitable early-short entries. Only long dead band is beneficial. |
| 563 | Slow EMA slope filter (slow > slow_20 + 0.05*ATR for dip entry) | 4.1801 | No | Z5=4.1801 H6=0.6478 Trades_Z5=717. Blocks 46 Z5 dip trades — slope filter too strict. The slope check fires right when good Z5 dip entries occur (slow briefly flat). |
| 564 | Tier-specific increasing timeouts (t1=60, t2=75, t3=90) | 4.4391 | No | Z5=4.4391 H6=0.6260. Essentially identical to champion — all tier2/tier3 dips resolve before 75 bars. Uniform 60-bar timeout confirmed. |
| 565 | Exit hysteresis: exit long when fast < slow - 0.05*ATR (not just slow) | 3.9782 | No | Z5=3.9782 H6=0.5385. IDENTICAL to exp_552 (symmetric dead band) — exit hysteresis is the dominant bad effect. Holding losing longs too long and missing short entries. |
| 566 | Median ATR (rolling 25-bar median of bar range instead of mean) | 4.2060 | No | Z5=4.2060 H6=0.5230. Worse on both. Median ATR is lower in skewed distributions, tightening all thresholds and causing premature stops/missed entries. Mean ATR confirmed optimal. |
| 567 | 1-bar rising confirmation for long entry (fast[i] >= fast[i-1]) | 4.4395 | No | Identical to champion. EMA(6) is smooth — when crossing slow+0.05*ATR, it's virtually always rising by 1 bar. Filter never fires. |
| 568 | fast_declining magnitude filter: threshold=0.10*ATR (5-bar decline must exceed 0.1*ATR) | 4.4347 | No | Z5=4.4347 H6=0.6572 Trades=764. H6 significantly improved (+0.031) but Z5 just below champion. Higher threshold consistently improves H6 at cost of Z5. |
| 569 | Magnitude threshold=0.05 | 4.4382 | No | Z5=4.4382 H6=0.6260. Essentially identical to champion — same H6 but small Z5 regression. Threshold too small to filter any trades. |
| 570 | threshold=0.10 + LOOKBACK1=131 | 4.4308 | No | Z5=4.4308 H6=0.6537. LOOKBACK1 increase with threshold doesn't help Z5. |
| 571 | threshold=0.10 + LOOKBACK1=132 | 4.4319 | No | Z5=4.4319 H6=0.6337. LOOKBACK1=132 improves Z5 vs 131 but still below champion. |
| 572 | threshold=0.10 + LOOKBACK1=135 | 4.4317 | No | Z5=4.4317 H6=0.5697. Fails gate. |
| 573 | threshold=0.10 + LOOKBACK1=128 | 4.4250 | No | Z5=4.4250 H6=0.6670. Shorter lookback hurts Z5 more. |
| 574 | threshold=0.10 + LOOKBACK1=125 | 4.4250 | No | Z5=4.4250 H6=0.6506. Same as 128. Full sweep: threshold+LOOKBACK combos max out at 4.4347 (LB1=130). |
| 575 | threshold=0.15 + LOOKBACK1=130 | 4.4332 | No | Z5=4.4332 H6=0.6609. Higher threshold helps H6 more but Z5 consistently below 4.44. Direction exhausted: boolean fast_declining is Z5-optimal. |
| 576 | threshold=0.15 + LOOKBACK1=132 | 4.4304 | No | Z5=4.4304 H6=0.6373. Both below champion. Fast_declining magnitude threshold direction fully exhausted. |
| 578 | threshold=0.01 | 4.4395 | No | Identical to boolean champion (pre-577). Threshold too small to filter any trades. |
| 579 | threshold=0.025 | 4.4451 | No | Just below champion. Peak is clearly at 0.02. |
| 580 | threshold=0.03 | 4.4451 | No | Tied with 0.025 but below champion. 0.02 confirmed as global peak. |
| 581 | Minimum slow rise for tier1: (slow - slow_prev1) > 0.02*ATR | 4.4410 | No | Z5=4.4410 H6=0.6269. Below champion. Slow EMA in real uptrends already rises much more than 0.02*ATR/130 bars — filter rarely fires. |
| 582 | threshold=0.02 + LOOKBACK1=131 | 4.4416 | No | Z5=4.4416 H6=0.6226. Both metrics worse. LB1=130 confirmed optimal with new threshold. |
| 583 | threshold=0.02 + LOOKBACK1=132 | 4.4427 | No | Z5=4.4427 H6=0.6028. Closer on Z5 but H6 drops. LB1=130 is the boundary. Magnitude threshold doesn't unlock longer lookbacks. |
| 584 | DIP_MULT1=3.88 | 4.3794 | No | Z5=4.3794 H6=0.6260 Trades=764. Significant drop — adds one bad Z5 dip trade. DIP_MULT1=3.9 confirmed optimal. |
| 585 | fast_declining window=4 + threshold=0.02 | 4.4399 | No | Z5=4.4399 H6=0.6146. Window=5 still optimal with threshold. |
| 586 | fast_declining window=6 + threshold=0.02 | 4.4330 | No | Z5=4.4330 H6=0.6003. Both worse. Window=5 confirmed global optimum even with threshold. |
| 587 | Entry dead band 0.06*ATR + threshold=0.02 | 4.3736 | No | Z5=4.3736 H6=0.6175. Worse. Dead band 0.05 confirmed optimal with new threshold. |
| 588 | Slow EMA center=424 (spans 379/424/469) | 4.3260 | No | Z5=4.3260 H6=0.6443. Z5 much worse. center=425 is a hard optimum. |
| 589 | Weighted slow EMA 0.25/0.50/0.25 blend | 4.3224 | No | Z5=4.3224. Falls between median and mean quality. Median confirmed better. |
| 590 | Weighted fast EMA (O+H+L+2C)/5 | 4.3387 | No | Z5=4.3387 H6=0.4789. Fails gate. OHLC4 equal weights confirmed optimal. |
| 591 | Tier-specific threshold: tier1=0.02, tier2/3=boolean | 4.4395 | No | Identical to old boolean champion — threshold must apply to ALL tiers uniformly via shared fast_declining gate. |
| 592 | Tier-specific threshold: tier1=boolean, tier2/3=0.02 | 4.4395 | No | Same — tier1 fires first, overriding tier2/3 threshold. Confirmed: shared gate is required. |
| 593 | threshold=0.015 | 4.4395 | No | Identical to boolean. Boundary is between 0.015 and 0.018. |
| 594 | threshold=0.018 | 4.4455 | No | Identical to champion 0.02. Effective window: [0.018, ~0.025] all give 4.4455. |
| 595 | Combined momentum: fast_declining AND close_declining (close[i] < close[i-5]) | 4.4187 | No | Z5=4.4187 H6=0.5954 Trades_Z5=770. Worse — close-based filter fires on correct dip entries too, creates churn. |
| 596 | ATR(24) | 4.4047 | No | Z5=4.4047 H6=0.5891 Trades_Z5=765. Both worse. ATR(25) confirmed optimal with threshold. |
| 597 | ATR(26) | 4.3707 | No | Z5=4.3707 H6=0.5939 Trades_Z5=764. Worse. ATR(25) fully confirmed. |
| 598 | STOP1=5.0 | 4.4207 | No | Z5=4.4207 H6=0.6215 Trades_Z5=765. Worse. STOP1=5.5 confirmed. |
| 599 | STOP1=6.0 | 4.4242 | No | Z5=4.4242 H6=0.5957 Trades_Z5=762. Worse. STOP1=5.5 confirmed. |
| 600 | STOP3=7.5 | 4.4416 | No | Z5=4.4416 H6=0.6260 Trades_Z5=763. Nearly tied but below champion. |
| 601 | STOP3=8.5 | 4.4455 | No | Z5=4.4455 H6=0.6260. Identical Calmar, slightly higher PnL+drawdown. No improvement. STOP3=8.0 preferred (lower absolute drawdown). |
| 602 | STOP3=9.0 | 4.4455 | No | Same as 601. STOP3 flat in [8.0, 9.0]. STOP3=8.0 confirmed as preferred value. |
| 603 | LOOKBACK1=125 | 4.4358 | No | Z5=4.4358 H6=0.6197 Trades_Z5=761. Worse. LOOKBACK1=130 confirmed. |
| 604 | Bullish bar filter on dip entry (close-to-high ratio > 0.30) | 4.1731 | No | Z5=4.1731 H6=0.6465 Trades_Z5=794. MORE trades — filter delays entry creating churn via re-entry. Microstructure single-bar filter is dead end. |
| 605 | DIP_MULT2=4.0 | 4.4317 | No | Z5=4.4317 H6=0.6260 Trades_Z5=762. Worse. DIP_MULT2=3.95 confirmed. |
| 606 | RSI(14) > 65 as short entry (replace vol-gate) | 2.3591 | No | Z5=2.3591 H6=-0.2066 Trades=1295. Terrible — RSI fires too often on 1-min NQ. H6 deeply negative. RSI-based short dead end. |
| 607 | 30-bar close momentum short (close < close[i-30]) | 1.3022 | No | Z5=1.3022 H6=-0.5213 Trades=2355. Both negative — extremely correlated with EMA signal, constant churn. |
| 608 | Pure Z-score mean reversion (standalone, 60-bar) | -0.4187 | No | Z5=-0.4187 H6=-0.3292. Both negative — NQ trends, not mean-reverts. Mean reversion direction is dead. |
| 609 | Donchian channel breakout (standalone, 60/30-bar) | 0.1000 | No | Z5=0.1000 H6=-0.39. Near-zero — false breakouts dominate without slow EMA trend filter. |
| 610 | Mean-reversion bounce-fade short (close > fast+2*ATR) | 1.3156 | No | Z5=1.3156 H6=-0.3850. Fires too rarely and in wrong situations. Short redesign direction exhausted. |
| 611 | Short-only standalone (vol-gate, diagnostic) | 1.9802 | No | Z5=1.9802 H6=0.8736 Trades=469. SHORT SIGNAL STANDALONE DIAGNOSTIC. Strong isolated alpha. H6 better than full system (0.87 vs 0.63) — longs drag H6. |
| 612 | Short-only + EMA-only exit (hold through vol dips) | 1.5618 | No | Z5=1.5618 H6=1.5471 Trades=370. Z5 slightly lower but H6 dramatically better (0.87→1.55)! Holding shorts in sustained downtrends captures full H6 bear moves. |
| 613 | Full system + EMA-only short exit | 3.7162 | No | Z5=3.7162 H6=0.9541 Trades=640. H6 greatly improved but Z5 drops 0.73. EMA-exit holds shorts through Z5 bull-market recoveries — gives back gains. |
| 614 | Full system + hybrid short exit (130-bar slow regime) | 3.9193 | No | Z5=3.9193 H6=0.9778 Trades=658. Better than pure EMA-exit but Z5 still 0.52 below champion. Regime check too coarse for Z5. |
| 615 | Full system + hybrid short exit (10-bar slow regime) | 3.7768 | No | Z5=3.7768 H6=0.9552. Worse than 130-bar. 10-bar is too reactive. Both hybrid variants fail to preserve Z5. |
| 616 | Short-only selective: slow declining 50-bar + EMA exit | 0.4813 | No | Z5=0.4813 H6=1.3917 Trades=274. Z5 terrible — slow-declining filter kills Z5 alpha (Z5's shorts are pullbacks in bull market where slow EMA is still rising). KEY INSIGHT: Current vol-gate short is Pareto-optimal for Z5/H6 trade-off. All short changes sacrifice Z5. |
| 618 | Dynamic timeout 90/60 | 4.4490 | No | Identical — all recovering dips resolve within 80 bars. |
| 619 | Dynamic timeout 70/60 | 4.4490 | No | Identical — all recovering dips resolve within 70 bars. |
| 620 | Dynamic timeout 65/60 | 4.4490 | No | Identical — critical dip resolves at bar 63 (extension of 2-3 bars is sufficient). Timeout of 62+ all equivalent. |
| 621 | Dynamic timeout 62/60 | 4.4490 | No | Identical to 65/70/80. 62-bar extended timeout is the minimum effective value. Champion at 80 is safely in flat region. |
| 622 | Dynamic timeout + near-slow extension (fast > slow - 0.5*ATR) | 4.4490 | No | Identical. Near-slow condition already captured by existing fast-above-entry condition. |
| 623 | Minimum fast EMA distance for dip entry: (slow - fast) > 0.5*ATR | 4.4473 | No | Z5=4.4473 H6=0.6638 Trades=764. H6 improved (+0.038) but Z5 slightly below champion. Filters "barely crossed" dip entries — good for H6 quality but removes some Z5 alpha. |
| 624 | Minimum fast EMA distance 0.3*ATR | 4.4473 | No | Z5=4.4473 H6=0.6572. Same Z5 as 0.5*ATR — same trades filtered. 0.5*ATR is better for H6. |
| 625 | LOOKBACK1=131 + dynamic timeout | 4.4451 | No | Z5=4.4451. Below champion — dynamic timeout adds +0.0035 offset but LB=130 remains peak. |
| 626 | Dynamic timeout when fast within 0.5*ATR of entry level | 4.4490 | No | Identical — in practice, fast always above entry level when dip stays open past 60 bars and benefits from extension. |
| 627 | Extend timeout when fast within 0.5*ATR of slow | 4.4455 | No | Reverts to old champion — different condition from fast-above-entry, doesn't capture the same beneficial dip. Original condition confirmed. |
| 633 | HLC3 for fast EMA (instead of OHLC4) | 4.2765 | No | Z5=4.2765 H6=0.5073 Trades=765/689. Significant Z5 drop — OHLC4 confirmed superior. HLC3 removes open-price contribution that adds information. |
| 634 | Slow EMA 20-bar slope gate on base_long entries | 3.8946 | No | Z5=3.8946 H6=0.7618 Trades=775. Gate fires during Z5 corrections — blocks good Z5 long re-entries. H6 improved but Z5 catastrophic. |
| 635 | Slow EMA 200-bar slope gate on base_long entries | 3.0785 | No | Z5=3.0785 H6=0.9386 Trades=707. Even worse — slow EMA declines for 200 bars even in Z5 bull corrections. Slope gate direction fully dead. |
| 636 | 480-bar slow EMA regime: EMA-only short exit in bear, vol-gate in bull | 4.1766 | No | Z5=4.1766 H6=1.0499 Trades=670/591. H6 dramatically improved (+0.42!) but Z5 drops -0.27. Best H6 of all short-exit variants. Cannot beat champion Z5. |
| 637 | 480-bar + 0.5*ATR magnitude threshold for regime switch | 4.1738 | No | Identical to exp_636 — regime is binary, no marginal cases near 0.5*ATR threshold. |
| 638 | 1440-bar slow EMA regime (24-hour lookback) | 3.9108 | No | Z5=3.9108 H6=0.9486. Worse than 480-bar on BOTH metrics. Longer lookback fires at different (worse) Z5 correction points. 480-bar was the best of this family. |
| 639 | ATR expansion exit: atr > 2.0*dip_entry_atr | 4.4490 | No | Identical to champion — condition never fires. ATR(25) cannot double within 60-80 bar dip hold. |
| 640 | ATR expansion exit: atr > 1.5*dip_entry_atr | 4.4381 | No | Z5=4.4381 H6=0.6260. Slightly worse — fires occasionally in Z5, removing a profitable dip. H6 unaffected. ATR expansion exit direction dead. |
| 641 | Volume-imbalance mild dip entry (up_vol_frac > 0.75, 2.5*ATR threshold) | 4.4490 | No | Identical — condition never fires. During deep dips, most bars close DOWN (selling pressure). 75% up-volume fraction is structurally impossible when close < slow - 2.5*ATR. |
| 642 | Re-sweep EXIT_ABOVE_SLOW=0.0 with current architecture | 4.4416 | No | Z5=4.4416 H6=0.6214. Slightly worse. EAS=0.25 confirmed optimal even with fast_declining+dynamic timeout. |
| 643 | EXIT_ABOVE_SLOW=0.5 re-sweep with current architecture | 4.4306 | No | Z5=4.4306 H6=0.6070. Worse. EAS sweep: 0.0→4.4416, 0.25→4.4490(peak), 0.5→4.4306. Peak at 0.25 confirmed with new architecture. |
| 644 | 48-hour (2880-bar) close momentum filter on dip entries | 4.1860 | No | Z5=4.1860 H6=0.6061 Trades=738. Removes BEST Z5 dip trades — deep corrections where 48h momentum turns negative are EXACTLY the most profitable dips (deepest = biggest bounces). All backward momentum filters remove the best dip alpha. |
| 645 | Fast EMA deceleration filter on dip entries | 4.2381 | No | Z5=4.2381 H6=0.5261 Trades=758. Worse. Removes V-bottom dip entries (accelerating declines = capitulation = best entries). Filtering for decelerating declines misses panic-bottom entries. |
| 646 | Slow EMA spread (e380-e470 > 0.5*ATR) quality filter on dip entries | 4.1480 | No | Z5=4.1480 H6=0.5565 Trades=733. Worse. e380 > e470 only after SUSTAINED trend — early-bull Z5 dips (best ones) don't have this spread yet. Removes early-trend dip entries. |
| 647 | Session gap-up dip entry (int64 ts diff) | 4.4490 | No | Identical to champion — condition never fires. int64 diff approach failed silently; ts parsing broken. |
| 648 | Session gap-up dip entry (fixed ts parsing, pd.to_datetime diff) | 4.4490 | No | Identical to champion — structurally cannot fire. System almost always in position at session open (position != 0), so elif gap_dip path never reached. |
| 649 | Drawdown-from-30d-high gate on base_long (7% threshold) | 3.8698 | No | Z5=3.8698 H6=0.4131. Fails H6 gate. Gate fires during Z5 corrections — blocks valid long re-entries in bull market. Drawdown gate direction dead. |
| 650 | Slow EMA = np.min(e380,e425,e470) | 3.7085 | No | Z5=3.7085 H6=0.0006. Catastrophic. Min threshold too low — premature exits and bad entry quality across both periods. |
| 651 | Slow EMA = np.max(e380,e425,e470) | 4.3059 | No | Z5=4.3059 H6=0.3418. Fails H6 gate. Max threshold too high — delays bull entries and misses Z5 dip recoveries. Median confirmed optimal. |
| 652 | roc_240-gated base_long exit (threshold=0) | 4.9698 | No | Z5=4.9698 H6=0.3982 Trades_Z5=666. Massive Z5 gain but H6 fails gate. 4h lookback too slow to detect H6 bear — holds longs during H6 declines. |
| 653 | roc_60-gated base_long exit (threshold=0) | 3.6052 | No | Z5=3.6052 H6=0.7946 Trades_Z5=690. Z5 drops badly — 1h exit too reactive in Z5 bull, exits valid trends. |
| 654 | roc_15-gated base_long exit (threshold=0) | 3.9994 | No | Z5=3.9994 H6=0.6551 Trades_Z5=735. Z5 still below champion. roc sweep: 240→Z5+,H6-; 60→Z5-,H6+; 15→in between. |
| 655 | roc_240 gate (0.005), no base_short override | 4.4793 | No | Z5=4.4793 H6=0.5417 Trades_Z5=761. Z5 beats champion but H6 below 0.6 target. Holds through H6 base_short (high-vol downmoves). |
| 656 | roc_240 gate (0.010) | 4.4490 | No | Identical to champion. 0.010 threshold never fires. |
| 657 | roc_240 gate (0.007) | 4.4490 | No | Z5=4.4490 H6=0.5920. Same Z5 trades but H6 slightly hurt. Some H6 holds through 4h bounces in bear market. |
| 658 | roc_240 gate (0.005) + base_short override | 4.4793 | No | Z5=4.4793 H6=0.5966 Trades_Z5=761. Z5 beats champion. H6 closer to target (0.60 vs 0.54 without override) but still just below. |
| 659 | roc_240 gate (0.003) + base_short override | 4.4892 | No | Z5=4.4892 H6=0.6033 Trades_Z5=758. Both beat champion! Z5 +0.04, H6 just above 0.6 target. |
| 660 | roc_240 gate (0.002) + base_short override | 4.4881 | No | Z5=4.4881 H6=0.6180 Trades_Z5=752. Slightly below 0.003 on Z5. |
| 661 | roc_240 gate (0.001) + base_short override | 4.5344 | No | Z5=4.5344 H6=0.6209 Trades_Z5=743. Better still. Trend continues as threshold drops. |
| 662 | roc_240 gate (0.0005) + base_short override | 4.7173 | YES | Z5=4.7173 H6=0.6538 Z5pnl=$123,788 H6pnl=$23,128 Trades_Z5=728 Trades_H6=660. NEW CHAMPION! Peak Z5 in sweep. |
| 663 | roc_240 gate (0.0001) + base_short override | 4.6996 | No | Z5=4.6996 H6=0.6636 Trades_Z5=717. Z5 slightly below 0.0005 peak — confirms 0.0005 is optimum. |
| 664 | roc_240 gate (0.0003) + base_short override | 4.6872 | No | Z5=4.6872 H6=0.6838 Trades_Z5=722. Z5 below peak. Higher H6 but 0.0005 remains best Z5. |
| 665 | roc_240 gate (0.0007) + base_short override | 4.7162 | No | Z5=4.7162 H6=0.6349 Trades_Z5=736. Nearly tied with 0.0005 but just below. 0.0005 confirmed optimal. |
| 666 | roc_240 gate on dip_tier→0 transition (base_long AND roc_240>0.0005) | 4.7173 | No | Identical to champion. dip_tier→normal transition is already well-specified; base_long is sufficient confirmation. roc_240 adds nothing here. |
| 667 | roc_240-gated short exit (hold when roc_240 < -0.0005) | 3.8347 | No | Z5=3.8347 H6=1.3639. H6 explodes to 1.36 but Z5 drops to 3.83. Holding shorts prevents long re-entries in Z5 bull — same cascading opportunity cost as all short extension experiments. |
| 668 | roc_240-boosted re-entry (dead band 0.01*ATR when roc_240>0.005) | 4.7173 | No | Identical Z5, H6=0.6564. The OR condition for entry never adds new trades. Dead band direction not promising. |
| 669 | Additive tier0 dip entry (roc_240>0.005, close<slow-2.5*ATR) | 4.7173 | No | Identical. roc_240>0.005 AND close<slow-2.5*ATR AND fast_declining rarely coexist — structurally impossible in most bull corrections. |
| 670 | Additive tier0 dip entry (roc_240>0, close<slow-2.0*ATR) | 4.5937 | No | Z5=4.5937 H6=0.6327 Trades_Z5=731. Fires 3 extra trades but they're lower quality (shallow 2.0*ATR dips don't bounce reliably). |
| 671 | Additive tier0 dip entry (roc_240>0, close<slow-3.0*ATR) | 4.5681 | No | Z5=4.5681 H6=0.6147. Still worse — additive dip entries add noise at any ATR level. |
| 672 | Re-sweep EXIT_ABOVE_SLOW=0.0 with roc_240 gate | 4.7121 | No | Slightly below champion. EAS=0.25 still optimal with new architecture. |
| 673 | Re-sweep EXIT_ABOVE_SLOW=0.5 with roc_240 gate | 4.6988 | No | Z5=4.6988 H6=0.6671. Worse. EAS=0.25 confirmed optimal with roc_240 gate. |
| 674 | Normal long stop loss (close < slow - 2.5*ATR) | 4.5818 | No | Z5=4.5818 H6=0.6621 Trades_Z5=746. Stop too tight — cuts profitable roc_240-held positions. |
| 675 | Normal long stop loss (close < slow - 3.5*ATR) | 4.7575 | No | Z5=4.7575 H6=0.6860 Trades_Z5=734. BEATS CHAMPION Z5! Stop catches runaway drawdowns in H6 bear. New best found. |
| 676 | Normal long stop loss (close < slow - 3.0*ATR) | 4.6808 | No | Z5=4.6808 H6=0.6965 Trades_Z5=740. Below 3.5 peak. |
| 677 | Normal long stop loss (close < slow - 4.0*ATR) | 4.7537 | No | Z5=4.7537 H6=0.6021 Trades_Z5=732. Slightly below 3.5. |
| 678 | Normal long stop loss (close < slow - 3.3*ATR) | 4.7213 | No | Z5=4.7213 H6=0.6972. Narrowing in on 3.5-3.7 range. |
| 679 | Normal long stop loss (close < slow - 3.6*ATR) | 4.7712 | YES | Z5=4.7712 H6=0.6837 Z5pnl=$125,202 H6pnl=$23,958 Trades_Z5=733 Trades_H6=664. NEW CHAMPION! Peak of sweep (3.6=3.7, chose 3.6 for better H6). |
| 680 | Normal long stop loss (close < slow - 3.8*ATR) | 4.7643 | No | Z5=4.7643 H6=0.6428. Slightly below 3.6 peak. |
| 681 | Normal long stop loss (close < slow - 3.7*ATR) | 4.7712 | No | Identical to 3.6 (same trades=733, PnL=$125,202). 3.6 preferred for tighter stop and H6=0.6837. |
| 682 | Re-sweep roc_240=0.001 with stop 3.6*ATR | 4.5671 | No | Z5=4.5671 H6=0.6907. roc_240=0.0005 remains optimal even with stop in place. |
| 683 | Re-sweep roc_240=0.0003 with stop 3.6*ATR | 4.7438 | No | Z5=4.7438 H6=0.7140. Below 0.0005 peak. roc_240 threshold confirmed. |
| 684 | roc_240=0.0 + stop 3.6*ATR | 4.7121 | No | Z5=4.7121 H6=0.7177. More aggressive hold but Z5 lower. 0.0005 confirmed optimal. |
| 685 | Hold when roc_240>0.0005 OR roc_60>0.001 + stop 3.6*ATR | 4.7839 | No | Z5=4.7839 H6=0.7174. Adding roc_60 OR condition improves beyond champion! Still below 4.8 though. |
| 686 | roc_60 threshold=0.002 (hold when roc_60>0.2%) | 4.7504 | No | Z5=4.7504. Stricter 1h gate → less improvement. |
| 687 | roc_60 threshold=0.0005 (hold when roc_60>0.05%) | 4.8192 | No | Z5=4.8192 H6=0.7128. Getting closer to peak. |
| 688 | roc_60 threshold=0.0002 (hold when roc_60>0.02%) | 4.8293 | YES | Z5=4.8293 H6=0.7305 Z5pnl=$126,192 H6pnl=$25,168 Trades_Z5=722 Trades_H6=650. NEW CHAMPION! Peak roc_60 sweep. |
| 689 | roc_60 threshold=0.0001 | 4.8119 | No | Z5=4.8119. Slightly below 0.0002 peak. |
| 690 | roc_60 threshold=0.00015 | 4.8100 | No | Z5=4.8100. Below 0.0002. |
| 691 | roc_60 threshold=0.00025 | 4.8148 | No | Z5=4.8148. Below 0.0002. 0.0002 confirmed as peak. |
| 692 | Triple-momentum gate: roc_240>0.0005 OR roc_60>0.0002 OR roc_15>0.0001 | 4.7785 | No | Z5=4.7785 H6=0.6967. Adding roc_15 hurts — too reactive, adds holds in declining 15m momentum cases. |
| 693 | Re-sweep roc_240=0.0003 with dual-momentum gate | 4.7934 | No | Z5=4.7934. roc_240=0.0005 still optimal with roc_60 in place. |
| 694 | Re-sweep stop=3.0*ATR with dual-momentum gate | 4.7313 | No | Stop too tight with dual gate. 3.6 still optimal. |
| 695 | Re-sweep stop=4.0*ATR with dual-momentum gate | 4.8064 | No | Looser stop slightly lower. 3.6 confirmed optimal. |
| 696 | No stop with dual-momentum gate | 4.7800 | No | Z5=4.7800. Stop at 3.6 confirmed necessary and optimal. |
| 697 | RSI<35 hold condition (3rd OR gate) | 5.0250 | No | Z5=5.0250 H6=0.8339. HUGE gain! Holding through oversold RSI captures Z5 bounce. RSI threshold sweep needed. |
| 698 | RSI<40 hold condition | 5.2556 | YES | Z5=5.2556 H6=0.8610 Z5pnl=$130,562 H6pnl=$28,468 Trades_Z5=694 Trades_H6=626. NEW CHAMPION! Peak RSI threshold. |
| 699 | RSI<45 hold condition | 5.2071 | No | Z5=5.2071. Below RSI<40 peak. |
| 700 | RSI<42 hold condition | 5.0989 | No | Z5=5.0989. Below peak. |
| 701 | RSI<38 hold condition | 5.1149 | No | Z5=5.1149. Below peak. |
| 702 | RSI<39 hold condition | 5.2470 | No | Z5=5.2470. Just below peak. RSI=40 confirmed optimal. |
| 703 | Re-sweep roc_240=0.0003 with RSI gate | 5.2527 | No | Z5=5.2527 H6=0.8770. Very close but just below champion. roc_240=0.0005 confirmed with RSI gate. |
| 704 | roc_240=0.0007 with RSI gate | 5.1826 | No | Z5=5.1826. Lower. roc_240=0.0005 confirmed. |
| 705 | roc_60=0.0001 with RSI gate | 5.2236 | No | Z5=5.2236. roc_60=0.0002 confirmed optimal. |
| 706 | roc_60=0.0003 with RSI gate | 5.2412 | No | Z5=5.2412. roc_60=0.0002 confirmed. |
| 707 | Re-sweep stop=4.0*ATR with RSI gate | 5.1939 | No | Z5=5.1939. stop=3.6 confirmed optimal. |
| 708 | Dynamic stop: 5.0*ATR when RSI<30, else 3.6*ATR | 5.0629 | No | Z5=5.0629. Looser stop in deeply oversold allows big losses in H6 bear. Fixed stop=3.6 confirmed. |
| 709 | RSI<30 additive dip entry (OR to existing dip_entry) | 5.5119 | Yes | Z5=5.5119 H6=0.6658 Z5pnl=$139,102 H6pnl=$27,752 composite=11.08. New champion! |
| 710 | RSI-only dip: assign tier1 stop (5.5*ATR) instead of tier3 (8.0*ATR) | 5.7034 | Yes | Z5=5.7034 H6=0.8133 Z5pnl=$141,088 H6pnl=$29,982 composite=11.35. New champion! |
| 711 | RSI dip threshold <35 (more entries) | 5.5241 | No | Z5=5.5241 H6=0.4232 Z5pnl=$140,668 H6pnl=$17,972 composite=11.14. H6 collapsed — threshold <30 confirmed optimal. |
| 712 | RSI dip threshold <25 (fewer, extreme entries) | 5.4465 | No | Z5=5.4465 H6=0.7415 Z5pnl=$136,698 composite=10.91. Too restrictive, fewer good trades. RSI<30 confirmed. |
| 713 | RSI dip requires fast_declining (filter bouncing entries) | 5.7238 | Yes | Z5=5.7238 H6=0.7956 Z5pnl=$141,592 H6pnl=$29,552 composite=11.39. New champion! |
| 714 | Dip timeout gate: skip if RSI<40 (still oversold) | 5.7238 | No | Z5=5.7238 identical. Timeout never binds in Z5 — other exits dominate. |
| 715 | Require roc_240<0 for shorts (bearish momentum filter) | 3.9990 | No | Z5=3.9990 H6=0.4939 Z5pnl=$112,442. Both Z5 and H6 collapsed. Changing base_short breaks dip_entry logic (more false dip entries when base_short suppressed). |
| 716 | EXIT_ABOVE_SLOW = 0.5 (was 0.25, let dip trades ride) | 5.6997 | No | Z5=5.6997 H6=0.7338 Z5pnl=$142,198 composite=11.39. Calmar and H6 both degraded despite tiny PnL gain. 0.25 confirmed optimal. |
| 717 | roc_15>0 required for base_long_enter (15-min momentum confirm) | 5.6350 | No | Z5=5.6350 H6=0.7373 Z5pnl=$142,522 composite=11.34. Filtering entries by 15-min momentum hurts — misses valid early trend entries. |
| 718 | Vol spike dip entry (vol_5 > 2.0*vol_240 + fast_declining) | 5.6675 | No | Z5=5.6675 H6=0.8054 Z5pnl=$139,948 composite=11.27. Capitulation signal adds noise — vol spike entries during neutral market are lower quality than RSI<30 entries. |
| 719 | Large bar range dip entry (bar_range > 2.5*ATR + fast_declining) | 5.6903 | No | Z5=5.6903 H6=0.8371 Z5pnl=$140,762 composite=11.32. Same pattern: extra entries improve H6 but reduce Z5 Calmar. RSI<30 is the best additive signal. |
| 720 | 3 consecutive down bars dip entry (microstructure) | 5.4342 | No | Z5=5.4342 H6=0.1847 Z5pnl=$139,282 1197 trades. H6 catastrophic — 3-bar down pattern fires constantly in bear market creating mass losing longs. Microstructure bar patterns banned. |
| 721 | RSI>=60 recovery exit for dip positions | 5.6836 | No | Z5=5.6836 H6=0.6858 Z5pnl=$140,092 composite=11.29. Exits ATR-tier dips too early before full recovery. EXIT_ABOVE_SLOW=0.25 already handles this cleanly. |
| 722 | EXIT_ABOVE_SLOW = 0.0 (exit dip exactly at slow EMA) | 5.5397 | No | Z5=5.5397 H6=0.7643 Z5pnl=$137,468 composite=11.04. Exits too early, misses recovery. EXIT_ABOVE_SLOW sweep: 0.0→11.04, 0.25→11.39(champion), 0.5→11.39-. 0.25 confirmed optimal. |
| 723 | RSI dip: roc_5<-0.001 instead of fast_declining | 5.4709 | No | Z5=5.4709 H6=1.0334 Z5pnl=$137,672 composite=10.98. Too restrictive: 768 trades vs 995. H6 outstanding (1.03) but Z5 drops. roc_5 threshold 10x larger than fast_declining EMA magnitude. fast_declining confirmed for RSI path. |
| 724 | RSI dip: require roc_60 > -0.001 (hourly momentum not negative) | 5.2201 | No | Z5=5.2201 H6=0.9303 Z5pnl=$130,898 composite=10.46. Z5 very bad. RSI<30 occurs BECAUSE of sharp declines — filtering by negative roc_60 removes most valid RSI dips. Momentum filters on RSI dip entries are contradictory. |
| 725 | RSI>70 dead-cat bounce short (exit when RSI<50 or base_long) | 5.0918 | No | Z5=5.0918 H6=1.0438 Z5pnl=$136,582 H6pnl=$35,448 composite=10.56. H6 excellent (1.04) but Z5 Calmar drops. RSI>70 fires during bull market continuation → false shorts in Z5. Needs bear regime filter. |
| 726 | RSI>70 short with roc_240<0 regime filter | 5.2975 | No | Z5=5.2975 H6=1.0582 Z5pnl=$141,982 composite=10.98. Z5 still hurt by corrections where roc_240<0 briefly. |
| 727 | RSI>70 short with roc_240<-0.002 regime filter | 5.5855 | No | Z5=5.5855 H6=1.0018 Z5pnl=$141,178 composite=11.23. Closer but still below champion. |
| 728 | RSI>70 short with roc_240<-0.005 regime filter | 5.6523 | No | Z5=5.6523 H6=1.0427 Z5pnl=$139,822 composite=11.25. Still below. RSI>70 mean-reversion short direction exhausted — even strict bear-regime filters hurt Z5 Calmar through Z5 corrections. BANNED. |
| 729 | fast_declining window=8 (beyond previous sweep of 3-7) | 5.6786 | No | Z5=5.6786 H6=0.7600 Z5pnl=$140,472 composite=11.30. Window=8 worse than 5 even with RSI dip. fast_declining window sweep fully confirmed at 5. |
| 730 | LOOKBACK1=125 (was 130) | 5.7172 | No | Z5=5.7172 H6=0.7991 Z5pnl=$141,428 composite=11.37. Essentially identical to champion. |
| 731 | LOOKBACK1=135 (was 130) | 5.7234 | No | Z5=5.7234 H6=0.7799 Z5pnl=$141,582 composite=11.387. Essentially identical. LOOKBACK1 extremely stable — 125/130/135 all equivalent. Confirmed at 130. |
| 732 | RSI-only dip: assign dip_tier=2 (STOP2=5.0 tighter) instead of tier1 | 5.7137 | No | Z5=5.7137 H6=0.7407 Z5pnl=$141,818 composite=11.386. Essentially identical to champion. Stop difference (5.0 vs 5.5) negligible. tier1 assignment confirmed. |
| 733 | DIP_MULT1=4.0 (was 3.9, deepen tier1 since RSI handles shallows) | 5.7161 | No | Z5=5.7161 H6=0.7954 Z5pnl=$141,528 composite=11.377. Essentially same. DIP_MULT1 extremely stable at 3.9. |
| 734 | Slow EMA base HLC3 = (H+L+C)/3 instead of HL2 = (H+L)/2 | 5.5471 | No | Z5=5.5471 H6=0.7833 Z5pnl=$139,098 composite=11.11. HL2 confirmed significantly better basis for slow EMA. |
| 735 | Add roc_5<=0.0002 as 4th AND condition to non-dip exit gate | 5.7713 | Yes | Z5=5.7713 H6=0.7926 Z5pnl=$141,998 H6pnl=$29,408 composite=11.45. New champion! 5-min momentum check prevents exits during brief 5-min dips within longer uptrends. |
| 736 | roc_5 threshold 0.0001 (tighter, only exit when 5-min clearly negative) | 5.8850 | Yes | Z5=5.8850 H6=0.7963 Z5pnl=$142,378 H6pnl=$29,588 composite=11.58. New champion! |
| 737 | roc_5 threshold 0.0 (must be negative) | 5.8083 | No | Z5=5.8083 H6=0.7847 composite=11.48. Too restrictive. |
| 738 | roc_5 threshold 0.00005 (between 0 and 0.0001) | 5.8686 | No | Z5=5.8686 H6=0.7862 composite=11.55. roc_5 sweep: 0.0→11.48, 0.00005→11.55, 0.0001→11.58(peak), 0.0002→11.45. roc_5=0.0001 confirmed optimal. |
| 739 | Add roc_15<=0.0001 as 5th AND condition to exit gate | 5.8884 | Yes | Z5=5.8884 H6=0.7271 Z5pnl=$143,408 H6pnl=$27,918 composite=11.62. New champion! H6 dipped (0.7271) but still passes gate. |
| 740 | roc_15 threshold 0.0002 (looser) | 5.8536 | No | Z5=5.8536 H6=0.7440 Z5pnl=$142,202 composite=11.54. Looser 15-min threshold is worse. |
| 741 | roc_15 threshold 0.00005 (tighter than 0.0001) | 5.8895 | Yes | Z5=5.8895 H6=0.7299 Z5pnl=$143,532 H6pnl=$28,018 composite=11.63. New champion! Marginal improvement. |
| 742 | roc_15 threshold 0.0 (must be negative) | 5.8858 | No | Z5=5.8858 H6=0.7578 Z5pnl=$143,442 composite=11.62. roc_15 sweep: 0.0→11.62, 0.00005→11.63(peak), 0.0001→11.62, 0.0002→11.54. roc_15=0.00005 confirmed. |
| 743 | roc_60 threshold 0.0001 (tighter from 0.0002) | 5.9170 | Yes | Z5=5.9170 H6=0.7063 Z5pnl=$143,742 H6pnl=$27,112 composite=11.67. New champion! H6=0.7063 near gate floor. |
| 744 | roc_60 threshold 0.00005 (even tighter) | 5.8536 | No | Z5=5.8536 H6=0.7012 Z5pnl=$143,568 composite=11.60. Z5 Calmar drops. roc_60=0.0001 confirmed optimal. |
| 745 | roc_240 threshold 0.0003 (tighter from 0.0005) | 5.9521 | Yes | Z5=5.9521 H6=0.7262 Z5pnl=$143,902 H6pnl=$27,668 composite=11.71. New champion! |
| 746 | roc_240 threshold 0.0001 (tighter from 0.0003) | 5.9522 | No | Z5=5.9522 H6=0.7114 Z5pnl=$143,938 composite=11.71. Essentially no change — tighter roc_240 has no effect. |
| 747 | roc_240 threshold 0.0002 (between 0.0001 and 0.0003) | 5.9745 | Yes | Z5=5.9745 H6=0.7233 Z5pnl=$143,912 H6pnl=$27,558 composite=11.73. New champion! roc_240 sweet spot at 0.0002. |
| 748 | RSI threshold 35 (lower from 40) in non-dip exit gate | 5.7858 | No | Z5=5.7858 H6=0.7492 Z5pnl=$142,612 composite=11.49. Looser RSI allows exits too early. |
| 749 | RSI threshold 45 (higher from 40) in non-dip exit gate | 5.5641 | No | Z5=5.5641 H6=0.7131 Z5pnl=$142,272 composite=11.26. Tighter RSI hurts. RSI>40 confirmed optimal. |
| 750 | non-dip hard stop 3.0*ATR (tighter from 3.6) | 5.7346 | No | Z5=5.7346 H6=0.6815 Z5pnl=$140,872 composite=11.37. Tighter stop hurts. |
| 751 | non-dip hard stop 4.0*ATR (looser from 3.6) | 5.8897 | No | Z5=5.8897 H6=0.6781 Z5pnl=$142,428 composite=11.59. Looser stop also worse. 3.6*ATR confirmed optimal. |
| 752 | require roc_240>0 for base_long entry (medium-term momentum filter) | 6.4560 | Yes | Z5=6.4560 H6=1.3046 Z5pnl=$149,992 H6pnl=$41,802 composite=12.46. MASSIVE win! +0.73 composite, H6 nearly doubles. |
| 753 | roc_240>0.001 for base_long entry (tighter) | 6.2764 | No | Z5=6.2764 H6=1.0804 Z5pnl=$147,492 composite=12.18. Too strict. |
| 754 | roc_240>-0.001 for base_long entry (looser) | 6.4049 | No | Z5=6.4049 H6=0.9855 Z5pnl=$150,478 composite=12.42. Allows some bad longs back. roc_240>0 confirmed optimal. |
| 755 | roc_240>0 filter added to all dip_entry tier conditions | 6.4047 | No | Z5=6.4047 H6=1.3076 Z5pnl=$148,802 composite=12.36. Tier dips already filtered by slow>slow_prev. |
| 756 | roc_240>0 filter on RSI dip path only | 5.6452 | No | Z5=5.6452 H6=1.3388 Z5pnl=$137,328 composite=11.14. Trades drop 695. RSI dips cut too aggressively. |
| 757 | roc_240>0 AND roc_60>0 for base_long_enter | 6.4587 | Yes | Z5=6.4587 H6=1.0750 Z5pnl=$150,952 H6pnl=$36,392 composite=12.50. New champion! Marginal improvement. H6 drops vs 752. |
| 758 | roc_60>0.001 for base_long_enter (tighter 60-min threshold) | 6.9889 | Yes | Z5=6.9889 H6=0.8689 Z5pnl=$149,148 H6pnl=$31,852 composite=12.95. New champion! +0.45 composite. |
| 759 | roc_60>0.002 for base_long_enter (even tighter) | 7.8443 | Yes | Z5=7.8443 H6=0.6620 Z5pnl=$142,848 H6pnl=$25,418 composite=13.56. New champion! +0.61. H6=0.66 near gate floor. |
| 760 | roc_60>0.003 for base_long_enter | 5.4209 | No | Z5=5.4209 H6=1.0443 Z5pnl=$115,415 composite=10.04. Cliff edge — PnL drops $27k. roc_60>0.002 confirmed optimal. |
| 761 | roc_240>0.001 (tighter, re-sweep with roc_60>0.002) | 7.2733 | No | Z5=7.2733 H6=0.5437 Z5pnl=$136,930 composite=12.75. Fails H6 gate. roc_240>0 confirmed even with roc_60>0.002. |
| 762 | add roc_15>0 to entry (3-TF: roc_240>0, roc_60>0.002, roc_15>0) | 7.6579 | No | Z5=7.6579 H6=0.7230 Z5pnl=$139,452 composite=13.24. Z5 calmar drops. roc_15 not helpful for entry. |
| 763 | exit gate roc_240 re-sweep: <=0.0003 (with roc_60>0.002 entry) | 7.8468 | Yes | Z5=7.8468 H6=0.6620 Z5pnl=$142,892 H6pnl=$25,418 composite=13.56. Marginal improvement. Exit roc_240 effectively same at 0.0002 vs 0.0003. |
| 764 | exit gate roc_240 <=0.0005 re-sweep | 7.8015 | No | Z5=7.8015 H6=0.6562 Z5pnl=$142,068 composite=13.48. Looser exit threshold slightly worse. 0.0003 confirmed. |
| 765 | exit gate roc_60 <=0.0002 (looser, re-sweep) | 7.8468 | No | Z5=7.8468 H6=0.6620 Z5pnl=$142,892 composite=13.562. Identical to champion — roc_60 exit threshold never binding between 0.0001 and 0.0002. |
| 766 | remove roc_60 from exit gate entirely | 7.7633 | No | Z5=7.7633 H6=0.6494 Z5pnl=$141,372 composite=13.42. Removing roc_60 hurts — prevents exits when roc_60>0.0002 but other conditions met. Keep it. |
| 767 | exit gate roc_5 <=0.0002 (looser) | 7.5680 | No | Z5=7.5680 H6=0.6695 Z5pnl=$142,688 composite=13.28. Looser roc_5 hurts. |
| 768 | exit gate roc_5 <=0.00005 (tighter) | 7.8446 | No | Z5=7.8446 H6=0.6618 Z5pnl=$142,852 composite=13.559. Marginally below threshold. roc_5=0.0001 confirmed optimal. |
| 769 | roc_60>0.0015 for base_long_enter (finer sweep) | 6.8118 | No | Z5=6.8118 H6=0.6908 Z5pnl=$147,108 composite=12.70. Non-monotonic — 0.0015 worse than both 0.001 and 0.002. roc_60>0.002 confirmed optimal. |
| 770 | add roc_5>0 to base_long_enter (4th entry filter) | 7.7306 | No | Z5=7.7306 H6=0.6885 Z5pnl=$142,835 composite=13.44. Drops Z5 calmar. roc_5 entry filter not helpful. |
| 771 | add RSI<70 overbought filter to base_long_enter | 7.5554 | No | Z5=7.5554 H6=0.7915 Z5pnl=$137,418 composite=13.05. Z5 drops. RSI<70 filter not helpful. |
| 772 | LOOKBACK1 = 140 (longer tier1 dip window) | 7.8438 | No | Z5=7.8438 H6=0.6344 Z5pnl=$142,838 composite=13.557. Essentially identical to champion. LOOKBACK1=130 confirmed. |
| 773 | EXIT_ABOVE_SLOW = 0.30 (looser dip profit exit) | 7.7676 | No | Z5=7.7676 H6=0.6097 Z5pnl=$142,872 composite=13.48. H6 drops. 0.25 confirmed. |
| 774 | EXIT_ABOVE_SLOW = 0.20 (tighter dip profit exit) | 7.8457 | No | Z5=7.8457 H6=0.6547 Z5pnl=$142,872 composite=13.561. Marginally below. 0.25 confirmed optimal. |
| 775 | add roc_240<-0.001 as trend-collapse exit for non-dip longs | 5.8741 | No | Z5=5.8741 H6=0.6945 Z5pnl=$129,368 composite=11.05. Too many exits — fires on Z5 corrections. |
| 776 | add roc_60<-0.002 as fast exit for non-dip longs | 7.1214 | No | Z5=7.1214 H6=0.9710 Z5pnl=$126,798 composite=12.19. PnL drops $16k. Momentum-based fast exits hurt Z5. |
| 777 | base_long requires roc_60>0 (tighter hold condition) | 6.5279 | No | Z5=6.5279 H6=1.1165 Z5pnl=$128,982 composite=11.69. 1216 trades — too many whipsaws. |
| 778 | relative momentum: roc_60>roc_240 for entry | 7.1742 | No | Z5=7.1742 H6=0.5384 Z5pnl=$126,622 composite=12.24. Fails H6 gate. |
| 779 | DIP_MULT1 = 3.8 re-sweep | 7.8468 | No | Z5=7.8468 H6=0.6896 Z5pnl=$142,892 composite=13.562. Identical — no dips in 3.8-3.9 ATR range fire. DIP_MULT1 insensitive here. |
| 780 | DIP_MULT1 = 4.0 re-sweep | 7.8441 | No | Z5=7.8441 H6=0.6617 Z5pnl=$142,842 composite=13.558. Marginally below. DIP_MULT1 insensitive in 3.8-4.0 range. |
| 781 | remove roc_240 from exit gate (test redundancy) | 7.6359 | No | Z5=7.6359 H6=0.6429 Z5pnl=$139,052 composite=13.20. roc_240 exit condition not redundant. Keep it. |
| 782 | fast EMA uses close price (instead of OHLC4) | 6.8103 | No | Z5=6.8103 H6=0.5447 Z5pnl=$141,518 composite=12.47. Fails H6 gate. OHLC4 confirmed for fast EMA. |
| 783 | fast EMA uses HL2 price instead of OHLC4 | 8.2066 | Yes | Z5=8.2066 H6=0.8029 Z5pnl=$145,118 H6pnl=$28,002 composite=14.01. MASSIVE win! +0.45 composite, H6 rebounds to 0.80. |
| 784 | fast EMA span = 5 (re-sweep with HL2) | 6.5347 | No | Z5=6.5347 H6=0.6154 Z5pnl=$136,408 composite=11.99. Span 5 too reactive. |
| 785 | fast EMA span = 7 (re-sweep with HL2) | 6.7294 | No | Z5=6.7294 H6=0.3711 Z5pnl=$139,202 composite=n/a. Fails H6 gate hard. Span 6 confirmed optimal with HL2. |
| 786 | roc_60>0.001 entry re-sweep with HL2 | 7.1017 | No | Z5=7.1017 H6=1.0752 Z5pnl=$150,058 composite=13.10. Lower Z5 calmar. roc_60>0.002 confirmed optimal. |
| 787 | roc_60>0.003 entry re-sweep with HL2 | 5.7302 | No | Z5=5.7302 H6=1.2094 Z5pnl=$118,565 composite=10.47. PnL drops $27k. roc_60>0.002 confirmed. |
| 788 | slow EMA center=426 re-sweep (was forbidden) | 8.1523 | No | Z5=8.1523 H6=0.8034 Z5pnl=$144,158 composite=13.92. Center=425 still optimal. |
| 789 | slow EMA center=424 re-sweep | 7.7358 | No | Z5=7.7358 H6=0.8575 Z5pnl=$142,588 composite=13.44. Center=425 confirmed optimal. |
| 790 | slow EMA spread ±44 (381/425/469) | 8.2066 | No | Z5=8.2066 H6=0.8029 Z5pnl=$145,118 composite=14.011. Identical to champion. Spread insensitive ±44 vs ±45. |
| 791 | slow EMA spread ±46 (379/425/471) | 8.2066 | No | Z5=8.2066 H6=0.8029 Z5pnl=$145,118 composite=14.011. Identical. Spread ±44-46 all equivalent. |
| 792 | fast EMA uses HLC3 instead of HL2 | 7.7319 | No | Z5=7.7319 H6=0.5776 Z5pnl=$145,262 composite=13.54. HL2 > HLC3 > OHLC4 > close confirmed ordering. |
| 793 | entry dead band 0.04*ATR (tighter re-sweep with HL2) | 8.2066 | No | Z5=8.2066 H6=0.7991 Z5pnl=$145,118 composite=14.011. Identical. Dead band 0.04/0.05 equivalent. |
| 794 | entry dead band 0.03*ATR | 8.1684 | No | Z5=8.1684 H6=0.7405 Z5pnl=$144,442 composite=13.95. Slightly worse. Dead band 0.04-0.05 optimal range. |
