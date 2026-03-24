# Trading Autoresearch — program.md

---

## CURRENT STATE

```
best_exp:        1466
best_z5_calmar:  12.68
best_h6_calmar:  (run uv run python run.py to verify)
best_composite:  19.183
next_exp:        1467
run_command:     uv run python run.py
editable_files:  [agent.py, program.md]
frozen_files:    [prepare.py]
```

Champion strategy (exp_1466):
Three-tier dip-buying + RSI<32 dip entry, HL2 fast EMA(6) / median(EMA380/425/470 of HL2) slow,
bidirectional with vol_240<pct85(480) long entry gate + vol_60>pct40(480) short gate,
momentum entry filters (roc_240>0.0001 AND roc_60>0.0025 AND slow_rising10),
momentum hold gates (roc_240>0.0003 OR roc_60>0.0001 OR RSI(14)<40),
not_freefall exit gates (multi-lag roc_5/roc_15/roc_60/roc_240 conditions),
Wilder ATR(14), non-dip stop 4.0*ATR, dip stops T1=5.25/T2=3.0/T3=10.0*ATR,
entry dead band 0.05*ATR, fast_declining filter (3-bar, 0.02*ATR threshold),
dip timeout 90/70 bars, EXIT_ABOVE_SLOW=0.12*ATR,
LOOKBACK1=240, LOOKBACK2=90, LOOKBACK3=45.

---

## OBJECTIVE

Maximize the **composite score** on Z5 validation:

```
composite = Z5_calmar + Z5_pnl / 25000
```

Current champion composite: **19.183** (calmar=12.68)

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
elif z5_composite <= 19.183:
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
|---|-----------|-----------|-------|-------|
| 000 | Random baseline | -0.9009 | No | Starting point |

| 943 | STOP3=10.0 | 11.1527 | Yes | calmar=11.1527 |
| 940 | STOP2=3.5 | 11.1255 | Yes | calmar=11.1255 |
| 939 | STOP2=4.0 | 11.0852 | Yes | calmar=11.0852 |
| 991 | fast_declining lookback=3 bars | 11.1753 | Yes | calmar=11.1753 |
| 1034 | STOP1=5.25 | 11.2538 | Yes | composite=17.310 H6=0.8400 |
| 1041 | EXIT_ABOVE_SLOW=0.12 | 11.2686 | Yes | composite=17.333 H6=0.8480 |
| 1044 | non-dip stop 3.75*ATR | 11.2735 | Yes | composite=17.340 H6=0.8558 |
| 1053 | dip timeout 90/70 | 11.2913 | Yes | composite=17.368 H6=0.8968 |
| 1084 | ohlc4 for non-dip stop | 11.3036 | Yes | composite=17.387 H6=0.8684 |
| 1103 | roc_240>0 for dip_tier transition | 11.3155 | Yes | composite=17.405 H6=0.8007 |
| 1113 | roc_60>0.0003 dip_tier transition | 11.3188 | Yes | composite=17.410 H6=0.8161 |
| 1116 | roc_60>0.0003 only for dip_tier transition | 11.3244 | Yes | composite=17.419 H6=0.8202 |
| 1125 | RSI tier4 EXIT_RSI=0.20 | 11.3903 | Yes | composite=17.479 H6=0.8295 |
| 1154 | slow EMA basis OHLC4 | 11.4504 | Yes | composite=17.526 H6=0.8499 |
| 1168 | EXIT_RSI=0.18 | 11.4523 | Yes | composite=17.529 H6=0.8362 |
| 1170 | EXIT_RSI=0.19 | 11.4542 | Yes | composite=17.532 H6=0.8425 |
| 1185 | single EMA(425) OHLC4 slow | 11.4565 | Yes | composite=17.535 H6=0.8425 |
| 1201 | dip entry roc_5>-0.005 filter | 11.5341 | Yes | composite=17.654 H6=0.8621 Z5pnl=$153,000 |
| 1202 | dip entry roc_5>-0.003 | 11.5511 | Yes | composite=17.680 H6=0.7982 Z5pnl=$153,225 |
| 1214 | exit dip on roc_5<-0.003 | 11.5556 | Yes | composite=17.687 H6=0.7115 Z5pnl=$153,285 |
| 1216 | entry roc_5>-0.0025 + exit roc_5<-0.003 | 11.5673 | Yes | composite=17.705 H6=0.7531 Z5pnl=$153,440 |
| 1219 | entry roc_5>-0.0025 only (no exit) | 11.5703 | Yes | composite=17.710 H6=0.8405 Z5pnl=$153,480 |
| 1223 | dip entry requires roc_5[i-1]<0 | 11.5304 | Yes | composite=17.741 H6=0.8309 Z5pnl=$155,255 |
| 1225 | dip entry roc_5[i-2]<0 also | 11.5714 | Yes | composite=17.860 H6=1.009 Z5pnl=$157,220 |
| 1250 | tier1/2/3 dip RSI>65 exit | — | Yes | composite=18.028 H6=1.157 |
| 1266 | not_freefall += roc_15[i-1,i-2]<0 | — | Yes | composite=18.155 H6=1.095 |
| 1272 | not_freefall += roc_60[i-1,i-2]<0 | — | Yes | composite=18.224 H6=1.148 |
| 1302 | LOOKBACK2=90 | — | Yes | composite=18.246 H6=— |
| 1310 | DIP_MULT1=4.5 | — | Yes | composite=18.313 H6=— |
| 1316 | STOP2=3.0 | — | Yes | composite=18.322 H6=— |
| 1320 | DIP_MULT3=5.0 | — | Yes | composite=18.366 H6=— |
| 1340 | not_freefall roc_5[i-1]<0.0005 | — | Yes | composite=18.574 H6=— |
| 1342 | roc_15[i-1]<0.0005 added | — | Yes | composite=18.590 H6=— |
| 1345 | not_freefall roc_5/roc_15[i-1]<0.001 | — | Yes | composite=18.630 H6=— |
| 1359 | roc_60[i-2]<0.0002 | — | Yes | composite=18.760 H6=— |
| 1371 | remove redundant EMA gap from base_long_enter | — | Yes | composite=18.785 H6=— |
| 1378 | DIP_MULT2 3.95→3.75 | 12.44 | Yes | composite=18.814 H6=— |
| 1380 | DIP_MULT2 3.75→3.85 | 12.44 | Yes | composite=18.818 H6=— |
| 1394 | not_freefall: add roc_240<0.002 | 12.53 | Yes | composite=18.956 H6=— |
| 1409 | not_freefall: add roc_60[i]<0 | 12.58 | Yes | composite=19.022 H6=— |
| 1412 | not_freefall: add roc_15[i]<0.0005 | 12.59 | Yes | composite=19.040 H6=— |
| 1435 | not_freefall: roc_60[i-2] 0.0002→0.0003 | 12.62 | Yes | composite=19.086 H6=— |
| 1453 | not_freefall: add roc_60[i-3]<0 | 12.68 | Yes | composite=19.139 H6=— |
| 1454 | not_freefall: roc_60[i-3] 0→0.0003 | 12.67 | Yes | composite=19.167 H6=— |
| 1458 | not_freefall: roc_60[i-2] 0.0003→0.0004 | 12.68 | Yes | composite=19.171 H6=— |
| 1466 | not_freefall: roc_5[i] -0.0025→-0.0026 | 12.68 | Yes | composite=19.183 H6=— |
