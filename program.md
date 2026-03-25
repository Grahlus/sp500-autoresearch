# SP500 Autoresearch — Research Program

## Objective
Beat SP500 buy-and-hold on the **validation set** (2022-07 → 2024-06, held-out from training).
Long AND short positions allowed. Full universe ~500 stocks, daily bars.

---

## Data Splits (hardcoded in prepare.py — never change these)

| Split | Period | Days | Purpose |
|-------|--------|------|---------|
| Train | 2014-01-01 → 2022-06-30 | ~2142 | Agent fits freely here |
| Validation | 2022-07-01 → 2024-06-30 | ~504 | **Agent optimises METRIC here** |
| Test | 2024-07-01 → present | ~190+ | **HIDDEN — run evaluate.py only when done** |

**Critical rules:**
- `run.py` shows Train + Validation only. Test is never shown during experiments.
- `evaluate.py` shows all 3 splits — run ONLY when a strategy is truly finalised.
- Looking at test results and then continuing to iterate invalidates the test set.
- The 47 experiments from session 1 used an older 2-year window. The new validation
  window (2022-07 → 2024-06) is partially fresh — re-establish baseline scores first.

---

## Execution Model

- Signal observed at **close[T]**
- Position entered at **open[T+1]** ← realistic fill price
- Position exited/rebalanced at **open[T+2]**
- `generate_signals()` must only use data up to row T
- `run.py` runs `validate_no_lookahead()` automatically — violations abort the run

**Lookahead rules for agent.py:**
- Never use `.shift(-N)` for N > 0
- Never index future rows: `close.iloc[i + k]` where k > 0 is forbidden
- Global stats like `close.mean()` over the full dataset are lookahead — use rolling windows
- `.iloc[a:i]` is safe (excludes row i). `.iloc[a:i+1]` includes today's close (also fine)

---

## Hard Constraints

- Max gross exposure: 1.0 (normalised by prepare.py)
- Transaction costs: 5 bps per turnover unit (enforced by prepare.py)
- Time budget: 60 minutes per experiment (hard timeout in prepare.py)
- **Do NOT edit prepare.py, run.py, or evaluate.py**

---

## What We Know (confirmed over 47 experiments — DO NOT re-test)

### Confirmed winners (additive — each improved on the previous)

| Finding | Effect | Exp |
|---------|--------|-----|
| JT skip=3 weeks (not 4) | Optimal with vol-accel filter | #024 |
| Vol top 50% pre-filter | +0.09 Sharpe vs no filter | #005 |
| Vol acceleration composite (JT × vol_accel rank) | +0.30 Sharpe, major | #007 |
| Top 2.5% concentration (not 2%, not 3%) | Optimal concentration | #020 |
| Trailing stop 15% from position HIGH (not from entry) | Better than fixed stop | #028 |
| F&G regime: skip new entries fear<25, top1.5% greed>75 | +0.32 Sharpe, huge | #039 |

### Confirmed dead ends (do not retry)

| Idea | Result | Why |
|------|--------|-----|
| Short bottom 3% momentum | Hurts | Bull market period, longs dominate |
| VIX position scaling | No effect | Engine normalises gross exposure to 1.0, scaling has no impact |
| VIX cash regime (VIX>35) | -0.20 Sharpe | Too many false positives |
| RSI < 70 filter | -1.1 Sharpe | Kills best momentum stocks |
| Stop-loss from entry (vs from HIGH) | Worse | Locked in less of each winner |
| Stop-loss 10-12.5% | Worse than 15% | Too tight, premature exits |
| 7 or 10 holdings (vs top 2.5%) | Worse | Dilutes the edge |
| Hard regime filter (SPY 200 MA) | No effect | SPY barely crossed in test period |
| Sector diversification cap | Reduces returns | Concentration is the edge |
| 3-day vol accel window | Worse | Noisy |
| 2-week rebalance | Worse | Too frequent, txn costs hurt |
| Multi-horizon momentum avg(13w+26w) | Worse | 26w alone is better |
| Dollar-volume (vs raw volume) | Slightly worse | Raw volume better signal |
| Intraday filter (close > open) | -1.1 Sharpe | Too restrictive |
| Vol accel as hard filter (binary) | -0.6 Sharpe | Rank combination is better |
| Sticky exit threshold | No improvement | Not worth the complexity |
| F&G thresholds 20/80 or 30/70 | Worse than 25/75 | 25/75 is optimal |
| Both VIX+F&G dual regime | Same as F&G alone | VIX adds nothing |

---

## Mandatory Logging Rule

After **every** `uv run python run.py`, immediately run this — before writing any new code:

```bash
./log_exp.sh NNN "hypothesis" "sharpe=X.XX calmar=X.XX alpha=X.XX maxdd=-XX.X% ret=+XXX%" Yes|No
```

- `NNN` = experiment number
- metrics = exact values from the `>>> SCORE` block
- `Yes` = committed (improved validation METRIC), `No` = reverted

**This is mandatory. Do not skip it.**

---

## Research Frontier (priority order — all untested on new val window)

### 1. Re-establish baseline (first run)
Re-run the current agent.py (exp #039 config) on the NEW validation window.
This is exp #001 in the new log. Record the score — it becomes the new benchmark.

### 2. Short side (revisit with tighter filter)
Previous attempt (#002) shorted broadly. Try a more surgical approach:
- Short only stocks that triggered trailing stop (momentum broke down)
- Short bottom 1% (not 3%) by composite rank, only when F&G > 60 (not fear)
- Short must also be below 20-week MA
- Target: improve Sharpe AND reduce MaxDD

### 3. ML signal layer
LightGBM or random forest trained on TRAIN set only (2014-2022):
- Features per stock per week: JT_26w, JT_13w, vol_accel_10/20, vol_accel_5/20,
  MA_distance (price/MA20w - 1), VIX level, F&G level, 4w return
- Target: top 2.5% composite rank label (binary classification)
- Use ML score as a weight modifier on the RSL composite rank
- Train on train split only — validate on validation split

### 4. Earnings/event avoidance
- Don't enter positions within 3 days before earnings (use a calendar or infer from
  abnormal volume spikes as proxy)
- Reduces gap-down risk on long positions

### 5. Sector momentum overlay
- Rank GICS sectors by 3-month momentum
- Overweight stocks in top 2 sectors by 1.5x, underweight bottom 2 by 0.5x
- Keep stock-level composite ranking as primary signal

### 6. Adaptive top% based on market breadth
- When > 60% of SP500 is above 200-day MA: use top 2.5% (normal)
- When < 40% of SP500 is above 200-day MA: shrink to top 1% (defensive)
- Breadth computed as of close[T], no lookahead

---

## Experiment Log (new validation window: 2022-07 → 2024-06)

| # | Hypothesis | Sharpe | Calmar | Alpha | MaxDD | Return | Kept |
|---|-----------|--------|--------|-------|-------|--------|------|
| 001 | Re-baseline: best config from session 1 on new val window | — | — | — | — | — | pending |
| 068 | F&G<25 AND VIX>20 skip entries dual filter — AND condition less effective than F&G alone | 1.924 | 3.789 | 0.575 | -23.4% | +256.5% | — | — | — | 0.657 | 0.410 | -36.7% | +318.0% | — | No |
