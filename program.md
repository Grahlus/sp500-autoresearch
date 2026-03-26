# SP500 Autoresearch — Research Program (Session 6)

## Walk-Forward Windows
| Window | Train | Test | Regime |
|--------|-------|------|--------|
| W1 | 2014-2017 | 2018 | late bull |
| W2 | 2015-2018 | 2019 | bull |
| W3 | 2016-2019 | 2020 | COVID crash+recovery |
| W4 | 2017-2020 | 2021 | bull |
| W5 | 2018-2021 | 2022 | bear |
| W6 | 2019-2022 | 2023 | recovery |
| W7 | 2020-2023 | 2024-H1 | AI bull |

## Scoring
- Primary: **mean Sharpe across all 7 windows**
- Target: mean Sharpe > 0.6, MaxDD < 25%

## Hard Constraints (enforced by run.py)
1. trades_per_year > 150 → REVERT
2. mean_sharpe < 0.3 → REVERT
3. neg_windows > 2 → REVERT
4. min_sharpe < -0.5 → REVERT
5. ONE change per experiment

---

## CRITICAL ARCHITECTURAL FINDINGS (Session 6)

### Finding 1: n_top = 2 with SP500 filter (not 12)
`int(len(filt) × 2.5%)` where `len(filt) ≈ 90` (SP500 after dvol q70 + above_ma)
gives only **n_top = 2 stocks**. The min(…,15) cap never binds. avg_positions = **1.6**.

The "3.9" figure was from the bad-revert code that used R1000 (841 stocks), where
`int(~176_filt × 2.5%) = 4` stocks per rebalance.

### Finding 2: The cost model caps positions at ~4-6
With REBAL_WEEKS=4 (13 rebalances/yr) and inv_vol sizing, each position generates
a resize trade at every rebalance: N × 13 trades/yr from resizing alone.
- N=2: 26 resize + ~10 entry/exit = 36 trades → feasible
- N=4: 52 + 20 = 72 trades → feasible
- N=10: 130 + 50 = 180 trades → OVER LIMIT
The practical ceiling is **avg_positions ≈ 6-8** before the 150 trade limit binds.

### Finding 3: More positions = worse Sharpe (this architecture is concentration-dependent)
Experiments tried in Session 6:
- TOP_PCT 5% of filt (n_top=4): avg_pos=3.4, Sharpe=-0.114, 78 trades — fails
- TOP_PCT 5% of universe (n_top=23): avg_pos=18.9, trades=523, Sharpe=-1.983 — fails
- Position persistence buffer (n_enter=10, n_exit=15): avg_pos=9.4, trades=229 — fails
- dvol blend sizing 50/50: avg_pos=1.6, Sharpe=0.68, MaxDD=-29.6% — worse than champion
- R1000 full universe + all S5-041 improvements: avg_pos=3.1, Sharpe=-0.066 — fails

**The S5-041 champion's alpha comes from extreme concentration in the #1-2 momentum stocks.**
Adding more positions dilutes quality AND increases costs. Both effects hurt Sharpe.

### Finding 4: SP500 filter is non-negotiable
R1000 full universe fails regardless of improvements applied:
- S5-014 (from early S5 baseline): 0.204
- S6 (from S5-041 improvements): -0.066
The R1000 mid-caps introduce low-quality momentum names that drag performance.

---

## Revised Session 6 Goals

| Goal | S5-041 baseline | Revised S6 target |
|------|----------------|-------------------|
| avg_positions | **1.6** (S5-041 actual; was 3.9 from bad-revert code) | > 4 (achievable) |
| MaxDD | -27.17% avg | < 25% avg |
| mean WF Sharpe | 0.751 | > 0.6 |

**Key MaxDD offenders:**
- 2023: -39.3% (worst — AI bull with single concentrated position)
- 2018: -33.2% (bear market, single stock)
- 2017: -25.5%

## S5-041 Per-Window Results (baseline for all S6 experiments)
```
2017: Sharpe=+0.050  MaxDD=-25.5%  Ret=-4.5%
2018: Sharpe=+0.598  MaxDD=-33.2%  Ret=+17.8%
2019: Sharpe=+0.954  MaxDD=-18.3%  Ret=+24.3%
2020: Sharpe=+2.408  MaxDD=-26.2%  Ret=+265.9%
2021: Sharpe=+0.469  MaxDD=-22.7%  Ret=+11.5%
2022: Sharpe=+0.524  MaxDD=-24.9%  Ret=+14.7%
2023: Sharpe=+0.252  MaxDD=-39.3%  Ret=+1.6%
MEAN: 0.751  avg_MaxDD=-27.17%
```

---

## Session 6 Research Directions

### Theme 1: MaxDD reduction (primary target)
The 2023 MaxDD of -39.3% is driven by single-stock concentration (n_top=1 in
bear/greed periods = 100% in one volatile name, adaptive 30% stop).

**S6-001: min(1,…) → min(2,…) — force at least 2 positions always**
```python
n_top = min(max(2, int(len(filt) * eff_pct)), 15)
```
Bear: n_top = max(2, int(90*0.01)) = max(2,0) = 2 stocks (was 1)
Normal: n_top = max(2, 2) = 2 stocks (unchanged)
Expected: avg_positions rises from 1.6 to ~2.5, MaxDD may improve in
concentrated years (2018, 2023). Mild trade increase from ~33 to ~45.

**S6-002: Adaptive stop 30% → 25% in uptrend**
Reduces the maximum per-position loss before stop triggers.
Hypothesis: the 2023 -39.3% MaxDD came from adaptive 30% stop in NVDA-type run.
Risk: may cut winning positions prematurely.

**S6-003: VIX entry gate — skip new entries when VIX > 25**
Note: VIX SIZE scaling is normalised away by the backtest engine. But a VIX
ENTRY GATE (skip rebalance when VIX > 25) does have real effect: positions
build up more slowly in volatile markets, reducing exposure to spike/crash sequences.

### Theme 2: Position count improvement (secondary)
**S6-004: Loosen dvol filter q70→q60 (top 30%→top 40%)**
len(filt) grows from ~90 to ~120. n_top at 2.5% = int(120*0.025) = 3.
avg_positions may rise from 1.6 to ~2.5-3.0.
Risk: includes some lower-quality names.

**S6-005: TOP_PCT 2.5%→3.5% (from larger filt after S6-004)**
If S6-004 improves filt to ~120 stocks: int(120*0.035) = 4 stocks.
avg_positions target ~3.5-5.0.

**S6-006: dvol blend sizing 30/70 (mild blend after understanding quality dynamics)**
Retest with smaller dvol weight. 50/50 gave Sharpe 0.68 (vs 0.751).
Maybe 20/80 or 30/70 finds a middle ground.

### Theme 3: Alpha improvement
**S6-007: Position persistence — only exit if rank > N+buffer**
Reduces churn but requires careful counting. Not yet tested successfully.

### Known failures (Session 6)
- TOP_PCT 5% of filt: Sharpe=-0.114, avg_pos=3.4 — dilutes alpha with bad picks
- TOP_PCT 5% of universe (n_top=23): Sharpe=-1.983, trades=523 — cost destruction
- Position persistence buffer (n_enter=10): trades=229, Sharpe=-3.366
- dvol blend 50/50: Sharpe=0.68, MaxDD=-29.6% — worse than champion
- R1000 full universe + S5-041: Sharpe=-0.066 — mid-caps fail
- Removing eff_pct dynamic: catastrophic (2022 bear = -0.688 without bear concentration)

---

## Experiment Log (Session 6)

| # | Hypothesis | WF Mean Sharpe | WF Sharpe Range | avg_pos | Trades/yr | Kept |
|---|-----------|---------------|-----------------|---------|-----------|------|
| S6-000 | [failed trial: dvol blend 50/50] | 0.68 | -0.176→+2.47 | 1.6 | 33.1 | No |
| S6-001 | min 2 positions always | — | — | — | — | baseline |
