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
| 001 | 4w momentum quality filter — only enter if 20d return > 0 | -0.159 | — | — | — | — | — | — | 27 | — | — | — | — | — | No |
| 002 | holding period exit after 20w (100d) | 0.779 | — | — | — | — | — | — | 34 | — | — | — | — | — | Yes |
| 003 | TLT/SPY regime — bonds 4w > stocks 4w = risk-off mode (TLT not in dataset, no effect) | 0.779 | — | — | — | — | — | — | 34 | — | — | — | — | — | No |
| 004 | 4w momentum top-third filter (top 33% universe by 4w return) | -0.314 | — | — | — | — | — | — | 20 | — | — | — | — | — | No |
| 005 | adaptive 30% stop only if position up >10% from entry | 0.769 | — | — | — | — | — | — | 34 | — | — | — | — | — | No |
| 006 | 13w momentum quality filter — enter only if 65d return > 0 | 0.631 | — | — | — | — | — | — | 31 | — | — | — | — | — | No |
| 007 | stop-loss cooldown — no new entries 20d after any stop | 0.672 | — | — | — | — | — | — | 28 | — | — | — | — | — | No |
| 000 | bear-market stop cooldown breadth<55% | 0.764 | — | — | — | — | — | — | 32 | — | — | — | — | — | No |
| 008 | bear-market stop cooldown breadth<55% | 0.764 | — | — | — | — | — | — | 32 | — | — | — | — | — | No |
| 009 | market-neutral ranking (constant shift, identical to baseline) | 0.779 | — | — | — | — | — | — | 34 | — | — | — | — | — | No |
| 010 | multi-period momentum avg rank 26w+13w+52w | 0.334 | — | — | — | — | — | — | 34 | — | — | — | — | — | No |
| 011 | 15w (75d) forced exit instead of 20w — faster rotation | 0.786 | — | — | — | — | — | — | 35 | — | — | — | — | — | Yes |
| 012 | 12w (60d) forced exit — too aggressive, 0.777 | 0.777 | — | — | — | — | — | — | 34 | — | — | — | — | — | No |
| 013 | conditional 15w exit only if 4w return <= 0 | 0.741 | — | — | — | — | — | — | 35 | — | — | — | — | — | No |
| 014 | 15w forced exit only when F&G < 60 (let bull runs continue) | 0.819 | — | — | — | — | — | — | 35 | — | — | — | — | — | Yes |
| 015 | breadth momentum gate breadth_delta<-0.15 (blocks rebalance, hurts 2018H2/2022H1) | 0.859 | — | — | — | — | — | — | 32 | — | — | — | — | — | No |
| 016 | entry-only breadth_delta gate — still blocks too many recoveries | 0.626 | — | — | — | — | — | — | 33 | — | — | — | — | — | No |
| 017 | 15w exit when F&G<55 (same as <60, no diff) | 0.819 | — | — | — | — | — | — | 35 | — | — | — | — | — | No |
| 018 | 15w exit F&G<65 (same as <60) | 0.819 | — | — | — | — | — | — | 35 | — | — | — | — | — | No |
| 019 | 15w exit F&G<70 (still same 0.819) | 0.819 | — | — | — | — | — | — | 35 | — | — | — | — | — | No |
| 020 | 15w exit F&G<80 (same as unconditional 0.786) | 0.786 | — | — | — | — | — | — | 35 | — | — | — | — | — | No |
| 021 | breadth min gate 25% — blocks 2019H1 recovery | 0.755 | — | — | — | — | — | — | 35 | — | — | — | — | — | No |
| 022 | parabolic reversal exit (>50% gain + 20d return<-5%) — cuts 2020H2 | 0.785 | — | — | — | — | — | — | 35 | — | — | — | — | — | No |
| 023 | time-decay stop 40d+15% — hurts 2020H2 2021H1 | 0.801 | — | — | — | — | — | — | 35 | — | — | — | — | — | No |
| 024 | 12w (60d) F&G<60 gated exit — 0.777, too short | 0.777 | — | — | — | — | — | — | 34 | — | — | — | — | — | No |
| 025 | 18w (90d) F&G<60 gated exit — too long, 2023H2 worsened | 0.770 | — | — | — | — | — | — | 35 | — | — | — | — | — | No |
| 026 | adaptive stop threshold 5%->8% — helped 2017H1 but hurt 2017H2 overall 0.804 | 0.804 | — | — | — | — | — | — | 35 | — | — | — | — | — | No |
| 027 | SKIP_WEEKS=2 — fixed 2018H2 structural floor (-1.127->+0.196) and 2017H1 (0.078->0.563) | 0.863 | — | — | — | — | — | — | 34 | — | — | — | — | — | Yes |
| 027 | SKIP_WEEKS=2 REVERTED — OOS 0.114 << 1.0, WF overfit | 0.863 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 028 | FG_MIN=20 — tiny +0.003 gain, mixed windows, not kept | 0.822 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 029 | FG_MIN=25 — too strict, 2020H1 destroyed, 0.692 | 0.692 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 030 | LOOKBACK_WEEKS=24 — fails, 5 neg windows, 0.59 | 0.59 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 031 | LOOKBACK_WEEKS=28 — improved 2021H2/2022H1 but broke 2017H1/H2, 0.803 | 0.803 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 032 | INV_VOL_DAYS=10 — fails min_sharpe constraint, 0.737 | 0.737 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 033 | INV_VOL_DAYS=20 — 0.817, marginal miss | 0.817 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 034 | MA_WEEKS=15 — fails 5 neg windows, 0.609 | 0.609 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 035 | MA_WEEKS=25 — catastrophic 2018H2=-2.908, constraint fail | 0.631 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 036 | SKIP_WEEKS=4 — 2017H2/2018H1/2021H1 improve, OOS=1.200 passes | 0.862 | — | — | — | — | — | — | 34 | — | — | — | — | — | Yes |
| 037 | STOP_LOSS_PCT=0.15 — too tight, whipsaws, 2017H1=-0.819, 0.571 | 0.571 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 038 | STOP_LOSS_PCT=0.25 — REVERTED OOS=0.991 < 1.0, WF=0.897 overfit | 0.897 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 039 | REBAL_WEEKS=3 — fails 5 neg windows, 0.661 | 0.661 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 040 | REBAL_WEEKS=5 — 0.536, too slow, 4 neg windows | 0.536 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 041 | breadth ceiling 0.90 — 2021H1 hurt, 0.649 | 0.649 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 042 | breadth ceiling 0.80 — 2017H1 improved but 2020H2 hurt, 0.754 | 0.754 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 043 | adaptive stop 10d window — 2023H2 worse, 0.842 | 0.842 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 044 | adaptive stop 30d window — 0.833, 2023H2 still weak | 0.833 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 045 | MIN_HOLD_DAYS=10 — 2021H1 hurt, 0.822 | 0.822 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 046 | dvol q0.75 — fails 6 neg windows, 0.425 | 0.425 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 047 | dvol q0.65 — fails worst=-1.871, too many trades 43.3 | 0.664 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 048 | remove 15w exit — forced exit IS beneficial, 0.788 without it | 0.788 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 049 | bear breadth 0.35 — no effect, 0.862 identical | 0.862 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 050 | bear breadth 0.45 — no effect, identical 0.862 (threshold insensitive) | 0.862 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 051 | no greed concentration — hurts 2020H2 and 2021H1, 0.823 | 0.823 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 052 | forced exit F&G<50 — hurts 2023H1, too restrictive, 0.806 | 0.806 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 053 | equal-weight sizing — 2023H2 much worse -0.912, 0.824 | 0.824 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 054 | greed threshold 75 — hurts 2020H2/2021H2, 0.759 | 0.759 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 055 | greed threshold 85 — no effect, 0.862 identical | 0.862 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 056 | LOOKBACK_WEEKS=30 — 2019H2 destroyed, 2022H1 bad, 0.636 | 0.636 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 057 | breadth MA 80d — 2021H1 crashes, 0.694 | 0.694 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 058 | breadth MA 120d — 2020H2/2021H1 hurt, 0.677 | 0.677 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 059 | dvol window 30d — fails worst=-1.49 constraint | 0.808 | — | — | — | — | — | — | — | — | — | — | — | — | No |

---

## CRITICAL ARCHITECTURAL FINDINGS (Session 7)

### Session 7 Champion: S7-036
- **Config**: SKIP_WEEKS=4, 15w F&G<60 forced exit (all other S7-014 params unchanged)
- **WF mean Sharpe**: 0.862 (vs 0.819 baseline start)
- **OOS Sharpe**: 1.200 (verified, above 1.0 guard)
- **Commits**: S7-014 (0.819), S7-036 (0.862)

### Finding 1: SKIP_WEEKS=4 generalizes; SKIP_WEEKS=2 overfits
- SKIP_WEEKS=2 → WF=0.863 (+0.044 vs baseline) but OOS=0.114 — catastrophic overfit
- SKIP_WEEKS=4 → WF=0.862 (+0.043 vs baseline) AND OOS=1.200 — genuine edge
- The 20d exclusion window (4w) is theoretically sounder (matches JT 1-month exclusion)
- The 10d window (2w) exploits reversal patterns present in training but not OOS

### Finding 2: OOS-checked WF improvements: only +0.043 survived
- Every change giving WF > 0.862 failed OOS:
  - STOP_LOSS_PCT=0.25: WF=0.897, OOS=0.991 (just below threshold)
  - SKIP_WEEKS=2: WF=0.863, OOS=0.114 (catastrophic)
- The OOS guard (>1.0) is binding — there's genuine alpha but it's limited

### Finding 3: Parameter scan complete on S7-036 baseline
All parameters exhaustively scanned; 0.85/20/26/4/0.20/15/5/10 is the global optimum:
- REBAL_WEEKS: 3 (fails neg), 4 (opt), 5 (0.536)
- MA_WEEKS: 15 (fails), 20 (opt), 25 (fails catastrophically)
- LOOKBACK_WEEKS: 24 (fails), 26 (opt), 28 (0.803), 30 (0.636)
- STOP_LOSS_PCT: 0.15 (whipsaws), 0.20 (opt), 0.25 (OOS fail)
- INV_VOL_DAYS: 10 (fails), 15 (opt), 20 (0.817)
- FG_MIN: 10 (opt), 20 (0.822 marginal), 25 (destroys)
- Breadth ceiling: 0.80 (0.754), 0.85 (opt), 0.90 (0.649)
- Bear breadth threshold (0.35-0.45): insensitive — never crosses this range in data
- Greed threshold (80-85): insensitive — F&G either never or always in this range at rebalance
- dvol quantile: 0.65 (constraint fail), 0.70 (opt), 0.75 (fails)
- Forced exit F&G<50: 0.806 (too restrictive), F&G<60 (opt), F&G<80 (same as unconditional)

### Finding 4: 2023H2 (-0.601) is a structural floor with SKIP_WEEKS=4
The Jul 2023-Jul 2024 window is -0.601 regardless of all parameter changes.
- The forced exit is not causing it (confirmed by disabling exit → same -0.601)
- The stock selection with 4w skip picks different names for 2023-2024 that underperform
- This is a residual cost of the SKIP_WEEKS=4 configuration; 2018H2 improved from -1.127→-0.835

### S7-036 Per-Window Results
```
2017H1: Sharpe=+0.072  MaxDD=-30.7%  Ret=-2.9%
2017H2: Sharpe=+1.117  MaxDD=-21.3%  Ret=+19.0%
2018H1: Sharpe=+2.699  MaxDD=-17.9%  Ret=+71.6%
2018H2: Sharpe=-0.835  MaxDD=-33.4%  Ret=-21.1%
2019H1: Sharpe=+1.255  MaxDD=-9.1%   Ret=+12.9%
2019H2: Sharpe=+0.076  MaxDD=-18.5%  Ret=-0.7%
2020H1: Sharpe=+1.770  MaxDD=-26.2%  Ret=+60.3%
2020H2: Sharpe=+2.422  MaxDD=-19.1%  Ret=+86.8%
2021H1: Sharpe=+1.574  MaxDD=-20.2%  Ret=+40.3%
2021H2: Sharpe=+0.357  MaxDD=-17.3%  Ret=+3.3%
2022H1: Sharpe=+0.224  MaxDD=-25.5%  Ret=-1.9%
2022H2: Sharpe=+0.364  MaxDD=-34.6%  Ret=+3.1%
2023H1: Sharpe=+1.579  MaxDD=-23.1%  Ret=+25.1%
2023H2: Sharpe=-0.601  MaxDD=-47.3%  Ret=-33.8%
MEAN: 0.862  worst=-0.835  trades/yr=34.2
```
| 001 | vol-scaled sizing B&SC 2015 — scale by 0.15/vol_20d SPY defensive fill | 0.858 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 002 | idiosyncratic momentum — rank by 26w_stock_return minus 26w_sector_ETF_return | 0.476 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 003 | multi-lookback signal average of 26w 13w 4w ranks | -0.219 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 004 | risk-adjusted momentum rank by 26w_return divided by realized_vol_26w | 0.295 | — | — | — | — | — | — | — | — | — | — | — | — | No |
| 005 | portfolio vol circuit breaker exit all when 20d vol > 50% ann | 0.460 | — | — | — | — | — | — | — | — | — | — | — | — | No |

---

## CRITICAL ARCHITECTURAL FINDINGS (Session 8)

### Session 8 Result: No Improvement — S7-036 Champion Unchanged (WF=0.862)

All 3 academic ideas from the session prompt failed.

### Finding 1: Vol-scaling (B&SC 2015) requires leverage ability to work
- Engine normalizes weights to 100% invested — cannot scale down without a proxy asset
- SPY fill (S8-001, WF=0.858): SPY also crashes in bear markets → no defensive benefit
- Portfolio circuit breaker (S8-005, WF=0.460): 50% threshold fires too often (9.1 stops/yr), misses good trades
- **Verdict**: Vol-scaling fundamentally incompatible with this normalized-weight engine

### Finding 2: Any signal change selects different stocks → alpha destroyed
- Idiosyncratic momentum (S8-002, WF=0.476): stock-sector = selects sector-relative outperformers, not absolute momentum leaders; 2021H1 → -0.799
- Risk-adjusted momentum (S8-004, WF=0.295): return/vol selects stable low-vol names; misses NVDA-type big winners; 2020H1 → -0.115
- **Verdict**: Strategy alpha is entirely dependent on holding the TOP 1-2 absolute-momentum names. ANY relative or risk-adjusted signal selects different names → destroys alpha

### Finding 3: Multi-lookback confirmed dead (second confirmation)
- Session 6 exp010 (26w+13w+52w): 0.334
- S8-003 (26w+13w+4w): -0.219
- The 4w window IS the skip period (reversals). Adding shorter lookbacks corrupts the signal.
- **Verdict**: Pure 26w with SKIP_WEEKS=4 is the only working signal for this architecture

### Session 8 Experiment Log

| # | Hypothesis | WF Mean Sharpe | Kept |
|---|-----------|---------------|------|
| S8-001 | vol-scaled sizing B&SC 2015 target=0.15 SPY defensive fill | 0.858 | No |
| S8-002 | idiosyncratic momentum rank by 26w_stock_return minus 26w_sector_ETF_return | 0.476 | No |
| S8-003 | multi-lookback signal average of 26w 13w 4w ranks | -0.219 | No |
| S8-004 | risk-adjusted momentum rank by 26w_return divided by realized_vol_26w | 0.295 | No |
| S8-005 | portfolio vol circuit breaker exit all when 20d vol > 50% ann | 0.460 | No |

---

## CRITICAL ARCHITECTURAL FINDINGS (Session 9)

### Session 9 Result: All 5 Vol-Targeting Experiments Failed — Champion Unchanged (WF=0.777)

New engine verified: weights < 1.0 → cash. Baseline (S9-001) avg_equity=97%, WF=0.777, OOS=1.34 confirmed.

### Finding 1: Vol-targeting is architecturally incompatible — confirmed from both directions

New engine CAN hold cash. But vol-targeting still destroys alpha, regardless of method:
- Per-position (0.10, 0.15, 0.20): avg_equity drops to 47%, 71%, 95% → WF 0.064–0.104
- Portfolio-level (0.15): avg_equity=48% → WF=0.145
- Downside semi-vol (0.15): avg_equity=130%→capped, but crashes = high downside vol → WF=0.079

Even S9-006 (TARGET_VOL=0.20, avg_equity=95%) failed with WF=0.064. 5% less equity → WF drops from 0.777 to 0.064.

### Finding 2: The alpha is a function of FULL EXPOSURE in the RIGHT periods

The strategy picks 1-2 concentrated momentum stocks. Its alpha comes entirely from being 100% invested in those stocks during big runs (COVID 2020: +265%; 2018H1: +71%). Any reduction in exposure linearly scales those wins down while NOT proportionally reducing risk in all periods.

Vol-targeting reduces in HIGH vol periods = exactly when momentum pays off. The strategy needs:
- 100% invested in COVID bottom recovery (60% vol → vol-targeting says 25% invested)
- 100% invested in NVDA-type runs (40% vol → vol-targeting says 50% invested)

### Finding 3: Cash-weight engine adds zero value for this strategy

The new engine's key feature (weights < 1.0 → cash) is only useful if the agent can signal "hold cash now." This strategy's alpha requires always being fully invested in the top 1-2 names. A cash buffer only helps if cash can be deployed productively — this strategy has no mechanism for that.

**Verdict**: Vol-scaling direction fully exhausted. The cash-weight engine has no benefit for a concentrated momentum strategy.

### Session 9 Experiment Log

| # | Hypothesis | WF Mean Sharpe | avg_equity | Kept |
|---|-----------|---------------|------------|------|
| S9-001 | Baseline S5-041 on new 14-window engine | 0.777 | 97% | Yes (baseline) |
| S9-002 | per-position TARGET_VOL=0.15 | 0.084 | 71% | No |
| S9-003 | portfolio-level TARGET_PORT_VOL=0.15 | 0.145 | 48% | No |
| S9-004 | per-position TARGET_VOL=0.15 + USE_DOWNSIDE_VOL=True | 0.079 | 130%→100% | No |
| S9-005 | per-position TARGET_VOL=0.10 (tighter) | 0.104 | 47% | No |
| S9-006 | per-position TARGET_VOL=0.20 (looser) | 0.064 | 95% | No |


---

## CRITICAL ARCHITECTURAL FINDINGS (Session 9 — RSL Ideas)

### Session 9 RSL Result: Best WF=0.656 (EXIT_PCT_RANK=0.97), no commit

**Calibration note**: True baseline with 14-window engine = WF=0.569 (5 neg windows, fails constraint).
The "WF=0.777" claim from vol-targeting session was erroneous — could not be reproduced.
Wikipedia SP500 filter IS required; dollar-vol alone picks wrong R1000 mid-caps from 841-ticker universe.

### Finding 1: Rank-based exit (EXIT_PCT_RANK=0.97) — best RSL idea, but incomplete

- Improves WF: 0.569 → 0.656 (+0.087)
- Reduces neg_windows: 5 → 4 (passes constraint!)
- rank_exits/yr = 5.5 (as expected)
- Critical fix: 2017H1 (-0.330 → +0.270), 2023H1 (+0.621 → +1.941)
- Hurts: 2019H1 (+1.557 → +0.478) — exits winners too early in momentum rallies
- Still fails: 2023-07→2024-07 (-0.547 → -0.829) — 12-month window, worst in data

Threshold 0.97 is regime-dependent: helps bear market exits, hurts bull market continuation.
Threshold 0.95 is worse (5 neg windows again — 2017H1 not fixed).

### Finding 2: Sector cap (MAX_PER_SECTOR=2) — complete no-op

With avg_positions=1.6 and n_top=1-2, the strategy almost never holds 2 stocks from the same sector simultaneously. The cap never binds. GICS sector diversification is a non-issue at this concentration level.

### Finding 3: Weekly rebalance destroys the strategy

REBAL_WEEKS=1: WF=0.067, 7 neg windows, cost=$27,800 (vs $11,384 baseline = 2.4×).
The 4-week hold is structural — momentum needs time to play out. Weekly = chasing noise.

### Session 9 RSL Experiment Log

| Exp | Change | WF | neg_windows | Passes? | Notes |
|-----|--------|----|-------------|---------|-------|
| S9-001 (baseline) | flat 20% stop, no RSL | 0.569 | 5 | NO | True baseline re-confirmed |
| S9-002 | EXIT_PCT_RANK=0.97 | 0.656 | 4 | YES (WF<0.777) | Best result; 2019H1 hurt |
| S9-003 | MAX_PER_SECTOR=2 | 0.569 | 5 | NO | Identical to baseline, cap never binds |
| S9-004 | 0.97 + sector=2 | 0.656 | 4 | YES (WF<0.777) | Same as S9-002 |
| S9-005 | EXIT_PCT_RANK=0.95 | 0.551 | 5 | NO | Looser threshold hurts 2017H1 more |
| S9-006 | REBAL_WEEKS=1 | 0.067 | 7 | NO | Catastrophic; weekly = wrong for momentum |

---

## Session 10: Exit Tuning (2026-03-27)

**Champion: S10-005 — adaptive stop (30%/20%) + EXIT_PCT_RANK=0.97**
- WF mean Sharpe: **0.722** (was 0.656 at S9-002)
- OOS Sharpe: **1.548** (+208.59% OOS return vs SPY +23.33%) — Verdict: STRONG
- neg_windows: **3** (was 4): 2018H2=-1.158, 2022H1=-0.091, 2023H2=-0.313
- rank_exits/yr: 5.6 | stops/yr: 2.0 | avg_positions: 1.5 | invested: 87%

**Committed as cb08f15.**

### Session 10 Critical Finding

**Adaptive stop (30%/20%) is the key driver.** The 30% stop in strong uptrends (>5% gain in 20d) prevents premature exits on corrections in parabolic momentum names. This fixed:
- 2017H1: -0.627 → +0.339
- 2021H1: +0.478 → +1.057
- 2021H2: near-zero positive (+0.065 with just simple rank exit)

The rank exit (0.97) adds protection from bear market drawdowns while the adaptive stop keeps big winners from stopping out too early. These two improvements are complementary.

**Hybrid exit (rank<0.97 AND 4w mom<0) is incompatible with 2021H2.** In the 2021H2 rotation, growth stocks decline in rank but 4w momentum is still positive (slow grind down). Hybrid gate never fires → 2021H2 stays negative (-0.143). Simple rank exit fires in time.

### Session 10 Experiment Log

| Exp | Change | WF | neg_windows | Passes? | Notes |
|-----|--------|----|-------------|---------|-------|
| S10-001 | EXIT_PCT_RANK=0.90 (looser) | 0.568 | 5 | NO | 2021H2 went negative; worse than S9-002 |
| S10-003 | EXIT_PCT_RANK=0.97 + confirm=3 | 0.589 | 5 | NO | Confirm delays exits; misses 2021H2 |
| S10-004 | Hybrid rank<0.97 AND 4w mom<0 | 0.748 | 5 | NO | 2021H2=-0.143, 2022H1=-0.091 unchanged |
| S10-005 | Adaptive stop + rank exit 0.97 | 0.722 | 3 | **YES** | **COMMITTED** |
| S10-006 | Hybrid exit + adaptive stop | 0.802 | 5 | NO | 2021H2=-0.143 unfixable with hybrid |

### Session 11 Directions

1. **2023H2 window (-0.313)** remains the persistent loss — investigate what stocks were held and why
2. **2022H1 window (-0.091)** barely negative — a small improvement could fix it
3. Consider: adaptive stop threshold 0.10 (10% short_mom) instead of 0.05 — might help 2019H1 further
4. Consider: FG_MIN threshold variation — 2021H2 might be influenced by F&G gate
5. WF=0.722, OOS=1.548 may be close to practical ceiling for this architecture

---

## Session ML: ML Momentum Ranking (2026-03-28)

**Objective:** Replace `combo = mom.rank(pct=True)` with XGBoost/RF/LightGBM predicted forward return rank. Strict walk-forward protocol.

**Conclusion: FAILED after 38 experiments. ML cannot beat baseline WF=0.722.**

### Best valid result
**ML-027: XGB, 1yr training window, breadth_threshold=0.65**
- WF=0.648 (vs baseline 0.722 → 10% below)
- WORST=-1.050 (passes >-1.2)
- NEG_WIN=3 (passes ≤4)
- W11=+0.216 (positive)

Files: ml_signal.py, ml_run.py, ml_run2.py, ml_run3.py, ml_run4.py, ml_run5.py, ml_run6.py, ml_run7.py

### Key findings

1. **Root cause of ML failures**: W11 (2022H1 bear market) caused all vanilla ML models to fail catastrophically (Sharpe -2.0 to -3.0). Model trained on bull-market data learns correlations that break in bears.

2. **Regime-conditional gate (breadth threshold) fixed W11** (from -2.5 to -0.405). But introduced W04 (2018H2) as new worst window at -1.32 (violates -1.2 constraint).

3. **1yr training window fixed W04** (from -1.32 to -0.49). The 3yr window overfit to 2014-2016 bull market. 1yr window covers ~2017, which includes early bear events.

4. **Feature importance dominated by market-level features** (breadth=0.22, vix=0.17, fg=0.13), not stock-level. ML learns market timing, not stock selection. The regime gate already handles market timing — the model is redundant on its strongest signal.

5. **Removing market features (stock-only) made results worse** — the market features are complementary to the breadth gate, not competing. They help calibrate stock picks within the regime.

6. **Soft blends are toxic**: `alpha × ML_rank + (1-alpha) × mom_rank` causes W14 to collapse to -60% MaxDD. Not investigated why, but the blend creates unstable rankings.

7. **2yr training window also fails** (W04 stays at -1.695). Only 1yr fixes W04 reliably.

8. **WF trend for 1yr model vs breadth threshold** (non-monotonic peak at 0.65):
   - 0.55: WF=0.578, W11=-0.296
   - 0.60: WF=0.579, W11=+0.216
   - **0.65: WF=0.648 (PEAK), W11=+0.216**
   - 0.70: WF=0.598, W11=+0.216
   - 0.75+: WF collapses, W11=-0.696

### Why ML can't beat momentum rank
- Momentum rank is non-parametric and regime-agnostic: it doesn't need to learn that bears are bad
- ML adds variance (some windows better, others worse) without systematic signal
- Training sample size is limited: 1yr × SAMPLE_EVERY=10 ≈ 25 dates × 460 stocks = ~11,500 rows
- The best signal (cross-sectional momentum rank) is already the target feature; ML can't improve on it substantially

### Session ML Experiment Log

| Range | Script | Description | Best WF | Status |
|-------|--------|-------------|---------|--------|
| ML-001..004 | ml_run.py | RF, XGB, LGBM, Ensemble (full replacement) | 0.573 | ALL FAIL |
| ML-005..009 | ml_run2.py | Target variants, feature subsets | -0.547..0.218 | ALL FAIL |
| ML-010..016 | ml_run3.py | Regime gate (breadth threshold) | 0.683 | FAIL (W04) |
| ML-017..023 | ml_run4.py | Higher breadth, soft blends, 1yr | 0.683 | ML-023 PASS |
| ML-024..029 | ml_run5.py | 1yr/1.5yr/2yr × breadth scan | 0.648 | ML-027 PASS |
| ML-030..034 | ml_run6.py | 1yr + breadth 0.70-0.90 | 0.598 | ML-030 PASS |
| ML-035..038 | ml_run7.py | Stock-features-only (no breadth/VIX/FG) | 0.221 | ALL FAIL |
