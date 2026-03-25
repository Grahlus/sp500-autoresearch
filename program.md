# SP500 Autoresearch — Research Program (Session 3)

## Engine Status — FIXED
The backtest engine was corrected in session 3:
- **Bug fixed**: daily dollar-drift rebalancing removed — engine now only trades when weights change
- **Bug fixed**: trailing stop now uses close-price HIGH, not intraday HIGH (intraday not tradeable)
- All session 1/2 Sharpe numbers were inflated by phantom daily churn — do not rely on them
- The correct engine is in prepare.py + evaluate.py as of the session 3 baseline commit

## Execution Model
- Signal observed at **close[T]** → filled at **open[T+1]** ± slippage
- Trades ONLY executed when weights change (new entry, exit, or stop trigger)
- No daily drift rebalancing — positions are held until intentionally changed

## Cost Model (enforced by prepare.py)
- Starting capital: $100,000
- Commission: $20 per ticker traded (IB-style flat fee)
- Slippage: 5bps one-way on execution price

## Hard Constraints (enforced by run.py — violations auto-flagged)
1. **trades_per_year > 150** → REVERT. Use rebalance_days >= 20.
2. **train sharpe < 0.5** → REVERT. No edge on 9 years.
3. **train_sharpe < val_sharpe / 1.6** → REVERT. Overfit.
4. Change ONE thing per experiment. Isolate variables.

## Benchmark to Beat
**OOS result (2024-07 → 2026-03, 431 days, never touched during training):**
- Champion exp140: Sharpe 0.564 | Calmar 0.516 | +27.2% vs SPY +25.0%
- Thin edge in a tough regime (mega-cap crowding, low momentum dispersion)
- Val Sharpe 1.643 | Train Sharpe 1.066 | ~55 trades/yr | $2,186 costs

**The goal: beat OOS Sharpe 0.564 on a strategy that is robust across regimes.**

---

## What We Know (Confirmed with Fixed Engine)

### Champion stack (exp140, val window 2022-07 → 2024-06):
- RSL Jegadeesh-Titman: 26w lookback, skip 3w, rebalance 4w
- Triple composite signal: 26w_rank × vol_accel_rank × 13w_rank (product)
- Volume filter: top 50% by 20d avg volume
- Vol acceleration: 5d avg / 10d avg volume ratio
- Adaptive concentration: top 1% bear (breadth<40%), top 1.5% greed (F&G>70), top 2.5% normal
- F&G entry filter: skip new entries when F&G < 22
- Trailing stop: 20% from close-price HIGH since entry
- Position sizing: inv-vol (6d window)

### Confirmed optimal parameters (do not re-test):
| Parameter | Optimal | Confirmed bad |
|-----------|---------|---------------|
| Lookback | 26w | 24w/28w overfit |
| Skip weeks | 3w | 2w destroys, 4w fails train |
| Rebalance | 4w | 5w/6w overfit |
| Concentration | 2.5% | 2% worse, 3% fails train |
| Stop-loss | 20% from close HIGH | 15% too tight, 25% fails train |
| Inv-vol window | 6d | 5d overfit |
| Composite | 26w×vol_accel×13w product | sum overfits, duo worse |
| Vol filter | top 50% | 30% too restrictive |
| F&G entry | ≥22 | ≥21 fails overfit, ≥25 too restrictive |
| 2nd factor | 13w | 8w/10w much worse |

### Why OOS underperformed val:
The 2024-2025 regime had low momentum dispersion — mega-cap tech dominated,
everything moved together. The strategy thrives on dispersion. In low-dispersion
regimes, the top 2.5% is crowded and correlated = high vol, lower Sharpe.

---

## Universe Change — Session 3

**SP500 → Russell 1000** (via iShares IWB ETF holdings, ~1000 names)

Why: The strategy scored only +27% OOS (2024-2026) vs possible +100%+ because:
1. SP500 excludes mid-cap miners, uranium, royalty streamers, materials that ran 60-100%
2. The volume filter (top 50% by share count) systematically excluded resource stocks
   — NVDA trades 500M shares/day, a gold miner trades 5M. Dollar-volume fixes this.
3. 26-week lookback is too slow for sector rotation — by the time gold ranked top 2.5%
   the move was mostly done.

Russell 1000 adds: AEM, AG, CCJ, UEC, WPM, RGLD, FNV, TECK, CLF, KTOS, MP, LAC and
~500 more mid-caps that were the actual winners in 2024-2026.

**Key agent changes for the new universe:**
- Replace share-volume filter with **dollar-volume filter** (price × volume)
  so $5 miners and $500 megacaps compete on equal footing
- Let the agent tune the optimal lookback — 26w may be too slow for this broader universe
- The dispersion problem partially solves itself: more diverse universe = more dispersion

## Research Frontier — Session 3

The fixed engine + Russell 1000 universe is a clean new starting point.
Val baseline (S3-001) was on SP500 — first priority is re-establishing baseline on R1000.

### Priority 1: R1000 baseline — re-run champion config on new universe
Just swap the universe. Keep all other parameters identical to exp140.
Expected: similar or better Sharpe since universe is richer.
Dollar-volume filter is mandatory — implement before running anything else.

### Priority 2: Dispersion-aware regime filter (highest impact)
The OOS underperformance is explained by low cross-sectional momentum dispersion.
Measure it directly and reduce exposure when it's low:
- Compute cross-sectional std of 13w returns across the universe each rebalance
- When dispersion (std) is below median historical → reduce top_pct or cut position size 50%
- When dispersion is high → full position
- This directly addresses the 2024-25 regime failure

### Priority 2: Mean reversion overlay (different regime, different edge)
When momentum dispersion is low (bad for RSL), mean reversion tends to work better.
Build a hybrid: RSL in high-dispersion, mean-reversion in low-dispersion:
- Mean reversion signal: stocks down 5-15% in past 10 days, above 200d MA, high volume
- Weight: 100% RSL in high-dispersion, 50/50 RSL+MR in low-dispersion
- Different alpha source, potentially smoother equity curve across regimes

### Priority 3: Sector rotation overlay
The 2024-25 AI bull was sector-specific. A sector momentum layer might capture it:
- Rank GICS sectors by 13w momentum
- Overweight top 2 sectors by 25%, underweight bottom 2 by 25%
- Use yfinance sector ETFs (XLK, XLF, XLE etc.) as sector proxies
- Add as weight modifier on top of individual stock RSL

### Priority 4: Volatility-adjusted concentration
When VIX > 25 (high fear), the top 2.5% are high-beta stocks that get punished hardest.
In high-VIX environments, wider diversification (top 4-5%) actually helps:
- top_pct = 2.5% when VIX < 20
- top_pct = 4.0% when VIX 20-30
- top_pct = 5.0% when VIX > 30
Different from breadth-based bear filter — this is vol-level not price-level

### Priority 5: Earnings blackout filter
High-vol stop triggers often happen around earnings. If earnings date is within 5 days,
skip new entries for that ticker. Reduces the "catching a falling knife on earnings" problem.
Use yfinance calendar data.

### Do NOT re-test (exhausted in sessions 1-2):
- All lookback/skip/rebalance/concentration variations (fully grid-searched)
- VIX as a binary skip filter (redundant with breadth)
- Dual-window vol_accel, LightGBM overlays, 52-week high factor
- Short leg (multiple failures, kills train sharpe)
- F&G threshold variations (all fail overfit or too restrictive)

---

## Mandatory Logging Rule

After **every** `uv run python run.py`, immediately run this — before writing any new code:

```bash
./log_exp.sh NNN "hypothesis" \
  "sharpe=X.XX calmar=X.XX alpha=X.XX maxdd=-XX.X% ret=+XXX% final=$XXXk cost=$XXX trades=XXX" \
  "sharpe=X.XX calmar=X.XX alpha=X.XX maxdd=-XX.X% ret=+XXX% final=$XXXk cost=$XXX trades=XXX" \
  Yes|No
```
- 3rd arg = VALIDATION metrics, 4th arg = TRAIN metrics
- Overfitting check: revert if train_sharpe < val_sharpe / 1.6
- This is mandatory. Do not skip it.

---

## R1000 Champion Config (Session 3 — Final)

**Val: 1.271 | Train: 0.844 | Trades/yr: ~85 | MaxDD: -27.4% | +129.4% val return**

```
LOOKBACK_WEEKS = 26    # Primary momentum lookback
SKIP_WEEKS     = 3     # Reversal skip
REBAL_WEEKS    = 4     # 20-day rebalance
TOP_PCT        = 0.025 # 2.5% of eligible universe
MA_WEEKS       = 20    # 100-day trend filter
STOP_LOSS_PCT  = 0.13  # 13% trailing stop from close HIGH
VOL_MA_DAYS    = 10    # Dollar-vol averaging window
MIN_HOLD_DAYS  = 5     # Minimum hold before stop eligible

Signal: 26w_rank + 16w_rank (sum of two momentum ranks)
Filter: dollar-vol top 30% (levels playing field for $5 miners vs $500 megacaps)
Sizing: inv-vol (12d window)
Regime: breadth<40% → 1% concentration (rarely fires on R1000)
No F&G gate (removed — was blocking recovery-period entries)
```

### R1000-Confirmed Optimal Parameters (do not re-test):
| Parameter | R1000 Optimal | Confirmed bad on R1000 |
|-----------|---------------|------------------------|
| Lookback | 26w | 13w destroys, 20w worse, 28w overfit, 30w overfit |
| Skip weeks | 3w | 2w destroys, 4w worse |
| Rebalance | 4w | 3w worse + higher cost, 5w worse |
| Concentration | 2.5% | 1.5% massive overfit, 2.0% worse, 3.0% worse, 4.0% worse |
| Stop-loss | 13% from close HIGH | 11% hurts train, 12% worse, 14-15% worse, 20% baseline |
| Inv-vol window | 12d | 6d worse, 8d worse, 10d worse, 20d worse, 30d worse |
| Composite | 26w+16w sum | product slightly worse, 3-factor sum worse, vol_accel hurts |
| Dollar-vol filter | top 30% | 20%/25% too tight, 35%/40% too loose, 50% baseline |
| Secondary factor | 16w (80d) | 8w/13w/14w/15w/17w all worse |
| F&G gate | NONE (removed) | F&G≥22 was blocking recovery entries |
| Greed reduction | NONE (removed) | F&G>70→1.5% hurts in bull markets |
| MA filter | 20w (100d) | 10w/15w/30w/40w all worse |
| Vol filter basis | DOLLAR-vol | Share-vol systematically excludes resource stocks |

### Key R1000 Discoveries vs SP500 Champion:
1. **Dollar-vol filter essential**: Share-vol favored NVDA (500M shares) over gold miners (5M shares) — unfair. Dollar-vol levels playing field.
2. **Tighter stop (13% vs 20%)**: R1000 mid-caps mean-revert faster after momentum pushes.
3. **16w secondary (vs 13w)**: Mid-cap sector rotation (gold, uranium, defense) has different timing.
4. **Remove F&G gate**: On R1000, recoveries from extreme fear ARE signal — blocking them hurts.
5. **Sum composite**: Allows partial factor compensation — better generalization across regimes.
6. **12d inv-vol (vs 6d)**: Smoother sizing on more volatile R1000 names reduces position flip noise.
7. **Breadth filter dead code**: R1000's breadth never drops below 40% — filter never fires.

---

## Experiment Log (Session 3 — Fixed Engine)

| # | Hypothesis | VAL Sharpe | TRAIN Sharpe | Kept |
|---|-----------|-----------|-------------|------|
| S3-001 | Baseline: exp140 champion on fixed engine (SP500) | 1.643 | 1.066 | baseline |
| S3-002 | R1000 universe + dollar-vol filter baseline | -0.077 | — | baseline |
| S3-003 | MA_WEEKS=30 | -0.147 | 0.723 | No |
| S3-004 | MA_WEEKS=40 | -0.163 | 0.715 | No |
| S3-005 | Dollar-vol top 40% | 0.477 | 0.758 | **Yes** |
| S3-006 | Dollar-vol top 30% | 0.669 | 0.868 | **Yes** |
| S3-007 | Dollar-vol top 20% | -0.096 | 0.626 | No |
| S3-008 | TOP_PCT=1.5% | 0.009 | 0.885 | No |
| S3-009 | TOP_PCT=4.0% | 0.482 | 0.978 | No |
| S3-010 | LOOKBACK=13w | -0.589 | 0.852 | No |
| S3-011 | LOOKBACK=20w | 0.242 | 0.774 | No |
| S3-012 | MA_WEEKS=10 | 0.090 | 0.745 | No |
| S3-013 | SKIP_WEEKS=2 | 0.083 | 0.569 | No |
| S3-014 | REBAL_WEEKS=3 | 0.227 | 0.753 | No |
| S3-015 | Dollar-vol top 25% | -0.102 | 0.896 | No |
| S3-016 | Dispersion-aware filter (position scaling) | 0.669 | 0.868 | No (neutral) |
| S3-017 | STOP_LOSS=15% | 0.687 | 0.906 | **Yes** |
| S3-018 | STOP_LOSS=12% | 0.590 | 0.889 | No |
| S3-019 | Drop vol_accel (26w×13w only) | 0.909 | 0.844 | **Yes** |
| S3-020 | Inv-vol window 10d | 0.922 | 0.852 | **Yes** |
| S3-021 | Equal weighting | 0.746 | 0.833 | No |
| S3-022 | TOP_PCT=3.0% | 0.788 | 0.887 | No |
| S3-023 | Remove greed concentration reduction | 0.953 | 0.712 | **Yes** |
| S3-024 | Remove F&G entry filter | 1.037 | 0.696 | **Yes** |
| S3-025 | Breadth threshold 35% | 0.810 | 0.668 | No |
| S3-026 | LOOKBACK=30w | 1.069 | 0.665 | No (overfit) |
| S3-027 | VOL_MA_DAYS=20 | 0.889 | 0.722 | No |
| S3-028 | Sum composite (26w+13w) | 1.037 | 0.745 | **Yes** (better train) |
| S3-029 | 3-factor: 26w+13w+52w | -0.203 | 0.481 | No (train<0.5) |
| S3-030 | Secondary=8w | 0.648 | 0.884 | No |
| S3-031 | STOP_LOSS=13% | 1.125 | 0.784 | **Yes** |
| S3-032 | STOP_LOSS=14% | 1.102 | 0.750 | No |
| S3-033 | MIN_HOLD_DAYS=7 | 1.088 | 0.793 | No |
| S3-034 | Dollar-vol accel in composite | 0.323 | 0.872 | No |
| S3-035 | Breadth threshold 45% | 0.763 | 0.832 | No |
| S3-036 | SKIP_WEEKS=4 | 1.047 | 0.969 | No |
| S3-037 | Dollar-vol top 35% | 1.111 | 0.757 | No |
| S3-038 | REBAL_WEEKS=5 | 0.714 | 1.057 | No |
| S3-039 | LOOKBACK=24w | 1.123 | 0.886 | No |
| S3-040 | No MA filter (above_ma=True) | 0.829 | 0.823 | No (trades>150) |
| S3-041 | MA_WEEKS=15 | 0.557 | 0.867 | No |
| S3-042 | TOP_PCT=2.0% | 0.482 | 0.889 | No |
| S3-043 | 13w secondary no skip | 0.682 | 0.684 | No |
| S3-044 | Secondary=16w (80d) | 1.209 | 0.804 | **Yes** |
| S3-045 | Secondary=17w | 1.070 | 1.155 | No |
| S3-046 | Secondary=15w | 0.434 | 0.849 | No |
| S3-047 | SKIP=4 + 16w secondary | 1.089 | 0.873 | No |
| S3-048 | STOP=12% + 16w secondary | 1.086 | 0.771 | No |
| S3-049 | Positive momentum filter | 0.989 | 0.906 | No |
| S3-050 | Secondary=14w | 0.927 | 0.855 | No |
| S3-051 | Deep-bear force-exit at breadth<30% | 1.109 | 0.734 | No |
| S3-052 | Product composite + 16w | 1.198 | 0.852 | No |
| S3-053 | Inv-vol window 20d | 1.237 | 0.833 | **Yes** |
| S3-054 | Inv-vol window 30d | 1.211 | 0.858 | No |
| S3-055 | Inv-vol window 15d | 1.257 | 0.825 | **Yes** |
| S3-056 | Inv-vol window 12d | 1.271 | 0.844 | **Yes** |
| S3-057 | Inv-vol window 8d | 1.227 | 0.795 | No |
| S3-058 | Inv-vol window 10d retest | 1.209 | 0.804 | No |
| S3-059 | LOOKBACK=28w | 1.188 | 0.908 | No |
| S3-060 | Bear mode eff_pct=0% | 1.271 | 0.844 | No (identical) |
| S3-061 | Breadth threshold 55% | 0.790 | 0.826 | No |
| S3-062 | Max position cap 40% | 1.211 | 0.818 | No |
| S3-063 | 3-factor: 26w+16w+8w | 0.751 | 0.900 | No |
| S3-064 | STOP=11% | 0.790 | 0.614 | No |
| S3-065 | Secondary=17w + 12d inv-vol retest | 1.130 | 1.172 | No |
