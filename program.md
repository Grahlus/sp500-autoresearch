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

## Experiment Log (Session 3 — Fixed Engine)

| # | Hypothesis | VAL Sharpe | VAL Calmar | VAL MaxDD | VAL Ret | TRAIN Sharpe | TRAIN Calmar | TRAIN MaxDD | Kept |
|---|-----------|-----------|-----------|---------|--------|-------------|-------------|-----------|------|
| S3-001 | Baseline: exp140 champion on fixed engine | 1.643 | 3.918 | -16.9% | +175.6% | 1.066 | 0.850 | -35.0% | baseline |
