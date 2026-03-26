# SP500 Autoresearch — Research Program (Session 5)

## Why Walk-Forward

Every previous session improved val Sharpe but degraded OOS:
| Session | Strategy | Val Sharpe | OOS Sharpe |
|---------|----------|-----------|-----------|
| S1/S2 | SP500 momentum | 1.643 | 0.564 |
| S3 | R1000 momentum | 1.271 | -0.410 |
| S4 | Hybrid MOM/MR | 1.843 | -1.316 |

Single-window validation is broken. Walk-forward forces the strategy to prove
itself across 7 independent years including different regimes.

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
| OOS | — | 2024-07→now | evaluate.py only |

## Scoring
- Primary: **mean Sharpe across all 7 windows**
- A strategy must survive COVID (2020), the 2022 bear, AND the 2019/2021 bulls
- Target: mean Sharpe > 0.8, no more than 2 negative windows

## Hard Constraints (enforced by run.py)
1. trades_per_year > 150 → REVERT
2. mean_sharpe < 0.3 → REVERT
3. neg_windows > 2 → REVERT (fails in more than 2 of 7 windows)
4. min_sharpe < -0.5 → REVERT (catastrophic failure in any window)
5. ONE change per experiment

## Engine
- Trades only on weight changes (no drift rebalancing)
- Trailing stop uses close-price HIGH (not intraday)
- $100k capital per window (independent resets)
- $20/trade commission | 5bps slippage

## Starting Point
SP500 champion (best OOS result across all sessions):
  JT 26w skip=3w rebal=4w top=2.5% stop=20%close-HIGH MA=20w F&G≥22
  This is S5-001. Run it first to establish the walk-forward baseline.

## Research Directions (try in order)
Each change must help the MEAN across all 7 windows — not just one.

### 1. Establish baseline (S5-001)
Run the SP500 champion. Check per-window breakdown — which years fail?
The 2020 COVID window and 2022 bear window will likely be the weak spots.

### 2. Improve bear/crash robustness
- VIX-based position sizing: reduce size when VIX > 25 (not a skip, a scale)
- Wider stop in bear (breadth < 40%): STOP=30% instead of 20%
- Cash when VIX > 35: full exit, wait for VIX < 25 to re-enter

### 3. Improve bull capture
- Remove F&G gate (session 3 showed it hurt on R1000)
- Shorter skip (2w vs 3w) during low-VIX periods
- Wider top_pct (3%) in strong breadth environments

### 4. Universe
- Try R1000 + ETFs (841 tickers) vs SP500 (468) — which gives better WF mean?
- Dollar-volume filter (session 3 improvement for R1000)

### 5. Signal improvements
- Triple composite 26w×13w (session 2 discovery, +0.33 Sharpe on SP500)
- Vol acceleration (worked on SP500, not R1000 — re-test on WF)

### Known failures (do not retry)
- MR signal — regime-specific, doesn't generalise
- Regime classifier — overfits to val window
- Session 3/4 configs — all OOS negative

## Mandatory Logging Rule

After every `uv run python run.py`:
```bash
./log_exp.sh NNN "hypothesis" \
  "sharpe=X.XX calmar=X.XX maxdd=-XX.X% ret=+XXX% beats=X/7 trades=XXX" \
  "n/a — walk-forward has no separate train split" \
  Yes|No
```
- Use val metrics for the walk-forward aggregate results
- Log "n/a" for train (walk-forward doesn't have a separate train metric)
- Commit if MEAN Sharpe improved

## Experiment Log (Session 5)

| # | Hypothesis | WF Mean Sharpe | WF Sharpe Range | Beat SPY | Trades/yr | Kept |
|---|-----------|---------------|-----------------|----------|-----------|------|
| S5-001 | SP500 champion baseline — 7-window WF | — | — | — | — | pending |
