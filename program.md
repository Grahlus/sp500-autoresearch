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
| S5-001 | SP500 champion baseline — 7-window WF | 0.357 | -1.13→+1.97 (2017=+0.04,2018=+0.56,2019=+0.75,2020=+1.97,2021=-1.13,2022=+0.28,2023=+0.02) | — | — | Yes (baseline) |
| S5-002 | valuation proxy filter — exclude stocks up >100% in past 26w | 0.155 | -0.82→+1.15 (4/7 neg) | 1/7 | 33.6 | No — catastrophic, 100% threshold too restrictive |
| S5-003 | trailing stop tighten to 15% when breadth>70% | 0.307 | -1.30→+1.92 (2/7 neg) | 3/7 | 33.0 | No — extra stops in normal dips hurt 2023; 2021 worse |
| S5-004 | max position age 12w (60d) force exit | 0.453 | -0.73→+1.97 (2/7 neg) | 3/7 | 31.7 | No — 2021=-0.73 still violates min_sharpe<-0.5; promising direction |
| S5-005 | max position age 8w (40d) force exit | 0.446 | -0.90→+2.11 (2/7 neg) | 3/7 | 31.7 | No — re-entry churn worse than 12w; 2021=-0.90 |
| S5-006 | 26w absolute momentum hold gate | 0.357 | -1.13→+1.97 (baseline) | 3/7 | 31.7 | No — zero effect; 26w base is COVID recovery low, never fires in 2021 |
| S5-007 | 13w absolute momentum hold gate | 0.136 | -1.49→+1.66 (3/7 neg) | 2/7 | 33.5 | No — fires on normal dips, kills 2017/2018 |
| S5-008 | MA_WEEKS 20→10 (faster trend filter) | -0.055 | -1.32→+1.47 (4/7 neg) | 2/7 | 29.4 | No — catastrophic; faster MA creates false exits everywhere |
| S5-009 | skip rebal when breadth>80% | 0.364 | -0.84→+1.46 (1/7 neg) | 3/7 | 22.3 | No — 2021=+0.40 but 2017=-0.84; 80% threshold blocks 2017's low-vol bull |
| S5-010 | skip rebal when breadth>85% | **0.547** | +0.016→+1.998 (0 neg) | 3/7 | 27.4 | **YES** — 2021 fixed (+0.016), 0 negative windows, mean 0.357→0.547 |
| S5-011 | dual momentum composite 26w+13w rank avg | 0.115 | -0.67→+1.08 (2/7 neg) | 1/7 | 29.2 | No — 13w signal too noisy; kills 2017/2018/2023 |
| S5-012 | VIX floor VIX<13 AND breadth>80% reduce to 1% | 0.407 | -1.08→+2.24 (1/7 neg) | 3/7 | 25.1 | No — VIX<13 fires all of 2017, over-concentrates to 4 stocks |
| S5-013 | STOP_LOSS_PCT 20%→25% | 0.493 | -0.009→+1.78 (0 neg) | 3/7 | 26.7 | No — 2023 improved but 2018/2020/2022 hurt; net worse (0.493<0.547) |
| S5-014 | remove SP500 filter, full 841-stock universe | 0.204 | -0.97→+1.79 (3/7 neg) | 2/7 | 67.1 | No — R1000 mid-caps too noisy; 2017/2018 catastrophic |
| S5-015 | REBAL_WEEKS 4→6 | 0.106 | -1.20→+1.63 (3/7 neg) | 2/7 | 22.2 | No — 6w rebal too slow; 2021/2022 negative again |
| S5-016 | LOOKBACK_WEEKS 26→13 | 0.069 | -0.84→+1.05 (3/7 neg) | 1/7 | 30.2 | No — 13w signal too noisy; 2018/2019 fail |
| S5-017 | adaptive stop 30% in uptrend (>10%/4w) else 20% | **0.568** | +0.016→+1.99 (0 neg) | 3/7 | 27.3 | **YES** — 2023 +0.174 (was +0.022), mean 0.547→0.568 |
| S5-018 | adaptive stop threshold 10%→5% | **0.581** | +0.016→+1.99 (0 neg) | 3/7 | 27.3 | **YES** — 2017 +0.136 (was +0.041), mean 0.568→0.581 |
| S5-019 | adaptive stop threshold 5%→0% | 0.581 | identical to S5-018 | 3/7 | 27.3 | No — zero effect; 0% threshold indistinguishable from 5% in practice |
| S5-020 | LOOKBACK_WEEKS 26→20 | 0.266 | -0.745→+2.352 (3/7 neg) | 3/7 | 28.6 | No — catastrophic; 2017=-0.454 2019=-0.745 2021=-0.335 |
| S5-021 | SKIP_WEEKS 3→2 | 0.476 | -0.282→+1.872 (1/7 neg) | 3/7 | 27.3 | No — 2019=+0.026 (was +0.750), 2021=-0.282 (was +0.016); net worse |
| S5-022 | FG_MIN 22→15 | **0.611** | +0.016→+2.020 (0 neg) | 3/7 | 28.8 | **YES** — 2018=+0.723 (+0.161), 2022=+0.455 (+0.174), mean 0.581→0.611 |
| S5-023 | greed top_pct threshold fg>70→fg>80 | **0.621** | +0.116→+2.046 (0 neg) | 3/7 | 31.7 | **YES** — 2021=+0.612 (was +0.016), 2018=-0.492 (was +0.723), net +0.010 |
| S5-024 | INV_VOL_DAYS 6→15 | **0.705** | +0.050→+2.170 (0 neg) | 3/7 | 31.7 | **YES** — 2018=+0.598, 2019=+0.792, 2021=+0.720; 2017 slight drop (+0.050); mean 0.621→0.705 |
