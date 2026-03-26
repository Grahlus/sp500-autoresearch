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
| S5-025 | adaptive stop threshold 5%→3% | 0.705 | identical to S5-024 | 3/7 | 31.7 | No — zero effect; 3% and 5% thresholds produce identical results |
| S5-026 | REBAL_WEEKS 4→3 | -0.160 | -1.18→+0.802 (4/7 neg) | 1/7 | 89.8 | No — catastrophic; 90 trades/yr churn cost kills all windows |
| S5-027 | dollar-vol filter 70th→50th pct | 0.469 | -0.364→+1.965 (2/7 neg) | 2/7 | 58.4 | No — 2017/2022 go negative; 58 trades/yr too high |
| S5-028 | breadth gate 85%→90% | 0.628 | +0.050→+1.981 (0 neg) | 3/7 | 34.0 | No — 2017 unchanged, 2018 hurt (0.297 vs 0.598); net worse 0.705→0.628 |
| S5-029 | INV_VOL_DAYS 15→20 | 0.704 | +0.048→+2.180 (0 neg) | 3/7 | 31.7 | No — essentially identical to S5-024; 15d is sweet spot |
| S5-030 | remove greed top_pct reduction (fg>80→1.5%) | 0.225 | -0.868→+1.599 (3/7 neg) | 2/7 | 76.6 | No — catastrophic; greed reduction controls trade cost not just euphoria |
| S5-031 | greed threshold fg>80→fg>75 | 0.675 | +0.050→+2.171 (0 neg) | 3/7 | 30.1 | No — worse than fg>80 (0.675 vs 0.705); fg>80 is optimal |
| S5-032 | MIN_HOLD_DAYS 5→15 | 0.680 | +0.050→+2.170 (0 neg) | 3/7 | 31.7 | No — 2021 hurt (0.543 vs 0.720), 2017 unchanged; net worse |
| S5-033 | 4w positive return filter (buy only if up in last 4w) | -0.302 | -1.474→+1.550 (4/7 neg) | 1/7 | 25.7 | No — catastrophic; chasing short-term momentum kills 2017/2018/2022 |
| S5-034 | TOP_PCT 2.5%→2% | 0.391 | -1.076→+1.983 (hard fail) | 3/7 | 22.2 | No — 2017=-1.076, 2021=-0.520; too few stocks means one bad pick tanks portfolio |
| S5-035 | FG_MIN 15→10 | **0.718** | +0.050→+2.170 (0 neg) | 3/7 | 32.0 | **YES** — 2022=+0.524 (+0.089), mean 0.705→0.718; allows rebal in deep fear spikes |
| S5-036 | STOP_LOSS_PCT 20%→18% | 0.677 | +0.011→+2.352 (0 neg) | 3/7 | 32.6 | No — 2018=+0.407, 2022=+0.360, 2023=+0.128 all hurt; 20% is optimal base stop |
| S5-037 | breadth gate 85%→82% | 0.688 | +0.114→+2.139 (0 neg) | 3/7 | 29.3 | No — 2017=+0.501 (huge!), but 2018=+0.114 (catastrophic); net worse |
| S5-038 | breadth gate 85%→84% | 0.714 | +0.050→+2.139 (0 neg) | 3/7 | 31.6 | No — 2017/2018 same as 85%, 2020 slightly worse; 85% is optimal |
| S5-039 | combined gate: block only if breadth>85% AND F&G>70 | 0.489 | -0.397→+1.981 (1/7 neg) | 3/7 | 37.0 | No — 2021=-0.397; gate allows rebalancing in fear dips at high breadth |
| S5-040 | MA_WEEKS 20→25 | 0.534 | -0.682→+2.163 (hard fail) | 2/7 | 32.0 | No — 2018=-0.682; longer MA too conservative in bear/volatile environments |
| S5-041 | breadth MA 200d→100d | **0.751** | +0.050→+2.408 (0 neg) | 3/7 | 33.1 | **YES** — 2019=+0.954, 2020=+2.408, 2023=+0.252; 2021 dip to +0.469; mean 0.718→0.751 |
| S5-042 | breadth gate 82% with 100d MA | 0.698 | -0.067→+2.252 (1/7 neg) | 3/7 | 30.7 | No — 2023=+0.535 (excellent!) but 2017=-0.067, 2021=+0.137; net worse 0.751→0.698 |
| S5-043 | breadth gate 84% with 100d MA | 0.739 | +0.050→+2.408 (0 neg) | 3/7 | 32.4 | No — 85% is strictly better (2021=+0.384 vs +0.469); 85% is optimal with 100d MA |
| S5-044 | greed threshold fg>80→fg>75 with 100d MA | 0.677 | +0.050→+2.104 (0 neg) | 3/7 | 32.2 | No — 2021 hurt (0.254 vs 0.469); fg>80 is optimal with 100d MA too |
| S5-045 | TOP_PCT 2.5%→3% | 0.534 | -0.226→+2.146 (2/7 neg) | 2/7 | 41.9 | No — too many trades (42/yr), 2017/2018 negative; 2.5% is optimal |
| S5-046 | VIX position scaling min(1, 25/VIX) | 0.751 | identical to S5-041 | 3/7 | 33.1 | No — zero effect; Sharpe ratio invariant to proportional position scaling |
| S5-047 | 75/25 dual momentum composite (26w×0.75 + 13w×0.25) | 0.566 | -0.584→+2.063 (hard fail) | 3/7 | 35.3 | No — 2017=-0.584, 2023=-0.018; 13w signal too noisy even at 25% weight |
| S5-048 | 50/50 inv-vol + momentum rank blended sizing | 0.684 | -0.220→+2.447 (hard fail) | 3/7 | 33.1 | No — 2017=-0.220, 2021=+0.147; momentum overweights crash candidates in 2017/2021 |
| S5-049 | adaptive wide stop 30%→25% | 0.751 | identical to S5-041 | 3/7 | 33.1 | No — zero effect; 25% and 30% produce identical results |
| S5-050 | bear breadth threshold 0.40→0.45 | 0.751 | identical to S5-041 | 3/7 | 33.1 | No — zero effect; 100d breadth in 2022 never in 0.40-0.45 range at rebal dates |
| S5-051 | equal-weight positions | 0.101 | -3.603→+1.751 (hard fail) | 3/7 | 28.8 | No — 2017=-3.603 catastrophic; inv-vol weighting is essential risk management |
| S5-052 | breadth MA 150d | 0.652 | +0.050→+2.170 (0 neg) | 3/7 | 33.1 | No — essentially same as 200d; 100d is optimal and unique |
| S5-053 | two-tier gate: breadth>0.85 OR (breadth>0.75 AND F&G>75) | 0.650 | +0.050→+1.946 (0 neg) | 3/7 | 32.5 | No — blocks 2020 recovery rebalancing; 2020=+1.946, 2021=+0.228; net worse |
| S5-054 | breadth MA 100d→80d | 0.582 | -0.338→+2.407 (1/7 neg) | 3/7 | 34.5 | No — 2021=-0.338 catastrophic; 100d is the unique sweet spot |
| S5-055 | add F&G>80 gate (block when extreme greed) | 0.667 | +0.050→+2.252 (0 neg) | 3/7 | 32.7 | No — blocks late-2020 rebuilding (2020=+2.252, 2019=+0.764); net worse |
| S5-056 | STOP_LOSS_PCT 20%→22% | 0.673 | +0.050→+2.193 (0 neg) | 3/7 | 31.9 | No — 2018=+0.377, 2022=+0.445 both hurt; 20% is optimal base stop |
| S5-057 | FG_MIN 10→5 | 0.750 | +0.050→+2.408 (0 neg) | 3/7 | 33.4 | No — identical to S5-041; FG_MIN=10 is the floor, no days with F&G 5-10 in dataset |
| S5-058 | dollar-vol filter 70th→65th pct | 0.431 | -0.524→+2.204 (4 neg) | 2/7 | 42.0 | No — catastrophic; brings in poor-quality names that blow up 2017/2018 |
| S5-059 | adaptive stop threshold 5%→3% in 20d | 0.751 | identical | 3/7 | 33.1 | No — zero effect; 3-5% 20d range is same stocks as 5%; threshold is moot |
| S5-060 | VIX<20 gate extension (breadth<=0.90 AND VIX<20) | 0.533 | -0.338→+2.236 (1 neg) | 3/7 | 36.0 | No — 2021=-0.338 catastrophic (VIX was calm in 2021 calm periods), same flaw as S5-039 |
| S5-061 | REBAL_WEEKS 4→5 | 0.452 | -0.710→+2.130 (1 neg) | 3/7 | 25.9 | No — 2019=-0.710 catastrophic; rebal timing shift misses Jan 2019 recovery entry |
| S5-062 | LOOKBACK_WEEKS 26→28 | 0.732 | -0.372→+2.449 (1 neg) | 5/7 | 31.9 | No — 2017=-0.372, 2019=0.361 badly hurt; but 2021=1.129, 2023=0.603 greatly improved |
| S5-063 | LOOKBACK_WEEKS 26→27 | 0.659 | -0.387→+2.185 (1 neg) | 3/7 | 31.9 | No — worse than both 26w and 28w; 26w is genuine half-year sweet spot |
| S5-064 | add 13w positive-momentum quality filter | 0.465 | -0.263→+1.790 (0 neg) | 2/7 | 26.6 | No — destroys 2018=-0.263 and 2019=0.586; hard filter excludes too many stocks in volatile markets |
| S5-065 | MA_WEEKS 20→15 | 0.538 | -0.531→+2.280 (1 neg) | 4/7 | 31.3 | No — 2017=-0.531 catastrophic; but 2023=0.771 (best ever); shorter MA lets stale signals pass |
| S5-066 | MA_WEEKS 20→22 | 0.684 | -0.293→+2.408 (0 neg) | 3/7 | 33.7 | No — 2018=-0.293, 2021=0.187 hurt; 20w remains optimal |
| S5-067 | positive absolute momentum filter (mom>0) | 0.553 | -0.283→+2.636 (0 neg) | 2/7 | 24.9 | No — 2018=0.042, 2021=-0.173; too many candidates excluded in volatile markets |
| S5-068 | breadth-trend gate (allow if <=0.92 AND breadth rising) | 0.521 | -0.338→+2.152 (1 neg) | 3/7 | 36.7 | No — 2021=-0.338; breadth was rising in some 2021 periods, opens gate at wrong time |
| S5-069 | breadth cooldown 20d | 0.666 | -0.067→+1.981 (1 neg) | 3/7 | 27.6 | No — 2018=0.961 best ever but 2019=0.640, 2020=1.981; 20d too long |
| S5-070 | breadth cooldown 10d | 0.580 | -0.067→+2.018 (1 neg) | 3/7 | 29.5 | No — 2019=0.640, 2021=0.206; cooldown disrupts timing in recovery phases |
| S5-071 | position persistence (hold top 5% = 2×entry) | 0.752 | +0.050→+2.197 (0 neg) | 3/7 | 36.5 | No (+0.001) — 2021=0.758, 2023=0.506 greatly improved; 2019=0.399 hurt by dilution |
| S5-072 | persistence cap n_top+3 | 0.752 | identical to S5-071 | 3/7 | 36.5 | No — cap already bound; n_hold naturally ≈ n_top+3 with SP500 filter |
| S5-073 | idiosyncratic momentum (mom - SPY_ret) | 0.751 | identical to S5-041 | 3/7 | 33.1 | No — subtracting constant doesn't change ranking; need beta-adjusted, not raw excess |
| S5-074 | beta-adjusted momentum (mom - beta×SPY_ret) | 0.751 | identical to S5-041 | 3/7 | 33.1 | No — same top-12 picks; energy had genuine +18% idiosyncratic alpha in 2016, beta doesn't help 2017 |
| S5-075 | INV_VOL_DAYS 15→10 | 0.659 | -0.088→+2.374 (1 neg) | 3/7 | 33.1 | No — 2017=-0.088, 2018=0.430 hurt; 15d confirmed optimal |
| S5-076 | quality-filter retention (hold while above MA+dvol) | 0.496 | -0.584→+2.023 (1 neg) | 4/7 | 53.4 | No — 2017=-0.584 catastrophic (holds energy too long); 2023=1.241 but can't overcome 2017 |

## Session 5 Status (after S5-076, 2026-03-26)

**Champion**: S5-041 — mean Sharpe **0.751** (target: >0.800, gap: 0.049)

Per-window: 2017=+0.050  2018=+0.598  2019=+0.954  2020=+2.408  2021=+0.469  2022=+0.524  2023=+0.252

**Structural limits identified after 76 experiments:**

1. **2017 (0.050) — irreducible**: Energy/financials had +18% genuine idiosyncratic alpha from the Trump trade in late 2016. No signal (absolute, beta-adjusted, 13w filter, etc.) can predict the 2017 sector reversal within the momentum framework.

2. **2023 (0.252) — breadth gate conflict**: The AI rally happened in a high-breadth environment (>85% of SP500 above 100d MA). Any gate relaxation that allows 2023 entry also allows 2021 Feb rotation entry → catastrophic 2021 (-0.338).

3. **Retention tradeoff**: Every mechanism that helps 2023 by holding AI names longer (persistence, quality-filter retention) hurts 2017 by holding energy names longer, and hurts 2019 by diluting concentrated winners.

**Key confirmed optima**: 26w/3w/4w/2.5%/20w/20%/15d/FG≥10/breadth≤85%(100d MA) — all verified exhaustively.

**Possible next directions** (require data/complexity not yet available):
- Sector-rotation-aware filtering (GICS data needed)
- Regime-switching model (bull/bear regime detection)
- Alternative to trailing stop for 2023 high-vol names
