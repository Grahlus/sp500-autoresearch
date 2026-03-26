# SP500 Autoresearch — Research Program (Session 4)

## Engine Status — FIXED (since session 3)
- Trades only on weight changes (no daily drift rebalancing)
- Trailing stop uses close-price HIGH only (no intraday)
- $100k capital | $20/trade commission | 5bps slippage

## Universe
Russell 1000 (iShares IWB) + SPDR sector ETFs + commodity ETFs = ~841 tickers
Key additions vs SP500: GLD, GDX, URA, XME, XLE, XLF, XLRE, XLU etc.

## Hard Constraints (enforced by run.py)
1. trades_per_year > 150 → REVERT
2. train sharpe < 0.5 → REVERT
3. train_sharpe < val_sharpe / 1.6 → REVERT (overfit)
4. ONE change per experiment

## The Problem We Are Solving

Momentum strategy OOS results by period:
| Period | Regime | Momentum Sharpe |
|--------|--------|----------------|
| Train 2014-2022 | Mixed cycles | 1.066 |
| Val 2022-2024 | Bear bottom + recovery | 1.643 |
| OOS 2024-2026 | AI exhaustion + tariff shock | -0.41 |

**Root cause**: Cross-sectional momentum requires return dispersion.
When dispersion collapses (all stocks move together = crowding), momentum
picks yesterday's leaders who become tomorrow's mean-reversion candidates.

2024-2026 was: mega-cap AI crowding → rotation → tariff shock → defensive rotation.
In this environment mean reversion works better than momentum.

**Solution**: Regime classifier that switches between momentum and mean reversion.

---

## Architecture: Dual-Mode Strategy

```
Each rebalance day:
  1. Measure regime (see below)
  2. IF momentum regime → run RSL signal (existing champion)
  3. IF mean-reversion regime → run MR signal (new)
  4. Mix or switch cleanly between the two
```

### Regime Detection — measure ONE of these (agent picks best):

**A. Cross-sectional dispersion** (recommended first attempt)
- Compute std of 13-week returns across full universe each rebalance
- Rolling 52-week median of that std = "normal dispersion"
- If current dispersion > median → MOMENTUM regime
- If current dispersion < median → MEAN REVERSION regime
- Logic: when stocks diverge (high std) momentum works; when they converge, MR works

**B. Serial autocorrelation**
- Compute 4-week autocorrelation of the equal-weight universe return
- Positive autocorr → trending → MOMENTUM
- Negative autocorr → mean-reverting → MR

**C. VIX term structure**
- VIX < 20 and falling → low fear, trending → MOMENTUM
- VIX > 25 or rising → fear/uncertainty → MR
- Simpler but less precise

**D. Market breadth trend**
- % stocks above 50d MA rising over 4 weeks → MOMENTUM
- % stocks above 50d MA falling → MR

### Mean Reversion Signal (when in MR regime):

**Weekly loser reversal** (simplest, most robust):
- Universe: stocks down 8-20% in past 15 trading days
- Filter: above 200d MA (not in structural downtrend)
- Filter: dollar-volume top 50% (liquid only)
- Rank by magnitude of drop (biggest losers = strongest MR signal)
- Long top 2-3% of universe by MR rank
- Hold for 1 rebalance period (4 weeks)
- No trailing stop in MR mode (exit is the next rebalance)

**Why this works**: Institutional forced selling, earnings overreaction, 
sector rotation selling creates temporary dislocation. Mean reverts within 2-6 weeks.

### Momentum Signal (when in MOM regime):
Same as R1000 champion (session 3):
- 26w+16w sum composite, no vol_accel, dollar-vol top 30%
- STOP=13% trailing from close HIGH
- inv-vol 12d sizing

---

## Session 4 Experiment Plan

### Phase 1: Baseline regime detection (exp S4-001 to ~S4-010)
Goal: find which regime detector correctly classifies 2014-2026 history

S4-001: Cross-sectional dispersion detector only — print regime each rebalance,
        run MOMENTUM signal in all regimes (verify detector is classifying correctly)
        Check: is 2022-2024 mostly MOM? is 2024-2026 mostly MR?

S4-002: Same detector + pure MR signal in MR regime
        First test of the hybrid system

S4-003 onwards: tune regime threshold, MR signal parameters, transition logic

### Phase 2: Tune MR signal parameters
- Lookback for "loser" definition (10d, 15d, 20d)
- Drop threshold (5%, 8%, 12%)
- Universe filter (dollar-vol percentile)
- Concentration (top 1%, 2%, 3%)

### Phase 3: Blend vs hard switch
- Hard switch: 100% MOM or 100% MR based on regime
- Soft blend: weight by regime confidence score
- Gradual transition: 4-week moving average of regime signal

---

## Confirmed Optimal Parameters (do not re-test)

### Momentum signal (R1000, session 3):
| Parameter | Optimal |
|-----------|---------|
| Lookback | 26w primary + 16w secondary (sum) |
| Skip weeks | 3w |
| Rebalance | 4w |
| Concentration | 2.5% |
| Stop-loss | 13% from close HIGH |
| Inv-vol window | 12d |
| Dollar-vol filter | top 30% |
| Vol_accel | removed (noise on R1000) |
| F&G gate | removed (fear = entry signal on R1000) |
| MA filter | 20w |

### Known failures (do not retry):
- Vol_accel on R1000 — share-vol noise pollutes signal
- F&G entry gate — fear periods ARE recovery entries on R1000
- Intraday HIGH for trailing stop — lookahead bias
- Daily rebalancing — commission drag fatal at $20/trade
- Stop < 13% — too tight for mid-cap volatility

---

## Mandatory Logging Rule

After every `uv run python run.py`, immediately run:
```bash
./log_exp.sh NNN "hypothesis" \
  "sharpe=X.XX calmar=X.XX alpha=X.XX maxdd=-XX.X% ret=+XXX% final=$XXXk cost=$XXX trades=XXX" \
  "sharpe=X.XX calmar=X.XX alpha=X.XX maxdd=-XX.X% ret=+XXX% final=$XXXk cost=$XXX trades=XXX" \
  Yes|No
```
3rd arg = VALIDATION | 4th arg = TRAIN | mandatory before any new code.

---

## Experiment Log (Session 4)

| # | Hypothesis | VAL Sharpe | VAL Calmar | VAL MaxDD | VAL Ret | TRAIN Sharpe | TRAIN Calmar | TRAIN MaxDD | Kept |
|---|-----------|-----------|-----------|---------|--------|-------------|-------------|-----------|------|
| S4-001 | Cross-sectional dispersion detector — verify regime classification | — | — | — | — | — | — | — | pending |
| S4-002 | hybrid MOM/MR — dispersion classifier switches signals | 0.046 | -0.147 | -51.1% | -14.4% | 0.638 | 0.371 | -55.2% | No |
| S4-003 | tune MR drop threshold to 10% min (was 8%) | 0.238 | 0.019 | -42.0% | +1.6% | 0.589 | 0.284 | -62.6% | Yes |
| S4-004 | tune MR drop threshold to 12% min (was 10%) | 0.437 | 0.236 | -42.0% | +20.8% | 0.674 | 0.482 | -45.3% | Yes |
| S4-005 | tune MR concentration to top 1% (was 2.5%) | 0.437 | 0.236 | -42.0% | +20.8% | 0.674 | 0.482 | -45.3% | Yes |
| S4-006 | tune MR concentration to top 3% (was 1%) | 0.437 | 0.236 | -42.0% | +20.8% | 0.674 | 0.482 | -45.3% | Yes |
| S4-007 | dispersion threshold to 40th percentile (was median/50th) | 0.544 | 0.345 | -42.9% | +31.6% | 0.736 | 0.560 | -45.3% | Yes |
| S4-008 | dispersion threshold to 60th percentile (was 40th) | 0.005 | -0.163 | -51.4% | -16.1% | 0.761 | 0.585 | -45.3% | No |
| S4-009 | autocorrelation regime detector (replace dispersion with autocorr) | 0.507 | 0.469 | -26.9% | +26.7% | 0.441 | 0.171 | -59.2% | No |
| S4-010 | blend MOM+MR weights 70/30 instead of hard switch | 0.567 | 0.445 | -35.9% | +34.3% | 0.948 | 0.828 | -41.8% | Yes |
| S4-011 | tune MR lookback from 15 days to 10 days | 0.754 | 0.785 | -32.1% | +56.5% | 0.943 | 0.786 | -45.1% | Yes |
| S4-012 | tune MR lookback from 10 days to 20 days | 1.397 | 1.910 | -34.7% | +175.2% | 1.026 | 0.874 | -44.4% | Yes |
| S4-013 | upper bound filter for MR candidates: ret >= -0.15 (was -0.20) | 1.330 | 2.274 | -22.8% | +129.9% | 0.976 | 0.712 | -48.5% | No |
| S4-014 | remove 200d MA filter from MR signal (all oversold stocks) | 0.999 | 1.234 | -27.3% | +78.2% | 0.926 | 0.767 | -43.5% | No |
| S4-015 | 30th percentile dispersion threshold (was 40th) | 1.357 | 1.827 | -34.7% | +165.8% | 1.066 | 0.912 | -44.4% | No |
| S4-016 | 50/50 blend instead of 70/30 | 1.391 | 1.922 | -35.9% | +184.3% | 1.058 | 0.933 | -42.96% | No |
| S4-017 | 60/40 blend (MOM: 60% MOM+40% MR; MR: 40% MOM+60% MR) | 1.405 | 1.925 | -35.3% | +180.9% | 1.048 | 0.907 | -43.7% | Yes |
| S4-018 | MR lookback 20d → 25d | 0.987 | 1.050 | -33.8% | +83.1% | 0.907 | 0.701 | -44.6% | No |
| S4-019 | MR lookback 20d → 15d | 0.592 | 0.491 | -34.8% | +36.9% | 0.955 | 0.833 | -41.2% | No |
| S4-020 | sector ETF regime: XLK vs XLV relative strength | 1.325 | 1.747 | -31.7% | +140.7% | 0.844 | 0.651 | -44.2% | No |
| S4-021 | widen MR drop window upper bound -0.20 → -0.30 | 1.553 | 2.146 | -33.8% | +196.4% | 1.051 | 0.966 | -42.1% | Yes |
| S4-022 | extend dispersion window 252 → 504 days | 1.529 | 2.097 | -33.8% | +190.8% | 0.923 | 0.721 | -46.4% | No (overfit) |
| S4-023 | shrink dispersion window 252 → 126 days | 1.383 | 1.809 | -33.8% | +158.7% | 1.095 | 1.015 | -42.1% | No |
| S4-024 | 35th percentile dispersion threshold (was 40th) | 1.532 | 2.101 | -33.8% | +191.3% | 1.049 | 0.961 | -42.1% | No |
| S4-025 | inv-vol sizing in MR mode (instead of equal weight) | 1.553 | 2.146 | -33.8% | +196.4% | 1.051 | 0.966 | -42.1% | No (tied) |
| S4-026 | fine-tune MR lower bound from -0.12 to -0.10 | 1.125 | 1.121 | -41.3% | +113.5% | 1.072 | 0.986 | -42.1% | No |
| S4-027 | 45th percentile dispersion threshold (was 40th) | 1.553 | 2.146 | -33.8% | +196.4% | 1.049 | 0.964 | -42.1% | No (tied) |
| S4-028 | blend 55/45 MOM/MR (was 60/40) | 1.575 | 2.309 | -32.5% | +205.0% | 1.068 | 0.984 | -42.0% | Yes |
| S4-029 | blend 50/50 MOM/MR (was 55/45) | 1.592 | 2.400 | -32.5% | +215.1% | 1.077 | 0.993 | -42.3% | Yes |
| S4-030 | blend 45/55 MOM/MR — MR dominant both regimes | 1.603 | 2.485 | -32.5% | +224.8% | 1.084 | 1.000 | -42.6% | Yes |
| S4-031 | blend 40/60 MOM/MR | 1.609 | 2.566 | -32.5% | +234.3% | 1.089 | 1.007 | -42.9% | Yes |
| S4-032 | blend 30/70 MOM/MR | 1.611 | 2.715 | -32.5% | +252.2% | 1.095 | 1.016 | -43.5% | Yes |
| S4-033 | blend 20/80 MOM/MR | 1.602 | 2.850 | -32.5% | +268.7% | 1.096 | 1.019 | -44.2% | No |
| S4-034 | asymmetric: MOM regime 60/40, MR regime 0/100 pure MR | 1.401 | 1.759 | -40.6% | +192.6% | 1.001 | 0.663 | -61.9% | No |
| S4-035 | MR regime blend 70/30 mr/mom (was 55/45) | 1.612 | 2.544 | -35.3% | +258.1% | 1.074 | 0.998 | -44.0% | Yes |
| S4-036 | MR regime blend 80/20 mr/mom (was 70/30) | 1.605 | 2.443 | -37.1% | +261.3% | 1.055 | 0.922 | -47.3% | No |
| S4-037 | MR drop window upper bound -35% (was -30%) | 1.845 | 3.271 | -35.3% | +360.9% | 1.119 | 1.034 | -46.8% | No (OVERFIT) |
| S4-038 | MR drop window upper bound -32% | 1.779 | 3.061 | -35.3% | +329.8% | 1.131 | 1.106 | -44.0% | Yes |
| S4-039 | MR drop window upper bound -33% | 1.833 | 3.229 | -35.3% | +354.6% | 1.131 | 1.106 | -44.0% | No (OVERFIT) |
| S4-040 | MOM regime blend 25/75 mom/mr (was 30/70) | 1.782 | 3.165 | -35.3% | +345.0% | 1.133 | 1.090 | -45.1% | Yes |
| S4-041 | MOM regime blend 20/80 mom/mr (was 25/75) | 1.783 | 3.266 | -35.3% | +360.1% | 1.133 | 1.074 | -46.1% | Yes |
| S4-042 | MOM regime pure MR 0/100 — both regimes use MR only | 1.524 | 2.685 | -35.3% | +276.9% | 0.948 | 0.732 | -45.9% | No (OVERFIT) |
| S4-043 | MR concentration top 4% (was 3%) | 1.783 | 3.266 | -35.3% | +360.1% | 1.133 | 1.074 | -46.1% | No (tied) |
| S4-044 | dispersion threshold 38th percentile (was 40th) | 1.783 | 3.266 | -35.3% | +360.1% | 1.133 | 1.074 | -46.1% | No (tied) |
| S4-045 | remove 200d MA filter from MR — allow downtrend stocks | 1.280 | 2.306 | -23.4% | +136.5% | 0.839 | 0.524 | -63.8% | No |
| S4-046 | 100d MA filter in MR signal (was 200d) | 1.870 | 4.336 | -30.1% | +428.3% | 1.048 | 0.935 | -44.0% | No (OVERFIT) |
| S4-047 | 150d MA filter in MR signal (was 200d) | 1.975 | 3.715 | -35.3% | +430.1% | 0.766 | 0.589 | -44.3% | No (OVERFIT) |
| S4-048 | MR liquidity filter top 30% dollar-vol (was 50%) | 1.709 | 3.392 | -32.6% | +340.1% | 0.931 | 0.850 | -41.6% | No (OVERFIT) |
| S4-049 | 3-week rebalance (was 4-week) | 1.270 | 1.396 | -39.1% | +138.1% | 0.891 | 0.743 | -45.3% | No |
| S4-050 | MR liquidity top 60% dollar-vol (loosen from 50%) | 1.947 | 3.598 | -35.3% | +411.3% | 1.148 | 1.066 | -47.9% | No (OVERFIT) |
| S4-051 | stop loss 10% (was 13%) | 1.809 | 3.366 | -35.3% | +375.3% | 0.911 | 0.745 | -47.0% | No (OVERFIT) |
| S4-052 | add MOM anti-filter to MR — exclude top-10% momentum | 1.661 | 2.373 | -38.0% | +259.8% | 0.717 | 0.422 | -53.5% | No (OVERFIT) |
| S4-053 | require 5-day bounce for MR candidates (ret_5d > 0) | 1.052 | 1.688 | -26.4% | +108.1% | 0.928 | 0.733 | -47.3% | No |
| S4-054 | dispersion lookback 8w/40d (was 13w/65d) | 1.694 | 2.927 | -35.3% | +310.5% | 1.155 | 1.287 | -40.0% | No |
| S4-055 | dispersion lookback 10w/50d (was 13w/65d) | 1.684 | 2.919 | -35.3% | +309.4% | 1.170 | 1.369 | -37.7% | No |
| S4-056 | trailing stop tracking in MR regime (clear pos_high) | 1.733 | 3.068 | -34.9% | +325.8% | 1.075 | 0.843 | -52.4% | No (OVERFIT) |
| S4-057 | dispersion window 300d (was 252d) | 1.792 | 3.292 | -35.3% | +364.0% | 1.113 | 0.930 | -51.3% | No (OVERFIT) |
| S4-058 | MR regime blend 65/35 mr/mom (fine-tune 70/30) | 1.784 | 3.332 | -34.3% | +356.9% | 1.138 | 1.107 | -44.5% | Yes |
| S4-059 | MR regime blend 60/40 mr/mom | 1.782 | 3.401 | -33.4% | +353.5% | 1.143 | 1.143 | -42.9% | No |
| S4-060 | MOM regime blend 15/85 mom/mr (was 20/80) | 1.782 | 3.432 | -34.3% | +371.7% | 1.138 | 1.089 | -45.6% | No |
| S4-061 | rank MR candidates by 20d vol (not drop magnitude) | 1.012 | 1.230 | -39.8% | +121.2% | 1.013 | 0.819 | -51.4% | No |
| S4-062 | require positive 26w return for MR candidates | 1.784 | 3.332 | -34.3% | +356.9% | 1.170 | 1.152 | -44.5% | Yes (train↑) |
| S4-063 | MR drop window -33% (retry with train=1.170) | 1.833 | 3.498 | -34.3% | +381.4% | 1.170 | 1.152 | -44.5% | Yes |
| S4-064 | MR drop window -34% | 1.833 | 3.498 | -34.3% | +381.4% | 1.154 | 1.077 | -47.4% | No (tied) |
| S4-065 | MR drop window -35% (retry with train=1.170) | 1.843 | 3.539 | -34.3% | +387.6% | 1.154 | 1.077 | -47.4% | Yes |
| S4-066 | MR drop window -36% | 1.843 | 3.539 | -34.3% | +387.6% | 1.154 | 1.077 | -47.4% | No (tied/-35% cap) |
| S4-067 | MR liquidity top 45% dollar-vol (loosen from 50%) | 1.965 | 3.809 | -34.3% | +429.1% | 1.054 | 0.947 | -47.4% | No (OVERFIT) |
| S4-068 | require positive 52w return for MR (stronger than 26w) | 1.784 | 2.899 | -39.8% | +361.0% | 1.015 | 0.877 | -47.8% | No (OVERFIT) |
| S4-069 | 13w uptrend requirement for MR (was 26w) | 2.256 | 6.347 | -25.4% | +575.4% | 0.985 | 0.803 | -47.8% | No (OVERFIT) |
| S4-070 | MR MA filter today > 1.05x 200d MA (not just above) | 1.939 | 4.574 | -28.5% | +426.7% | 0.971 | 0.877 | -43.8% | No (OVERFIT) |
| S4-071 | no regime detection — always 20/80 mom/mr blend | 1.820 | 3.338 | -35.5% | +373.9% | 1.081 | 0.858 | -53.9% | No (OVERFIT) |
| S4-072 | dispersion window 180d (was 252d) | 1.770 | 3.288 | -34.3% | +350.5% | 1.143 | 1.064 | -47.4% | No |
| S4-073 | composite MR score: drop × 26w uptrend strength | 1.353 | 2.107 | -37.7% | +220.3% | 1.167 | 1.199 | -43.8% | No |
| S4-074 | dispersion window 210d (was 252d) | 1.843 | 3.539 | -34.3% | +387.6% | 1.142 | 1.060 | -47.4% | No (OVERFIT) |
