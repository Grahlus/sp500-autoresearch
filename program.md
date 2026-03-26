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
