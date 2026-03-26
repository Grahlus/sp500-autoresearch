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
