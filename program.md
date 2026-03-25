# SP500 Autoresearch — Research Program

## Objective
Beat the RSL baseline on the **validation set** (last 2 years, held-out).
Long AND short positions allowed. Full universe ~500 stocks, daily bars.
Metric: agent chooses per experiment (Sharpe, Calmar, alpha_ann, or composite).

## Execution Model
- Signal observed at **close[T]**
- Position entered at **open[T+1]**  ← realistic fill price
- Position exited / rebalanced at **open[T+2]**
- No lookahead: `generate_signals()` must only use data up to row T

## Hard Constraints
- No lookahead bias (enforced above)
- Max gross exposure: 1.0 (normalised by prepare.py)
- Transaction costs: 5 bps per turnover unit (enforced by prepare.py)
- Time budget: 60 minutes per experiment (hard timeout in prepare.py)
- **Do NOT edit prepare.py or run.py**

## Data Available (via `data` dict in generate_signals)
| Key | Type | Description |
|-----|------|-------------|
| `data["close"]` | DataFrame | Daily close, all SP500 tickers |
| `data["open"]` | DataFrame | Daily open prices |
| `data["high"]` | DataFrame | Daily high prices |
| `data["low"]` | DataFrame | Daily low prices |
| `data["volume"]` | DataFrame | Daily volume |
| `data["vix"]` | Series | VIX index aligned to trading days |
| `data["fear_greed"]` | Series | Fear & Greed index 0–100 |
| `data["train_end"]` | Timestamp | Train/validation cutoff |

---

## What We Already Know (DO NOT Repeat These)

### 10-Year Backtest Results (~400 stocks, $100k capital, 2015–2026)

| Config | Return | Sharpe | MaxDD | Trades | Final |
|--------|--------|--------|-------|--------|-------|
| **skip=4 + top3% + inv-vol + MA + stop20%** | **+4,455%** | **1.37** | **-33.7%** | 343 | **$4.6M** ← CURRENT BASELINE |
| skip=4 + MA filter | +3,780% | 1.35 | -32.5% | 332 | $3.9M |
| skip=4 + stop20% | +3,016% | 1.26 | -27.7% | 330 | $3.1M |
| skip=4 + inv-vol + top3% (no MA/stop) | +3,052% | 1.27 | -32.1% | 315 | $3.2M |
| skip=4 baseline | +2,471% | 1.21 | -28.8% | 271 | $2.6M |
| skip=0 original | +1,202% | 1.00 | -30.1% | 270 | $1.3M |
| top5% (wider) | +971% | 0.91 | -40.3% | 226 | $1.1M |

### Key Findings (already confirmed — no need to re-test)
- **skip=4 (Jegadeesh-Titman)**: essential — avoids short-term mean reversion. 2x baseline.
- **top3% is better than top5% or top10%**: concentration is the edge.
- **Inverse-vol sizing**: +250% over equal-weight for same selection.
- **20-week MA filter**: unexpected winner — adds +$700k with minimal DD change.
- **stop-loss 20%**: sweet spot — cuts MaxDD from -32% to -27.7% at minimal PnL cost.
- **stop-loss 10%**: too tight, hurts both sides. Don't test below 15%.
- **7 or 10 holdings**: worse than top3% — dilutes the edge.
- **Hard regime filter (SPY 200 MA)**: zero effect over 10 years, skip it.
- **Sector diversification cap**: reduces returns, not worth it.

---

## Research Frontier — What to Try Next

These have NOT been tested. Prioritise in order:

### 1. Short Side (highest priority)
Add a symmetric short leg to the RSL baseline:
- Short bottom 3% by momentum (worst ranked stocks below MA)
- Short stocks that triggered stop-loss (momentum broke down)
- VIX-gated shorts only: only short when VIX > 20 (avoids shorting bull markets)
- Test: does adding shorts improve Sharpe and reduce MaxDD?

### 2. VIX Regime Filter
- VIX < 15: full long momentum (current baseline)
- VIX 15–25: reduce position size to 50%, no new longs
- VIX > 25: go defensive — cash or short-only mode
- Test separately from the short-side experiment

### 3. Fear & Greed Index Filter
- Extreme Fear (F&G < 20): boost position size or buy dips
- Extreme Greed (F&G > 80): reduce longs, add shorts
- Combine with VIX for dual-filter regime

### 4. Momentum + Reversal Hybrid
- Long top 3% momentum (JT skip=4) as before
- Short last week's biggest losers (-5%+ in 5 days) for mean reversion
- Different holding periods: momentum monthly, reversal weekly

### 5. Volume Confirmation Filter
- Only enter momentum longs where volume is > 1.5x 20-day avg
- Volume spike on breakout = stronger signal

### 6. 52-Week High Proximity
- Weight stocks by proximity to 52-week high (George & Hwang factor)
- Combine with JT momentum for double confirmation

### 7. ML Signal Layer (60-min budget allows this)
- Train LightGBM on rolling 60-day feature window (first 9 years)
- Features: momentum at multiple lookbacks, vol, MA distance, VIX, F&G
- Predict: sign of next 4-week return
- Use ML score as a weight modifier on top of RSL rankings

### 8. Turtle Trading Short Side
- 20-day breakdown → short entry
- 10-day recovery → short exit
- ATR-based position sizing
- Combine with RSL longs for a long/short book

---

## Mandatory Logging Rule

After **every** `uv run python run.py`, immediately run this — before writing any new code:

```bash
./log_exp.sh NNN "hypothesis" "sharpe=X.XX calmar=X.XX alpha=X.XX maxdd=-XX.X% ret=+XXX%" Yes|No
```

- `NNN` = experiment number (e.g. 002)
- metrics = copy from the `>>> SCORE` and results block printed by run.py
- `Yes` = committed (new best), `No` = reverted

This is mandatory. Do not skip it.

---

## Experiment Log

| # | Hypothesis | Sharpe | Calmar | Alpha | MaxDD | Return | Kept |
|---|-----------|--------|--------|-------|-------|--------|------|
| 001 | RSL benchmark: JT skip=4 + top3% + inv-vol + MA20 + stop20% | — | — | — | — | — | pending |
| 001 | RSL benchmark: JT skip=4 + top3% + inv-vol + MA20 + stop20% | 1.309 | 1.397 | 0.347 | -36.17% | +126.6% | No |
| 002 | RSL long top3% + VIX-gated short bottom3% below MA VIX>20, inv-vol both sides, stop20% | 1.151 | 1.104 | 0.279 | -37.2% | +98.96% | No |
| 003 | RSL + VIX regime scaling VIX<20=100% VIX20-30=60% VIX>30=20% (engine normalizes so no effect) | 1.309 | 1.397 | 0.347 | -36.17% | +126.6% | No |
| 004 | RSL + explicit cash VIX>35 with recovery rebalance trigger at VIX<30 | 1.109 | 1.012 | 0.257 | -37.86% | +91.3% | No |
| 005 | RSL + volume confirmation: only hold stocks in top 50% universe by 20-day avg volume | 1.402 | 1.514 | 0.483 | -44.57% | +180.5% | Yes |
| 006 | RSL + vol filter top50% + 52wkH proximity combo rank 50/50 | 1.167 | 1.632 | 0.272 | -24.9% | +97.8% | No |
| 007 | RSL + vol filter top50% + vol acceleration composite: JT_rank × vol_accel_rank | 1.706 | 2.795 | 0.411 | -22.51% | +165.4% | Yes |
| 000 | RSL + vol top50% + vol accel 10d/20d + RSI<70 filter (RSI kills momentum) | 0.645 | 0.785 | 0.081 | -21.55% | +36.7% | No |
| 009 | RSL + vol filter top50% + vol accel composite: top 2% (more concentrated) | 1.770 | 3.38 | 0.502 | -22.52% | +210.2% | Yes |
| 010 | RSL + vol top50% + vol accel composite: top 1.5% ultra-concentrated (worse than 2%) | 1.412 | 1.982 | 0.453 | -32.39% | +169.6% | No |
| 011 | RSL + vol accel + F&G weight (no effect: scaling doesnt change rankings) | 1.770 | 3.38 | 0.502 | -22.52% | +210.2% | No |
| 012 | RSL + vol accel top2% + above median mom filter (over-filters good stocks) | 1.168 | 1.49 | 0.386 | -34.69% | +130.1% | No |
| 013 | RSL + vol accel composite top2%: 3d window + 2-week rebalance (txn costs hurt) | 0.635 | 0.546 | 0.095 | -31.97% | +38.0% | No |
| 014 | RSL + dollar-vol filter top50% + dollar-vol accel composite top2% (raw vol better) | 1.577 | 3.066 | 0.534 | -25.28% | +215.0% | No |
| 015 | RSL + vol accel as hard filter >1.0 + JT rank top2% (binary filter hurts) | 0.747 | 0.576 | 0.305 | -53.25% | +70.7% | No |
| 016 | triple composite JT x vol_accel x 4wk_mom (too restrictive) | 1.052 | 1.98 | 0.289 | -20.39% | +97.0% | No |
| 017 | RSL + vol accel composite top2%: 10w MA filter (20w is better) | 1.665 | 3.468 | 0.589 | -24.99% | +248.4% | No |
| 018 | RSL + vol accel composite top2%: 15% stop-loss (tighter) | 1.790 | 3.399 | 0.508 | -22.7% | +213.9% | Yes |
| 019 | RSL + vol accel composite: 12.5% stop-loss (too tight vs 15%) | 1.724 | 2.789 | 0.501 | -27.04% | +207.8% | No |
| 020 | RSL + vol accel composite 15%stop: top 2.5% concentration | 1.809 | 3.466 | 0.471 | -20.83% | +196.6% | Yes |
| 021 | RSL + vol accel top2.5% 15%stop: 52-week lookback (too long, 26w is better) | 1.324 | 1.664 | 0.255 | -24.35% | +97.5% | No |
| 022 | RSL + vol top40% + vol accel composite (top 50% is better) | 1.702 | 2.802 | 0.482 | -25.86% | +197.4% | No |
| 023 | RSL + vol top60% + vol accel composite (top 50% is optimal) | 1.327 | 1.806 | 0.251 | -22.25% | +96.5% | No |
| 024 | RSL + vol accel composite 15%stop top2.5%: JT skip 3 weeks (vs 4) | 1.844 | 3.991 | 0.493 | -18.94% | +208.4% | Yes |
| 025 | vol accel top2.5% 15%stop: skip 2 weeks (too short, skip 3 is optimal) | 1.325 | 2.06 | 0.317 | -23.0% | +117.2% | No |
| 026 | multi-horizon JT avg(13wk+26wk) + vol accel (26wk alone is better) | 1.629 | 3.157 | 0.421 | -20.11% | +167.3% | No |
| 027 | vol accel 15%stop skip3 top2.5%: 7d window (5d is better) | 1.657 | 3.217 | 0.457 | -21.31% | +184.1% | No |
| 028 | RSL + vol accel skip3 top2.5%: trailing stop 15% from position HIGH | 1.853 | 4.176 | 0.516 | -18.94% | +220.8% | Yes |
| 029 | trailing stop 12% from high (too tight vs 15%) | 1.262 | 1.815 | 0.254 | -21.98% | +95.7% | No |
| 030 | trailing stop 15% from high + 5-week rebalance (too infrequent, 4w is better) | 1.149 | 1.566 | 0.292 | -27.24% | +103.5% | No |
| 031 | trailing stop 15%high + 3-week rebalance (4w is optimal) | 1.328 | 2.075 | 0.315 | -22.78% | +116.9% | No |
| 032 | trailing stop 15%high + 30d vol baseline (20d is better) | 1.842 | 3.903 | 0.514 | -20.18% | +219.6% | No |
| 033 | size by composite score (inv-vol is better) | 1.710 | 3.616 | 0.492 | -20.44% | +202.5% | No |
| 034 | vol accel skip3 top2.5%: 17% trailing stop from high (15% is better) | 1.826 | 3.87 | 0.506 | -19.98% | +214.3% | No |
| 035 | intraday filter close>open at rebalance (too restrictive) | 0.721 | 0.664 | 0.127 | -31.76% | +46.6% | No |
| 036 | require absolute positive momentum (1.805 vs 1.853, slight improvement in calmar) | 1.805 | 4.389 | 0.553 | -19.1% | +237.97% | No |
| 037 | raw product ranking JT×vol_accel (rank×rank is better for recent period) | 1.461 | 2.175 | 0.521 | -33.78% | +200.9% | No |
| 038 | sticky exit top5% threshold (doesn't improve vs regular exit) | 1.825 | 4.045 | 0.501 | -18.94% | +211.96% | No |
| 039 | F&G regime: top1.5% when F&G>75, skip new entries when F&G<25 | 2.176 | 5.146 | 0.622 | -19.29% | +297.1% | Yes |
| 040 | F&G regime: fear<30 greed>70 (25/75 is better) | 1.697 | 3.312 | 0.419 | -19.29% | +168.7% | No |
| 041 | F&G fear<20 greed>80 (25/75 is optimal) | 2.093 | 4.82 | 0.589 | -19.29% | +272.5% | No |
| 042 | F&G fear-only skip entries (2.044) - both components needed for 2.176 | 2.044 | 4.75 | 0.574 | -18.94% | +261.0% | No |
| 043 | F&G exit all + skip entries fear<25 (skip-entries-only is better) | 2.030 | 3.957 | 0.358 | -14.64% | +149.5% | No |
| 044 | F&G with recovery rebalance trigger (disrupts signal) | 1.537 | 2.375 | 0.346 | -22.33% | +134.2% | No |
| 045 | F&G top1.0% greed (1.5% is better) | 2.020 | 4.669 | 0.577 | -19.33% | +262.0% | No |
| 046 | F&G: 2.0% normal + 1.5% greed>75 (2.5% normal is better) | 1.746 | 3.843 | 0.497 | -19.53% | +206.4% | No |
| 047 | RSL + vol top50% + vol accel skip3 trailing 15%high: combined fear: skip entries when F&G<25 OR VIX>30; greed top1.5% when F&G>75 | 2.176 | 5.146 | 0.622 | -19.3% | +297.1% | No |
| 048 | F&G 3-tier: exit all F&G<15, skip entries F&G<25, top1.5% F&G>75 | 1.968 | 4.527 | 0.440 | -15.3% | +186.2% | No |
| 049 | RSL + vol top50% + vol accel skip3 trailing 15%high + F&G: require absolute positive momentum filter | 1.864 | 4.530 | 0.578 | -19.5% | +255.2% | No |
| 050 | triple composite JT_rank × vol_accel_rank × 52wkHigh_proximity_rank + F&G regime | 1.466 | 2.798 | 0.345 | -18.7% | +131.9% | No |
| 051 | RSL + vol top50% + vol accel + F&G regime + VIX-gated short leg: short bottom 1.5% JT below MA when VIX>20 | 1.860 | 3.919 | 0.491 | -19.3% | +208.4% | No |
| 052 | trailing stop uses daily HIGH for watermark (vs close); exit on close<high_wm*(1-15%) | 1.432 | 2.370 | 0.331 | -21.2% | +125.5% | No |
| 053 | F&G regime: top1.5% when F&G>70 greed, skip entries F&G<25 fear — lower greed threshold | 2.248 | 5.622 | 0.671 | -19.3% | +334.6% | Yes |
