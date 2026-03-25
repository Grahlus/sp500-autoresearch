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
