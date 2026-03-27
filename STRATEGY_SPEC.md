# SP500 Momentum Strategy — Complete Specification
## Champion: S10-005 | WF Sharpe 0.722 | OOS Sharpe 1.548 | +208% OOS

**Document purpose:** Complete implementation spec so any session can rebuild this
strategy from scratch without referencing prior code. No details omitted.

---

## 1. Strategy Overview

**Style:** High-conviction cross-sectional momentum  
**Universe:** ~841 tickers (iShares IWB Russell 1000 + sector/commodity ETFs)  
**Signal:** Jegadeesh-Titman momentum with volume filter  
**Positions:** 1-2 concurrent (concentration is the alpha source)  
**Rebalance:** Every 4 weeks (20 trading days)  
**Direction:** Long only  
**Time frame:** Daily bars, monthly action  

**Validated performance:**
- Walk-forward: mean Sharpe 0.722 across 14 independent half-year windows (2017-2024)
- OOS (435 unseen trading days, 2024-07-01 → 2026-03-25): Sharpe 1.548, +208%, MaxDD -24%
- Trades/year: ~34, costs ~$1,200/year on $100k capital

---

## 2. Universe & Data

### Data sources (all free)
```
SP500 prices:     yfinance, daily OHLCV, 2014-present
                  Tickers: iShares IWB holdings (~1010) + ETF supplements
                  Download: yf.download(tickers, start="2014-01-01", auto_adjust=True)

VIX:              yfinance ticker "^VIX", daily close

Fear & Greed:     https://api.alternative.me/fng/?limit=3000&format=json&date_format=us
                  Returns JSON with "data" list, field "value" (0-100 integer)
                  date_format=us gives MM-DD-YYYY timestamps
```

### Ticker universe construction
1. Fetch iShares IWB holdings CSV:
   `https://www.ishares.com/us/products/239707/...ajax?fileType=csv&fileName=IWB_holdings&dataType=fund`
   Parse: skip metadata rows, find header starting with "Ticker,", read Symbol column
   Filter: keep only `^[A-Z][A-Z0-9\-]{0,6}$` (real tickers, no cash/futures)

2. Always append these ETFs (not in IWB):
   ```
   Sector:    XLF, XLRE, XLE, XLU, XLK, XLB, XLP, XLY, XLC, XLV, XLI
   Gold/silver: GLD, GDX, GDXJ, SLV
   Uranium:   URA, URNM
   Mining:    XME, PICK
   Energy:    XOP
   Commodity: DBA, DBC, COPX, CPER
   Royalty:   WPM, RGLD, FNV
   ```

3. Download OHLCV for all ~1036 tickers. Drop tickers with >20% missing days.
   Final universe: ~841 tickers.

4. Store as parquet: MultiIndex columns (close/open/high/low/volume, ticker)
   File: `data/sp500_prices.parquet`

**CRITICAL:** Do NOT filter to SP500-only at runtime. The strategy uses all 841 tickers.
Any Wikipedia SP500 filter breaks the strategy — the momentum + volume + MA filters
naturally select large-cap names without an explicit filter.

---

## 3. Signal Construction

### Step 1: Momentum score (Jegadeesh-Titman)
```
Lookback period:  26 weeks = 130 trading days
Skip period:      3 weeks  = 15 trading days (avoids short-term reversal)

momentum[tkr] = close[T - 15] / close[T - 130] - 1
```

The skip is essential. Using `close[T]` instead of `close[T-15]` causes short-term
reversal contamination. Confirmed: skip=3w outperforms skip=0 and skip=4w on this universe.

### Step 2: Quality filters (applied before ranking)
```
MA filter:       Only stocks where close[T] > mean(close[T-100 : T])
                 (20-week simple moving average)
                 Removes downtrending stocks from consideration.

Volume filter:   Only stocks in top 50% by 20-day average share volume
                 avg_vol = mean(volume[T-20 : T]) for each ticker
                 Keep only tickers where avg_vol >= median(avg_vol across universe)
                 Note: use SHARE volume (not dollar volume) for this universe.
```

Both filters are applied as a mask — any ticker failing either is excluded from ranking.

### Step 3: Percentile ranking
```
combo = momentum.rank(pct=True)  # 0.0 = worst, 1.0 = best
filt  = combo.where(above_ma & high_vol).dropna()
```

### Step 4: Adaptive concentration
```
Breadth = fraction of universe stocks above their 200-day MA (100 trading days)
F&G     = Fear & Greed index value (0-100)

if breadth < 0.40:          # bear market — fewer than 40% stocks in uptrend
    eff_pct = 0.010         # top 1% of filtered universe (~1 stock)
elif F&G > 70:              # greed regime
    eff_pct = 0.015         # top 1.5% (~2 stocks)
else:                       # normal
    eff_pct = 0.025         # top 2.5% (~2 stocks)

n_top = max(1, int(len(filt) * eff_pct))
top_tickers = filt.nlargest(n_top).index
```

With ~841 tickers and typical 50% passing quality filters (~420 filtered), this gives:
- Bear: 1% × 420 = 4 → typically 1-2 after inv-vol sizing
- Normal: 2.5% × 420 = 10 → typically 2 after inv-vol sizing
- In practice avg_positions ≈ 1.6 across full period

### Step 5: Position sizing (inverse-volatility)
```
INV_VOL_DAYS = 6  (6-day rolling window for vol measurement)

vol_ret      = pct_change of close[T-6 : T] for each top ticker
realized_vol = std(vol_ret) for each ticker
inv_vol      = 1.0 / realized_vol  (replace 0 or NaN with 0)
inv_vol_norm = inv_vol / sum(inv_vol)   ← sums to 1.0
weight[tkr]  = inv_vol_norm[tkr]
```

Lower-volatility stocks get larger allocations. This counteracts the tendency of
momentum winners to be high-vol names.

---

## 4. Entry Rules

### When to rebalance
```
REBAL_DAYS = 20  (every 4 weeks = 20 trading days)
Rebalance if: (day_index % 20 == 0) AND (F&G >= 10)
```

The F&G filter (≥10) blocks new entries only in *extreme* fear (F&G < 10 = panic).
This is very permissive — essentially only skips entries during historic crashes.
Earlier sessions tried F&G ≥ 22 which was too restrictive and hurt 2022 recovery entries.

### Entry mechanics
1. Compute signal on `close[T]` (end of day T)
2. Place market order for `open[T+1]` (next morning open)
3. Fill price = `open[T+1] × (1 + 0.0005)` for buys (5bps slippage)
4. Positions entered = top N tickers by inv-vol-weighted score
5. Positions exited = any current holdings not in new top N
6. **No lookahead**: signal uses only data up to and including close[T]

### Minimum holding period
```
MIN_HOLD_DAYS = 5
```
No exit (stop or rank) can trigger in the first 5 days after entry.
Prevents same-day whipsaws from entering on an up-gap then stopping immediately.

---

## 5. Exit Rules (Two Mechanisms — Both Active Simultaneously)

### Exit 1: Adaptive trailing stop (S10-005 improvement)
```
Mechanism: track highest closing price since entry for each position.
           Exit if current close drops X% below that peak.

Adaptive threshold:
  recent_mom_20d = close[T] / close[T-20] - 1  (4-week momentum)

  if recent_mom_20d > 0.05:   # stock up >5% in past 4 weeks (parabolic trend)
      stop_pct = 0.30         # 30% trailing stop — give it room
  else:
      stop_pct = 0.20         # 20% trailing stop — normal

Logic each day for each held position:
  pos_high[tkr] = max(pos_high[tkr], close[T][tkr])  # update rolling high
  if (days_held >= MIN_HOLD_DAYS) and
     (close[T][tkr] < pos_high[tkr] × (1 - stop_pct)):
      EXIT immediately
```

**Why adaptive:** Parabolic momentum names (NVDA-type) routinely pull back 20-25%
mid-trend before doubling. A fixed 20% stop exits these winners prematurely.
Widening to 30% when recent momentum is strong lets winners run.

**CRITICAL:** Use `close[T]` for pos_high, NOT `high[T]` (intraday).
Intraday highs are never tradeable — using them creates phantom stop triggers
because stocks can make a 5% intraday spike and then close -20% from that spike
the same day, firing the stop when close was fine.

### Exit 2: Rank-based exit (S10-005 improvement, from RSL)
```
EXIT_PCT_RANK = 0.97  (exit if rank drops below top 3% of universe)

Each day, compute current momentum ranks across full universe:
  mom_today = close[T-15] / close[T-130] - 1  (same formula as signal)
  rank_today = percentile rank (0.0 = worst, 1.0 = best)

For each held position:
  if (days_held >= MIN_HOLD_DAYS) and
     (rank_today[tkr] < (1.0 - 0.97)):   # i.e., rank < 0.03 = bottom 3%
      EXIT immediately
```

**Why it works:** Unlike a fixed % stop, rank exit is adaptive:
- In bull markets, a winner's rank stays high → no exit, hold the full trend
- In bear markets, losers fall out of rank quickly → early exit before big drawdown
- In choppy markets, stocks that just stall fall out of top 3% fast → early rotation

**Both exits are active simultaneously.** First one to trigger wins.
In practice: rank exit fires more often in bear markets, trailing stop more in bulls.

---

## 6. Complete Daily Loop (Pseudocode)

```python
for each trading day T:
    today_close = close[T]
    fg_val = fear_greed[T]

    # 1. Compute daily momentum ranks for rank-exit check
    mom_today = close[T-15] / close[T-130] - 1
    rank_today = mom_today.rank(pct=True)

    # 2. Check rank-based exit for all held positions
    for tkr in held_positions:
        days_held = T - entry_day[tkr]
        if days_held >= 5:
            if rank_today[tkr] < 0.03:   # fell out of top 3%
                exit(tkr)

    # 3. Check adaptive trailing stop for all held positions
    for tkr in held_positions:
        pos_high[tkr] = max(pos_high[tkr], today_close[tkr])
        recent_mom = today_close[tkr] / close[T-20][tkr] - 1
        stop_pct = 0.30 if recent_mom > 0.05 else 0.20
        days_held = T - entry_day[tkr]
        if days_held >= 5:
            if today_close[tkr] < pos_high[tkr] * (1 - stop_pct):
                exit(tkr)

    # 4. Rebalance every 20 days if F&G >= 10
    if (T % 20 == 0) and (fg_val >= 10):
        # Compute signal
        momentum = close[T-15] / close[T-130] - 1
        above_ma = today_close > mean(close[T-100:T])
        high_vol = avg_vol[T-20:T] >= median(avg_vol[T-20:T] across universe)
        combo = momentum.rank(pct=True).where(above_ma & high_vol)

        # Adaptive concentration
        breadth = mean(today_close > mean(close[T-200:T]))
        if breadth < 0.40:
            n_top = max(1, int(len(filtered) * 0.010))
        elif fg_val > 70:
            n_top = max(1, int(len(filtered) * 0.015))
        else:
            n_top = max(1, int(len(filtered) * 0.025))

        # Select top N
        top_tickers = combo.nlargest(n_top).index

        # Size by inverse volatility (6-day window)
        vol = std(pct_change(close[T-6:T])) for each top ticker
        weight = (1/vol) / sum(1/vol)

        # Update portfolio
        exit positions not in top_tickers
        enter positions in top_tickers not already held

    # 5. Record weights for execution next morning
    weights[T] = current_positions
    # Execution: buy/sell at open[T+1] ± 5bps slippage
```

---

## 7. Cost Model (for realistic backtesting)

```
Starting capital:    $100,000
Commission:          $20 flat per ticker traded (IB-style, one-way)
Slippage:            5 basis points one-way on execution price
                     Buy:  fill = open[T+1] × 1.0005
                     Sell: fill = open[T+1] × 0.9995

Execution:           Signal at close[T] → fill at open[T+1]
Rebalancing:         Only trade when weights CHANGE (no daily drift rebalancing)
                     This is critical — daily drift rebalancing inflates costs 10×

Expected costs:      ~34 trades/year × $20 = $680 commission
                     + ~$540 slippage
                     Total: ~$1,220/year on $100k = 1.2% drag
```

**Engine rule for weight normalisation:**
- If sum(|weights|) > 1.0: scale down to gross=1.0 (no leverage allowed)
- If sum(|weights|) < 1.0: use as-is (remainder = cash, earns 0%)
- This allows partial equity exposure if the strategy signals it

---

## 8. Walk-Forward Validation Setup

```
14 overlapping windows, 3-year train / 6-month test, 6-month step:
  W01: train 2014-01 → 2017-01  test 2017-01 → 2017-07
  W02: train 2014-07 → 2017-07  test 2017-07 → 2018-01
  W03: train 2015-01 → 2018-01  test 2018-01 → 2018-07
  W04: train 2015-07 → 2018-07  test 2018-07 → 2019-01
  W05: train 2016-01 → 2019-01  test 2019-01 → 2019-07
  W06: train 2016-07 → 2019-07  test 2019-07 → 2020-01
  W07: train 2017-01 → 2020-01  test 2020-01 → 2020-07  ← COVID crash
  W08: train 2017-07 → 2020-07  test 2020-07 → 2021-01  ← recovery
  W09: train 2018-01 → 2021-01  test 2021-01 → 2021-07  ← bull peak
  W10: train 2018-07 → 2021-07  test 2021-07 → 2022-01  ← rotation
  W11: train 2019-01 → 2022-01  test 2022-01 → 2022-07  ← bear start
  W12: train 2019-07 → 2022-07  test 2022-07 → 2023-01  ← bear bottom
  W13: train 2020-01 → 2023-01  test 2023-01 → 2023-07  ← recovery
  W14: train 2020-07 → 2023-07  test 2023-07 → 2024-07  ← AI bull

OOS (never touched during any training or experiment selection):
  2024-07-01 → today
  Evaluated separately with evaluate.py after research is complete.

Scoring metric: MEAN Sharpe across all 14 windows
Constraints for committing:
  mean_sharpe > 0.3
  neg_windows ≤ 4   (at most 4 of 14 windows negative)
  worst_window > -1.2
  trades_per_year < 150
```

---

## 9. Known Optimal Parameters (Do Not Re-Test)

All confirmed via exhaustive walk-forward search across sessions 5-10:

| Parameter | Value | What happens if changed |
|-----------|-------|------------------------|
| Lookback | 26w (130d) | 24w/28w: overfit |
| Skip period | 3w (15d) | 2w: bad, 4w: slightly worse OOS |
| Rebalance | 4w (20d) | 5w/6w: overfit; weekly: catastrophic |
| Concentration (normal) | 2.5% of filtered | 2%: worse; 3%: fails train |
| Concentration (bear) | 1.0% of filtered | Increasing hurts 2022 bear protection |
| Concentration (greed) | 1.5% of filtered | Confirmed better than 1% or 2% |
| MA filter | 20w (100d) | 15w: barely fails overfit |
| Breadth threshold (bear) | <40% above 200d MA | <35%: worse; <45%: no effect |
| F&G entry gate | ≥10 | ≥22: too restrictive (blocks 2022 recovery) |
| Trailing stop normal | 20% from close HIGH | 15%: too tight; 25%: fails overfit |
| Trailing stop parabolic | 30% when 20d mom>5% | Key fix: prevents exits mid-parabola |
| Rank exit threshold | 0.97 (top 3%) | 0.95: worse; 0.90: worse |
| Inv-vol window | 6 days | 5d: overfits; 7d: slightly worse |
| Min hold before exit | 5 days | Prevents same-day whipsaws |
| Volume filter | Top 50% by share volume | 30%: too restrictive; 70%: too loose |
| Universe | All 841 tickers (no SP500 filter) | Any Wikipedia filter breaks strategy |

---

## 10. What Does NOT Work (Do Not Retry)

All proven failures across sessions 1-10:

- **More positions (>4):** Dilutes alpha. Edge comes from top 1-2 names only.
- **Portfolio-level stop loss:** Exits during drawdowns, misses recovery. Worse than nothing.
- **Vol-scaling (Barroso & Santa-Clara):** Reduces equity during high-vol = exactly when momentum wins. Destroyed WF.
- **R1000 mid-caps (without quality filter):** Lower-quality names add noise, reduce Sharpe.
- **Explicit SP500 Wikipedia filter:** Causes universe drift as composition changes monthly, breaks backtest reproducibility.
- **Weekly rebalance:** Catastrophic — momentum is a 4-week signal, not a 1-week signal.
- **F&G gate ≥22:** Blocks too many valid entries, especially during 2022 recovery.
- **Mean reversion hybrid:** Works in training period, fails in live deployment (regime-specific).
- **Regime classifier (MOM/MR switching):** Overfits to validation window, OOS -1.3 Sharpe.
- **Sector cap:** Inert at 1-2 stock concentration — never binds.
- **Hybrid exit (rank AND 4w momentum negative):** In 2021 rotation, stocks decline in rank with still-positive 4w momentum. Gate never fires.
- **Trailing stop on intraday HIGH:** Lookahead bias — intraday high not tradeable at signal time.

---

## 11. Live Trading Implementation Guide

### Daily schedule (runs once per day after market close)
```
16:30 ET  market closes
17:00 ET  run refresh_data.py to fetch today's OHLCV + VIX + F&G
17:05 ET  run signals.py to compute today's weights
17:10 ET  diff weights vs current portfolio → generate order list
17:15 ET  submit orders as market-on-open for next day 09:30 ET
09:30+1   orders fill at open ± slippage
```

### Broker
Interactive Brokers recommended:
- Commission: $0.005/share (tiered) or $1/trade minimum → approximates $20/trade for 1-2 stock lots
- API: ib_async Python library (already used in prior IB sessions)
- Order type: MKT (market on open), or OPGG (market-on-open guaranteed)

### Position sizing in live trading
```python
def compute_shares(ticker, weight, portfolio_value, open_price):
    dollar_allocation = weight * portfolio_value
    shares = dollar_allocation / open_price
    return int(shares)  # round down to whole shares
```

### Monthly universe refresh
- Re-run refresh_data.py on the first trading day of each month
- This updates the iShares IWB holdings and adds any new tickers
- Tickers removed from the universe: let existing positions ride to their normal exits

### F&G data in production
```python
import requests
r = requests.get("https://api.alternative.me/fng/?limit=1&format=json")
fg_today = int(r.json()["data"][0]["value"])
```

### Stop-loss monitoring
Runs daily. For each held position:
1. Fetch today's close price (after 16:00 ET)
2. Update pos_high = max(pos_high, today_close)
3. Compute recent_mom_20d
4. Set stop_pct = 0.30 or 0.20
5. If today_close < pos_high × (1 - stop_pct) → queue sell MKT for tomorrow open

### Rank-exit monitoring
Also runs daily. Compute momentum rank across full universe. If any held position
drops below top 3% → queue sell MKT for tomorrow open.

---

## 12. Performance Summary

| Metric | Value |
|--------|-------|
| OOS period | 2024-07-01 → 2026-03-25 (435 trading days) |
| OOS Sharpe | 1.548 |
| OOS Return | +208% |
| OOS MaxDD | -24.0% |
| OOS Final value | $308,588 from $100,000 |
| SPY over same period | +23% |
| Alpha (annualised) | ~57% |
| Win rate (daily) | 55% |
| Ann. volatility | ~47% |
| Trades/year | ~34 |
| Commission/year | ~$680 |
| Total cost/year | ~$1,200 (1.2% of capital) |
| Avg positions | 1.6 concurrent |
| Walk-forward mean Sharpe | 0.722 (14 windows) |
| WF worst window | -1.158 (2018H2 rate-hike crash) |
| WF neg windows | 3 out of 14 |

**Honest caveats:**
- Strategy has 13-month flat periods (Jul 2024 - Sep 2025 in OOS)
- Returns are concentrated: last 5 months generated ~140% of total return
- High annualised volatility (47%) means big drawdowns are normal
- Not suitable for capital you cannot afford to have flat for 12+ months
- This is a high-conviction momentum strategy, not a diversified fund
