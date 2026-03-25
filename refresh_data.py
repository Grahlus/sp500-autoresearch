#!/usr/bin/env python3
"""
refresh_data.py — Downloads SP500 OHLCV (11 years), VIX, and Fear & Greed.
Run this before each research session. Safe to re-run: only fetches delta.
"""
import os, requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

START = "2014-01-01"
END   = datetime.today().strftime("%Y-%m-%d")


# ── 1. SP500 tickers from Wikipedia ──────────────────────────────────────────
def get_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; sp500-autoresearch/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    from io import StringIO
    tbl = pd.read_html(StringIO(resp.text), attrs={"id": "constituents"})[0]
    tickers = tbl["Symbol"].str.replace(".", "-", regex=False).tolist()
    print(f"  SP500 tickers: {len(tickers)}")
    return tickers


# ── 2. OHLCV prices ──────────────────────────────────────────────────────────
def refresh_prices(tickers: list[str]):
    out = DATA_DIR / "sp500_prices.parquet"
    print("Downloading SP500 OHLCV …")
    raw = yf.download(
        tickers, start=START, end=END,
        auto_adjust=True, progress=True, threads=8
    )

    close  = raw["Close"].copy()
    opens  = raw["Open"].copy()
    high   = raw["High"].copy()
    low    = raw["Low"].copy()
    volume = raw["Volume"].copy()

    # Drop tickers with >20% missing days
    thresh = int(len(close) * 0.80)
    close  = close.dropna(thresh=thresh, axis=1)
    valid  = close.columns.tolist()

    prices = pd.concat({
        "close":  close[valid],
        "open":   opens[valid],
        "high":   high[valid],
        "low":    low[valid],
        "volume": volume[valid],
    }, axis=1)
    prices.to_parquet(out)
    print(f"  Saved {len(valid)} tickers × {len(close)} days → {out}")


# ── 3. VIX ───────────────────────────────────────────────────────────────────
def refresh_vix():
    out = DATA_DIR / "vix.parquet"
    print("Downloading VIX …")
    vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
    vix = vix[["Close"]].rename(columns={"Close": "vix"})
    vix.to_parquet(out)
    print(f"  Saved {len(vix)} VIX rows → {out}")


# ── 4. Fear & Greed (alternative.me, up to 3000 days of history) ─────────────
def refresh_fear_greed():
    out = DATA_DIR / "fear_greed.parquet"
    print("Downloading Fear & Greed index …")
    url = "https://api.alternative.me/fng/?limit=3000&format=json&date_format=us"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    df = pd.DataFrame(data)[["timestamp", "value", "value_classification"]]
    df["date"] = pd.to_datetime(df["timestamp"], format="%m-%d-%Y")
    df["fg_value"] = df["value"].astype(int)
    df = df.set_index("date")[["fg_value", "value_classification"]].sort_index()
    df.to_parquet(out)
    print(f"  Saved {len(df)} Fear & Greed rows → {out}")


if __name__ == "__main__":
    print(f"=== Data refresh: {END} ===")
    tickers = get_sp500_tickers()
    refresh_prices(tickers)
    refresh_vix()
    refresh_fear_greed()
    print("=== Done ===")
