#!/usr/bin/env python3
"""
refresh_data.py — Downloads Russell 1000 OHLCV (11 years), VIX, Fear & Greed.
Universe: iShares IWB ETF holdings (~1000 large+mid cap US stocks).
Includes mid-cap miners, energy, materials, defense missing from SP500-only.
Safe to re-run: overwrites parquet files with latest data.
"""
import io, requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

START = "2014-01-01"
END   = datetime.today().strftime("%Y-%m-%d")


def _ensure_spy_in_tickers(tickers: list[str]) -> list[str]:
    """Always include SPY exactly once for benchmark compatibility."""
    return list(dict.fromkeys(list(tickers) + ["SPY"]))


def _extract_ohlcv_field(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """Handle yfinance batch and single-symbol shapes consistently."""
    if isinstance(raw.columns, pd.MultiIndex):
        if field not in raw.columns.get_level_values(0):
            raise KeyError(f"Field '{field}' missing from download output.")
        frame = raw[field].copy()
    else:
        if field not in raw.columns:
            raise KeyError(f"Field '{field}' missing from download output.")
        frame = raw[[field]].copy()

    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    return frame


def _download_symbol_ohlcv(symbol: str) -> dict[str, pd.DataFrame]:
    raw = yf.download(symbol, start=START, end=END, auto_adjust=True, progress=False, threads=False)
    fields = {}
    for field in ("Close", "Open", "High", "Low", "Volume"):
        frame = _extract_ohlcv_field(raw, field)
        if symbol in frame.columns:
            frame = frame[[symbol]]
        elif len(frame.columns) == 1:
            frame.columns = [symbol]
        else:
            raise KeyError(f"Could not isolate {symbol} {field} from download output.")
        fields[field.lower()] = frame
    return fields


# ── 1. Russell 1000 tickers via iShares IWB holdings ─────────────────────────
def get_russell1000_tickers() -> list[str]:
    """
    Fetch current Russell 1000 constituents from iShares IWB ETF holdings CSV.
    Falls back to SP500 + supplemental list if iShares is unavailable.
    """
    print("Fetching Russell 1000 tickers from iShares IWB …")
    try:
        url = "https://www.ishares.com/us/products/239707/ISHARES-RUSSELL-1000-ETF/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; sp500-autoresearch/1.0)"}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        # iShares CSV has metadata rows at top — find the header row
        lines = r.text.splitlines()
        header_i = next(i for i, l in enumerate(lines) if l.startswith("Ticker,"))
        df = pd.read_csv(io.StringIO("\n".join(lines[header_i:])))
        tickers = (
            df["Ticker"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
        )
        # Filter: only real ticker symbols (exclude cash, "-", blank)
        tickers = tickers[tickers.str.match(r"^[A-Z][A-Z0-9\-]{0,6}$")].tolist()
        # Always append ETFs — not in IWB holdings but essential for sector rotation
        tickers = _ensure_spy_in_tickers(tickers + _etf_supplements())
        print(f"  Russell 1000 + ETFs: {len(tickers)} tickers")
        return tickers

    except Exception as e:
        print(f"  iShares fetch failed ({e}), falling back to SP500 + supplements …")
        return _sp500_plus_supplements()


def _etf_supplements() -> list[str]:
    """ETFs to always add — not in IWB holdings but critical for sector/commodity rotation."""
    return [
        # SPDR Sector ETFs (all 11)
        "XLF","XLRE","XLE","XLU","XLK","XLB","XLP","XLY","XLC","XLV","XLI",
        # Gold / silver
        "GLD","GDX","GDXJ","SLV",
        # Uranium
        "URA","URNM",
        # Metals / mining broad
        "XME","PICK",
        # Energy E&P
        "XOP",
        # Commodities
        "DBA","DBC","COPX","CPER",
        # Royalty / streaming (stocks, not ETFs — often missing from IWB)
        "WPM","RGLD","FNV",
    ]


def _sp500_plus_supplements() -> list[str]:
    """Fallback: SP500 + key mid-cap resource/mining/defense names missing from it."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; sp500-autoresearch/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    from io import StringIO
    tbl     = pd.read_html(StringIO(resp.text), attrs={"id": "constituents"})[0]
    sp500   = tbl["Symbol"].str.replace(".", "-", regex=False).tolist()

    # Key mid-cap names the SP500 misses — miners, uranium, energy, defense
    supplements = [
        # Gold / silver miners
        "AEM","AG","EGO","KGC","PAAS","BTG","CDE","HL","FSM","MAG",
        # Uranium
        "CCJ","UEC","DNN","URG","NXE","UUUU",
        # Copper / base metals
        "TECK","HBM","ERO","CENX","SCCO",
        # Oil & gas mid-cap
        "SM","CIVI","CHRD","MGY","PR","MTDR",
        # Defense mid-cap
        "KTOS","HII","TDY","MOOG","CW",
        # Materials
        "MP","LTHM","LAC","PLL",      # rare earth / lithium
        "CLF","STLD","CMC",            # steel
        # Royalty / streaming
        "WPM","RGLD","FNV",
    ]
    combined = _ensure_spy_in_tickers(sp500 + supplements + _etf_supplements())
    print(f"  SP500 + supplements: {len(combined)} tickers")
    return combined


# ── 2. OHLCV prices ───────────────────────────────────────────────────────────
def refresh_prices(tickers: list[str]):
    tickers = _ensure_spy_in_tickers(tickers)
    out = DATA_DIR / "sp500_prices.parquet"   # keep filename for compatibility
    print(f"Downloading OHLCV for {len(tickers)} tickers …")
    raw = yf.download(
        tickers, start=START, end=END,
        auto_adjust=True, progress=True, threads=8
    )

    close  = _extract_ohlcv_field(raw, "Close")
    opens  = _extract_ohlcv_field(raw, "Open")
    high   = _extract_ohlcv_field(raw, "High")
    low    = _extract_ohlcv_field(raw, "Low")
    volume = _extract_ohlcv_field(raw, "Volume")

    # Drop tickers with >20% missing days
    thresh = int(len(close) * 0.80)
    close  = close.dropna(thresh=thresh, axis=1)
    valid  = close.columns.tolist()

    if "SPY" not in valid:
        print("  SPY missing from batch download, fetching separately …")
        spy = _download_symbol_ohlcv("SPY")
        spy_close = spy["close"]
        if spy_close["SPY"].notna().sum() < thresh:
            raise RuntimeError("SPY download did not produce enough history for benchmark use.")
        close["SPY"] = spy_close["SPY"]
        opens["SPY"] = spy["open"]["SPY"]
        high["SPY"] = spy["high"]["SPY"]
        low["SPY"] = spy["low"]["SPY"]
        volume["SPY"] = spy["volume"]["SPY"]
        valid = list(dict.fromkeys(valid + ["SPY"]))

    prices = pd.concat({
        "close":  close[valid],
        "open":   opens[valid],
        "high":   high[valid],
        "low":    low[valid],
        "volume": volume[valid],
    }, axis=1)
    prices.to_parquet(out)
    print(f"  Saved {len(valid)} tickers × {len(close)} days → {out}")
    return valid


# ── 3. VIX ────────────────────────────────────────────────────────────────────
def refresh_vix():
    out = DATA_DIR / "vix.parquet"
    print("Downloading VIX …")
    vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
    vix = vix[["Close"]].rename(columns={"Close": "vix"})
    vix.to_parquet(out)
    print(f"  Saved {len(vix)} VIX rows → {out}")


# ── 4. Fear & Greed ───────────────────────────────────────────────────────────
def refresh_fear_greed():
    out = DATA_DIR / "fear_greed.parquet"
    print("Downloading Fear & Greed index …")
    url = "https://api.alternative.me/fng/?limit=3000&format=json&date_format=us"
    r   = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    df   = pd.DataFrame(data)[["timestamp", "value", "value_classification"]]
    df["date"]     = pd.to_datetime(df["timestamp"], format="%m-%d-%Y")
    df["fg_value"] = df["value"].astype(int)
    df   = df.set_index("date")[["fg_value", "value_classification"]].sort_index()
    df.to_parquet(out)
    print(f"  Saved {len(df)} Fear & Greed rows → {out}")


if __name__ == "__main__":
    print(f"=== Data refresh: {END} ===")
    tickers = get_russell1000_tickers()
    valid   = refresh_prices(tickers)
    refresh_vix()
    refresh_fear_greed()
    print(f"=== Done — {len(valid)} tickers in universe ===")
