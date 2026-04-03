import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import refresh_data


def _make_multi_ohlcv(index: pd.Index, payload: dict[str, list[float]]) -> pd.DataFrame:
    cols = []
    data = {}
    for field, series in payload.items():
        for symbol, values in series.items():
            cols.append((field, symbol))
            data[(field, symbol)] = values
    return pd.DataFrame(data, index=index, columns=pd.MultiIndex.from_tuples(cols))


class RefreshDataTests(unittest.TestCase):
    def test_ensure_spy_in_tickers_deduplicates(self):
        tickers = refresh_data._ensure_spy_in_tickers(["AAA", "SPY", "BBB"])
        self.assertEqual(tickers.count("SPY"), 1)
        self.assertEqual(tickers[-1], "BBB")
        self.assertIn("SPY", tickers)

    def test_refresh_prices_backfills_spy_when_missing_from_batch(self):
        idx = pd.bdate_range("2024-01-01", periods=5)
        batch_raw = _make_multi_ohlcv(
            idx,
            {
                "Close": {"AAA": [10, 11, 12, 13, 14]},
                "Open": {"AAA": [9, 10, 11, 12, 13]},
                "High": {"AAA": [11, 12, 13, 14, 15]},
                "Low": {"AAA": [8, 9, 10, 11, 12]},
                "Volume": {"AAA": [100, 100, 100, 100, 100]},
            },
        )
        spy_raw = _make_multi_ohlcv(
            idx,
            {
                "Close": {"SPY": [100, 101, 102, 103, 104]},
                "Open": {"SPY": [99, 100, 101, 102, 103]},
                "High": {"SPY": [101, 102, 103, 104, 105]},
                "Low": {"SPY": [98, 99, 100, 101, 102]},
                "Volume": {"SPY": [1000, 1000, 1000, 1000, 1000]},
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(refresh_data, "DATA_DIR", Path(tmpdir)):
                with patch.object(refresh_data.yf, "download", side_effect=[batch_raw, spy_raw]):
                    valid = refresh_data.refresh_prices(["AAA"])

                out = pd.read_parquet(Path(tmpdir) / "sp500_prices.parquet")
                self.assertIn("AAA", valid)
                self.assertIn("SPY", valid)
                self.assertIn("SPY", out["close"].columns)
                self.assertEqual(float(out["close"].iloc[-1]["SPY"]), 104.0)


if __name__ == "__main__":
    unittest.main()
