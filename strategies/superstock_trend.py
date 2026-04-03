import pandas as pd

from .screening import ScreeningResult, combine_masks
from .superstock_weekly import build_superstock_weekly_features, to_daily_feature_map, SuperstockWeeklyFeatures


def build_superstock_trend_template(
    data: dict,
    weekly: SuperstockWeeklyFeatures | None = None,
    config: dict | None = None,
) -> ScreeningResult:
    config = config or {}
    close = data["close"]
    weekly = weekly or build_superstock_weekly_features(data)
    weekly_daily = to_daily_feature_map(weekly, close.index)

    sma_50d = close.rolling(50, min_periods=50).mean()
    sma_150d = close.rolling(150, min_periods=150).mean()
    sma_200d = close.rolling(200, min_periods=200).mean()
    low_52w = close.rolling(252, min_periods=252).min()
    high_52w = close.rolling(252, min_periods=252).max()

    rule_masks = {
        "daily_above_50d": close > sma_50d,
        "daily_above_150d": close > sma_150d,
        "daily_above_200d": close > sma_200d,
        "ma_150d_above_200d": sma_150d > sma_200d,
        "ma_200d_uptrend": sma_200d > sma_200d.shift(20),
        "ma_stack_bullish": (sma_50d > sma_150d) & (sma_150d > sma_200d),
        "above_52w_low_enough": close >= (low_52w * float(config.get("above_52w_low_mult", 1.25))),
        "near_52w_high_enough": close >= (high_52w * float(config.get("near_52w_high_mult", 0.75))),
        "rs_rank_26w_min": weekly_daily["rs_rank_26w"] >= float(config.get("rs_rank_26w_min", 0.70)),
        "rs_rank_52w_min": weekly_daily["rs_rank_52w"] >= float(config.get("rs_rank_52w_min", 0.70)),
        "weekly_above_30w": weekly_daily["weekly_close"] > weekly_daily["ma_30w"],
        "weekly_above_10w": weekly_daily["weekly_close"] > weekly_daily["ma_10w"],
    }

    eligible_rule_names = [
        "daily_above_50d",
        "daily_above_150d",
        "daily_above_200d",
        "ma_150d_above_200d",
        "ma_200d_uptrend",
        "ma_stack_bullish",
        "above_52w_low_enough",
        "near_52w_high_enough",
        "rs_rank_26w_min",
        "rs_rank_52w_min",
        "weekly_above_30w",
    ]
    eligible = combine_masks({name: rule_masks[name] for name in eligible_rule_names})

    diagnostics: dict[str, pd.DataFrame | pd.Series | str | None] = {
        "sma_50d": sma_50d,
        "sma_150d": sma_150d,
        "sma_200d": sma_200d,
        "low_52w": low_52w,
        "high_52w": high_52w,
        "benchmark_source": weekly.benchmark_source,
        "benchmark_weekly_close": weekly_daily["benchmark_weekly_close"],
        "spy_weekly_close": weekly_daily["spy_weekly_close"],
        "weekly_close": weekly_daily["weekly_close"],
        "ma_10w": weekly_daily["ma_10w"],
        "ma_30w": weekly_daily["ma_30w"],
        "rs_rank_26w": weekly_daily["rs_rank_26w"],
        "rs_rank_52w": weekly_daily["rs_rank_52w"],
    }

    return ScreeningResult(eligible=eligible, rule_masks=rule_masks, diagnostics=diagnostics)
