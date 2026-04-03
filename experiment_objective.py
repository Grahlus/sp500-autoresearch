from __future__ import annotations

import re
from typing import Any


def _parse_beat_count(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        match = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", value)
        if match:
            return int(match.group(1)), int(match.group(2))
    return 0, 0


def evaluate_objective(metrics: dict[str, Any], family: str) -> float:
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    calmar = float(metrics.get("calmar", 0.0) or 0.0)
    total_return = float(metrics.get("total_return", metrics.get("total_return_pct", 0.0)) or 0.0)
    trades_per_year = float(metrics.get("trades_per_year", 0.0) or 0.0)
    max_drawdown = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    return (
        sharpe
        + (0.15 * calmar)
        + (0.02 * total_return)
        - (0.01 * max(0.0, trades_per_year - 80.0))
        - (0.02 * max(0.0, max_drawdown - 25.0))
    )


def evaluate_robustness(metrics: dict[str, Any]) -> dict[str, bool | int | float]:
    windows = metrics.get("windows", []) or []
    negative_windows = sum(1 for window in windows if float(window.get("sharpe", 0.0) or 0.0) < 0.0)
    beat_wins, beat_total = _parse_beat_count(metrics.get("windows_beat_spy"))
    robustness: dict[str, bool | int | float] = {
        "mean_sharpe_ok": float(metrics.get("sharpe", 0.0) or 0.0) >= 0.3,
        "trades_per_year_ok": float(metrics.get("trades_per_year", 0.0) or 0.0) <= 150.0,
        "negative_windows_ok": negative_windows <= 4,
        "worst_window_ok": float(metrics.get("sharpe_min", 0.0) or 0.0) >= -1.2,
        "negative_windows": negative_windows,
        "beat_spy_wins": beat_wins,
        "beat_spy_total": beat_total,
    }
    robustness["viable"] = all(
        bool(robustness[name])
        for name in ("mean_sharpe_ok", "trades_per_year_ok", "negative_windows_ok", "worst_window_ok")
    )
    return robustness


def is_experiment_viable(metrics: dict[str, Any]) -> bool:
    return bool(evaluate_robustness(metrics)["viable"])


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(result: dict[str, Any]):
        robustness = result.get("robustness", {})
        metrics = result.get("metrics", {})
        return (
            0 if robustness.get("viable") else 1,
            -float(result.get("objective_score", result.get("score", float("-inf")))),
            -float(metrics.get("sharpe_min", float("-inf"))),
            int(robustness.get("negative_windows", 999)),
            float(metrics.get("trades_per_year", float("inf"))),
            abs(float(metrics.get("max_drawdown", float("inf")))),
        )

    return sorted(results, key=sort_key)
