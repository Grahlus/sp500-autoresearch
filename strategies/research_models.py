from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment_spaces import normalize_experiment_config

from .base import StrategyFamily


def _series_to_frame(series: pd.Series, columns: pd.Index) -> pd.DataFrame:
    values = np.repeat(series.to_numpy(dtype=float)[:, None], len(columns), axis=1)
    return pd.DataFrame(values, index=series.index, columns=columns)


def _resolve_benchmark_series(data: dict[str, Any], close_index: pd.Index) -> pd.Series:
    close = data["close"]
    if "SPY" in close.columns:
        return close["SPY"].reindex(close_index).ffill()
    benchmark = data.get("spy_close")
    if isinstance(benchmark, pd.Series):
        return benchmark.reindex(close_index).ffill()
    benchmark = data.get("benchmark_close")
    if isinstance(benchmark, pd.Series):
        return benchmark.reindex(close_index).ffill()
    return close.ffill().mean(axis=1)


def _long_only_weights(scores: pd.Series, max_positions: int, top_pct: float, gross_target: float = 1.0) -> pd.Series:
    valid = scores.dropna().sort_values(ascending=False, kind="mergesort")
    if valid.empty:
        return pd.Series(0.0, index=scores.index)
    count = max(1, min(int(round(len(valid) * top_pct)), max_positions))
    chosen = valid.head(count).index
    weights = pd.Series(0.0, index=scores.index)
    weights.loc[chosen] = gross_target / count
    return weights


def _long_short_weights(scores: pd.Series, max_positions: int, top_pct: float) -> pd.Series:
    valid = scores.dropna().sort_values(ascending=False, kind="mergesort")
    if valid.empty:
        return pd.Series(0.0, index=scores.index)
    count = max(1, min(int(round(len(valid) * top_pct)), max_positions))
    weights = pd.Series(0.0, index=scores.index)
    long_names = valid.head(count).index
    short_names = valid.tail(count).index
    weights.loc[long_names] = 0.5 / count
    weights.loc[short_names] = -0.5 / count
    return weights


def _ml_feature_dict(data: dict[str, Any], horizon_days: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    close = data["close"].sort_index()
    high = data["high"].reindex(index=close.index, columns=close.columns)
    low = data["low"].reindex(index=close.index, columns=close.columns)
    volume = data["volume"].reindex(index=close.index, columns=close.columns)
    index = close.index
    columns = close.columns

    returns = close.pct_change()
    benchmark = _resolve_benchmark_series(data, index)
    vix = data.get("vix")
    fear_greed = data.get("fear_greed")
    vix_series = vix.reindex(index).ffill() if isinstance(vix, pd.Series) else pd.Series(np.nan, index=index)
    fg_series = fear_greed.reindex(index).ffill() if isinstance(fear_greed, pd.Series) else pd.Series(np.nan, index=index)
    breadth = (close > close.rolling(50, min_periods=50).mean()).mean(axis=1).fillna(0.0)

    feature_frames: dict[str, pd.DataFrame] = {
        "ret_1d": returns,
        "ret_5d": close.pct_change(5),
        "ret_20d": close.pct_change(20),
        "ret_63d": close.pct_change(63),
        "ma_20_gap": close.div(close.rolling(20, min_periods=20).mean()) - 1.0,
        "ma_50_gap": close.div(close.rolling(50, min_periods=50).mean()) - 1.0,
        "vol_20": returns.rolling(20, min_periods=20).std(),
        "vol_63": returns.rolling(63, min_periods=63).std(),
        "volume_ratio_20": volume.div(volume.rolling(20, min_periods=20).mean()) - 1.0,
        "range_5": (high.rolling(5, min_periods=5).max() - low.rolling(5, min_periods=5).min()).div(close.replace(0, pd.NA)),
        "spy_20d": _series_to_frame(benchmark.pct_change(20).fillna(0.0), columns),
        "vix": _series_to_frame(vix_series.ffill().fillna(0.0), columns),
        "fear_greed": _series_to_frame(fg_series.ffill().fillna(0.0), columns),
        "breadth": _series_to_frame(breadth, columns),
    }

    target = close.shift(-horizon_days).div(close) - 1.0
    return feature_frames, target


def _build_ml_panel(data: dict[str, Any], feature_set: str, horizon_days: int) -> tuple[pd.DataFrame, pd.Series]:
    feature_frames, target = _ml_feature_dict(data, horizon_days)
    allowed = {
        "trend": ["ret_1d", "ret_5d", "ret_20d", "ret_63d", "ma_20_gap", "ma_50_gap"],
        "trend_volume": ["ret_1d", "ret_5d", "ret_20d", "ret_63d", "ma_20_gap", "ma_50_gap", "vol_20", "vol_63", "volume_ratio_20", "range_5"],
        "full": ["ret_1d", "ret_5d", "ret_20d", "ret_63d", "ma_20_gap", "ma_50_gap", "vol_20", "vol_63", "volume_ratio_20", "range_5", "spy_20d", "vix", "fear_greed", "breadth"],
    }[feature_set]
    panel = pd.concat({name: feature_frames[name].stack() for name in allowed}, axis=1)
    panel.index.names = ["date", "ticker"]
    target_series = target.stack()
    return panel.sort_index(), target_series.sort_index()


def _fit_ml_model(model_type: str, x: pd.DataFrame, y: pd.Series, seed: int):
    if model_type == "hgb":
        model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=40, random_state=seed)
        return model.fit(x, y)
    if model_type == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        return model.fit(x, y)
    raise ValueError(f"Unknown ml_ranker model_type '{model_type}'.")


def generate_ml_ranker_signals_with_config(data: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    cfg = normalize_experiment_config("ml_ranker", config)
    close = data["close"].sort_index()
    dates = close.index
    weights = pd.DataFrame(0.0, index=dates, columns=close.columns)
    panel, target = _build_ml_panel(data, cfg["feature_set"], int(cfg["horizon_days"]))
    rebalance_step = int(cfg["rebalance_days"])
    lookback = int(cfg["lookback_days"])
    top_pct = float(cfg["top_pct"])
    max_positions = int(cfg["max_positions"])
    allow_short = bool(cfg["allow_short"])
    use_vix_gate = bool(cfg["use_vix_gate"])
    use_fg_gate = bool(cfg["use_fear_greed_gate"])
    model_type = str(cfg["model_type"])
    feature_names = list(panel.columns)

    if panel.empty or target.empty:
        return weights

    close_pct_20 = close.pct_change(20)
    vix = data.get("vix")
    fear_greed = data.get("fear_greed")
    vix_series = vix.reindex(dates).ffill() if isinstance(vix, pd.Series) else pd.Series(np.nan, index=dates)
    fg_series = fear_greed.reindex(dates).ffill() if isinstance(fear_greed, pd.Series) else pd.Series(np.nan, index=dates)

    rebalance_positions = list(range(max(lookback, 1), len(dates), rebalance_step))
    current_weights = pd.Series(0.0, index=close.columns)
    for idx, rebalance_pos in enumerate(rebalance_positions):
        current_date = dates[rebalance_pos]
        next_pos = rebalance_positions[idx + 1] if idx + 1 < len(rebalance_positions) else len(dates)
        start_date = dates[max(0, rebalance_pos - lookback)]

        train_mask = (
            (panel.index.get_level_values(0) >= start_date)
            & (panel.index.get_level_values(0) < current_date)
            & target.notna()
        )
        train_panel = panel.loc[train_mask].fillna(0.0)
        train_target = target.loc[train_mask].astype(float)
        # Keep the ML path CPU-feasible on the 8-core VM by capping the training window.
        # This preserves the search space while preventing each rebalance from fitting
        # on the full historical panel.
        max_train_rows = 1500
        if len(train_panel) > max_train_rows:
            train_panel = train_panel.iloc[-max_train_rows:]
            train_target = train_target.iloc[-max_train_rows:]
        if len(train_panel) < max(200, len(close.columns) * 2):
            scores = close_pct_20.loc[current_date].fillna(0.0)
        else:
            model = _fit_ml_model(model_type, train_panel[feature_names].astype(float), train_target, seed=rebalance_pos)
            scores = pd.Series(model.predict(panel.loc[current_date][feature_names].fillna(0.0).astype(float)), index=close.columns)

        scale = 1.0
        if use_vix_gate and pd.notna(vix_series.loc[current_date]) and float(vix_series.loc[current_date]) >= 30.0:
            scale *= 0.5
        if use_fg_gate and pd.notna(fg_series.loc[current_date]) and float(fg_series.loc[current_date]) <= 20.0:
            scale *= 0.5

        if allow_short:
            current_weights = _long_short_weights(scores, max_positions=max_positions, top_pct=top_pct) * scale
        else:
            current_weights = _long_only_weights(scores, max_positions=max_positions, top_pct=top_pct) * scale

        weights.iloc[rebalance_pos:next_pos] = current_weights.to_numpy(dtype=float)

    return weights.reindex(index=dates, columns=close.columns).fillna(0.0)


def _arm_momentum_scores(data: dict[str, Any], current_date: pd.Timestamp, lookback_days: int) -> pd.Series:
    close = data["close"].sort_index()
    shifted = close.pct_change(lookback_days)
    return shifted.loc[current_date].fillna(0.0)


def _arm_superstock_scores(data: dict[str, Any], current_date: pd.Timestamp) -> pd.Series:
    close = data["close"].sort_index()
    volume = data["volume"].reindex(index=close.index, columns=close.columns)
    ret_126 = close.pct_change(126).loc[current_date].fillna(0.0)
    ret_252 = close.pct_change(252).loc[current_date].fillna(0.0)
    vol_ratio = (volume / volume.rolling(20, min_periods=20).mean()).loc[current_date].fillna(1.0)
    high_252 = close.rolling(252, min_periods=252).max().loc[current_date].replace(0, pd.NA)
    proximity = (close.loc[current_date] / high_252).fillna(0.0)
    return (0.35 * ret_126) + (0.35 * ret_252) + (0.20 * vol_ratio) + (0.10 * proximity)


def _state_bucket(value: float, bounds: tuple[float, float]) -> str:
    low, high = bounds
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "mid"


def _bandit_state(data: dict[str, Any], date: pd.Timestamp) -> tuple[str, str, str, str]:
    close = data["close"]
    vix = data.get("vix")
    fear_greed = data.get("fear_greed")
    vix_value = float(vix.reindex(close.index).ffill().loc[date]) if isinstance(vix, pd.Series) else float("nan")
    fg_value = float(fear_greed.reindex(close.index).ffill().loc[date]) if isinstance(fear_greed, pd.Series) else float("nan")
    spy = close["SPY"] if "SPY" in close.columns else close.mean(axis=1)
    trend = float(spy.pct_change(20).loc[date]) if date in spy.index else 0.0
    ma_50 = close.rolling(50, min_periods=50).mean()
    breadth = float((close.loc[date] > ma_50.loc[date]).mean()) if date in close.index and date in ma_50.index else 0.0
    return (
        _state_bucket(vix_value, (20.0, 30.0)),
        _state_bucket(fg_value, (30.0, 60.0)),
        _state_bucket(trend, (-0.05, 0.05)),
        _state_bucket(breadth, (0.35, 0.65)),
    )


def _choose_bandit_arm(
    stats: dict[tuple[str, str, str, str], dict[str, dict[str, float]]],
    state: tuple[str, str, str, str],
    *,
    policy_type: str,
    epsilon: float,
    ucb_bonus: float,
    rng: np.random.Generator,
    data: dict[str, Any],
    date: pd.Timestamp,
) -> str:
    arms = ["momentum", "superstock", "cash"]
    state_stats = stats.get(state, {})
    if not state_stats:
        vix_bucket, fg_bucket, trend_bucket, breadth_bucket = state
        if vix_bucket == "high" or fg_bucket == "low":
            return "cash"
        if trend_bucket == "high" and breadth_bucket != "low":
            return "momentum"
        return "superstock"
    if policy_type == "epsilon_greedy" and rng.random() < epsilon:
        return str(rng.choice(arms))
    total = sum(item.get("count", 0.0) for item in state_stats.values()) + 1.0
    scored: list[tuple[float, str]] = []
    for arm in arms:
        arm_stats = state_stats.get(arm, {"count": 0.0, "reward": 0.0})
        count = max(float(arm_stats.get("count", 0.0)), 0.0)
        reward = float(arm_stats.get("reward", 0.0))
        bonus = ucb_bonus * np.sqrt(np.log(total + 1.0) / (count + 1.0))
        scored.append((reward + bonus, arm))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][1]


def _update_bandit_state(
    stats: dict[tuple[str, str, str, str], dict[str, dict[str, float]]],
    state: tuple[str, str, str, str],
    arm: str,
    reward: float,
) -> None:
    state_stats = stats.setdefault(state, {})
    arm_stats = state_stats.setdefault(arm, {"count": 0.0, "reward": 0.0})
    count = float(arm_stats["count"]) + 1.0
    mean_reward = float(arm_stats["reward"])
    arm_stats["count"] = count
    arm_stats["reward"] = mean_reward + ((reward - mean_reward) / count)


def _arm_weights_from_scores(scores: pd.Series, top_pct: float, max_positions: int) -> pd.Series:
    return _long_only_weights(scores, max_positions=max_positions, top_pct=top_pct)


def generate_rl_bandit_signals_with_config(data: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    cfg = normalize_experiment_config("rl_bandit", config)
    close = data["close"].sort_index()
    open_ = data["open"].reindex(index=close.index, columns=close.columns)
    dates = close.index
    weights = pd.DataFrame(0.0, index=dates, columns=close.columns)
    policy_type = str(cfg["policy_type"])
    lookback_days = int(cfg["lookback_days"])
    rebalance_days = int(cfg["rebalance_days"])
    epsilon = float(cfg["epsilon"])
    ucb_bonus = float(cfg["ucb_bonus"])
    max_positions = int(cfg["max_positions"])
    momentum_top_pct = float(cfg["momentum_top_pct"])
    superstock_top_pct = float(cfg["superstock_top_pct"])
    use_vix_gate = bool(cfg["use_vix_gate"])
    use_fg_gate = bool(cfg["use_fear_greed_gate"])
    rng = np.random.default_rng(lookback_days + rebalance_days)
    stats: dict[tuple[str, str, str, str], dict[str, dict[str, float]]] = {}

    open_returns = open_.shift(-1).div(open_) - 1.0
    rebalance_positions = list(range(max(lookback_days, 1), len(dates), rebalance_days))
    current_weights = pd.Series(0.0, index=close.columns)
    current_arm = "cash"
    current_state = ("mid", "mid", "mid", "mid")

    for idx, rebalance_pos in enumerate(rebalance_positions):
        current_date = dates[rebalance_pos]
        next_pos = rebalance_positions[idx + 1] if idx + 1 < len(rebalance_positions) else len(dates)
        if idx > 0:
            prev_start = dates[rebalance_positions[idx - 1]]
            window_returns = open_returns.loc[prev_start:current_date].fillna(0.0)
            if current_arm == "cash":
                reward = 0.0
            else:
                reward_series = window_returns.mul(current_weights, axis=1).sum(axis=1)
                reward = float(reward_series.mean()) if not reward_series.empty else 0.0
            _update_bandit_state(stats, current_state, current_arm, reward)

        state = _bandit_state(data, current_date)
        current_state = state
        chosen_arm = _choose_bandit_arm(
            stats,
            state,
            policy_type=policy_type,
            epsilon=epsilon,
            ucb_bonus=ucb_bonus,
            rng=rng,
            data=data,
            date=current_date,
        )
        if chosen_arm == "momentum":
            scores = _arm_momentum_scores(data, current_date=current_date, lookback_days=lookback_days)
            current_weights = _arm_weights_from_scores(scores, top_pct=momentum_top_pct, max_positions=max_positions)
        elif chosen_arm == "superstock":
            scores = _arm_superstock_scores(data, current_date=current_date)
            current_weights = _arm_weights_from_scores(scores, top_pct=superstock_top_pct, max_positions=max_positions)
        else:
            current_weights = pd.Series(0.0, index=close.columns)
        scale = 1.0
        if use_vix_gate:
            vix = data.get("vix")
            if isinstance(vix, pd.Series):
                vix_value = float(vix.reindex(close.index).ffill().loc[current_date])
                if vix_value >= 30.0:
                    scale *= 0.5
        if use_fg_gate:
            fg = data.get("fear_greed")
            if isinstance(fg, pd.Series):
                fg_value = float(fg.reindex(close.index).ffill().loc[current_date])
                if fg_value <= 20.0:
                    scale *= 0.5
        current_weights = current_weights * scale
        weights.iloc[rebalance_pos:next_pos] = current_weights.to_numpy(dtype=float)
        current_arm = chosen_arm

    return weights.reindex(index=dates, columns=close.columns).fillna(0.0)


def load_ml_ranker() -> StrategyFamily:
    return StrategyFamily(
        name="ml_ranker",
        metric="sharpe",
        hypothesis="CPU-friendly cross-sectional ML ranker using ridge or light gradient boosting.",
        generate_signals=generate_ml_ranker_signals_with_config,
        generate_signals_with_config=generate_ml_ranker_signals_with_config,
    )


def load_rl_bandit() -> StrategyFamily:
    return StrategyFamily(
        name="rl_bandit",
        metric="sharpe",
        hypothesis="CPU-light contextual bandit that chooses between momentum, breakout, and cash arms.",
        generate_signals=generate_rl_bandit_signals_with_config,
        generate_signals_with_config=generate_rl_bandit_signals_with_config,
    )
