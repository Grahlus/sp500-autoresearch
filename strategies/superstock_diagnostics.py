from __future__ import annotations

import math

import pandas as pd

from .superstock import MAX_POSITIONS, build_superstock_pipeline


def _bool_df(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.fillna(False).astype(bool)


def _series_from_shared_mask(value, columns: pd.Index, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        return value.reindex(index=index, columns=columns).iloc[:, 0].fillna(False).astype(bool)
    return pd.Series(value, index=index).fillna(False).astype(bool)


def _resolve_benchmark_exec_series(data: dict, benchmark_source: str) -> pd.Series:
    open_ = data["open"]
    close = data["close"]

    if benchmark_source == "spy_symbol" and "SPY" in open_.columns:
        return open_["SPY"]
    if benchmark_source == "spy_close_column" and isinstance(data.get("spy_close"), pd.Series):
        return data["spy_close"].reindex(close.index).ffill()
    if benchmark_source == "benchmark_close_column" and isinstance(data.get("benchmark_close"), pd.Series):
        return data["benchmark_close"].reindex(close.index).ffill()

    bench_ret = open_.shift(-1).div(open_).sub(1.0).fillna(0.0).mean(axis=1)
    return bench_ret.add(1.0).cumprod()


def _exit_reason_frame(exits, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    support = _bool_df(exits.diagnostics["exit_reason_support"]).reindex(index=index, columns=columns, fill_value=False)
    late_stage_hv = _bool_df(exits.diagnostics["exit_reason_late_stage_hv"]).reindex(index=index, columns=columns, fill_value=False)
    parabolic = _bool_df(exits.diagnostics["exit_reason_parabolic"]).reindex(index=index, columns=columns, fill_value=False)
    any_exit = _bool_df(exits.diagnostics["exit_reason_any"]).reindex(index=index, columns=columns, fill_value=False)

    reasons = pd.DataFrame("", index=index, columns=columns, dtype=object)
    reasons = reasons.mask(support, "support_break")
    reasons = reasons.mask(~support & late_stage_hv, "late_stage_hv")
    reasons = reasons.mask(~support & ~late_stage_hv & parabolic, "parabolic_extension")
    reasons = reasons.mask(~any_exit, "")
    return reasons


def _classify_vix_regime(vix_weekly_close: pd.Series) -> pd.Series:
    if vix_weekly_close.isna().all():
        return pd.Series("unknown", index=vix_weekly_close.index, name="vix_regime")

    regime = pd.Series("medium", index=vix_weekly_close.index, name="vix_regime", dtype=object)
    regime.loc[vix_weekly_close.isna()] = "unknown"
    regime.loc[vix_weekly_close < 20] = "low"
    regime.loc[vix_weekly_close >= 30] = "high"
    return regime


def _extract_first_series(value, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        return value.reindex(index=index).iloc[:, 0]
    return value.reindex(index)


def build_superstock_trade_attribution(
    data: dict,
    entries,
    exits,
    positions: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    trade_columns = [
        "ticker",
        "entry_signal_date",
        "entry_exec_date",
        "entry_price",
        "entry_weight",
        "entry_rs_rank_26w",
        "entry_rs_rank_52w",
        "entry_breakout_pct_above_pivot",
        "entry_benchmark_price",
        "exit_signal_date",
        "exit_exec_date",
        "exit_price",
        "hold_days",
        "gross_return_pct",
        "benchmark_return_pct_same_window",
        "alpha_pct_same_window",
        "exit_reason",
        "benchmark_source",
    ]
    index = weights.index
    columns = weights.columns
    open_ = data["open"].reindex(index=index, columns=columns)

    actual_weights = weights.shift(1).fillna(0.0)
    actual_held = actual_weights > 0
    prior_held = actual_held.shift(1, fill_value=False)
    opened = actual_held & ~prior_held
    closed = ~actual_held & prior_held

    exit_reasons = _exit_reason_frame(exits, index, columns)
    benchmark_source = str(exits.diagnostics.get("benchmark_source") or "unknown")
    benchmark_exec = _resolve_benchmark_exec_series(data, benchmark_source).reindex(index).ffill()
    rs_rank_26w = entries.diagnostics.get("rs_rank_26w")
    rs_rank_52w = entries.diagnostics.get("rs_rank_52w")
    breakout_pct = entries.diagnostics.get("breakout_pct_above_pivot")

    rows: list[dict[str, object]] = []
    open_trades: dict[str, dict[str, object]] = {}

    for i, dt in enumerate(index):
        signal_dt = index[i - 1] if i > 0 else pd.NaT

        for symbol in opened.loc[dt][opened.loc[dt]].index:
            open_trades[symbol] = {
                "ticker": symbol,
                "entry_signal_date": signal_dt.date().isoformat() if i > 0 else None,
                "entry_exec_date": dt.date().isoformat(),
                "entry_price": float(open_.loc[dt, symbol]),
                "entry_weight": float(actual_weights.loc[dt, symbol]),
                "entry_rs_rank_26w": float(rs_rank_26w.loc[signal_dt, symbol]) if i > 0 and isinstance(rs_rank_26w, pd.DataFrame) else math.nan,
                "entry_rs_rank_52w": float(rs_rank_52w.loc[signal_dt, symbol]) if i > 0 and isinstance(rs_rank_52w, pd.DataFrame) else math.nan,
                "entry_breakout_pct_above_pivot": float(breakout_pct.loc[signal_dt, symbol]) if i > 0 and isinstance(breakout_pct, pd.DataFrame) else math.nan,
                "entry_benchmark_price": float(benchmark_exec.loc[dt]) if pd.notna(benchmark_exec.loc[dt]) else math.nan,
            }

        for symbol in closed.loc[dt][closed.loc[dt]].index:
            trade = open_trades.pop(symbol, None)
            if trade is None:
                continue

            exit_signal_date = signal_dt.date().isoformat() if i > 0 else None
            reason = ""
            if i > 0:
                reason = str(exit_reasons.loc[signal_dt, symbol])
            if not reason:
                reason = "selection_drop" if bool(positions.loc[dt, symbol]) else "unknown"

            entry_exec_date = pd.Timestamp(trade["entry_exec_date"])
            hold_days = int(index.get_loc(dt) - index.get_loc(entry_exec_date))
            entry_price = float(trade["entry_price"])
            exit_price = float(open_.loc[dt, symbol])
            gross_return_pct = (exit_price / entry_price - 1.0) * 100.0 if entry_price > 0 else math.nan

            entry_benchmark_price = float(trade["entry_benchmark_price"])
            exit_benchmark_price = float(benchmark_exec.loc[dt]) if pd.notna(benchmark_exec.loc[dt]) else math.nan
            benchmark_return_pct = (
                (exit_benchmark_price / entry_benchmark_price - 1.0) * 100.0
                if entry_benchmark_price > 0 and pd.notna(exit_benchmark_price)
                else math.nan
            )

            rows.append(
                {
                    **trade,
                    "exit_signal_date": exit_signal_date,
                    "exit_exec_date": dt.date().isoformat(),
                    "exit_price": exit_price,
                    "hold_days": hold_days,
                    "gross_return_pct": gross_return_pct,
                    "benchmark_return_pct_same_window": benchmark_return_pct,
                    "alpha_pct_same_window": gross_return_pct - benchmark_return_pct if pd.notna(benchmark_return_pct) else math.nan,
                    "exit_reason": reason,
                    "benchmark_source": benchmark_source,
                }
            )

    for symbol, trade in open_trades.items():
        rows.append(
            {
                **trade,
                "exit_signal_date": None,
                "exit_exec_date": None,
                "exit_price": math.nan,
                "hold_days": math.nan,
                "gross_return_pct": math.nan,
                "benchmark_return_pct_same_window": math.nan,
                "alpha_pct_same_window": math.nan,
                "exit_reason": "open_position",
                "benchmark_source": benchmark_source,
            }
        )

    return pd.DataFrame(rows, columns=trade_columns)


def build_superstock_daily_diagnostics(
    data: dict,
    screen,
    entries,
    exits,
    positions: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    close = data["close"]
    open_ = data["open"].reindex(index=weights.index, columns=weights.columns)
    index = close.index
    columns = close.columns

    screened = _bool_df(screen.eligible)
    entry_signals = _bool_df(entries.eligible)
    lifecycle_positions = _bool_df(positions)
    target_weights = weights.reindex(index=index, columns=columns).fillna(0.0)
    held_weights = target_weights.shift(1).fillna(0.0)
    held_mask = held_weights > 0
    prior_held_mask = held_mask.shift(1, fill_value=False)
    opened_count = (held_mask & ~prior_held_mask).sum(axis=1)
    closed_mask = ~held_mask & prior_held_mask

    exit_reasons = _exit_reason_frame(exits, index, columns)
    benchmark_source = str(screen.diagnostics.get("benchmark_source") or "unknown")
    benchmark_not_broken = _extract_first_series(screen.diagnostics["benchmark_not_broken"], index).fillna(False).astype(bool)
    vix_weekly_close = _extract_first_series(screen.diagnostics["vix_weekly_close"], index)
    vix_regime = _classify_vix_regime(vix_weekly_close)

    open_next_ret = open_.shift(-1).div(open_).sub(1.0).fillna(0.0)
    portfolio_return = held_weights.mul(open_next_ret).sum(axis=1)
    benchmark_exec = _resolve_benchmark_exec_series(data, benchmark_source).reindex(index).ffill()
    benchmark_return = benchmark_exec.shift(-1).div(benchmark_exec).sub(1.0).fillna(0.0)

    avg_position_weight = held_weights.where(held_mask).mean(axis=1, skipna=True).fillna(0.0)
    turnover = held_weights.sub(held_weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    selected_but_not_held = lifecycle_positions & ~held_mask

    daily = pd.DataFrame(index=index)
    daily["benchmark_source"] = benchmark_source
    daily["screen_pass_count"] = screened.sum(axis=1)
    daily["screen_pass_rate"] = daily["screen_pass_count"] / len(columns)
    daily["entry_trigger_count"] = entry_signals.sum(axis=1)
    daily["entry_trigger_rate"] = daily["entry_trigger_count"].div(daily["screen_pass_count"].replace(0, pd.NA)).fillna(0.0)
    daily["held_count"] = held_mask.sum(axis=1)
    daily["opened_count"] = opened_count
    daily["exit_count"] = closed_mask.sum(axis=1)
    daily["selection_drop_count"] = selected_but_not_held.sum(axis=1)
    daily["gross_exposure"] = held_weights.sum(axis=1)
    daily["target_exposure"] = target_weights.sum(axis=1)
    daily["avg_position_weight"] = avg_position_weight
    daily["turnover"] = turnover
    daily["simultaneous_positions"] = daily["held_count"]
    daily["portfolio_return"] = portfolio_return
    daily["benchmark_return"] = benchmark_return
    daily["alpha_return"] = portfolio_return - benchmark_return
    daily["benchmark_not_broken"] = benchmark_not_broken
    daily["vix_regime"] = vix_regime

    if isinstance(screen.diagnostics.get("rs_rank_26w"), pd.DataFrame):
        rs26 = screen.diagnostics["rs_rank_26w"].reindex(index=index, columns=columns)
        daily["avg_rs_rank_26w_screened"] = rs26.where(screened).mean(axis=1, skipna=True).fillna(0.0)
        daily["avg_rs_rank_26w_entered"] = rs26.where(entry_signals).mean(axis=1, skipna=True).fillna(0.0)

    exit_reasons_exec = exit_reasons.shift(1, fill_value="")
    support_mask = exit_reasons_exec.eq("support_break") & closed_mask
    hv_mask = exit_reasons_exec.eq("late_stage_hv") & closed_mask
    parabolic_mask = exit_reasons_exec.eq("parabolic_extension") & closed_mask
    unknown_exit_mask = exit_reasons_exec.eq("unknown") & closed_mask

    daily["exit_support_count"] = support_mask.sum(axis=1)
    daily["exit_late_stage_hv_count"] = hv_mask.sum(axis=1)
    daily["exit_parabolic_count"] = parabolic_mask.sum(axis=1)
    daily["exit_unknown_count"] = unknown_exit_mask.sum(axis=1)

    for name, mask in screen.rule_masks.items():
        daily[f"screen_rule_{name}_count"] = _bool_df(mask).sum(axis=1)
    for name, mask in entries.rule_masks.items():
        daily[f"entry_rule_{name}_count"] = _bool_df(mask).sum(axis=1)
    for name, mask in exits.rule_masks.items():
        daily[f"exit_rule_{name}_count"] = _bool_df(mask).sum(axis=1)

    return daily.reset_index(names="date")


def summarize_superstock_diagnostics(diag: dict, comparison_metrics: dict | None = None) -> pd.DataFrame:
    daily = diag["daily_diagnostics"]
    trades = diag["trade_attribution"]
    benchmark_source = str(diag["metadata"]["benchmark_source"])

    completed = trades[trades["exit_exec_date"].notna()].copy()
    winners = completed[completed["gross_return_pct"] > 0]
    losers = completed[completed["gross_return_pct"] <= 0]

    rows: list[dict[str, object]] = []

    def add(section: str, metric: str, value):
        rows.append({"section": section, "metric": metric, "value": value})

    add("core", "benchmark_source", benchmark_source)
    add("core", "screen_pass_rate_mean", round(float(daily["screen_pass_rate"].mean()), 6))
    add("core", "entry_trigger_rate_mean", round(float(daily["entry_trigger_rate"].mean()), 6))
    add("core", "avg_held_count", round(float(daily["held_count"].mean()), 4))
    add("core", "avg_gross_exposure", round(float(daily["gross_exposure"].mean()), 6))
    add("core", "avg_turnover", round(float(daily["turnover"].mean()), 6))
    add("core", "avg_hold_days", round(float(completed["hold_days"].mean()), 4) if not completed.empty else 0.0)
    add("core", "win_rate", round(float((completed["gross_return_pct"] > 0).mean()), 6) if not completed.empty else 0.0)
    add("core", "avg_winner_pct", round(float(winners["gross_return_pct"].mean()), 4) if not winners.empty else 0.0)
    add("core", "avg_loser_pct", round(float(losers["gross_return_pct"].mean()), 4) if not losers.empty else 0.0)

    funnel = {
        "screened_names_avg": float(daily["screen_pass_count"].mean()),
        "entry_names_avg": float(daily["entry_trigger_count"].mean()),
        "held_names_avg": float(daily["held_count"].mean()),
        "exit_names_avg": float(daily["exit_count"].mean()),
        "selection_drop_avg": float(daily["selection_drop_count"].mean()),
    }
    for metric, value in funnel.items():
        add("funnel", metric, round(value, 4))

    if not completed.empty:
        exit_dist = completed["exit_reason"].value_counts(normalize=True).sort_index()
        for reason, value in exit_dist.items():
            add("exit_reason_distribution", reason, round(float(value), 6))

    regime_summary = (
        daily.groupby("vix_regime", dropna=False)
        .agg(
            days=("date", "count"),
            avg_screen_pass_count=("screen_pass_count", "mean"),
            avg_entry_count=("entry_trigger_count", "mean"),
            avg_held_count=("held_count", "mean"),
            avg_gross_exposure=("gross_exposure", "mean"),
            total_portfolio_return=("portfolio_return", "sum"),
            total_alpha_return=("alpha_return", "sum"),
        )
        .reset_index()
    )
    for _, row in regime_summary.iterrows():
        for metric in regime_summary.columns[1:]:
            add(f"regime:{row['vix_regime']}", metric, round(float(row[metric]), 6))

    benchmark_table = (
        completed.groupby("benchmark_source", dropna=False)
        .agg(
            trades=("ticker", "count"),
            avg_trade_return=("gross_return_pct", "mean"),
            avg_alpha=("alpha_pct_same_window", "mean"),
        )
        .reset_index()
        if not completed.empty
        else pd.DataFrame([{"benchmark_source": benchmark_source, "trades": 0, "avg_trade_return": 0.0, "avg_alpha": 0.0}])
    )
    for _, row in benchmark_table.iterrows():
        add(f"benchmark_source:{row['benchmark_source']}", "trades", int(row["trades"]))
        add(f"benchmark_source:{row['benchmark_source']}", "avg_trade_return", round(float(row["avg_trade_return"]), 6))
        add(f"benchmark_source:{row['benchmark_source']}", "avg_alpha", round(float(row["avg_alpha"]), 6))

    if comparison_metrics:
        for strategy_name, metrics in comparison_metrics.items():
            for metric_name, metric_value in metrics.items():
                add(f"comparison:{strategy_name}", metric_name, metric_value)

    return pd.DataFrame(rows)


def build_superstock_diagnostics(
    data: dict,
    max_positions: int = MAX_POSITIONS,
    pipeline: dict[str, object] | None = None,
    comparison_metrics: dict | None = None,
    start_date: str | pd.Timestamp | None = None,
) -> dict:
    pipeline = pipeline or build_superstock_pipeline(data, max_positions=max_positions)

    trade_attribution = build_superstock_trade_attribution(
        data,
        pipeline["entries"],
        pipeline["exits"],
        pipeline["positions"],
        pipeline["weights"],
    )
    daily_diagnostics = build_superstock_daily_diagnostics(
        data,
        pipeline["screen"],
        pipeline["entries"],
        pipeline["exits"],
        pipeline["positions"],
        pipeline["weights"],
    )
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        daily_diagnostics = daily_diagnostics[daily_diagnostics["date"] >= start_ts].reset_index(drop=True)
        if not trade_attribution.empty:
            entry_exec_dates = pd.to_datetime(trade_attribution["entry_exec_date"], errors="coerce")
            trade_attribution = trade_attribution[entry_exec_dates >= start_ts].reset_index(drop=True)

    diag = {
        **pipeline,
        "daily_diagnostics": daily_diagnostics,
        "trade_attribution": trade_attribution,
        "metadata": {
            "family": "superstock",
            "max_positions": max_positions,
            "benchmark_source": pipeline["weekly"].benchmark_source,
            "date_start": str(data["close"].index[0].date()),
            "date_end": str(data["close"].index[-1].date()),
        },
    }
    diag["summary"] = summarize_superstock_diagnostics(diag, comparison_metrics=comparison_metrics)
    return diag
