from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_best_results import BEST_RESULTS_COLUMNS, latest_non_empty_batch, load_best_results
from experiment_scorecards import build_family_scorecards, scorecards_to_records
from experiment_store import load_results_index


DASHBOARD_RESULT_COLUMNS = [
    "experiment_id",
    "timestamp_utc",
    "strategy_family",
    "strategy_type",
    "config_hash",
    "objective_score",
    "sharpe",
    "calmar",
    "total_return",
    "max_drawdown",
    "trades_per_year",
    "status",
    "viable",
    "baseline_name",
    "comparison_status",
    "baseline_verified",
    "baseline_metric_source",
    "baseline_comparison_kind",
    "delta_sharpe",
    "delta_calmar",
    "delta_return",
    "beats_baseline_objective",
    "beats_baseline_guardrails",
    "source_type",
    "template_id",
    "hypothesis",
    "reason_selected",
    "source_proposal_id",
    "source_idea_ids",
    "result_dir",
]

NUMERIC_COLUMNS = [
    "objective_score",
    "sharpe",
    "calmar",
    "total_return",
    "max_drawdown",
    "trades_per_year",
    "delta_sharpe",
    "delta_calmar",
    "delta_return",
]

BOOL_COLUMNS = [
    "viable",
    "baseline_verified",
    "baseline_comparison_eligible",
    "beats_baseline_objective",
    "beats_baseline_guardrails",
]


@dataclass(frozen=True)
class BestResultsDashboard:
    generated_at_utc: str
    base_dir: str
    ranking_policy: str
    counts: dict[str, Any]
    top_overall: list[dict[str, Any]]
    top_viable: list[dict[str, Any]]
    top_baseline_beating: list[dict[str, Any]]
    top_per_family: dict[str, list[dict[str, Any]]]
    latest_non_empty_batch: dict[str, Any] | None
    family_scorecards: dict[str, dict[str, Any]]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"true", "1", "yes"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            return value
    return value


def _normalize_results(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in set(BEST_RESULTS_COLUMNS + DASHBOARD_RESULT_COLUMNS + BOOL_COLUMNS):
        if column not in normalized.columns:
            normalized[column] = None
    for column in NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for column in BOOL_COLUMNS:
        normalized[column] = normalized[column].map(_truthy)
    return normalized


def _records(frame: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    selected = frame.head(limit) if limit is not None else frame
    selected = selected.copy()
    for column in DASHBOARD_RESULT_COLUMNS:
        if column not in selected.columns:
            selected[column] = None
    return _json_safe(selected[DASHBOARD_RESULT_COLUMNS].to_dict("records"))


def _sort_dashboard_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sortable = frame.copy()
    return sortable.sort_values(
        by=["viable", "objective_score", "sharpe", "calmar", "total_return", "timestamp_utc", "experiment_id"],
        ascending=[False, False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _families_from_index(index: pd.DataFrame, additional_families: list[str] | None) -> list[str]:
    families = set(additional_families or [])
    if not index.empty and "strategy_family" in index.columns:
        families.update(
            str(family)
            for family in index["strategy_family"].dropna().astype(str).tolist()
            if str(family).strip() and str(family).strip().lower() != "nan"
        )
    return sorted(families)


def build_best_results_dashboard(
    *,
    base_dir: str = "experiments",
    overall_limit: int = 20,
    viable_limit: int = 20,
    baseline_limit: int = 20,
    per_family_limit: int = 10,
    families: list[str] | None = None,
) -> BestResultsDashboard:
    index = load_results_index(base_dir)
    best = _normalize_results(load_best_results(base_dir))
    top_overall = _sort_dashboard_results(best)
    top_viable = _sort_dashboard_results(best[best["viable"]].copy())
    baseline_mask = best["beats_baseline_objective"] | best["beats_baseline_guardrails"]
    top_baseline = _sort_dashboard_results(best[baseline_mask].copy())
    top_per_family = {
        family: _records(_sort_dashboard_results(frame), limit=per_family_limit)
        for family, frame in top_overall.groupby(top_overall["strategy_family"].astype(str), sort=True)
        if family and family.lower() != "nan"
    }

    scorecard_families = _families_from_index(index, families)
    scorecards = build_family_scorecards(families=scorecard_families, base_dir=base_dir) if scorecard_families else {}
    scorecard_records = _json_safe(scorecards_to_records(scorecards))
    family_counts = (
        index["strategy_family"].dropna().astype(str).value_counts().sort_index().to_dict()
        if not index.empty and "strategy_family" in index.columns
        else {}
    )
    viable_count = int(_normalize_results(index)["viable"].sum()) if not index.empty else 0

    return BestResultsDashboard(
        generated_at_utc=datetime.now(UTC).isoformat(),
        base_dir=str(base_dir),
        ranking_policy="viable-first, then objective_score/sharpe/calmar/total_return descending",
        counts={
            "official_result_rows": int(len(index)),
            "eligible_result_rows": int(len(best)),
            "viable_result_rows": viable_count,
            "baseline_beating_rows": int(len(top_baseline)),
            "families": family_counts,
        },
        top_overall=_records(top_overall, limit=overall_limit),
        top_viable=_records(top_viable, limit=viable_limit),
        top_baseline_beating=_records(top_baseline, limit=baseline_limit),
        top_per_family=top_per_family,
        latest_non_empty_batch=latest_non_empty_batch(base_dir=base_dir),
        family_scorecards=scorecard_records,
    )


def dashboard_to_dict(dashboard: BestResultsDashboard) -> dict[str, Any]:
    return _json_safe(asdict(dashboard))


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(none)"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [_format_value(row.get(column)).replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_dashboard_markdown(dashboard: BestResultsDashboard) -> str:
    result_columns = [
        "experiment_id",
        "strategy_family",
        "objective_score",
        "sharpe",
        "calmar",
        "total_return",
        "max_drawdown",
        "viable",
        "baseline_name",
        "comparison_status",
    ]
    scorecard_columns = [
        "family",
        "total_experiments",
        "viable_rate",
        "win_rate_vs_baseline",
        "best_objective_score",
        "median_objective_score",
        "recent_objective_trend",
        "last_improvement_timestamp",
        "exploration_budget_recommendation",
        "exploitation_budget_recommendation",
    ]
    scorecard_rows = [record for _, record in sorted(dashboard.family_scorecards.items())]
    latest = dashboard.latest_non_empty_batch or {}
    lines = [
        "# Best Results Dashboard",
        "",
        f"generated_at_utc={dashboard.generated_at_utc}",
        f"base_dir={dashboard.base_dir}",
        f"ranking_policy={dashboard.ranking_policy}",
        "",
        f"## Top {len(dashboard.top_overall)} Overall",
        _markdown_table(dashboard.top_overall, result_columns),
        "",
        f"## Top {len(dashboard.top_viable)} Viable",
        _markdown_table(dashboard.top_viable, result_columns),
        "",
        "## Top Baseline-Beating",
        _markdown_table(dashboard.top_baseline_beating, result_columns),
        "",
        "## Top Per Family",
    ]
    if dashboard.top_per_family:
        for family, rows in sorted(dashboard.top_per_family.items()):
            lines.extend(["", f"### {family}", _markdown_table(rows, result_columns)])
    else:
        lines.append("(none)")
    lines.extend(
        [
            "",
            "## Latest Non-Empty Batch",
            "(none)"
            if not latest
            else "\n".join(f"{key}={latest.get(key)}" for key in ("batch_id", "executed_count", "leaderboard_path", "summary_path")),
            "",
            "## Family Scorecards",
            _markdown_table(scorecard_rows, scorecard_columns),
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, prefix=path.name + ".", suffix=".tmp", encoding="utf-8") as handle:
        handle.write(content)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def save_dashboard_reports(dashboard: BestResultsDashboard, *, reports_dir: str = "reports") -> dict[str, str]:
    reports = Path(reports_dir)
    payload = dashboard_to_dict(dashboard)
    best_results_json = reports / "best_results.json"
    best_results_md = reports / "best_results.md"
    family_scorecards_json = reports / "family_scorecards.json"
    _atomic_write_text(best_results_json, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    _atomic_write_text(best_results_md, format_dashboard_markdown(dashboard))
    _atomic_write_text(
        family_scorecards_json,
        json.dumps(
            {
                "timestamp_utc": dashboard.generated_at_utc,
                "families": dashboard.family_scorecards,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
    )
    return {
        "best_results_json": str(best_results_json),
        "best_results_md": str(best_results_md),
        "family_scorecards_json": str(family_scorecards_json),
    }
