from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_store import init_store, load_results_index


BEST_RESULTS_COLUMNS = [
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
    "baseline_comparison_eligible",
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

ELIGIBLE_STATUSES = {"success", "no_trades"}
OPTIONAL_TEXT_COLUMNS = [
    "strategy_type",
    "baseline_name",
    "comparison_status",
    "source_type",
    "template_id",
    "hypothesis",
    "reason_selected",
    "result_dir",
]


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


def ensure_global_results_index(base_dir: str = "experiments") -> Path:
    init_store(base_dir)
    return Path(base_dir) / "index.csv"


def _empty_best_results_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=BEST_RESULTS_COLUMNS)


def _prepare_rankable_results(index: pd.DataFrame) -> pd.DataFrame:
    if index.empty:
        return _empty_best_results_frame()

    eligible = index[index["status"].isin(ELIGIBLE_STATUSES)].copy()
    if eligible.empty:
        return _empty_best_results_frame()

    eligible["objective_score"] = pd.to_numeric(eligible["objective_score"], errors="coerce")
    eligible["sharpe"] = pd.to_numeric(eligible["sharpe"], errors="coerce")
    eligible["calmar"] = pd.to_numeric(eligible["calmar"], errors="coerce")
    eligible["total_return"] = pd.to_numeric(eligible["total_return"], errors="coerce")
    eligible["viable"] = eligible["viable"].map(_truthy)
    eligible = eligible[eligible["objective_score"].notna()].copy()
    if eligible.empty:
        return _empty_best_results_frame()
    return eligible


def rank_best_results(
    index: pd.DataFrame,
    *,
    require_viable: bool = False,
    require_baseline_beating: bool = False,
) -> pd.DataFrame:
    ranked_source = _prepare_rankable_results(index)
    if ranked_source.empty:
        return _empty_best_results_frame()

    ranked = ranked_source
    if require_viable:
        ranked = ranked[ranked["viable"]]
    if require_baseline_beating:
        baseline_mask = ranked["beats_baseline_objective"].map(_truthy) | ranked["beats_baseline_guardrails"].map(_truthy)
        ranked = ranked[baseline_mask]
    if ranked.empty:
        return _empty_best_results_frame()

    ranked = ranked.sort_values(
        by=[
            "viable",
            "objective_score",
            "sharpe",
            "calmar",
            "total_return",
            "timestamp_utc",
            "experiment_id",
        ],
        ascending=[False, False, False, False, False, True, True],
        kind="mergesort",
    )
    for column in BEST_RESULTS_COLUMNS:
        if column not in ranked.columns:
            ranked[column] = None
    best = ranked[BEST_RESULTS_COLUMNS].reset_index(drop=True)
    for column in OPTIONAL_TEXT_COLUMNS:
        if column in best.columns:
            best[column] = best[column].fillna("")
    return best


def load_best_results(base_dir: str = "experiments") -> pd.DataFrame:
    ensure_global_results_index(base_dir)
    return rank_best_results(load_results_index(base_dir))


def top_results_overall(limit: int = 20, base_dir: str = "experiments") -> pd.DataFrame:
    return load_best_results(base_dir).head(limit).reset_index(drop=True)


def top_results_per_family(limit: int = 10, base_dir: str = "experiments") -> dict[str, pd.DataFrame]:
    ranked = load_best_results(base_dir)
    if ranked.empty:
        return {}
    return {
        family: group.head(limit).reset_index(drop=True)
        for family, group in ranked.groupby("strategy_family", sort=True)
    }


def _batch_dirs(base_dir: str) -> list[Path]:
    batch_root = Path(base_dir) / "batches"
    if not batch_root.exists():
        return []
    return sorted([path for path in batch_root.iterdir() if path.is_dir()], key=lambda path: path.name, reverse=True)


def _batch_executed_count(batch_dir: Path) -> int:
    summary_path = batch_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = {}
        value = summary.get("total_executed")
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    leaderboard_path = batch_dir / "leaderboard.csv"
    if not leaderboard_path.exists():
        return 0
    leaderboard = pd.read_csv(leaderboard_path)
    if leaderboard.empty or "status" not in leaderboard.columns:
        return 0
    return int(leaderboard["status"].isin(ELIGIBLE_STATUSES).sum())


def latest_non_empty_batch(base_dir: str = "experiments") -> dict[str, Any] | None:
    for batch_dir in _batch_dirs(base_dir):
        executed = _batch_executed_count(batch_dir)
        if executed <= 0:
            continue
        leaderboard_path = batch_dir / "leaderboard.csv"
        summary_path = batch_dir / "summary.json"
        return {
            "batch_id": batch_dir.name,
            "executed_count": executed,
            "leaderboard_path": str(leaderboard_path) if leaderboard_path.exists() else None,
            "summary_path": str(summary_path) if summary_path.exists() else None,
        }
    return None


def save_best_results_reports(
    *,
    base_dir: str = "experiments",
    overall_limit: int = 20,
    per_family_limit: int = 10,
) -> dict[str, str]:
    reports_dir = Path(base_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    overall = top_results_overall(limit=overall_limit, base_dir=base_dir)
    overall_path = reports_dir / "top_results_overall.csv"
    overall.to_csv(overall_path, index=False)

    family_frames = top_results_per_family(limit=per_family_limit, base_dir=base_dir)
    family_rows: list[pd.DataFrame] = []
    for family, frame in family_frames.items():
        family_frame = frame.copy()
        family_frame.insert(0, "family_rank", range(1, len(family_frame) + 1))
        family_frame.insert(0, "family", family)
        family_rows.append(family_frame)
    per_family = pd.concat(family_rows, ignore_index=True) if family_rows else _empty_best_results_frame()
    per_family_path = reports_dir / "top_results_per_family.csv"
    per_family.to_csv(per_family_path, index=False)

    latest = latest_non_empty_batch(base_dir=base_dir) or {}
    latest_path = reports_dir / "latest_non_empty_batch.json"
    latest_path.write_text(json.dumps(latest, indent=2, sort_keys=True))

    return {
        "overall_path": str(overall_path),
        "per_family_path": str(per_family_path),
        "latest_batch_path": str(latest_path),
    }
