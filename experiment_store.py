from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


INDEX_COLUMNS = [
    "experiment_id",
    "timestamp_utc",
    "strategy_family",
    "config_hash",
    "status",
    "objective_score",
    "sharpe",
    "calmar",
    "total_return",
    "annual_return",
    "max_drawdown",
    "trades_per_year",
    "turnover",
    "exposure",
    "final_value",
    "trade_count",
    "avg_hold_days",
    "sharpe_min",
    "negative_windows",
    "viable",
    "benchmark_source",
    "dataset_id",
    "data_start",
    "data_end",
    "git_commit",
    "family_version",
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
    "baseline_sharpe_delta",
    "result_dir",
    "error_message",
]

PROPOSAL_INDEX_COLUMNS = [
    "proposal_id",
    "timestamp_utc",
    "strategy_families",
    "source_batch_ids",
    "objective_name",
    "baseline_name",
    "seed",
    "exploration_fraction",
    "exploitation_fraction",
    "max_experiments",
    "status",
    "proposal_dir",
]


def compute_config_hash(family: str, params: dict[str, Any]) -> str:
    canonical = json.dumps({"family": family, "config": params}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def init_store(base_dir: str = "experiments") -> None:
    base = Path(base_dir)
    (base / "runs").mkdir(parents=True, exist_ok=True)
    (base / "proposals").mkdir(parents=True, exist_ok=True)
    index_path = base / "index.csv"
    if not index_path.exists():
        pd.DataFrame(columns=INDEX_COLUMNS).to_csv(index_path, index=False)
    proposal_index_path = base / "proposals" / "index.csv"
    if not proposal_index_path.exists():
        pd.DataFrame(columns=PROPOSAL_INDEX_COLUMNS).to_csv(proposal_index_path, index=False)


def load_results_index(base_dir: str = "experiments") -> pd.DataFrame:
    init_store(base_dir)
    return pd.read_csv(Path(base_dir) / "index.csv")


def load_proposals_index(base_dir: str = "experiments") -> pd.DataFrame:
    init_store(base_dir)
    return pd.read_csv(Path(base_dir) / "proposals" / "index.csv")


def _to_jsonable(value: Any):
    if is_dataclass(value):
        return asdict(value)
    return value


def _result_dir(base_dir: str, experiment_id: str) -> Path:
    return Path(base_dir) / "runs" / experiment_id


def _flatten_result(result: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    spec = result["spec"]
    metrics = result.get("metrics", {})
    robustness = result.get("robustness", {})
    baseline = result.get("baseline_comparison") or {}
    row = {
        "experiment_id": spec.get("experiment_id"),
        "timestamp_utc": spec.get("timestamp_utc"),
        "strategy_family": spec.get("strategy_family", spec.get("family")),
        "config_hash": spec.get("config_hash"),
        "status": result.get("status"),
        "objective_score": result.get("objective_score", result.get("score")),
        "sharpe": metrics.get("sharpe"),
        "calmar": metrics.get("calmar"),
        "total_return": metrics.get("total_return"),
        "annual_return": metrics.get("annual_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "trades_per_year": metrics.get("trades_per_year"),
        "turnover": metrics.get("turnover"),
        "exposure": metrics.get("exposure"),
        "final_value": metrics.get("final_value"),
        "trade_count": metrics.get("trade_count"),
        "avg_hold_days": metrics.get("avg_hold_days"),
        "sharpe_min": metrics.get("sharpe_min"),
        "negative_windows": robustness.get("negative_windows"),
        "viable": robustness.get("viable"),
        "benchmark_source": spec.get("benchmark_source"),
        "dataset_id": spec.get("dataset_id"),
        "data_start": spec.get("data_start"),
        "data_end": spec.get("data_end"),
        "git_commit": spec.get("git_commit"),
        "family_version": spec.get("family_version"),
        "baseline_name": baseline.get("baseline_name"),
        "comparison_status": baseline.get("comparison_status"),
        "baseline_verified": baseline.get("baseline_verified"),
        "baseline_metric_source": baseline.get("baseline_metric_source"),
        "baseline_comparison_kind": baseline.get("comparison_kind"),
        "baseline_comparison_eligible": baseline.get("comparison_eligible"),
        "delta_sharpe": baseline.get("delta_sharpe"),
        "delta_calmar": baseline.get("delta_calmar"),
        "delta_return": baseline.get("delta_return"),
        "beats_baseline_objective": baseline.get("beats_baseline_objective"),
        "beats_baseline_guardrails": baseline.get("beats_baseline_guardrails"),
        "baseline_sharpe_delta": baseline.get("sharpe_delta"),
        "result_dir": str(result_dir),
        "error_message": result.get("error_message", result.get("error")),
    }
    return row


def _serialize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    payload = dict(spec)
    payload["strategy_family"] = payload.get("family")
    payload["config"] = payload.get("params")
    return payload


def _serialize_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload["spec"] = _serialize_spec(payload["spec"])
    if "objective_score" not in payload and "score" in payload:
        payload["objective_score"] = payload["score"]
    if "error_message" not in payload and "error" in payload:
        payload["error_message"] = payload["error"]
    return payload


def has_experiment_result(config_hash: str, family: str | None = None, base_dir: str = "experiments") -> bool:
    index = load_results_index(base_dir)
    if index.empty:
        return False
    matches = index[index["config_hash"] == config_hash]
    if family is not None:
        matches = matches[matches["strategy_family"] == family]
    if matches.empty:
        return False
    return bool((matches["status"] == "success").any())


def load_experiment_result(config_hash: str, family: str | None = None, base_dir: str = "experiments") -> dict[str, Any] | None:
    index = load_results_index(base_dir)
    if index.empty:
        return None
    matches = index[index["config_hash"] == config_hash]
    if family is not None:
        matches = matches[matches["strategy_family"] == family]
    matches = matches[matches["status"] == "success"]
    if matches.empty:
        return None
    result_dir = Path(matches.iloc[-1]["result_dir"])
    result_path = result_dir / "result.json"
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text())


def save_experiment_result(result: dict[str, Any], base_dir: str = "experiments") -> None:
    init_store(base_dir)
    result = _serialize_result(result)
    spec = result["spec"]
    experiment_id = spec["experiment_id"]
    result_dir = _result_dir(base_dir, experiment_id)
    result_dir.mkdir(parents=True, exist_ok=True)

    (result_dir / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True, default=_to_jsonable))
    (result_dir / "metrics.json").write_text(json.dumps(result.get("metrics", {}), indent=2, sort_keys=True, default=_to_jsonable))
    (result_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_to_jsonable))

    index = load_results_index(base_dir)
    row = _flatten_result(result, result_dir)
    index = pd.concat([index, pd.DataFrame([row], columns=INDEX_COLUMNS)], ignore_index=True)
    index.to_csv(Path(base_dir) / "index.csv", index=False)


def load_prior_results(family: str | None = None, base_dir: str = "experiments") -> pd.DataFrame:
    index = load_results_index(base_dir)
    if family is None or index.empty:
        return index
    return index[index["strategy_family"] == family].reset_index(drop=True)


def save_proposal_result(proposal: dict[str, Any], base_dir: str = "experiments") -> dict[str, str]:
    init_store(base_dir)
    request = proposal["request"]
    proposal_id = request["proposal_id"]
    proposal_dir = Path(base_dir) / "proposals" / proposal_id
    proposal_dir.mkdir(parents=True, exist_ok=True)

    proposal_path = proposal_dir / "proposal.json"
    summary_path = proposal_dir / "summary.json"
    candidates_path = proposal_dir / "candidate_configs.json"

    proposal_path.write_text(json.dumps(proposal, indent=2, sort_keys=True, default=_to_jsonable))
    summary_path.write_text(
        json.dumps(proposal.get("reasoning_summary", {}), indent=2, sort_keys=True, default=_to_jsonable)
    )
    candidates_path.write_text(
        json.dumps(proposal.get("candidate_configs", {}), indent=2, sort_keys=True, default=_to_jsonable)
    )

    index = load_proposals_index(base_dir)
    row = {
        "proposal_id": proposal_id,
        "timestamp_utc": request.get("timestamp_utc"),
        "strategy_families": ",".join(request.get("strategy_families", [])),
        "source_batch_ids": ",".join(request.get("source_batch_ids", [])),
        "objective_name": request.get("objective_name"),
        "baseline_name": request.get("baseline_name"),
        "seed": request.get("seed"),
        "exploration_fraction": request.get("exploration_fraction"),
        "exploitation_fraction": request.get("exploitation_fraction"),
        "max_experiments": request.get("max_experiments"),
        "status": proposal.get("status"),
        "proposal_dir": str(proposal_dir),
    }
    index = pd.concat([index, pd.DataFrame([row], columns=PROPOSAL_INDEX_COLUMNS)], ignore_index=True)
    index.to_csv(Path(base_dir) / "proposals" / "index.csv", index=False)
    return {
        "proposal_path": str(proposal_path),
        "summary_path": str(summary_path),
        "candidate_configs_path": str(candidates_path),
    }
