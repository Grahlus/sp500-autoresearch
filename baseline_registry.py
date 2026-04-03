from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiment_store import compute_config_hash


BASELINES_DIR = Path("baselines")
BASELINES_INDEX = BASELINES_DIR / "index.json"
VERIFIED_BASELINES_DIR = BASELINES_DIR / "verified"


def load_baseline_registry(index_path: Path = BASELINES_INDEX) -> list[dict[str, Any]]:
    return json.loads(index_path.read_text())


def _verified_baseline_path(name: str, baselines_dir: Path = VERIFIED_BASELINES_DIR) -> Path:
    return baselines_dir / f"{name}.json"


def load_verified_baseline_result(name: str, baselines_dir: Path = VERIFIED_BASELINES_DIR) -> dict[str, Any] | None:
    path = _verified_baseline_path(name, baselines_dir=baselines_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_verified_baseline_result(
    name: str,
    result: dict[str, Any],
    baselines_dir: Path = VERIFIED_BASELINES_DIR,
) -> None:
    baselines_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "verified_at": result["spec"]["timestamp_utc"],
        "status": result["status"],
        "experiment_id": result["spec"]["experiment_id"],
        "strategy_family": result["spec"]["strategy_family"],
        "config_hash": result["spec"]["config_hash"],
        "dataset_id": result["spec"].get("dataset_id"),
        "data_start": result["spec"].get("data_start"),
        "data_end": result["spec"].get("data_end"),
        "split": result["spec"].get("split", "walk-forward"),
        "benchmark_source": result["spec"].get("benchmark_source"),
        "git_commit": result["spec"].get("git_commit"),
        "family_version": result["spec"].get("family_version"),
        "objective_score": result.get("objective_score"),
        "verified_metrics": result.get("metrics", {}),
        "robustness": result.get("robustness", {}),
        "runtime_seconds": result.get("runtime_seconds"),
        "artifacts": result.get("artifacts", {}),
    }
    _verified_baseline_path(name, baselines_dir=baselines_dir).write_text(json.dumps(payload, indent=2, sort_keys=True))


def get_baseline(
    name: str,
    *,
    index_path: Path = BASELINES_INDEX,
    verified_dir: Path = VERIFIED_BASELINES_DIR,
) -> dict[str, Any]:
    for record in load_baseline_registry(index_path=index_path):
        if record.get("name") == name:
            record = dict(record)
            record["config_hash"] = record.get("config_hash") or compute_config_hash(record["family"], record["config"])
            record["config_source"] = record.get("config_source", record.get("baseline_config_source"))
            if "historical_metrics" not in record:
                note = record.get("historical_provenance", {}).get("starting_point_champion_note", {})
                record["historical_metrics"] = {
                    "val": note.get("reported_val"),
                    "train": note.get("reported_train"),
                    "oos": note.get("reported_oos"),
                    "survived_unseen_trading_days": note.get("reported_survived_unseen_days"),
                }
            if "historical_metrics_note" not in record:
                record["historical_metrics_note"] = (
                    "Historical note metrics are provenance only and are not treated as verified current-engine "
                    "results until this baseline is reproduced."
                )
            record["metrics_verified_by_current_engine"] = None
            verified = load_verified_baseline_result(name, baselines_dir=verified_dir)
            if verified is not None:
                record["verified_metrics"] = verified.get("verified_metrics")
                record["metrics_verified_by_current_engine"] = verified.get("verified_metrics")
                record["verified_run"] = verified
                comparison_policy = dict(record.get("comparison_policy", {}))
                comparison_policy["metrics_verification_status"] = "verified_current_engine"
                comparison_policy["comparison_eligible"] = True
                comparison_policy["comparison_rule"] = (
                    "Use verified current-engine baseline metrics for exact comparison when dataset_id, split, and "
                    "benchmark_source match. Otherwise treat as partial comparison."
                )
                record["comparison_policy"] = comparison_policy
            return record
    raise ValueError(f"Unknown baseline '{name}'.")


def reproduce_baseline(
    name: str,
    *,
    data: dict[str, Any] | None = None,
    persist: bool = True,
    base_dir: str = "experiments",
    index_path: Path = BASELINES_INDEX,
    verified_dir: Path = VERIFIED_BASELINES_DIR,
) -> dict[str, Any]:
    from experiment_runner import run_single_experiment

    baseline = get_baseline(name, index_path=index_path, verified_dir=verified_dir)
    result = run_single_experiment(
        baseline["family"],
        baseline["config"],
        data=data,
        persist=persist,
        base_dir=base_dir,
    )
    if result.status in {"success", "no_trades", "duplicate"}:
        verified_payload = {
            "spec": {
                "timestamp_utc": result.spec.timestamp_utc,
                "experiment_id": result.spec.experiment_id,
                "strategy_family": result.spec.strategy_family,
                "config_hash": result.spec.config_hash,
                "dataset_id": result.spec.dataset_id,
                "data_start": result.spec.data_start,
                "data_end": result.spec.data_end,
                "split": result.spec.split,
                "benchmark_source": result.spec.benchmark_source,
                "git_commit": result.spec.git_commit,
                "family_version": result.spec.family_version,
            },
            "status": result.status,
            "objective_score": result.objective_score,
            "metrics": result.metrics,
            "robustness": result.robustness,
            "runtime_seconds": result.runtime_seconds,
            "artifacts": result.artifacts,
        }
        save_verified_baseline_result(name, verified_payload, baselines_dir=verified_dir)
    return {
        "baseline": get_baseline(name, index_path=index_path, verified_dir=verified_dir),
        "result": result,
    }
