from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from proposal_artifacts import compact_request, compact_summary_for_disk

HOT_INDEX_FILENAME = "hot_index.sqlite3"
HOT_INDEX_EXPERIMENT_LIMIT_PER_FAMILY = 256
HOT_INDEX_BOOTSTRAP_LIMIT_PER_FAMILY = 128
HOT_INDEX_PIN_TOP_PER_FAMILY = 20  # top-N by objective_score that are never evicted by recency
HOT_INDEX_BATCH_LIMIT = 100
HOT_INDEX_BATCH_JSON_LIMIT_BYTES = 64_000
HOT_INDEX_PROPOSAL_LIMIT = 500
BATCH_SUMMARY_FULL_PARSE_LIMIT_BYTES = 32_000_000
BATCH_SUMMARY_WINDOW_BYTES = 8_000_000
BATCH_ARTIFACT_CLEANUP_THRESHOLD_BYTES = 25_000_000


def _hot_index_path(base_dir: str) -> Path:
    return Path(base_dir) / HOT_INDEX_FILENAME


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _connect(base_dir: str) -> sqlite3.Connection:
    path = _hot_index_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _compute_config_hash(family: str, params: dict[str, Any]) -> str:
    canonical = json.dumps({"family": family, "config": params}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _compact_runtime_decision_reference(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    keep_keys = {
        "decision_id",
        "timestamp_utc",
        "status",
        "cycle_mode",
        "max_experiments",
        "family_budgets",
        "confirmation_family_budgets",
        "large_search_mode",
        "winner_family",
        "winner_config_hash",
        "winner_experiment_id",
        "winner_promotion_status",
        "promotion_state",
        "confirmation_required",
        "confirmation_reason",
        "confirmation_batch_id",
        "targeted_follow_up_required",
        "targeted_follow_up_reason",
        "targeted_follow_up_type",
        "holdout_check_required",
        "holdout_check_type",
        "holdout_check_status",
        "holdout_check_outcome",
        "holdout_check_scope",
        "new_idea_budget",
        "refinement_budget",
        "confirmation_budget",
        "report_path",
    }
    return {key: payload.get(key) for key in sorted(keep_keys) if key in payload}


def _compact_batch_proposal_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in (
        "proposal_id",
        "proposal_quality",
        "runtime_memory_caps",
        "runtime_decision_report",
        "source_proposal_id",
    ):
        if key in payload:
            compact[key] = payload.get(key)
    if isinstance(payload.get("runtime_decision"), dict):
        compact["runtime_decision"] = _compact_runtime_decision_reference(payload.get("runtime_decision"))
    return compact


def _compact_batch_summary_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    compact = {
        key: payload.get(key)
        for key in (
            "batch_id",
            "timestamp_utc",
            "status",
            "total_sampled",
            "total_executed",
            "total_skipped",
            "total_failed",
            "status_counts",
            "family_summary",
            "throughput_diagnostics",
            "top_results",
            "viable_count",
            "baseline_beating_count",
            "artifact_compaction",
            "leaderboard_path",
            "raw_results_path",
            "summary_path",
        )
        if key in payload
    }
    compact["proposal_metadata"] = _compact_batch_proposal_metadata(payload.get("proposal_metadata") or {})
    return compact


def _compact_if_oversized(payload: Any, *, limit_bytes: int = HOT_INDEX_BATCH_JSON_LIMIT_BYTES) -> Any:
    if not isinstance(payload, dict):
        return payload
    try:
        if len(_json_dumps(payload).encode("utf-8")) <= int(limit_bytes):
            return payload
    except Exception:
        pass
    return _compact_batch_summary_payload(payload)


def compact_batch_summary_for_disk(payload: dict[str, Any], *, limit_bytes: int = HOT_INDEX_BATCH_JSON_LIMIT_BYTES) -> dict[str, Any]:
    """Return a bounded batch summary suitable for JSON debug/export output."""
    compacted = _compact_if_oversized(payload, limit_bytes=limit_bytes)
    return dict(compacted) if isinstance(compacted, dict) else {}


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


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS experiment_summaries (
            experiment_id TEXT PRIMARY KEY,
            timestamp_utc TEXT,
            strategy_family TEXT,
            config_hash TEXT,
            batch_id TEXT,
            status TEXT,
            objective_score REAL,
            viable INTEGER,
            source_type TEXT,
            template_id TEXT,
            strategy_type TEXT,
            parent_config_hash TEXT,
            lineage_root_config_hash TEXT,
            branch_root_config_hash TEXT,
            promotion_state TEXT,
            holdout_check_type TEXT,
            holdout_check_status TEXT,
            holdout_check_outcome TEXT,
            idea_kind TEXT,
            idea_source TEXT,
            result_dir TEXT,
            updated_at TEXT,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_experiment_summaries_family_timestamp
            ON experiment_summaries(strategy_family, timestamp_utc DESC, experiment_id DESC);
        CREATE INDEX IF NOT EXISTS idx_experiment_summaries_batch_timestamp
            ON experiment_summaries(batch_id, timestamp_utc DESC, experiment_id DESC);

        CREATE TABLE IF NOT EXISTS batch_summaries (
            batch_id TEXT PRIMARY KEY,
            timestamp_utc TEXT,
            total_sampled INTEGER,
            total_executed INTEGER,
            total_skipped INTEGER,
            total_failed INTEGER,
            status_counts_json TEXT,
            family_summary_json TEXT,
            proposal_metadata_json TEXT,
            throughput_json TEXT,
            summary_json TEXT NOT NULL,
            updated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_batch_summaries_timestamp
            ON batch_summaries(timestamp_utc DESC, batch_id DESC);

        CREATE TABLE IF NOT EXISTS proposal_summaries (
            proposal_id TEXT PRIMARY KEY,
            timestamp_utc TEXT,
            status TEXT,
            strategy_families_json TEXT,
            source_batch_ids_json TEXT,
            objective_name TEXT,
            baseline_name TEXT,
            seed INTEGER,
            exploration_fraction REAL,
            exploitation_fraction REAL,
            max_experiments INTEGER,
            candidate_count INTEGER,
            candidate_family_counts_json TEXT,
            source_idea_ids_json TEXT,
            proposal_quality_json TEXT,
            reasoning_summary_json TEXT,
            request_json TEXT,
            proposal_dir TEXT,
            proposal_path TEXT,
            summary_path TEXT,
            candidate_configs_path TEXT,
            json_export_status TEXT DEFAULT 'active',
            json_export_deleted_at TEXT,
            updated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_proposal_summaries_timestamp
            ON proposal_summaries(timestamp_utc DESC, proposal_id DESC);
        CREATE INDEX IF NOT EXISTS idx_proposal_summaries_status_timestamp
            ON proposal_summaries(status, timestamp_utc DESC, proposal_id DESC);

        CREATE TABLE IF NOT EXISTS proposal_candidates (
            proposal_id TEXT NOT NULL,
            candidate_index INTEGER NOT NULL,
            family TEXT,
            family_candidate_index INTEGER,
            config_hash TEXT,
            source_type TEXT,
            template_id TEXT,
            strategy_type TEXT,
            proposal_role TEXT,
            exploration_mode TEXT,
            parent_config_hash TEXT,
            near_duplicate_of TEXT,
            novelty_score REAL,
            selection_score REAL,
            duplicate_risk TEXT,
            dead_zone_risk REAL,
            is_new_idea INTEGER,
            is_uncommon_idea INTEGER,
            source_idea_ids_json TEXT,
            config_json TEXT NOT NULL,
            metadata_json TEXT,
            updated_at TEXT,
            PRIMARY KEY(proposal_id, candidate_index),
            FOREIGN KEY(proposal_id) REFERENCES proposal_summaries(proposal_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_proposal_candidates_proposal_family
            ON proposal_candidates(proposal_id, family, family_candidate_index);
        CREATE INDEX IF NOT EXISTS idx_proposal_candidates_config_hash
            ON proposal_candidates(config_hash);
        CREATE INDEX IF NOT EXISTS idx_proposal_candidates_family_template
            ON proposal_candidates(family, template_id, proposal_id);

        CREATE TABLE IF NOT EXISTS summary_items (
            item_type TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            timestamp_utc TEXT,
            source_batch_id TEXT,
            payload_json TEXT NOT NULL,
            updated_at TEXT,
            PRIMARY KEY(item_type, scope_key)
        );
        CREATE INDEX IF NOT EXISTS idx_summary_items_type_updated
            ON summary_items(item_type, updated_at DESC);

        CREATE TABLE IF NOT EXISTS archive_manifest (
            item_type TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            retained_count INTEGER NOT NULL,
            archived_count INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            note TEXT,
            PRIMARY KEY(item_type, scope_key, updated_at)
        );
        """
    )


def init_hot_index(base_dir: str = "experiments") -> Path:
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
    return _hot_index_path(base_dir)


def _payload_from_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload["updated_at"] = payload.get("updated_at") or _now()
    return payload


def _bootstrap_from_archive(
    base_dir: str,
    *,
    families: Iterable[str] | None = None,
    limit_per_family: int = HOT_INDEX_BOOTSTRAP_LIMIT_PER_FAMILY,
) -> pd.DataFrame:
    index_path = Path(base_dir) / "index.csv"
    if not index_path.exists():
        return pd.DataFrame()
    family_set = {str(family).strip().lower() for family in (families or []) if str(family).strip()}
    keep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    try:
        for chunk in pd.read_csv(index_path, chunksize=2000):
            if chunk.empty or "strategy_family" not in chunk.columns:
                continue
            chunk = chunk.copy()
            chunk["strategy_family"] = chunk["strategy_family"].astype(str).str.strip().str.lower()
            if family_set:
                chunk = chunk[chunk["strategy_family"].isin(family_set)]
            if chunk.empty:
                continue
            chunk = chunk.sort_values(["timestamp_utc", "experiment_id"], kind="mergesort")
            for _, row in chunk.iterrows():
                family = str(row.get("strategy_family") or "").strip().lower()
                if not family:
                    continue
                keep[family].append(dict(row.to_dict()))
                if len(keep[family]) > limit_per_family:
                    keep[family] = keep[family][-limit_per_family:]
    except Exception:
        return pd.DataFrame()
    rows = [row for family_rows in keep.values() for row in family_rows]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "timestamp_utc" in frame.columns:
        frame = frame.sort_values(["timestamp_utc", "experiment_id"], kind="mergesort")
    return frame.reset_index(drop=True)


def _archive_is_newer(base_dir: str) -> bool:
    archive_path = Path(base_dir) / "index.csv"
    hot_path = _hot_index_path(base_dir)
    if not archive_path.exists():
        return False
    if not hot_path.exists():
        return True
    try:
        return archive_path.stat().st_mtime > (hot_path.stat().st_mtime + 0.01)
    except FileNotFoundError:
        return True


def _upsert_dataframe(conn: sqlite3.Connection, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    for _, row in frame.iterrows():
        upsert_experiment_summary(conn, dict(row.to_dict()))


def upsert_experiment_summary(
    base_dir_or_conn: str | sqlite3.Connection,
    row: dict[str, Any],
    *,
    keep_recent_per_family: int = HOT_INDEX_EXPERIMENT_LIMIT_PER_FAMILY,
) -> None:
    if isinstance(base_dir_or_conn, sqlite3.Connection):
        conn = base_dir_or_conn
        close_conn = False
    else:
        conn = _connect(base_dir_or_conn)
        close_conn = True
    try:
        _ensure_schema(conn)
        payload = _payload_from_row(row)
        family = str(payload.get("strategy_family") or "").strip().lower()
        experiment_id = str(payload.get("experiment_id") or "").strip()
        if not family or not experiment_id:
            return
        viable = payload.get("viable")
        viable_value = 1 if _truthy(viable) else 0 if viable is not None else None
        record = {
            "experiment_id": experiment_id,
            "timestamp_utc": str(payload.get("timestamp_utc") or ""),
            "strategy_family": family,
            "config_hash": str(payload.get("config_hash") or ""),
            "batch_id": str(payload.get("batch_id") or payload.get("source_batch_id") or payload.get("proposal_batch_id") or ""),
            "status": str(payload.get("status") or ""),
            "objective_score": payload.get("objective_score"),
            "viable": viable_value,
            "source_type": payload.get("source_type"),
            "template_id": payload.get("template_id"),
            "strategy_type": payload.get("strategy_type"),
            "parent_config_hash": payload.get("parent_config_hash"),
            "lineage_root_config_hash": payload.get("lineage_root_config_hash"),
            "branch_root_config_hash": payload.get("branch_root_config_hash") or payload.get("lineage_root_config_hash"),
            "promotion_state": payload.get("promotion_state"),
            "holdout_check_type": payload.get("holdout_check_type"),
            "holdout_check_status": payload.get("holdout_check_status"),
            "holdout_check_outcome": payload.get("holdout_check_outcome"),
            "idea_kind": payload.get("idea_kind"),
            "idea_source": payload.get("idea_source"),
            "result_dir": payload.get("result_dir"),
            "updated_at": payload.get("updated_at") or _now(),
            "payload_json": _json_dumps(payload),
        }
        columns = ", ".join(record.keys())
        placeholders = ", ".join(["?"] * len(record))
        conn.execute(
            f"INSERT INTO experiment_summaries ({columns}) VALUES ({placeholders}) "
            "ON CONFLICT(experiment_id) DO UPDATE SET "
            + ", ".join(f"{column}=excluded.{column}" for column in record.keys() if column != "experiment_id"),
            tuple(record.values()),
        )
        conn.commit()
        _compact_family(conn, family, keep_recent_per_family=keep_recent_per_family)
    finally:
        if close_conn:
            conn.close()


def _compact_family(
    conn: sqlite3.Connection,
    family: str,
    *,
    keep_recent_per_family: int = HOT_INDEX_EXPERIMENT_LIMIT_PER_FAMILY,
    pin_top_per_family: int = HOT_INDEX_PIN_TOP_PER_FAMILY,
) -> dict[str, int]:
    # Pin the top-N rows by objective_score so they are never evicted by recency alone.
    top_rows = conn.execute(
        """
        SELECT experiment_id
        FROM experiment_summaries
        WHERE strategy_family = ?
        ORDER BY objective_score DESC, timestamp_utc DESC, experiment_id DESC
        LIMIT ?
        """,
        (family, int(pin_top_per_family)),
    ).fetchall()
    pinned_ids = {str(row["experiment_id"]) for row in top_rows}

    cur = conn.execute(
        """
        SELECT experiment_id
        FROM experiment_summaries
        WHERE strategy_family = ?
        ORDER BY timestamp_utc DESC, experiment_id DESC
        """,
        (family,),
    )
    experiment_ids = [str(row["experiment_id"]) for row in cur.fetchall()]
    retained = experiment_ids[:keep_recent_per_family]
    # Append pinned top-scorers that fell outside the recency window.
    retained_set = set(retained)
    for pid in pinned_ids:
        if pid not in retained_set:
            retained.append(pid)
            retained_set.add(pid)
    archived = [eid for eid in experiment_ids if eid not in retained_set]
    if archived:
        conn.executemany(
            "DELETE FROM experiment_summaries WHERE experiment_id = ?",
            [(experiment_id,) for experiment_id in archived],
        )
        conn.execute(
            "INSERT OR REPLACE INTO archive_manifest (item_type, scope_key, retained_count, archived_count, updated_at, note) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "experiment_summaries",
                family,
                len(retained),
                len(archived),
                _now(),
                "compact_recent_family_window",
            ),
        )
        conn.commit()
    return {"retained_count": len(retained), "archived_count": len(archived)}


def upsert_batch_summary(base_dir: str, summary: dict[str, Any]) -> None:
    init_hot_index(base_dir)
    payload = dict(summary)
    batch_id = str(payload.get("batch_id") or "").strip()
    if not batch_id:
        return
    compact_payload = _compact_if_oversized(payload)
    compact_metadata = _compact_batch_proposal_metadata(payload.get("proposal_metadata") or {})
    record = {
        "batch_id": batch_id,
        "timestamp_utc": payload.get("timestamp_utc"),
        "total_sampled": payload.get("total_sampled"),
        "total_executed": payload.get("total_executed"),
        "total_skipped": payload.get("total_skipped"),
        "total_failed": payload.get("total_failed"),
        "status_counts_json": _json_dumps(payload.get("status_counts") or {}),
        "family_summary_json": _json_dumps(payload.get("family_summary") or {}),
        "proposal_metadata_json": _json_dumps(compact_metadata),
        "throughput_json": _json_dumps(payload.get("throughput_diagnostics") or {}),
        "summary_json": _json_dumps(compact_payload),
        "updated_at": _now(),
    }
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        columns = ", ".join(record.keys())
        placeholders = ", ".join(["?"] * len(record))
        conn.execute(
            f"INSERT INTO batch_summaries ({columns}) VALUES ({placeholders}) "
            "ON CONFLICT(batch_id) DO UPDATE SET "
            + ", ".join(f"{column}=excluded.{column}" for column in record.keys() if column != "batch_id"),
            tuple(record.values()),
        )
        conn.commit()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _read_summary_windows(path: Path, *, window_bytes: int = BATCH_SUMMARY_WINDOW_BYTES) -> str:
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(min(size, int(window_bytes)))
        if size <= int(window_bytes):
            suffix = b""
        else:
            handle.seek(max(0, size - int(window_bytes)))
            suffix = handle.read(int(window_bytes))
    if suffix:
        return prefix.decode("utf-8", errors="replace") + "\n" + suffix.decode("utf-8", errors="replace")
    return prefix.decode("utf-8", errors="replace")


def _extract_top_level_json_value(text: str, key: str) -> Any:
    decoder = json.JSONDecoder()
    patterns = (f'{{\n  "{key}":', f'\n  "{key}":')
    start = -1
    pattern = ""
    for candidate in patterns:
        start = text.find(candidate)
        if start >= 0:
            pattern = candidate
            break
    if start < 0:
        return None
    colon = text.find(":", start + len(pattern) - 1)
    if colon < 0:
        return None
    value_start = colon + 1
    while value_start < len(text) and text[value_start].isspace():
        value_start += 1
    try:
        value, _end = decoder.raw_decode(text, value_start)
        return value
    except json.JSONDecodeError:
        return None


def _summarize_batch_results_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "row_count": 0,
            "status_counts": {},
            "family_summary": {},
            "top_results": [],
            "viable_count": 0,
            "baseline_beating_count": 0,
        }
    rows = frame.copy()
    status_counts = rows["status"].fillna("unknown").astype(str).value_counts().to_dict() if "status" in rows else {}
    family_summary: dict[str, dict[str, Any]] = {}
    family_col = "strategy_family" if "strategy_family" in rows.columns else "family" if "family" in rows.columns else None
    if family_col:
        for family, group in rows.groupby(family_col, dropna=False):
            family_key = str(family)
            objective = pd.to_numeric(group.get("objective_score"), errors="coerce") if "objective_score" in group else pd.Series(dtype=float)
            best_idx = objective.idxmax() if not objective.empty and objective.notna().any() else None
            best_row = group.loc[best_idx] if best_idx is not None else {}
            status_series = group["status"].fillna("").astype(str) if "status" in group else pd.Series(dtype=str)
            bucket: dict[str, Any] = {
                "results": int(len(group)),
                "executed": int(status_series.isin(["success", "no_trades", "duplicate"]).sum()) if not status_series.empty else int(len(group)),
                "skipped": int(status_series.isin(["duplicate", "skipped"]).sum()) if not status_series.empty else 0,
                "failed": int(status_series.isin(["error", "invalid"]).sum()) if not status_series.empty else 0,
                "best_objective_score": None if best_idx is None else float(objective.loc[best_idx]),
                "best_experiment_id": None if best_idx is None else best_row.get("experiment_id"),
                "strategy_type_counts": {},
                "source_type_counts": {},
                "template_counts": {},
            }
            for source_col, target_key in (
                ("strategy_type", "strategy_type_counts"),
                ("source_type", "source_type_counts"),
                ("template_id", "template_counts"),
            ):
                if source_col in group.columns:
                    bucket[target_key] = {
                        str(name): int(count)
                        for name, count in group[source_col].fillna("unspecified").astype(str).value_counts().to_dict().items()
                    }
            family_summary[family_key] = bucket
    top_results: list[dict[str, Any]] = []
    if "objective_score" in rows.columns:
        sortable = rows.copy()
        sortable["_objective_score"] = pd.to_numeric(sortable["objective_score"], errors="coerce")
        for _, row in sortable.sort_values("_objective_score", ascending=False, na_position="last").head(5).iterrows():
            top_results.append(
                {
                    "experiment_id": row.get("experiment_id"),
                    "strategy_family": row.get("strategy_family", row.get("family")),
                    "config_hash": row.get("config_hash"),
                    "objective_score": None if pd.isna(row.get("_objective_score")) else float(row.get("_objective_score")),
                    "status": row.get("status"),
                    "viable": row.get("viable"),
                    "template_id": row.get("template_id"),
                    "source_type": row.get("source_type"),
                }
            )
    viable_count = 0
    if "viable" in rows.columns:
        viable_count = int(rows["viable"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
    baseline_beating_count = 0
    if "beats_baseline_objective" in rows.columns:
        baseline_beating_count = int(rows["beats_baseline_objective"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
    return {
        "row_count": int(len(rows)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "family_summary": family_summary,
        "top_results": top_results,
        "viable_count": viable_count,
        "baseline_beating_count": baseline_beating_count,
    }


def _batch_results_summary(batch_dir: Path) -> dict[str, Any]:
    for name in ("raw_results.csv", "leaderboard.csv"):
        path = batch_dir / name
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        result = _summarize_batch_results_frame(frame)
        result["source_path"] = str(path)
        return result
    return {
        "row_count": 0,
        "status_counts": {},
        "family_summary": {},
        "top_results": [],
        "viable_count": 0,
        "baseline_beating_count": 0,
        "source_path": None,
    }


def _compact_batch_summary_from_artifacts(batch_dir: Path) -> dict[str, Any] | None:
    summary_path = batch_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary_size = summary_path.stat().st_size
    if summary_size <= BATCH_SUMMARY_FULL_PARSE_LIMIT_BYTES:
        try:
            payload = json.loads(summary_path.read_text())
            if isinstance(payload, dict):
                result_summary = _batch_results_summary(batch_dir)
                payload.setdefault("status_counts", result_summary.get("status_counts") or {})
                payload.setdefault("family_summary", result_summary.get("family_summary") or {})
                payload.setdefault("top_results", result_summary.get("top_results") or [])
                payload.setdefault("viable_count", result_summary.get("viable_count") or 0)
                payload.setdefault("baseline_beating_count", result_summary.get("baseline_beating_count") or 0)
                payload.setdefault(
                    "artifact_compaction",
                    {
                        "source": "legacy_batch_summary_full_parse",
                        "original_summary_bytes": summary_size,
                        "result_rows": result_summary.get("row_count") or 0,
                        "result_source_path": result_summary.get("source_path"),
                    },
                )
                payload = _compact_if_oversized(payload)
                payload.setdefault("summary_path", str(summary_path))
                return payload
        except Exception:
            pass

    windows = _read_summary_windows(summary_path)
    result_summary = _batch_results_summary(batch_dir)
    compact: dict[str, Any] = {
        "batch_id": _extract_top_level_json_value(windows, "batch_id") or batch_dir.name,
        "timestamp_utc": _extract_top_level_json_value(windows, "timestamp_utc"),
        "strategy_families": _extract_top_level_json_value(windows, "strategy_families") or [],
        "sampler_type": _extract_top_level_json_value(windows, "sampler_type"),
        "max_workers": _extract_top_level_json_value(windows, "max_workers"),
        "execution_mode": _extract_top_level_json_value(windows, "execution_mode"),
        "worker_failures": _extract_top_level_json_value(windows, "worker_failures"),
        "source_proposal_id": _extract_top_level_json_value(windows, "source_proposal_id"),
        "total_sampled": _extract_top_level_json_value(windows, "total_sampled"),
        "total_executed": _extract_top_level_json_value(windows, "total_executed"),
        "total_skipped": _extract_top_level_json_value(windows, "total_skipped"),
        "total_failed": _extract_top_level_json_value(windows, "total_failed"),
        "status_counts": _extract_top_level_json_value(windows, "status_counts") or result_summary.get("status_counts") or {},
        "family_summary": _extract_top_level_json_value(windows, "family_summary") or result_summary.get("family_summary") or {},
        "proposal_quality": _extract_top_level_json_value(windows, "proposal_quality") or {},
        "throughput_diagnostics": _extract_top_level_json_value(windows, "throughput_diagnostics") or {},
        "leaderboard_path": str(batch_dir / "leaderboard.csv") if (batch_dir / "leaderboard.csv").exists() else None,
        "raw_results_path": str(batch_dir / "raw_results.csv") if (batch_dir / "raw_results.csv").exists() else None,
        "summary_path": str(summary_path),
        "confirmation_state": _extract_top_level_json_value(windows, "confirmation_state"),
        "confirmation_required": _extract_top_level_json_value(windows, "confirmation_required"),
        "confirmation_reason": _extract_top_level_json_value(windows, "confirmation_reason"),
        "confirmation_batch_id": _extract_top_level_json_value(windows, "confirmation_batch_id"),
        "confirmation_outcome": _extract_top_level_json_value(windows, "confirmation_outcome"),
        "promotion_state": _extract_top_level_json_value(windows, "promotion_state"),
        "targeted_follow_up_required": _extract_top_level_json_value(windows, "targeted_follow_up_required"),
        "targeted_follow_up_reason": _extract_top_level_json_value(windows, "targeted_follow_up_reason"),
        "targeted_follow_up_type": _extract_top_level_json_value(windows, "targeted_follow_up_type"),
        "targeted_follow_up_priority": _extract_top_level_json_value(windows, "targeted_follow_up_priority"),
        "targeted_follow_up_batch_id": _extract_top_level_json_value(windows, "targeted_follow_up_batch_id"),
        "holdout_check_required": _extract_top_level_json_value(windows, "holdout_check_required"),
        "holdout_check_type": _extract_top_level_json_value(windows, "holdout_check_type"),
        "holdout_check_status": _extract_top_level_json_value(windows, "holdout_check_status"),
        "holdout_check_outcome": _extract_top_level_json_value(windows, "holdout_check_outcome"),
        "holdout_check_scope": _extract_top_level_json_value(windows, "holdout_check_scope"),
        "holdout_check_batch_id": _extract_top_level_json_value(windows, "holdout_check_batch_id"),
        "holdout_horizon_tags": _extract_top_level_json_value(windows, "holdout_horizon_tags") or [],
        "holdout_regime_tags": _extract_top_level_json_value(windows, "holdout_regime_tags") or [],
        "top_results": result_summary.get("top_results") or [],
        "viable_count": result_summary.get("viable_count") or 0,
        "baseline_beating_count": result_summary.get("baseline_beating_count") or 0,
        "artifact_compaction": {
            "source": "legacy_batch_summary_windows",
            "original_summary_bytes": summary_size,
            "result_rows": result_summary.get("row_count") or 0,
            "result_source_path": result_summary.get("source_path"),
        },
    }
    if compact["total_sampled"] is None:
        compact["total_sampled"] = result_summary.get("row_count") or 0
    if compact["total_executed"] is None:
        statuses = result_summary.get("status_counts") or {}
        compact["total_executed"] = int(statuses.get("success", 0) or 0) + int(statuses.get("no_trades", 0) or 0)
    if compact["total_failed"] is None:
        statuses = result_summary.get("status_counts") or {}
        compact["total_failed"] = int(statuses.get("error", 0) or 0) + int(statuses.get("invalid", 0) or 0)
    if compact["total_skipped"] is None:
        statuses = result_summary.get("status_counts") or {}
        compact["total_skipped"] = int(statuses.get("duplicate", 0) or 0) + int(statuses.get("skipped", 0) or 0)
    compact["proposal_metadata"] = {
        "source_proposal_id": compact.get("source_proposal_id"),
        "proposal_quality": compact.get("proposal_quality") or {},
    }
    return compact


def backfill_batch_metadata_from_artifacts(
    base_dir: str = "experiments",
    *,
    limit: int | None = None,
    batch_ids: Iterable[str] | None = None,
    force: bool = True,
) -> dict[str, Any]:
    batches_dir = Path(base_dir) / "batches"
    if not batches_dir.exists():
        return {"processed_count": 0, "written_count": 0, "skipped_count": 0, "error_count": 0, "errors": []}
    requested_ids = {str(batch_id) for batch_id in (batch_ids or []) if str(batch_id).strip()}
    batch_dirs = sorted([path for path in batches_dir.iterdir() if path.is_dir()])
    if requested_ids:
        batch_dirs = [path for path in batch_dirs if path.name in requested_ids]
    if limit is not None:
        batch_dirs = batch_dirs[: max(0, int(limit))]
    existing: set[str] = set()
    if not force and _hot_index_path(base_dir).exists():
        with _connect(base_dir) as conn:
            _ensure_schema(conn)
            existing = {str(row["batch_id"]) for row in conn.execute("SELECT batch_id FROM batch_summaries").fetchall()}
    processed = 0
    written = 0
    skipped = 0
    errors: list[dict[str, str]] = []
    for batch_dir in batch_dirs:
        processed += 1
        if not force and batch_dir.name in existing:
            skipped += 1
            continue
        try:
            summary = _compact_batch_summary_from_artifacts(batch_dir)
            if not summary:
                skipped += 1
                continue
            upsert_batch_summary(base_dir, summary)
            written += 1
        except Exception as exc:
            errors.append({"batch_id": batch_dir.name, "error": str(exc)})
    return {
        "processed_count": processed,
        "written_count": written,
        "skipped_count": skipped,
        "error_count": len(errors),
        "errors": errors[:20],
    }


def load_hot_batch_summary_by_id(base_dir: str = "experiments", batch_id: str | None = None) -> dict[str, Any] | None:
    if not batch_id or not _hot_index_path(base_dir).exists():
        return None
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        row = conn.execute("SELECT summary_json FROM batch_summaries WHERE batch_id = ?", (str(batch_id),)).fetchone()
        if row is None:
            return None
        payload = _json_loads(row["summary_json"])
        return dict(payload) if isinstance(payload, dict) else None


def _hot_batch_row(base_dir: str, batch_id: str) -> dict[str, Any] | None:
    if not _hot_index_path(base_dir).exists():
        return None
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT batch_id, timestamp_utc, total_sampled, total_executed, total_skipped, total_failed, summary_json FROM batch_summaries WHERE batch_id = ?",
            (str(batch_id),),
        ).fetchone()
    return dict(row) if row is not None else None


def validate_hot_batch_metadata(base_dir: str = "experiments", batch_id: str | None = None) -> dict[str, Any]:
    batches_dir = Path(base_dir) / "batches"
    batch_dirs = [batches_dir / str(batch_id)] if batch_id else sorted([path for path in batches_dir.iterdir() if path.is_dir()])
    checked = 0
    valid = 0
    missing_db: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for batch_dir in batch_dirs:
        summary_path = batch_dir / "summary.json"
        if not summary_path.exists():
            continue
        checked += 1
        summary = _compact_batch_summary_from_artifacts(batch_dir)
        row = _hot_batch_row(base_dir, batch_dir.name)
        if row is None:
            missing_db.append(batch_dir.name)
            continue
        fields = ("total_sampled", "total_executed", "total_skipped", "total_failed")
        row_mismatches: dict[str, Any] = {}
        for field in fields:
            expected = summary.get(field) if summary else None
            observed = row.get(field)
            if expected is not None and observed is not None and int(expected) != int(observed):
                row_mismatches[field] = {"disk": int(expected), "sqlite": int(observed)}
        if row_mismatches:
            mismatches.append({"batch_id": batch_dir.name, "fields": row_mismatches})
        else:
            valid += 1
    return {
        "checked_count": checked,
        "valid_count": valid,
        "missing_db_count": len(missing_db),
        "mismatch_count": len(mismatches),
        "missing_db": missing_db[:20],
        "mismatches": mismatches[:20],
    }


def _recent_batch_ids(base_dir: str, keep_recent: int) -> set[str]:
    path = _hot_index_path(base_dir)
    if not path.exists() or keep_recent <= 0:
        return set()
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            "SELECT batch_id FROM batch_summaries ORDER BY timestamp_utc DESC, batch_id DESC LIMIT ?",
            (int(keep_recent),),
        ).fetchall()
    return {str(row["batch_id"]) for row in rows}


def cleanup_validated_batch_artifacts(
    base_dir: str = "experiments",
    *,
    keep_recent: int = 200,
    min_summary_bytes: int = BATCH_ARTIFACT_CLEANUP_THRESHOLD_BYTES,
    dry_run: bool = True,
    mode: str = "thin",
) -> dict[str, Any]:
    if mode not in {"thin", "delete_summary"}:
        raise ValueError("mode must be 'thin' or 'delete_summary'")
    batches_dir = Path(base_dir) / "batches"
    recent_ids = _recent_batch_ids(base_dir, keep_recent)
    eligible: list[dict[str, Any]] = []
    skipped_recent = 0
    skipped_invalid = 0
    for summary_path in sorted(batches_dir.glob("*/summary.json")):
        try:
            size = summary_path.stat().st_size
        except FileNotFoundError:
            continue
        if size < int(min_summary_bytes):
            continue
        batch_id = summary_path.parent.name
        if batch_id in recent_ids:
            skipped_recent += 1
            continue
        validation = validate_hot_batch_metadata(base_dir, batch_id)
        if validation.get("valid_count") != 1 or validation.get("missing_db_count") or validation.get("mismatch_count"):
            skipped_invalid += 1
            continue
        compact = load_hot_batch_summary_by_id(base_dir, batch_id) or _compact_batch_summary_from_artifacts(summary_path.parent)
        compact = dict(compact or {})
        compact["summary_path"] = str(summary_path)
        compact["artifact_compaction"] = {
            **(compact.get("artifact_compaction") if isinstance(compact.get("artifact_compaction"), dict) else {}),
            "mode": mode,
            "original_summary_bytes": size,
            "compacted_at": _now(),
            "sqlite_validated": True,
        }
        compact_bytes = len(json.dumps(compact, sort_keys=True, default=str).encode("utf-8")) if compact else 0
        eligible.append(
            {
                "batch_id": batch_id,
                "summary_path": str(summary_path),
                "original_bytes": size,
                "replacement_bytes": 0 if mode == "delete_summary" else compact_bytes,
                "estimated_recoverable_bytes": max(0, size - (0 if mode == "delete_summary" else compact_bytes)),
                "compact_summary": compact,
            }
        )
    changed: list[dict[str, Any]] = []
    if not dry_run:
        for item in eligible:
            summary_path = Path(item["summary_path"])
            before = summary_path.stat().st_size if summary_path.exists() else 0
            if mode == "delete_summary":
                summary_path.unlink(missing_ok=True)
                after = 0
            else:
                _atomic_write_json(summary_path, item["compact_summary"])
                after = summary_path.stat().st_size
            changed.append(
                {
                    "batch_id": item["batch_id"],
                    "summary_path": item["summary_path"],
                    "before_bytes": before,
                    "after_bytes": after,
                    "recovered_bytes": max(0, before - after),
                }
            )
    total_estimated = sum(int(item["estimated_recoverable_bytes"]) for item in eligible)
    total_recovered = sum(int(item["recovered_bytes"]) for item in changed)
    return {
        "dry_run": dry_run,
        "mode": mode,
        "eligible_count": len(eligible),
        "changed_count": len(changed),
        "skipped_recent_count": skipped_recent,
        "skipped_invalid_count": skipped_invalid,
        "estimated_recoverable_bytes": total_estimated,
        "actual_recovered_bytes": total_recovered,
        "eligible": [{k: v for k, v in item.items() if k != "compact_summary"} for item in eligible[:50]],
        "changed": changed[:50],
    }


def _proposal_candidate_rows(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    request = proposal.get("request") if isinstance(proposal.get("request"), dict) else {}
    proposal_id = str(request.get("proposal_id") or proposal.get("proposal_id") or "").strip()
    candidate_configs = proposal.get("candidate_configs") if isinstance(proposal.get("candidate_configs"), dict) else {}
    candidate_metadata = proposal.get("candidate_metadata") if isinstance(proposal.get("candidate_metadata"), dict) else {}
    rows: list[dict[str, Any]] = []
    candidate_index = 0
    for family in sorted(candidate_configs.keys()):
        configs = candidate_configs.get(family) or []
        metadata_items = candidate_metadata.get(family) or []
        if not isinstance(configs, list):
            continue
        for family_idx, config in enumerate(configs):
            if not isinstance(config, dict):
                continue
            metadata = metadata_items[family_idx] if isinstance(metadata_items, list) and family_idx < len(metadata_items) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            source_idea_ids = metadata.get("source_idea_ids")
            config_hash = metadata.get("config_hash") or _compute_config_hash(str(family), config)
            rows.append(
                {
                    "proposal_id": proposal_id,
                    "candidate_index": candidate_index,
                    "family": str(family),
                    "family_candidate_index": family_idx,
                    "config_hash": str(config_hash),
                    "source_type": metadata.get("source_type"),
                    "template_id": metadata.get("template_id"),
                    "strategy_type": metadata.get("strategy_type"),
                    "proposal_role": metadata.get("proposal_role"),
                    "exploration_mode": metadata.get("exploration_mode"),
                    "parent_config_hash": metadata.get("parent_config_hash"),
                    "near_duplicate_of": metadata.get("near_duplicate_of"),
                    "novelty_score": metadata.get("novelty_score"),
                    "selection_score": metadata.get("selection_score"),
                    "duplicate_risk": metadata.get("duplicate_risk"),
                    "dead_zone_risk": metadata.get("dead_zone_risk"),
                    "is_new_idea": 1 if bool(metadata.get("is_new_idea")) else 0,
                    "is_uncommon_idea": 1 if bool(metadata.get("is_uncommon_idea")) else 0,
                    "source_idea_ids_json": _json_dumps(source_idea_ids or []),
                    "config_json": _json_dumps(config),
                    "metadata_json": _json_dumps(metadata),
                    "updated_at": _now(),
                }
            )
            candidate_index += 1
    return rows


def upsert_proposal_metadata(
    base_dir: str,
    proposal: dict[str, Any],
    *,
    proposal_path: str | None = None,
    summary_path: str | None = None,
    candidate_configs_path: str | None = None,
) -> None:
    """Store proposal metadata and normalized candidate rows in the hot SQLite index.

    JSON exports remain the compatibility layer; this function adds a queryable hot
    metadata layer without changing proposal execution behavior.
    """
    init_hot_index(base_dir)
    if not isinstance(proposal, dict):
        return
    request = proposal.get("request") if isinstance(proposal.get("request"), dict) else {}
    proposal_id = str(request.get("proposal_id") or proposal.get("proposal_id") or "").strip()
    if not proposal_id:
        return
    candidate_rows = _proposal_candidate_rows(proposal)
    candidate_family_counts: dict[str, int] = {}
    for row in candidate_rows:
        family = str(row.get("family") or "")
        candidate_family_counts[family] = candidate_family_counts.get(family, 0) + 1
    reasoning_summary = proposal.get("reasoning_summary") if isinstance(proposal.get("reasoning_summary"), dict) else {}
    compact_reasoning_summary = compact_summary_for_disk(reasoning_summary, proposal)
    compact_request_payload = compact_request(request)
    proposal_quality = reasoning_summary.get("proposal_quality") if isinstance(reasoning_summary.get("proposal_quality"), dict) else {}
    proposal_dir = str(Path(proposal_path).parent) if proposal_path else None
    record = {
        "proposal_id": proposal_id,
        "timestamp_utc": request.get("timestamp_utc"),
        "status": proposal.get("status"),
        "strategy_families_json": _json_dumps(request.get("strategy_families") or []),
        "source_batch_ids_json": _json_dumps(request.get("source_batch_ids") or []),
        "objective_name": request.get("objective_name"),
        "baseline_name": request.get("baseline_name"),
        "seed": request.get("seed"),
        "exploration_fraction": request.get("exploration_fraction"),
        "exploitation_fraction": request.get("exploitation_fraction"),
        "max_experiments": request.get("max_experiments"),
        "candidate_count": len(candidate_rows),
        "candidate_family_counts_json": _json_dumps(candidate_family_counts),
        "source_idea_ids_json": _json_dumps(request.get("source_idea_ids") or []),
        "proposal_quality_json": _json_dumps(proposal_quality),
        "reasoning_summary_json": _json_dumps(compact_reasoning_summary),
        "request_json": _json_dumps(compact_request_payload),
        "proposal_dir": proposal_dir,
        "proposal_path": proposal_path,
        "summary_path": summary_path,
        "candidate_configs_path": candidate_configs_path,
        "json_export_status": "active",
        "json_export_deleted_at": None,
        "updated_at": _now(),
    }
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        columns = ", ".join(record.keys())
        placeholders = ", ".join(["?"] * len(record))
        conn.execute(
            f"INSERT INTO proposal_summaries ({columns}) VALUES ({placeholders}) "
            "ON CONFLICT(proposal_id) DO UPDATE SET "
            + ", ".join(f"{column}=excluded.{column}" for column in record.keys() if column != "proposal_id"),
            tuple(record.values()),
        )
        conn.execute("DELETE FROM proposal_candidates WHERE proposal_id = ?", (proposal_id,))
        if candidate_rows:
            candidate_columns = list(candidate_rows[0].keys())
            candidate_placeholders = ", ".join(["?"] * len(candidate_columns))
            conn.executemany(
                f"INSERT INTO proposal_candidates ({', '.join(candidate_columns)}) VALUES ({candidate_placeholders})",
                [tuple(row[column] for column in candidate_columns) for row in candidate_rows],
            )
        conn.commit()


def _proposal_record_from_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    decode_fields = {
        "strategy_families_json": "strategy_families",
        "source_batch_ids_json": "source_batch_ids",
        "candidate_family_counts_json": "candidate_family_counts",
        "source_idea_ids_json": "source_idea_ids",
        "proposal_quality_json": "proposal_quality",
        "reasoning_summary_json": "reasoning_summary",
        "request_json": "request",
    }
    for source, target in decode_fields.items():
        payload[target] = _json_loads(payload.pop(source, None))
    return payload


def load_hot_proposal(base_dir: str = "experiments", proposal_id: str | None = None) -> dict[str, Any] | None:
    path = _hot_index_path(base_dir)
    if not path.exists():
        return None
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        if proposal_id:
            row = conn.execute(
                "SELECT * FROM proposal_summaries WHERE proposal_id = ?",
                (str(proposal_id),),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM proposal_summaries ORDER BY timestamp_utc DESC, proposal_id DESC LIMIT 1"
            ).fetchone()
    return _proposal_record_from_row(row) if row is not None else None


def load_hot_proposals(
    base_dir: str = "experiments",
    *,
    limit: int = HOT_INDEX_PROPOSAL_LIMIT,
    status: str | None = None,
    families: Iterable[str] | None = None,
) -> pd.DataFrame:
    path = _hot_index_path(base_dir)
    if not path.exists():
        return pd.DataFrame()
    family_set = {str(family).strip().lower() for family in (families or []) if str(family).strip()}
    params: list[Any] = []
    where: list[str] = []
    if status:
        where.append("status = ?")
        params.append(str(status))
    query = "SELECT * FROM proposal_summaries"
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY timestamp_utc DESC, proposal_id DESC LIMIT ?"
    params.append(max(0, int(limit)))
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        rows = conn.execute(query, tuple(params)).fetchall()
    records = [_proposal_record_from_row(row) for row in rows]
    if family_set:
        records = [
            record
            for record in records
            if family_set & {str(family).strip().lower() for family in (record.get("strategy_families") or [])}
        ]
    return pd.DataFrame(records)


def load_hot_proposal_candidates(
    base_dir: str = "experiments",
    proposal_id: str | None = None,
    *,
    family: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    path = _hot_index_path(base_dir)
    if not path.exists():
        return pd.DataFrame()
    params: list[Any] = []
    where: list[str] = []
    if proposal_id:
        where.append("proposal_id = ?")
        params.append(str(proposal_id))
    if family:
        where.append("family = ?")
        params.append(str(family).strip().lower())
    query = "SELECT * FROM proposal_candidates"
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY proposal_id DESC, candidate_index ASC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(max(0, int(limit)))
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        rows = conn.execute(query, tuple(params)).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record["source_idea_ids"] = _json_loads(record.pop("source_idea_ids_json", None)) or []
        record["config"] = _json_loads(record.pop("config_json", None)) or {}
        record["metadata"] = _json_loads(record.pop("metadata_json", None)) or {}
        record["is_new_idea"] = bool(record.get("is_new_idea"))
        record["is_uncommon_idea"] = bool(record.get("is_uncommon_idea"))
        records.append(record)
    return pd.DataFrame(records)


def validate_hot_proposal_write(base_dir: str = "experiments", proposal_id: str | None = None) -> dict[str, Any]:
    proposal = load_hot_proposal(base_dir, proposal_id)
    if proposal is None:
        return {
            "proposal_id": proposal_id,
            "exists": False,
            "candidate_count": 0,
            "candidate_rows": 0,
            "valid": False,
        }
    candidate_count = int(proposal.get("candidate_count") or 0)
    candidates = load_hot_proposal_candidates(base_dir, str(proposal.get("proposal_id")))
    candidate_rows = int(len(candidates))
    valid = candidate_rows == candidate_count
    return {
        "proposal_id": proposal.get("proposal_id"),
        "exists": True,
        "candidate_count": candidate_count,
        "candidate_rows": candidate_rows,
        "valid": valid,
    }


def backfill_proposal_metadata_from_json(
    base_dir: str = "experiments",
    *,
    limit: int | None = None,
    proposal_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Import existing proposal JSON exports into SQLite without deleting files."""
    proposals_dir = Path(base_dir) / "proposals"
    if not proposals_dir.exists():
        return {"processed_count": 0, "written_count": 0, "skipped_count": 0, "error_count": 0, "errors": []}
    requested_ids = {str(proposal_id) for proposal_id in (proposal_ids or []) if str(proposal_id).strip()}
    proposal_paths = sorted(proposals_dir.glob("*/proposal.json"))
    if requested_ids:
        proposal_paths = [path for path in proposal_paths if path.parent.name in requested_ids]
    if limit is not None:
        proposal_paths = proposal_paths[: max(0, int(limit))]
    processed = 0
    written = 0
    skipped = 0
    errors: list[dict[str, str]] = []
    for proposal_path in proposal_paths:
        processed += 1
        try:
            proposal = json.loads(proposal_path.read_text())
            if not isinstance(proposal, dict):
                skipped += 1
                continue
            proposal_dir = proposal_path.parent
            summary_path = proposal_dir / "summary.json"
            candidate_configs_path = proposal_dir / "candidate_configs.json"
            upsert_proposal_metadata(
                base_dir,
                proposal,
                proposal_path=str(proposal_path),
                summary_path=str(summary_path) if summary_path.exists() else None,
                candidate_configs_path=str(candidate_configs_path) if candidate_configs_path.exists() else None,
            )
            written += 1
        except Exception as exc:
            errors.append({"proposal_path": str(proposal_path), "error": str(exc)})
    return {
        "processed_count": processed,
        "written_count": written,
        "skipped_count": skipped,
        "error_count": len(errors),
        "errors": errors,
    }


def proposal_storage_stats(base_dir: str = "experiments") -> dict[str, Any]:
    base = Path(base_dir)
    proposals_dir = base / "proposals"
    proposal_dirs = [path for path in proposals_dir.iterdir() if path.is_dir()] if proposals_dir.exists() else []
    json_files = list(proposals_dir.glob("*/proposal.json")) if proposals_dir.exists() else []
    summary_files = list(proposals_dir.glob("*/summary.json")) if proposals_dir.exists() else []
    candidate_files = list(proposals_dir.glob("*/candidate_configs.json")) if proposals_dir.exists() else []
    export_files = json_files + summary_files + candidate_files
    export_bytes = 0
    for path in export_files:
        try:
            export_bytes += path.stat().st_size
        except FileNotFoundError:
            pass
    sqlite_path = _hot_index_path(base_dir)
    sqlite_bytes = sqlite_path.stat().st_size if sqlite_path.exists() else 0
    proposal_rows = 0
    candidate_rows = 0
    if sqlite_path.exists():
        with _connect(base_dir) as conn:
            _ensure_schema(conn)
            proposal_rows = int(conn.execute("SELECT COUNT(*) FROM proposal_summaries").fetchone()[0])
            candidate_rows = int(conn.execute("SELECT COUNT(*) FROM proposal_candidates").fetchone()[0])
    return {
        "proposal_dir_count": len(proposal_dirs),
        "proposal_json_file_count": len(json_files),
        "summary_json_file_count": len(summary_files),
        "candidate_configs_json_file_count": len(candidate_files),
        "export_file_count": len(export_files),
        "export_bytes": export_bytes,
        "sqlite_path": str(sqlite_path),
        "sqlite_bytes": sqlite_bytes,
        "sqlite_proposal_rows": proposal_rows,
        "sqlite_candidate_rows": candidate_rows,
        "filesystem_entries_replaceable_after_validation": len(proposal_dirs) + len(export_files),
    }


def cleanup_validated_proposal_json_exports(
    base_dir: str = "experiments",
    *,
    keep_recent: int = HOT_INDEX_PROPOSAL_LIMIT,
    older_than_days: int | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Remove old proposal JSON folders only when their SQLite rows validate.

    This is intentionally opt-in and defaults to dry-run so the migration keeps
    existing proposal behavior unchanged until operators decide to prune exports.
    """
    path = _hot_index_path(base_dir)
    if not path.exists():
        return {"dry_run": dry_run, "deleted_count": 0, "eligible_count": 0, "deleted_dirs": [], "eligible_dirs": []}
    cutoff: datetime | None = None
    if older_than_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=max(0, int(older_than_days)))
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT proposal_id, timestamp_utc, proposal_dir, json_export_status
            FROM proposal_summaries
            ORDER BY timestamp_utc DESC, proposal_id DESC
            """
        ).fetchall()
    eligible: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if idx < max(0, int(keep_recent)):
            continue
        proposal_id = str(row["proposal_id"])
        if str(row["json_export_status"] or "active") == "deleted":
            continue
        proposal_dir = Path(str(row["proposal_dir"] or ""))
        if not proposal_dir.exists() or not proposal_dir.is_dir():
            continue
        if cutoff is not None:
            timestamp_text = str(row["timestamp_utc"] or "")
            try:
                timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=UTC)
            except ValueError:
                continue
            if timestamp >= cutoff:
                continue
        validation = validate_hot_proposal_write(base_dir, proposal_id)
        if not validation.get("valid"):
            continue
        eligible.append({"proposal_id": proposal_id, "proposal_dir": str(proposal_dir)})
    deleted: list[dict[str, Any]] = []
    if not dry_run and eligible:
        with _connect(base_dir) as conn:
            _ensure_schema(conn)
            for item in eligible:
                proposal_dir = Path(item["proposal_dir"])
                if proposal_dir.exists() and proposal_dir.is_dir():
                    shutil.rmtree(proposal_dir)
                    deleted.append(item)
                    conn.execute(
                        """
                        UPDATE proposal_summaries
                        SET json_export_status = ?, json_export_deleted_at = ?, updated_at = ?
                        WHERE proposal_id = ?
                        """,
                        ("deleted", _now(), _now(), item["proposal_id"]),
                    )
            conn.commit()
    return {
        "dry_run": dry_run,
        "deleted_count": len(deleted),
        "eligible_count": len(eligible),
        "deleted_dirs": [item["proposal_dir"] for item in deleted],
        "eligible_dirs": [item["proposal_dir"] for item in eligible],
    }


def archive_hot_batch_summaries(base_dir: str = "experiments", *, keep_recent: int = HOT_INDEX_BATCH_LIMIT) -> dict[str, int]:
    path = _hot_index_path(base_dir)
    if not path.exists():
        return {"retained_count": 0, "archived_count": 0}
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT batch_id
            FROM batch_summaries
            ORDER BY timestamp_utc DESC, batch_id DESC
            """
        ).fetchall()
        batch_ids = [str(row["batch_id"]) for row in rows]
        retained = batch_ids[: max(0, int(keep_recent))]
        archived = batch_ids[max(0, int(keep_recent)) :]
        if archived:
            conn.executemany("DELETE FROM batch_summaries WHERE batch_id = ?", [(batch_id,) for batch_id in archived])
            conn.execute(
                "INSERT OR REPLACE INTO archive_manifest (item_type, scope_key, retained_count, archived_count, updated_at, note) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "batch_summaries",
                    "global",
                    len(retained),
                    len(archived),
                    _now(),
                    "compact_recent_batch_window",
                ),
            )
            conn.commit()
    return {"retained_count": len(retained), "archived_count": len(archived)}


def compact_hot_index_storage(
    base_dir: str = "experiments",
    *,
    keep_recent_batches: int = HOT_INDEX_BATCH_LIMIT,
    max_json_bytes: int = HOT_INDEX_BATCH_JSON_LIMIT_BYTES,
    vacuum: bool = True,
) -> dict[str, Any]:
    path = _hot_index_path(base_dir)
    if not path.exists():
        return {
            "path": str(path),
            "before_bytes": 0,
            "after_bytes": 0,
            "compacted_batch_rows": 0,
            "retained_batch_rows": 0,
            "archived_batch_rows": 0,
        }
    before_bytes = path.stat().st_size
    compacted_rows = 0
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            "SELECT batch_id, proposal_metadata_json, summary_json FROM batch_summaries"
        ).fetchall()
        for row in rows:
            metadata = _json_loads(row["proposal_metadata_json"]) or {}
            summary = _json_loads(row["summary_json"]) or {}
            compact_metadata = _compact_batch_proposal_metadata(metadata)
            compact_summary = _compact_if_oversized(summary, limit_bytes=max_json_bytes)
            metadata_json = _json_dumps(compact_metadata)
            summary_json = _json_dumps(compact_summary)
            if (
                metadata_json != (row["proposal_metadata_json"] or "")
                or summary_json != (row["summary_json"] or "")
                or len((row["summary_json"] or "").encode("utf-8")) > max_json_bytes
                or len((row["proposal_metadata_json"] or "").encode("utf-8")) > max_json_bytes
            ):
                conn.execute(
                    "UPDATE batch_summaries SET proposal_metadata_json = ?, summary_json = ?, updated_at = ? WHERE batch_id = ?",
                    (metadata_json, summary_json, _now(), row["batch_id"]),
                )
                compacted_rows += 1
        conn.commit()
    archive_counts = archive_hot_batch_summaries(base_dir, keep_recent=keep_recent_batches)
    if vacuum:
        with sqlite3.connect(path, timeout=30.0) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")
    after_bytes = path.stat().st_size
    return {
        "path": str(path),
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "compacted_batch_rows": compacted_rows,
        "retained_batch_rows": archive_counts.get("retained_count", 0),
        "archived_batch_rows": archive_counts.get("archived_count", 0),
    }


def upsert_summary_item(
    base_dir: str,
    item_type: str,
    scope_key: str,
    payload: dict[str, Any],
    *,
    source_batch_id: str | None = None,
) -> None:
    init_hot_index(base_dir)
    record = {
        "item_type": str(item_type),
        "scope_key": str(scope_key),
        "timestamp_utc": str(payload.get("timestamp_utc") or _now()),
        "source_batch_id": source_batch_id,
        "payload_json": _json_dumps(payload),
        "updated_at": _now(),
    }
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        columns = ", ".join(record.keys())
        placeholders = ", ".join(["?"] * len(record))
        conn.execute(
            f"INSERT INTO summary_items ({columns}) VALUES ({placeholders}) "
            "ON CONFLICT(item_type, scope_key) DO UPDATE SET "
            + ", ".join(f"{column}=excluded.{column}" for column in record.keys() if column not in {"item_type", "scope_key"}),
            tuple(record.values()),
        )
        conn.commit()


def load_hot_batch_summary(base_dir: str = "experiments") -> dict[str, Any] | None:
    path = _hot_index_path(base_dir)
    if not path.exists():
        return None
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT summary_json FROM batch_summaries ORDER BY timestamp_utc DESC, batch_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        payload = _json_loads(row["summary_json"])
        return dict(payload) if isinstance(payload, dict) else None


def load_hot_summary_item(
    base_dir: str,
    item_type: str,
    scope_key: str = "global",
) -> dict[str, Any] | None:
    path = _hot_index_path(base_dir)
    if not path.exists():
        return None
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT payload_json FROM summary_items WHERE item_type = ? AND scope_key = ?",
            (str(item_type), str(scope_key)),
        ).fetchone()
        if row is None:
            return None
        payload = _json_loads(row["payload_json"])
        return dict(payload) if isinstance(payload, dict) else None


def load_hot_experiment_index(
    base_dir: str = "experiments",
    *,
    families: Iterable[str] | None = None,
    recent_limit_per_family: int = HOT_INDEX_EXPERIMENT_LIMIT_PER_FAMILY,
    use_archive_bootstrap: bool = True,
) -> pd.DataFrame:
    families = [str(family).strip().lower() for family in (families or []) if str(family).strip()]
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        has_rows = conn.execute("SELECT 1 FROM experiment_summaries LIMIT 1").fetchone() is not None
    if (not has_rows) or (use_archive_bootstrap and _archive_is_newer(base_dir)):
        if not use_archive_bootstrap:
            return pd.DataFrame()
        bootstrap = _bootstrap_from_archive(base_dir, families=families or None, limit_per_family=recent_limit_per_family)
        if bootstrap.empty:
            return pd.DataFrame()
        with _connect(base_dir) as conn:
            _ensure_schema(conn)
            _upsert_dataframe(conn, bootstrap)
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        params: list[Any] = []
        family_filter = ""
        if families:
            family_filter = f"WHERE strategy_family IN ({', '.join(['?'] * len(families))})"
            params.extend(families)
        query = f"""
            WITH rows AS (
                SELECT
                    experiment_id,
                    timestamp_utc,
                    strategy_family,
                    config_hash,
                    batch_id,
                    status,
                    objective_score,
                    viable,
                    source_type,
                    template_id,
                    strategy_type,
                    parent_config_hash,
                    lineage_root_config_hash,
                    branch_root_config_hash,
                    promotion_state,
                    holdout_check_type,
                    holdout_check_status,
                    holdout_check_outcome,
                    idea_kind,
                    idea_source,
                    result_dir,
                    updated_at,
                    payload_json
                FROM experiment_summaries
                {family_filter}
            ),
            recent_ranked AS (
                SELECT experiment_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY strategy_family
                        ORDER BY timestamp_utc DESC, experiment_id DESC
                    ) AS rn
                FROM rows
            ),
            best_per_family AS (
                SELECT experiment_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY strategy_family
                        ORDER BY objective_score DESC, timestamp_utc DESC, experiment_id DESC
                    ) AS br
                FROM rows
            )
            SELECT r.*
            FROM rows r
            JOIN recent_ranked rr ON r.experiment_id = rr.experiment_id
            WHERE rr.rn <= ?
            UNION
            SELECT r.*
            FROM rows r
            JOIN best_per_family bpf ON r.experiment_id = bpf.experiment_id
            WHERE bpf.br = 1
            ORDER BY timestamp_utc DESC, experiment_id DESC
        """
        params.append(int(recent_limit_per_family))
        frame = pd.read_sql_query(query, conn, params=params)
    if frame.empty:
        return frame
    payload_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        payload = _json_loads(row.get("payload_json")) or {}
        if not isinstance(payload, dict):
            payload = {}
        merged = dict(payload)
        merged.setdefault("experiment_id", row.get("experiment_id"))
        merged.setdefault("timestamp_utc", row.get("timestamp_utc"))
        merged.setdefault("strategy_family", row.get("strategy_family"))
        merged.setdefault("config_hash", row.get("config_hash"))
        merged.setdefault("batch_id", row.get("batch_id"))
        merged.setdefault("status", row.get("status"))
        merged.setdefault("objective_score", row.get("objective_score"))
        merged.setdefault("viable", bool(row.get("viable")))
        merged.setdefault("source_type", row.get("source_type"))
        merged.setdefault("template_id", row.get("template_id"))
        merged.setdefault("strategy_type", row.get("strategy_type"))
        merged.setdefault("parent_config_hash", row.get("parent_config_hash"))
        merged.setdefault("lineage_root_config_hash", row.get("lineage_root_config_hash"))
        merged.setdefault("branch_root_config_hash", row.get("branch_root_config_hash"))
        merged.setdefault("promotion_state", row.get("promotion_state"))
        merged.setdefault("holdout_check_type", row.get("holdout_check_type"))
        merged.setdefault("holdout_check_status", row.get("holdout_check_status"))
        merged.setdefault("holdout_check_outcome", row.get("holdout_check_outcome"))
        merged.setdefault("idea_kind", row.get("idea_kind"))
        merged.setdefault("idea_source", row.get("idea_source"))
        merged.setdefault("result_dir", row.get("result_dir"))
        merged.setdefault("updated_at", row.get("updated_at"))
        payload_rows.append(merged)
    result = pd.DataFrame(payload_rows)
    # Augment with any per-family all-time bests from index.csv that were evicted from SQLite.
    result = _augment_with_index_bests(result, base_dir)
    return result


def _augment_with_index_bests(
    frame: pd.DataFrame,
    base_dir: str,
    *,
    pin_top_per_family: int = HOT_INDEX_PIN_TOP_PER_FAMILY,
) -> pd.DataFrame:
    """Inject per-family top-N best rows from index.csv that are absent from the hot index.

    This recovers rows that were evicted from the SQLite hot index by the recency window.
    Only rows whose config_hash is entirely missing from ``frame`` are injected (one row
    per unique config_hash, the best-scoring run), so the hot-index payload always wins
    for configs that are still present in SQLite.
    """
    index_path = Path(base_dir) / "index.csv"
    if not index_path.exists():
        return frame
    try:
        idx = pd.read_csv(index_path, low_memory=False)
    except Exception:
        return frame
    if idx.empty or "strategy_family" not in idx.columns or "objective_score" not in idx.columns:
        return frame
    idx["objective_score"] = pd.to_numeric(idx["objective_score"], errors="coerce")
    idx = idx[idx["objective_score"].notna() & idx["status"].astype(str).isin({"success", "no_trades"})]
    if idx.empty:
        return frame

    known_hashes: set[str] = set()
    if not frame.empty and "config_hash" in frame.columns:
        known_hashes = set(frame["config_hash"].dropna().astype(str).unique())

    rows_to_inject: list[dict[str, Any]] = []
    for _fam, grp in idx.groupby("strategy_family"):
        # Deduplicate to one row per config_hash (keep the best-scoring run).
        best_per_hash = (
            grp.sort_values("objective_score", ascending=False)
            .drop_duplicates(subset=["config_hash"], keep="first")
        )
        # Take top-N by objective_score; inject only those missing from the hot index.
        top_n = best_per_hash.nlargest(int(pin_top_per_family), "objective_score")
        for _, row in top_n.iterrows():
            if str(row.get("config_hash", "")) not in known_hashes:
                rows_to_inject.append(row.to_dict())

    if not rows_to_inject:
        return frame
    inject_df = pd.DataFrame(rows_to_inject)
    return pd.concat([inject_df, frame], ignore_index=True)


def load_hot_batch_index(base_dir: str = "experiments") -> pd.DataFrame:
    path = _hot_index_path(base_dir)
    if not path.exists():
        return pd.DataFrame()
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        return pd.read_sql_query(
            "SELECT * FROM batch_summaries ORDER BY timestamp_utc DESC, batch_id DESC LIMIT ?",
            conn,
            params=(HOT_INDEX_BATCH_LIMIT,),
        )


def archive_hot_experiment_summaries(
    base_dir: str = "experiments",
    *,
    keep_recent_per_family: int = HOT_INDEX_EXPERIMENT_LIMIT_PER_FAMILY,
    pin_top_per_family: int = HOT_INDEX_PIN_TOP_PER_FAMILY,
) -> dict[str, int]:
    path = _hot_index_path(base_dir)
    if not path.exists():
        return {"retained_count": 0, "archived_count": 0}
    total_retained = 0
    total_archived = 0
    with _connect(base_dir) as conn:
        _ensure_schema(conn)
        families = [str(row["strategy_family"]) for row in conn.execute("SELECT DISTINCT strategy_family FROM experiment_summaries").fetchall()]
        for family in families:
            counts = _compact_family(conn, family, keep_recent_per_family=keep_recent_per_family, pin_top_per_family=pin_top_per_family)
            total_retained += counts["retained_count"]
            total_archived += counts["archived_count"]
    return {"retained_count": total_retained, "archived_count": total_archived}


def _compact_family(
    conn: sqlite3.Connection,
    family: str,
    *,
    keep_recent_per_family: int = HOT_INDEX_EXPERIMENT_LIMIT_PER_FAMILY,
    pin_top_per_family: int = HOT_INDEX_PIN_TOP_PER_FAMILY,
) -> dict[str, int]:
    # Pin the top-N rows by objective_score so they are never evicted by recency alone.
    top_rows = conn.execute(
        """
        SELECT experiment_id
        FROM experiment_summaries
        WHERE strategy_family = ?
        ORDER BY objective_score DESC, timestamp_utc DESC, experiment_id DESC
        LIMIT ?
        """,
        (family, int(pin_top_per_family)),
    ).fetchall()
    pinned_ids = {str(row["experiment_id"]) for row in top_rows}

    cur = conn.execute(
        """
        SELECT experiment_id
        FROM experiment_summaries
        WHERE strategy_family = ?
        ORDER BY timestamp_utc DESC, experiment_id DESC
        """,
        (family,),
    )
    experiment_ids = [str(row["experiment_id"]) for row in cur.fetchall()]
    retained = experiment_ids[:keep_recent_per_family]
    # Append pinned top-scorers that fell outside the recency window.
    retained_set = set(retained)
    for pid in pinned_ids:
        if pid not in retained_set:
            retained.append(pid)
            retained_set.add(pid)
    archived = [eid for eid in experiment_ids if eid not in retained_set]
    if archived:
        conn.executemany(
            "DELETE FROM experiment_summaries WHERE experiment_id = ?",
            [(experiment_id,) for experiment_id in archived],
        )
        conn.execute(
            "INSERT OR REPLACE INTO archive_manifest (item_type, scope_key, retained_count, archived_count, updated_at, note) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "experiment_summaries",
                family,
                len(retained),
                len(archived),
                _now(),
                "compact_recent_family_window",
            ),
        )
        conn.commit()
    return {"retained_count": len(retained), "archived_count": len(archived)}


def refresh_hot_index_reports(
    *,
    base_dir: str = "experiments",
    families: list[str] | None = None,
    recent_limit_per_family: int = HOT_INDEX_EXPERIMENT_LIMIT_PER_FAMILY,
) -> dict[str, Any]:
    from experiment_dashboard import build_best_results_dashboard, dashboard_to_dict
    from experiment_idea_yield import build_idea_yield_summary
    from experiment_lineage import build_lineage_summary
    from experiment_memory import load_lineage_state_records
    from experiment_scorecards import build_family_scorecards, scorecards_to_records

    frame = load_hot_experiment_index(base_dir, families=families, recent_limit_per_family=recent_limit_per_family)
    if families:
        family_list = list(families)
    elif not frame.empty:
        family_list = sorted(set(frame["strategy_family"].dropna().astype(str).tolist()))
    else:
        family_list = []
    latest_batch = load_hot_batch_summary(base_dir)
    scorecards = (
        build_family_scorecards(
            families=family_list,
            base_dir=base_dir,
            recent_window=recent_limit_per_family,
        )
        if family_list
        else {}
    )
    idea_yield = (
        build_idea_yield_summary(
            families=family_list,
            base_dir=base_dir,
            recent_window=recent_limit_per_family,
            persist_memory=True,
            index=frame,
        )
        if family_list
        else {}
    )
    lineage = (
        build_lineage_summary(
            frame,
            persisted_records=load_lineage_state_records(base_dir),
            latest_batch=latest_batch,
            include_histories=False,
            include_records=False,
        )
        if not frame.empty
        else {
            "result_count": 0,
            "latest_batch_id": latest_batch.get("batch_id") if latest_batch else None,
            "records": {},
            "by_config_hash": {},
            "family_summaries": {},
            "branch_summaries": {},
            "lineage_status_counts": {},
        }
    )
    dashboard = build_best_results_dashboard(
        base_dir=base_dir,
        overall_limit=20,
        viable_limit=20,
        baseline_limit=20,
        per_family_limit=10,
        families=family_list or None,
    )
    dashboard_payload = dashboard_to_dict(dashboard)
    scorecard_records = scorecards_to_records(scorecards) if scorecards else {}
    if scorecards:
        for family, scorecard in scorecards.items():
            upsert_summary_item(
                base_dir,
                "family_scorecard",
                family,
                scorecard_records.get(family) or {},
                source_batch_id=latest_batch.get("batch_id") if latest_batch else None,
            )
    if latest_batch:
        upsert_summary_item(base_dir, "latest_batch", "global", latest_batch, source_batch_id=latest_batch.get("batch_id"))
    if idea_yield:
        upsert_summary_item(
            base_dir,
            "idea_yield_summary",
            "global",
            idea_yield,
            source_batch_id=latest_batch.get("batch_id") if latest_batch else None,
        )
    if lineage:
        upsert_summary_item(base_dir, "lineage_summary", "global", lineage, source_batch_id=latest_batch.get("batch_id") if latest_batch else None)
    upsert_summary_item(base_dir, "best_results_dashboard", "global", dashboard_payload, source_batch_id=latest_batch.get("batch_id") if latest_batch else None)
    if not frame.empty:
        from experiment_store import load_results_index
        from experiment_template_tracking import TRACKED_TEMPLATES, build_full_template_tracking_report
        # Use full index filtered to tracked templates so counts are not capped
        # by the hot-index per-family limit.  The filter keeps load small.
        full_index = load_results_index(base_dir)
        if not full_index.empty and "template_id" in full_index.columns:
            template_index = full_index[full_index["template_id"].isin(TRACKED_TEMPLATES)].copy()
            tracking_frame = template_index if not template_index.empty else frame
        else:
            tracking_frame = frame
        template_report = build_full_template_tracking_report(
            tracking_frame,
            idea_yield_summary=idea_yield,
            branch_summaries=lineage.get("branch_summaries") if lineage else None,
        )
        upsert_summary_item(
            base_dir,
            "template_tracking_report",
            "momentum",
            template_report,
            source_batch_id=latest_batch.get("batch_id") if latest_batch else None,
        )
    return {
        "dashboard": dashboard_payload,
        "family_scorecards": scorecard_records,
        "idea_yield_summary": idea_yield,
        "lineage_summary": lineage,
        "latest_batch": latest_batch,
    }
