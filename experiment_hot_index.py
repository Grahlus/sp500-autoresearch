from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

HOT_INDEX_FILENAME = "hot_index.sqlite3"
HOT_INDEX_EXPERIMENT_LIMIT_PER_FAMILY = 256
HOT_INDEX_BOOTSTRAP_LIMIT_PER_FAMILY = 128
HOT_INDEX_PIN_TOP_PER_FAMILY = 20  # top-N by objective_score that are never evicted by recency
HOT_INDEX_BATCH_LIMIT = 100
HOT_INDEX_BATCH_JSON_LIMIT_BYTES = 64_000


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
            include_idea_yield=True,
            use_hot_index=True,
            hot_limit_per_family=recent_limit_per_family,
            index=frame,
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
        include_idea_yield=True,
        index=frame,
        latest_batch=latest_batch,
        scorecards=scorecards if scorecards else None,
        lineage_summary=lineage,
        idea_yield_summary=idea_yield,
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
