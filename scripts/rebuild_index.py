#!/usr/bin/env python3
"""
Rebuild experiments/index.csv from all batch leaderboard.csv files on disk.

Run with the autoresearch loop stopped:
    uv run python scripts/rebuild_index.py

Safety: reads only, then writes atomically via tempfile + os.replace.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure the project root (parent of scripts/) is on the path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)  # also chdir so BASE_DIR = "experiments" resolves correctly

import pandas as pd

from experiment_store import INDEX_COLUMNS

BASE_DIR = "experiments"


def enrich_from_run_dir(experiment_id: str) -> dict:
    """Pull extra fields that batch leaderboards don't always carry."""
    run_dir = Path(BASE_DIR) / "runs" / str(experiment_id)
    extras: dict = {"result_dir": str(run_dir) if run_dir.exists() else None}
    spec_path = run_dir / "spec.json"
    if spec_path.exists():
        try:
            spec = json.loads(spec_path.read_text())
            for field in ("timestamp_utc", "git_commit", "family_version",
                          "dataset_id", "data_start", "data_end", "benchmark_source"):
                if field in spec:
                    extras[field] = spec[field]
        except Exception:
            pass
    result_path = run_dir / "result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
            metrics = result.get("metrics") or {}
            robustness = result.get("robustness") or {}
            for field, src in [
                ("annual_return", metrics), ("final_value", metrics),
                ("trade_count", metrics), ("avg_hold_days", metrics),
                ("sharpe_min", metrics), ("turnover", metrics),
                ("negative_windows", robustness),
            ]:
                if field in src:
                    extras[field] = src[field]
        except Exception:
            pass
    return extras


def main() -> None:
    # ---- 1. Load all leaderboard CSVs ----
    lb_paths = sorted(glob.glob(f"{BASE_DIR}/batches/*/leaderboard.csv"))
    print(f"Scanning {len(lb_paths)} leaderboard files...")

    frames = []
    errors = 0
    for path in lb_paths:
        try:
            df = pd.read_csv(path, low_memory=False)
            if df.empty or "experiment_id" not in df.columns:
                continue
            if "source_batch_id" not in df.columns:
                df["source_batch_id"] = Path(path).parent.name
            frames.append(df)
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  WARNING: {path}: {exc}")

    if not frames:
        print("ERROR: No leaderboard CSVs found. Aborting.")
        return

    all_lb = pd.concat(frames, ignore_index=True)
    print(f"Total rows loaded: {len(all_lb)}  (errors reading: {errors})")

    # ---- 2. Enrich from per-run directories (timestamp_utc etc.) ----
    print("Enriching from run directories (may take 60–90 s for 26k experiments)...")
    run_extras: dict[str, dict] = {}
    unique_eids = all_lb["experiment_id"].dropna().unique()
    for eid in unique_eids:
        run_extras[str(eid)] = enrich_from_run_dir(str(eid))

    enrich_cols = [
        "result_dir", "timestamp_utc", "git_commit", "family_version",
        "dataset_id", "data_start", "data_end", "benchmark_source",
        "annual_return", "final_value", "trade_count", "avg_hold_days",
        "sharpe_min", "turnover", "negative_windows",
    ]
    for col in enrich_cols:
        if col not in all_lb.columns:
            all_lb[col] = None
        enriched = all_lb["experiment_id"].map(
            lambda eid, c=col: (run_extras.get(str(eid)) or {}).get(c)
        )
        all_lb[col] = enriched.combine_first(all_lb[col])

    # ---- 3. Deduplicate: keep best objective_score per experiment_id ----
    all_lb["_obj_sort"] = pd.to_numeric(
        all_lb["objective_score"], errors="coerce"
    ).fillna(-999.0)
    all_lb = (
        all_lb.sort_values("_obj_sort", ascending=False)
        .drop_duplicates(subset=["experiment_id"], keep="first")
        .drop(columns=["_obj_sort"])
    )

    # Sort by family + timestamp + id for stable ordering
    sort_cols = [c for c in ["strategy_family", "timestamp_utc", "experiment_id"]
                 if c in all_lb.columns]
    if sort_cols:
        all_lb = all_lb.sort_values(sort_cols, na_position="last")
    all_lb = all_lb.reset_index(drop=True)
    print(f"Rows after deduplication: {len(all_lb)}")

    # ---- 4. Ensure all INDEX_COLUMNS present; keep extras too ----
    for col in INDEX_COLUMNS:
        if col not in all_lb.columns:
            all_lb[col] = None
    extra_cols = [c for c in all_lb.columns
                  if c not in INDEX_COLUMNS and c not in ("source_batch_id",)]
    final_cols = INDEX_COLUMNS + extra_cols
    final = all_lb[final_cols]

    # ---- 5. Quick sanity check ----
    missing_required = [c for c in INDEX_COLUMNS if c not in final.columns]
    if missing_required:
        print(f"ERROR: Missing required columns: {missing_required}")
        return

    # ---- 6. Atomic write ----
    index_path = Path(BASE_DIR) / "index.csv"
    with tempfile.NamedTemporaryFile(
        "w", delete=False,
        dir=index_path.parent,
        prefix="index.csv.", suffix=".tmp",
        encoding="utf-8",
    ) as fh:
        final.to_csv(fh, index=False)
        tmp_path = fh.name
    os.replace(tmp_path, str(index_path))

    print(f"\nWrote: {index_path}  ({len(final)} rows, {len(final_cols)} columns)")

    # ---- 7. Verification summary ----
    print("\n=== VERIFICATION ===")
    print(f"Total rows: {len(final)}")
    if "strategy_family" in final.columns:
        print("By family:")
        print(final["strategy_family"].value_counts().to_string())

    viable_mask = final["viable"].astype(str).str.lower().isin({"true", "1", "yes"}) if "viable" in final.columns else None
    if viable_mask is not None and "objective_score" in final.columns:
        top5 = final[viable_mask].nlargest(5, "objective_score")[
            ["experiment_id", "config_hash", "objective_score"]
        ]
        print("\nTop 5 viable by WF Sharpe:")
        print(top5.to_string())

    # Check known best config is present
    EXPECTED_BEST = "e59f68426635b89e"
    if EXPECTED_BEST in final.get("config_hash", pd.Series()).values:
        score = final[final["config_hash"] == EXPECTED_BEST]["objective_score"].max()
        print(f"\nKnown best {EXPECTED_BEST}: present, score={score:.4f}  ✓")
    else:
        print(f"\nWARNING: Known best config {EXPECTED_BEST} NOT FOUND in rebuilt index!")

    EXPECTED_HOT = "d200e40d61a7e10f"
    if EXPECTED_HOT in final.get("config_hash", pd.Series()).values:
        score = final[final["config_hash"] == EXPECTED_HOT]["objective_score"].max()
        print(f"Hot-index best {EXPECTED_HOT}: present, score={score:.4f}  ✓")
    else:
        print(f"NOTE: {EXPECTED_HOT} not in rebuilt index (may only be in hot_index SQLite)")


if __name__ == "__main__":
    main()
