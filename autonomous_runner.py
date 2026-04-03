#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from experiment_objective import rank_results
from experiment_proposer import propose_initial_batch, propose_next_batch
from experiment_runner import run_experiment_batch
from experiment_spaces import list_searchable_families
from experiment_store import init_store, load_prior_results
from prepare import load_data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous strategy batch runner")
    parser.add_argument("--family", default="all", help="momentum, superstock, or all")
    parser.add_argument("--method", default="random", choices=["random", "grid"], help="Search method for first round")
    parser.add_argument("--n", type=int, default=20, help="Experiments per family per round")
    parser.add_argument("--rounds", type=int, default=1, help="Autonomous rounds to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base-dir", default="experiments", help="Experiment store root")
    return parser.parse_args(argv)


def _families_from_arg(value: str) -> list[str]:
    if value == "all":
        return list_searchable_families()
    return [value]


def _leaderboard_frame(results) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "family": result.spec.family,
                "config_hash": result.spec.config_hash,
                "status": result.status,
                "score": result.score,
                "sharpe": result.metrics.get("sharpe"),
                "sharpe_min": result.metrics.get("sharpe_min"),
                "total_return_pct": result.metrics.get("total_return_pct"),
                "trades_per_year": result.metrics.get("trades_per_year"),
                "viable": result.robustness.get("viable"),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    init_store(args.base_dir)
    families = _families_from_arg(args.family)
    data = load_data()
    all_batch_results = []

    for round_idx in range(args.rounds):
        batch_id = datetime.now(UTC).strftime(f"%Y%m%d_%H%M%S_r{round_idx + 1}")
        print(f"\n=== Autonomous round {round_idx + 1}/{args.rounds} | batch {batch_id} ===")
        for family in families:
            prior = load_prior_results(family=family, base_dir=args.base_dir)
            if prior.empty or round_idx == 0:
                specs = propose_initial_batch(
                    family=family,
                    method=args.method,
                    n=args.n,
                    seed=args.seed + round_idx,
                    batch_id=batch_id,
                )
            else:
                specs = propose_next_batch(
                    family=family,
                    prior_results=prior,
                    n=args.n,
                    seed=args.seed + round_idx,
                    batch_id=batch_id,
                )
            print(f"[{family}] proposed {len(specs)} experiments")
            results = run_experiment_batch(family, specs, data=data, base_dir=args.base_dir)
            all_batch_results.extend(results)
            if results:
                ranked = rank_results([{"score": r.score, "metrics": r.metrics, "robustness": r.robustness, "spec": {"config_hash": r.spec.config_hash}} for r in results])
                best = ranked[0]
                print(
                    f"[{family}] completed {len(results)} | best score={best['score']:.3f} "
                    f"sharpe={best['metrics'].get('sharpe')} viable={best['robustness'].get('viable')}"
                )
            else:
                print(f"[{family}] no new experiments executed")

        round_rows = _leaderboard_frame(all_batch_results)
        if not round_rows.empty:
            leaderboard_dir = Path(args.base_dir) / "batches" / batch_id
            leaderboard_dir.mkdir(parents=True, exist_ok=True)
            round_rows.sort_values(["status", "score"], ascending=[True, False]).to_csv(
                leaderboard_dir / "leaderboard.csv", index=False
            )

    if all_batch_results:
        print("\n=== Batch summary ===")
        summary = _leaderboard_frame(all_batch_results)
        summary = summary.sort_values(["viable", "score"], ascending=[False, False]).head(10)
        print(summary.to_string(index=False))
    else:
        print("\nNo experiments were executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
