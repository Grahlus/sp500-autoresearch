#!/usr/bin/env python3
from __future__ import annotations

import argparse

from experiment_batch import (
    build_batch_leaderboard,
    build_batch_request,
    proposal_to_batch_request,
    run_batch_experiments,
)
from experiment_refinement import build_proposal_request, generate_next_round_proposal
from experiment_spaces import list_searchable_families


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--family", default="all", help="momentum, superstock, or all")
    parser.add_argument("--method", default="random", choices=["random", "grid"], help="Sampling method")
    parser.add_argument("--n", type=int, default=20, help="Maximum experiments to run")
    parser.add_argument("--max-per-family", type=int, default=None, help="Maximum sampled configs per family")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base-dir", default="experiments", help="Experiment store root")
    parser.add_argument("--no-persist", action="store_true", help="Do not persist single-experiment results")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip prior successful configs")
    parser.add_argument("--proposal-next", action="store_true", help="Generate the next-round proposal from history")
    parser.add_argument("--run-proposal", action="store_true", help="Run the generated proposal through the batch runner")
    parser.add_argument("--exploration-fraction", type=float, default=0.30, help="Proposal exploration fraction")
    parser.add_argument("--exploitation-fraction", type=float, default=0.70, help="Proposal exploitation fraction")
    return parser.parse_args(argv)


def _families_from_arg(value: str) -> list[str]:
    normalized = value.strip().lower()
    if normalized == "all":
        return list_searchable_families()
    return [normalized]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    families = _families_from_arg(args.family)
    if args.proposal_next:
        proposal_request = build_proposal_request(
            strategy_families=families,
            seed=args.seed,
            exploration_fraction=args.exploration_fraction,
            exploitation_fraction=args.exploitation_fraction,
            max_experiments=args.n,
            resume=not args.no_resume,
        )
        proposal = generate_next_round_proposal(proposal_request, base_dir=args.base_dir)
        print(f"proposal_id={proposal.request.proposal_id}")
        print(f"proposal_status={proposal.status}")
        print(proposal.reasoning_summary)
        if proposal.proposal_path:
            print(f"proposal={proposal.proposal_path}")
        if proposal.summary_path:
            print(f"proposal_summary={proposal.summary_path}")

        if not args.run_proposal:
            return 0

        request = proposal_to_batch_request(
            proposal,
            persist=not args.no_persist,
            resume=not args.no_resume,
        )
    else:
        request = build_batch_request(
            strategy_families=families,
            sampler_type=args.method,
            max_experiments=args.n,
            max_per_family=args.max_per_family,
            seed=args.seed,
            persist=not args.no_persist,
            resume=not args.no_resume,
        )

    batch_result = run_batch_experiments(request, base_dir=args.base_dir)

    print(f"batch_id={batch_result.request.batch_id}")
    print(
        f"sampled={batch_result.total_sampled} executed={batch_result.total_executed} "
        f"skipped={batch_result.total_skipped} failed={batch_result.total_failed}"
    )
    leaderboard = build_batch_leaderboard(batch_result.results)
    if not leaderboard.empty:
        print(leaderboard.head(10).to_string(index=False))
    if batch_result.leaderboard_path:
        print(f"leaderboard={batch_result.leaderboard_path}")
    if batch_result.summary_path:
        print(f"summary={batch_result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
