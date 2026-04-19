# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

```bash
# Install deps (first time or after pyproject.toml changes)
bash setup.sh                         # also inits git and installs cron

# Refresh market data (~500 tickers × 11 years, ~5–10 min)
uv run python refresh_data.py

# Run backtest for the current agent.py strategy
uv run python run.py

# Run true OOS evaluation (do NOT use to tune — contaminates the holdout)
uv run python evaluate.py

# Run a batch of experiments via the autonomous runner
uv run python autonomous_runner.py --n 24 --max-workers 6 --run-proposal

# Run with specific family only
uv run python autonomous_runner.py --family momentum --n 12 --run-proposal

# Show current research status
bash research_status.sh

# Check best results so far
uv run python best_results.py

# Start/resume unattended autonomous loop in tmux
bash ensure_research_tmux.sh
tmux attach -t autoresearch

# Run tests (pytest must be installed via uv add --dev pytest)
uv run python -m pytest tests/
uv run python -m pytest tests/test_experiment_batch.py   # single test file
uv run python -m unittest tests/test_experiment_batch.py  # alternative without pytest
```

---

## Architecture Overview

This repo implements a self-improving research loop for SP500/Russell-1000 equity strategies. Claude Code acts as the autonomous researcher: it proposes experiments, runs walk-forward backtests in parallel, and commits improvements.

### Data flow

```
refresh_data.py   →  data/*.parquet   (prices, VIX, fear & greed)
prepare.py        →  load_data()       (FROZEN — loads parquet, enforces 60-min timeout)
strategies/       →  generate_signals() per family
experiment_runner →  single backtest   (walk-forward or full-period)
experiment_batch  →  parallel runs     (multiprocessing, up to 8 workers)
experiment_store  →  experiments/      (index.csv + per-batch JSON)
autonomous_runner →  orchestrates full proposal→batch→persist cycle
```

### Key modules

| File | Role |
|------|------|
| `prepare.py` | **FROZEN.** Data loader + backtest engine + cost model. Never edit. |
| `agent.py` | Legacy single-strategy agent; original research target. Still runnable via `run.py`. |
| `strategies/` | Multi-family strategy registry. `momentum.py` is the champion family. |
| `strategies/registry.py` | `get_strategy_family(name)` — entry point for all family lookups. |
| `experiment_spaces.py` | Parameter search spaces per family. Defines grid and random sampling. |
| `experiment_runner.py` | Runs a single `ExperimentSpec` → `ExperimentResult`. |
| `experiment_batch.py` | Batches specs, runs in parallel, builds leaderboard and summaries. |
| `experiment_store.py` | Persists results to `experiments/index.csv` and per-batch dirs. |
| `experiment_types.py` | Frozen dataclasses: `ExperimentSpec`, `ExperimentResult`, `BatchRequest`, `ProposalRecord`. |
| `autonomous_runner.py` | CLI entry: plan → propose → batch → summarise cycle. |
| `agents/planning_agent.py` | Claude-based planning agent that constructs `ProposalRecord` from history. |
| `agents/analysis_agent.py` | Claude-based analysis agent; post-batch insight generation. |
| `agents/idea_agent.py` | Claude-based idea generator for queued helper ideas. |
| `agents/schemas.py` | Persistence layer for `IdeaRecord` and `ProposalRecord` (JSON files in `experiments/proposals/`). |
| `experiment_novelty.py` | Config-signature deduplication and near-duplicate suppression. |
| `experiment_refinement.py` | Builds next-round proposals by refining prior winners. |
| `experiment_runtime_decision.py` | Decides execution mode (parallel vs sequential) and worker count. |
| `evaluate.py` | True OOS evaluation on 2024-07-01 → present. Run once, never for tuning. |
| `baseline_registry.py` | Named baselines (e.g. `momentum_champion_s10005`) for fair comparison. |

### Strategy families

- **momentum** — Jegadeesh-Titman cross-sectional momentum; champion family. Params in `strategies/momentum.py`.
- **superstock** — High-conviction fundamental/technical screen. Multi-file in `strategies/superstock_*.py`.
- **ml_ranker** — LightGBM/XGBoost signal layer on top of momentum features.
- **rl_bandit** — Reinforcement learning bandit for position sizing (CPU-only; use sparingly).

### Experiment lifecycle

1. `autonomous_runner.py` calls `experiment_refinement.build_proposal_request()` → AI planning agent → `ProposalRecord` (saved to `experiments/proposals/`).
2. `proposal_to_batch_request()` converts the record to a `BatchRequest`.
3. `run_batch_experiments()` dispatches `ExperimentSpec` items via `multiprocessing`.
4. Results persisted atomically to `experiments/index.csv` and `experiments/batches/<batch_id>/`.
5. Hot index (`experiments/hot_index.sqlite3`) updated for fast leaderboard queries.
6. Scorecards and best-results dashboards updated via `experiment_scorecards.py` / `experiment_best_results.py`.

### Walk-forward validation

14 overlapping windows, 3-year train / 6-month test. Scoring metric is **mean Sharpe across all 14 windows** (`wf_v1_score`). Commit threshold: `mean_sharpe > 0.3`, `neg_windows ≤ 4`, `worst_window > -1.2`, `trades_per_year < 150`.

OOS window (2024-07-01 → present) is evaluated only via `evaluate.py` — never used for tuning.

### Champion strategy (S10-005)

Walk-forward Sharpe 0.722 | OOS Sharpe 1.548. Full specification in `STRATEGY_SPEC.md`. Known-good parameters are documented there — do not re-scan them.

### Novelty and exploration scoring (`experiment_novelty.py`)

Candidate configs are scored on two axes — novelty and objective proxy — then combined into a `selection_score`. All weights are **mode-dependent** so that refinement cycles stay objective-first and escape cycles stay novelty-first.

**Selection weights** (`_get_selection_weights(exploration_mode) → (obj_w, nov_w, dz_w)`):

| Mode | obj_w | nov_w | dz_w |
|------|-------|-------|------|
| confirmation / holdout | 0.65 | 0.35 | 0.28 |
| local_refinement | 0.62 | 0.38 | 0.30 |
| branch_refinement | 0.55 | 0.45 | 0.30 |
| template_expansion | 0.52 | 0.48 | 0.30 |
| broader_exploration | 0.45 | 0.55 | 0.26 |
| large_search | 0.38 | 0.62 | 0.24 |
| stagnation_escape / structural_exploration | 0.33 | 0.67 | 0.20 |

**Dead-zone penalties** — proportional, not binary:
- Exact signature match in dead zone → `dead_zone_risk = 1.0` (always)
- Partial overlap → `min(0.60, 0.15 × n_overlapping_params)`
- Escape/structural modes apply a further `0.70×` multiplier on partial overlap

**Near-duplicate penalties** (`_near_dup_penalties(mode, near_dead_zone)`) — reduced in refinement modes, increased in exploration modes, with an extra increment when the near-dup also overlaps a dead zone:
- confirmation / local_refinement / branch_refinement: `nov -0.12, sel -0.06`
- stagnation_escape / structural_exploration: `nov -0.20, sel -0.10`
- broader_exploration / idea_seed: `nov -0.26, sel -0.13`
- default: `nov -0.30, sel -0.15`
- near dead-zone adds `+0.14 / +0.07` on top

Exact duplicates are always blocked (`score_candidate()` returns `None`). Near-duplicates are not blocked — they receive a score penalty only.

**Structural novelty** (`_structural_novelty_profile()` in `experiment_refinement.py`):
- Cross-family borrowing (`source_type="cross_family_hybrid"`) → `+0.12` structural bonus
- Under-represented template (run share < 50% of median) → `+0.18` bonus
- Over-represented template (run share > 2.5× median) → `−0.08` penalty (does not trigger for structurally-novel sources)

### Adaptive novelty floor and budget fractions (`experiment_refinement.py`)

**Novelty floor** (`_get_novelty_floor(mode, stagnation_batches, is_confirmation, is_holdout)`):
- Confirmation or holdout: always `0.0` — preserves the small-batch bypass
- Stagnation (`stagnation_batches ≥ 2`): `max(0.05, base × 0.50)` regardless of mode
- stagnation_escape / structural_exploration: `max(0.05, base × 0.40)`
- broader_exploration / large_search / idea_seed: `max(0.08, base × 0.65)`
- local_refinement / branch_refinement: `max(0.10, base)` (no reduction)
- Default: `base_floor` (typically `0.15`)

**Adaptive budget fractions** (`_adaptive_budget_fractions(...)`):
- `stagnation_batches ≥ 2`: `template_fraction × 1.25`, `cross_family_fraction × 1.30`
- `branch_dominance ≥ 0.50`: `template_fraction × 1.15` (prevents one lineage crowding out exploration)
- `viable_rate < 0.10`: `template_fraction + 0.05`
- `large-search` mode: `template_fraction = max(current, 0.40)`
- All fractions capped at 0.65 (template) / 0.30 (cross-family)

### Autonomous ops

- Tmux session `autoresearch` is the long-lived container.
- `research_loop.sh` is the inner worker loop (flock-guarded).
- `ensure_research_tmux.sh` is idempotent — safe to re-run.
- Cron watchdog fires every 10 min to ensure the session exists (does not run experiments directly).
- Lock: `run/autonomous_research.lock` | Heartbeat: `run/last_heartbeat.txt`.

---

## Important Constraints

- **`prepare.py` is frozen.** Do not edit the backtest engine, data loader, or cost model.
- **No lookahead.** Signals must only use `close[T]` and earlier; `open[T+1]` is execution-only.
- **Universe**: all ~841 IWB tickers + supplemental ETFs. Do NOT filter to SP500-only at runtime.
- **Costs**: $20 commission per trade + 5 bps slippage, enforced by `prepare.py`.
- **OOS holdout**: 2024-07-01 → present is sacred. Do not use `evaluate.py` output to tune.
- `log_exp.sh` must be called after every `run.py` invocation when using the legacy single-agent loop.
