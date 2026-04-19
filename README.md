# sp500-autoresearch

An autonomous equity research engine that iterates on SP500 momentum strategies around the clock. An LLM planning agent proposes experiments, a batch backtester runs them in parallel, and improvements are committed to git — while you sleep.

**4,500+ experiments · 4 strategy families · Real-time dashboard**

---

## What It Does

The system explores a space of quantitative equity strategies using a closed loop:

1. **Planning agent** (LLM) proposes a batch of experiments based on history
2. **Batch runner** executes them in parallel (up to 6 workers on CPU)
3. **Results** are persisted to `experiments/index.csv` with full metrics
4. **Champion registry** tracks the best viable config per family
5. **Storage maintenance** keeps disk usage bounded

The target: beat the **JT momentum baseline** (Sharpe ~1.37 on in-sample) with a strategy that holds up on the held-out 2024–2026 period.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  research_loop.sh  (flock-gated while loop)             │
│  ├── disk_guard    — pause if <5 GB free               │
│  ├── planner       — Layer 2 LLM planner (hourly)      │
│  └── run_cycle     — autonomous_runner.py               │
│       ├── build_proposal_request  (planning_agent LLM) │
│       ├── run_batch_experiments   (multiprocessing)     │
│       └── storage_maintenance     (post-cycle cleanup)  │
└─────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  prepare.py  (FROZEN — never edit)                     │
│  load_data() → walk-forward backtest → metrics + costs  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  Strategy families (strategies/)                        │
│  momentum   — Jegadeesh-Titman cross-sectional momentum │
│  superstock  — fundamental/technical multi-screen        │
│  ml_ranker   — LightGBM/XGBoost signal layer            │
│  rl_bandit   — reinforcement-learning position sizing    │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  agents/                                                 │
│  planning_agent.py  — proposes next experiment batch    │
│  analysis_agent.py   — post-batch insight generation     │
│  idea_agent.py       — queued helper idea generator      │
│  schemas.py          — IdeaRecord / ProposalRecord JSON   │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  dashboard/app.py  (FastAPI — :8000)                     │
│  leaderboard · live runner status · family scorecards    │
└──────────────────────────────────────────────────────────┘
```

### Execution Model

```
close[T]   →   generate_signals()   →   weight[T]
open[T+1]  →   entry fill                          ← realistic
open[T+2]  →   exit / rebalance
P&L[T+1]   =   weight[T] * (open[T+2] / open[T+1] - 1)
```

No lookahead: signals never see future prices. Costs: 5 bps per unit of turnover.

---

## How to Run

### Prerequisites

```bash
# Ubuntu 22.04+, 8 vCPU, 24 GB RAM
apt update && apt install -y git curl build-essential python3-pip tmux

# uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Node.js 20+ (for LLM agents)
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# Claude Code
npm install -g @anthropic-ai/claude-code
```

### Setup

```bash
git clone https://github.com/Grahlus/sp500-autoresearch
cd sp500-autoresearch
bash setup.sh              # installs deps, inits git, cron at 16:30 ET Mon–Fri
export ANTHROPIC_API_KEY="sk-ant-..."
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
```

### First Data Pull

Downloads ~500 tickers × 11 years + VIX + Fear & Greed. Takes ~5–10 min.

```bash
uv run python refresh_data.py
```

### Quick Test

```bash
uv run python run.py              # run single strategy backtest
uv run python evaluate.py        # true OOS evaluation (2024-07-01→present)
```

### Autonomous Loop

```bash
# Start in tmux (auto-resumes after disconnect)
bash ensure_research_tmux.sh
tmux attach -t autoresearch

# Or run the raw loop directly
./research_loop.sh
```

### Dashboard

```bash
uv run python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
# Open: http://localhost:8000
```

---

## Directory Structure

```
sp500-autoresearch/
├── research_loop.sh          # Main loop: lock-gated, disk-aware, planner-aware
├── autonomous_runner.py      # CLI entry: propose → batch → persist
├── experiment_batch.py       # Parallel batch runner (multiprocessing)
├── experiment_runner.py      # Single experiment execution
├── experiment_store.py       # Persists results to experiments/index.csv
├── experiment_refinement.py  # Builds next-round proposals from winners
├── experiment_spaces.py      # Parameter search spaces per family
├── experiment_types.py       # Frozen dataclasses
├── experiment_novelty.py     # Config deduplication / dead-zone suppression
├── experiment_runtime_decision.py  # Auto-selects parallel vs sequential
├── experiment_hot_index.py   # SQLite index for fast leaderboard queries
├── experiment_scorecards.py # Per-family scorecard generation
├── experiment_best_results.py  # Best-results dashboard
├── prepare.py               # FROZEN. Data loader + backtest engine + costs.
├── agent.py                 # Legacy single-strategy agent (run via run.py)
├── evaluate.py              # True OOS evaluation (holdout = 2024-07-01→)

├── strategies/
│   ├── registry.py           # get_strategy_family(name) entry point
│   ├── momentum.py           # Champion family — JT cross-sectional momentum
│   ├── superstock*.py         # Multi-screen fundamental/technical
│   ├── ml_ranker.py          # LightGBM/XGBoost signal layer
│   └── rl_bandit.py          # RL-based position sizing
├── agents/
│   ├── planning_agent.py     # LLM planning: proposes experiment batches
│   ├── analysis_agent.py     # LLM analysis: post-batch insight extraction
│   ├── idea_agent.py          # LLM idea generator for queued ideas
│   └── schemas.py            # IdeaRecord / ProposalRecord JSON persistence
├── dashboard/
│   ├── app.py                # FastAPI app — leaderboard, runner status
│   └── static/                # CSS/JS assets
├── storage_maintenance.py    # Log rotation + stale experiment cleanup
├── baseline_registry.py      # Named baselines (momentum_champion_s10005, etc.)

├── data/                     # Parquet files (gitignored, from refresh_data.py)
├── logs/                     # Session logs (gitignored)
├── experiments/
│   ├── index.csv             # All experiment results (append-only)
│   ├── hot_index.sqlite3     # SQLite for fast leaderboard queries
│   ├── proposals/            # ProposalRecord JSON (LLM decisions)
│   ├── batches/              # Per-batch run artifacts
│   ├── runs/                  # Per-experiment run dirs
│   └── reports/              # family_scorecards.json, best_results.json
├── queues/                   # Queued ideas and helper tasks
├── plans/                   # Layer 2 planner outputs
├── llm/                     # LLM-related state
├── references/              # Papers, notes
├── docs/
│   └── autonomous_ops.md     # Detailed unattended operations guide
├── daily_run.sh             # Cron entry: refresh → launch loop
├── ensure_research_tmux.sh  # Safe tmux session launcher
├── log_exp.sh               # Agent calls this post-run (legacy)
└── pyproject.toml           # Python deps (uv)
```

---

## Current Performance

**As of 2026-04-19 · 4,519 experiments · 16 days of runtime**

| Metric | Value |
|--------|-------|
| Total experiments | 4,519 |
| Unique configs | 2,488 |
| Success rate | 63% (2,849 success / 1,633 no-trades / 37 errors) |
| Viable strategies | 895 (19.8%) |
| Beat baseline | 684 |

### By Strategy Family

| Family | Experiments | Viable Rate | Best Sharpe | Status |
|--------|------------|-------------|-------------|--------|
| **momentum** | 2,694 | 33.3% (896) | **1.062** | Active champion |
| superstock | 1,742 | 0% | 0.280 | Stalled — all no-trade |
| rl_bandit | 42 | 0% | −0.205 | Low priority |
| ml_ranker | 41 | 0% | −1.605 | Holdout check pending |

### Champion Config (momentum)

Config hash: `8e89d89098a61297`

| Metric | Value |
|--------|-------|
| Objective score | 2.059 |
| Sharpe ratio | 1.062 |
| Calmar ratio | 3.92 |
| Annual return | 41.7% |
| Total return | 20.5% (holdout) |
| Max drawdown | −22.3% |
| Trades/year | 27.5 |
| vs RSL baseline (1.37) | −0.31 Sharpe delta |

### Baseline Reference (pre-established, in-sample 2015–2026)

| Config | Sharpe | Calmar | MaxDD |
|--------|--------|--------|-------|
| JT skip=4 + top3% + inv-vol + MA20 + stop20% | **1.37** | — | −33.7% |
| JT skip=4 (no filters) | 1.21 | — | −28.8% |
| skip=0 (no JT) | 1.00 | — | −30.1% |

The champion is **not yet above the RSL baseline** on Sharpe — the loop is still searching. The RSL baseline itself was pre-established and is not re-testable.

---

## Dashboard

Starts on port 8000. Shows:

- **Leaderboard** — top experiments by objective score, filterable by family and viability
- **Runner status** — live PID, heartbeat, cycle count, state (running / success / error / paused)
- **Family scorecards** — per-family viable rate, best score, stagnation signals, exploration/exploitation budget recommendations
- **Best results** — top viable configs per family with full metrics
- **Research log** — tail of the autonomous loop log

---

## Key Files

| File | Role |
|------|------|
| `prepare.py` | **FROZEN.** Data loader + walk-forward backtest engine + cost model. Never edit. |
| `autonomous_runner.py` | Core CLI: `--proposal-next --run-proposal --n 24 --max-workers 6` |
| `research_loop.sh` | Production loop wrapper: lock, disk guard, planner, maintenance |
| `strategies/momentum.py` | Champion family — JT momentum with all active filters |
| `experiment_refinement.py` | Builds next proposal round by refining prior winners |
| `experiment_batch.py` | Multiprocessing batch execution + leaderboard |
| `baseline_registry.py` | Named baselines for comparison |
| `evaluate.py` | True OOS evaluation — run **once**, never for tuning |

---

## Research Frontier

Priority order (see `program.md` for full tracking):

1. **Short side** — short bottom 3% momentum, VIX-gated
2. **VIX regime filter** — reduce/exit longs when VIX > 25
3. **Fear & Greed contrarian** — extreme readings as position-size modifier
4. **Momentum + short-term reversal hybrid** — different holding periods
5. **Volume breakout confirmation** — filter weak momentum signals
6. **52-week high proximity** — George & Hwang factor overlay
7. **ML signal layer** — LightGBM on rolling features, trained on first 9 years
8. **Turtle short side** — 20-day breakdown entries, ATR sizing
