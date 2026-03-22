# trading-autoresearch

Karpathy's autoresearch loop adapted for NQ futures. An AI agent iterates
on `agent.py`, runs backtests, and commits improvements — while you sleep.

**Metric**: Calmar ratio on a held-out validation window  
**Hardware**: CPU only, Proxmox VM on Hetzner  
**Time budget**: 15 minutes per experiment

---

## 1. Provision Proxmox VM

In the Proxmox web UI:
- Create VM: Debian 12, 8+ vCPUs, 16–32 GB RAM, 50 GB disk
- Enable SSH access

```bash
ssh root@<vm-ip>
```

---

## 2. System dependencies

```bash
apt update && apt install -y git curl build-essential python3-pip

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env   # or restart shell

# Install Node.js 18+ (required for Claude Code)
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# Install Claude Code
npm install -g @anthropic-ai/claude-code
```

---

## 3. Clone this repo

```bash
git clone https://github.com/YOUR_FORK/trading-autoresearch
cd trading-autoresearch
git init   # if starting fresh from these files
git add . && git commit -m "initial: baseline setup"
```

---

## 4. Copy your NQ data

```bash
# From your local machine or existing server:
rsync -avz /path/to/nq_data/*.parquet root@<vm-ip>:/root/trading-autoresearch/data/

# Or mount via NFS/CIFS if data lives on Proxmox host
```

Expected parquet schema:
```
timestamp  datetime64[ns]
open       float64
high       float64
low        float64
close      float64
volume     float64  (or int64)
```

Column names are normalised to lowercase automatically.

---

## 5. Install Python dependencies

```bash
cd /root/trading-autoresearch
uv sync
```

---

## 6. Verify setup

```bash
# Sanity check: loads data, prints train/val split
uv run python prepare.py

# Run one experiment with the random baseline
uv run python run.py
# Should print Calmar ≈ 0.00 (random)
```

---

## 7. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Add to ~/.bashrc to persist
```

---

## 8. Kick off the autoresearch loop

```bash
cd /root/trading-autoresearch
claude   # opens Claude Code in this repo
```

Then paste this prompt:

```
Read program.md carefully, then start experiment #001.
Form a hypothesis, edit agent.py, run `uv run python run.py`,
record the Calmar result, commit if improved or revert if not,
and continue the loop indefinitely.
```

---

## File structure

```
prepare.py    — FROZEN. Data loader, backtest engine, Calmar metric, timeout
agent.py      — AGENT EDITS THIS. Strategy / model / signals
program.md    — HUMAN EDITS THIS. Research direction, constraints, experiment log
run.py        — Thin wrapper: runs one experiment, prints Calmar
pyproject.toml
data/         — Your NQ parquet files go here (or set NQ_DATA_DIR env var)
```

---

## Tuning the time budget

The 15-minute budget is set in `prepare.py`:
```python
EXPERIMENT_TIMEOUT_SECS = 15 * 60
```

Adjust based on your VM's speed. Goal: ~4 experiments/hour minimum.
If simple rule-based strategies run in <1 min, you can lower it to 5 min
for faster iteration. If you want RL training, 15–30 min is more appropriate.

---

## Tips

- Check `git log --oneline` in the morning to see what the agent found
- The experiment log in `program.md` is your research notebook
- If the agent gets stuck, edit `program.md` to nudge the research direction
- Multiple VMs = multiple parallel agents = faster exploration (just use separate git branches)
