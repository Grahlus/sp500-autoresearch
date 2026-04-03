# AGENTS.md

## Environment
- Ubuntu Server on Proxmox VM
- CPU only
- Use uv
- Keep runtime practical for ~500 stocks daily data

## Rules
- Keep changes minimal and isolated
- Do not rewrite unrelated files
- Show a plan before editing
- Preserve the current stock engine unless explicitly changing it

## Strategy direction
- General daily stock research engine
- Superstock is a first-class strategy family
- Long and short supported by the engine
- SPY is the benchmark
- VIX and Fear & Greed are context inputs

## Commands
- Setup: `bash setup.sh`
- Refresh data: `uv run python refresh_data.py`
- Run backtest: `uv run python run.py`