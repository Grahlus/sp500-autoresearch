# AGENTS.md

## Environment
- Ubuntu Server on Proxmox VM
- CPU only
- Use uv
- Keep runtime practical for ~500 stocks of daily data

## Output style
- Be concise and high-signal
- No long preambles, padding, or repeated summaries
- Show only the changed code with minimal surrounding context
- Do not restate the request unless needed for clarity

## Rules
- Keep changes minimal and isolated
- Do not rewrite unrelated files
- Preserve the current stock engine unless explicitly changing it
- Preserve existing public APIs and behavior unless the task requires a change
- Prefer simple, readable, DRY solutions
- For Python, use type hints and keep code PEP 8 clean
- Do not show a plan by default
- Only use a plan for larger or multi-step work, and keep it short

## Strategy direction
- General daily stock research engine
- Superstock is a first-class strategy family
- Long and short are supported by the engine
- SPY is the benchmark
- VIX and Fear & Greed are context inputs

## Commands
- Setup: `bash setup.sh`
- Refresh data: `uv run python refresh_data.py`
- Run backtest: `uv run python run.py`

## Validation
- Run the narrowest relevant checks first
- Do not run broad or expensive commands unless needed
- State exactly what changed, what was verified, and any blockers

## Done when
- The requested change is implemented with a minimal diff
- Relevant validation has been run
- The response stays compact and focused on the changed parts