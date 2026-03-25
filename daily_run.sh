#!/bin/bash
# daily_run.sh — triggered by cron at 16:30 ET (Mon–Fri)
# Refreshes data, launches Claude Code in a tmux session for the evening.
set -e

cd /root/sp500-autoresearch
source $HOME/.cargo/env   # uv in PATH

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
DATE=$(date +%Y%m%d)
LOGFILE="$LOG_DIR/session_${DATE}.log"

echo "[$(date)] ── Daily SP500 autoresearch starting ──" | tee -a "$LOGFILE"

# 1. Refresh market data
echo "[$(date)] Refreshing data …" | tee -a "$LOGFILE"
uv run python refresh_data.py 2>&1 | tee -a "$LOGFILE"

# 2. Kill any stale tmux session from previous day
tmux kill-session -t sp500 2>/dev/null || true

# 3. Prompt for Claude Code — runs experiments until midnight ET
CLAUDE_PROMPT='Read program.md carefully. \
Find the next experiment number in the log. \
Form a hypothesis, edit agent.py (set HYPOTHESIS, METRIC, and generate_signals), \
run `uv run python run.py`, \
then IMMEDIATELY run: \
  ./log_exp.sh NNN "hypothesis" "sharpe=X.XX calmar=X.XX alpha=X.XX maxdd=-XX.X% ret=+XXX%" Yes|No \
using the exact numbers from the run.py output. This is MANDATORY before writing any new code. \
If the chosen METRIC improved vs the previous best on validation, \
commit with message "exp#NNN: <hypothesis> | <metric>=<score>", otherwise revert agent.py. \
Immediately start the next experiment without waiting. \
Stop only when the system clock reaches 23:55 local time.'

# 4. Launch detached tmux session
tmux new-session -d -s sp500 -x 220 -y 50 \
  "bash -c 'export ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY; \
  cd /root/sp500-autoresearch && \
  claude --print \"$CLAUDE_PROMPT\" 2>&1 | tee -a $LOGFILE; \
  echo \"[session ended at \$(date)]\" >> $LOGFILE'"

echo "[$(date)] Session launched. Attach with: tmux attach -t sp500" | tee -a "$LOGFILE"
echo "[$(date)] Log: $LOGFILE" | tee -a "$LOGFILE"
