#!/bin/bash
set -euo pipefail

cd /root/sp500-autoresearch
source "$HOME/.cargo/env"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
DATE=$(date +%Y%m%d)
LOGFILE="$LOG_DIR/autonomous_${DATE}.log"

echo "[$(date)] ── Autonomous strategy batch starting ──" | tee -a "$LOGFILE"
echo "[$(date)] Refreshing data …" | tee -a "$LOGFILE"
uv run python refresh_data.py 2>&1 | tee -a "$LOGFILE"

echo "[$(date)] Running autonomous experiment batches …" | tee -a "$LOGFILE"
uv run python autonomous_runner.py --family all --method random --n 20 --rounds 2 --seed 42 2>&1 | tee -a "$LOGFILE"

echo "[$(date)] Autonomous batch complete." | tee -a "$LOGFILE"
