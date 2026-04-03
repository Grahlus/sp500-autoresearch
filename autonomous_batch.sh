#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-$HOME/sp500-autoresearch}

cd "$REPO_DIR"
exec bash "$REPO_DIR/ensure_research_tmux.sh"
