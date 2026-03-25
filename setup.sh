#!/bin/bash
# setup.sh — Run once on the VM to initialise the SP500 autoresearch project.
# Assumes: Python 3.11+, uv, git are already installed.
set -e

echo "=== SP500 Autoresearch Setup ==="

# 1. Install tmux if missing
if ! command -v tmux &>/dev/null; then
    echo "Installing tmux …"
    apt-get install -y tmux
fi

# 2. uv sync (install Python deps from pyproject.toml)
echo "Syncing Python dependencies …"
uv sync

# 3. Make scripts executable
chmod +x daily_run.sh

# 4. Set up git (if not already a repo)
if [ ! -d ".git" ]; then
    git init
    git add .
    git commit -m "initial: sp500 autoresearch scaffold"
fi

# 5. Prompt for API key if not set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "⚠  ANTHROPIC_API_KEY is not set."
    echo "   Add this to ~/.bashrc and re-source:"
    echo '   export ANTHROPIC_API_KEY="sk-ant-..."'
    echo ""
fi

# 6. Add cron job: 16:30 ET Mon–Fri (= 21:30 UTC, adjust if on DST)
CRON_LINE='TZ=America/New_York'$'\n''30 16 * * 1-5 /root/sp500-autoresearch/daily_run.sh >> /root/sp500-autoresearch/logs/cron.log 2>&1'

(crontab -l 2>/dev/null | grep -v 'daily_run.sh'; echo "$CRON_LINE") | crontab -
echo "Cron job installed (16:30 ET Mon–Fri)."
crontab -l

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. export ANTHROPIC_API_KEY='sk-ant-...'"
echo "  2. uv run python refresh_data.py   # ~5-10 min first run"
echo "  3. uv run python run.py            # sanity check baseline"
echo "  4. ./daily_run.sh                  # manual first launch"
echo "  5. tmux attach -t sp500            # watch live"
