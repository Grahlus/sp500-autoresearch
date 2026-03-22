"""
run.py — autoresearch loop runner.

Usage:
  python run.py           # runs one experiment and prints Calmar
  python run.py --run     # same (alias used by program.md instructions)

The agent (Claude Code) calls this after each edit to agent.py.
Git commit/revert logic is handled by the agent itself per program.md.
"""

import sys
import prepare

if __name__ == "__main__":
    score = prepare.run_experiment()
    # Exit code 0 always — agent reads stdout for the Calmar score
    sys.exit(0)
