#!/bin/bash
# log_exp.sh — Log one experiment result to program.md
#
# Usage:
#   ./log_exp.sh <exp#> "<hypothesis>" "<metrics string>" <kept>
#
# Example:
#   ./log_exp.sh 002 "VIX-gated shorts on bottom 3% momentum" \
#     "sharpe=1.41 calmar=0.89 alpha=0.042 maxdd=-31.2% ret=+187%" Yes
#
# Metrics string can contain any of:
#   sharpe=X.XX  calmar=X.XX  alpha=X.XX  maxdd=-XX.X%  ret=+XXX%
#
# Kept: Yes = committed, No = reverted

EXP="$1"
HYPOTHESIS="$2"
METRICS="$3"
KEPT="$4"

if [[ -z "$EXP" || -z "$HYPOTHESIS" || -z "$METRICS" || -z "$KEPT" ]]; then
    echo "Usage: ./log_exp.sh <exp#> \"<hypothesis>\" \"<metrics>\" Yes|No"
    exit 1
fi

# Extract individual metrics (safe defaults if not present)
SHARPE=$(echo "$METRICS" | grep -oP 'sharpe=\K[0-9.\-]+' || echo "—")
CALMAR=$(echo "$METRICS" | grep -oP 'calmar=\K[0-9.\-]+' || echo "—")
ALPHA=$(echo "$METRICS"  | grep -oP 'alpha=\K[0-9.\-]+' || echo "—")
MAXDD=$(echo "$METRICS"  | grep -oP 'maxdd=\K[-0-9.%]+' || echo "—")
RET=$(echo "$METRICS"    | grep -oP 'ret=\K[+\-0-9.%]+' || echo "—")

# Format experiment number with leading zeros
EXP_FMT=$(printf "%03d" "$EXP")

# Build table row
ROW="| $EXP_FMT | $HYPOTHESIS | $SHARPE | $CALMAR | $ALPHA | $MAXDD | $RET | $KEPT |"

# Append to program.md experiment log table
echo "$ROW" >> program.md

echo "Logged exp#$EXP_FMT: $ROW"
