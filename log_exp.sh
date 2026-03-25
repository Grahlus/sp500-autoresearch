#!/bin/bash
# log_exp.sh — Log one experiment result to program.md
#
# Usage:
#   ./log_exp.sh <exp#> "<hypothesis>" "<val metrics>" "<train metrics>" <kept>
#
# Metrics strings contain any of:
#   sharpe=X.XX  calmar=X.XX  alpha=X.XX  maxdd=-XX.X%
#   ret=+XXX%    final=$XXXk  cost=$XXX   trades=XXX
#
# Kept: Yes = committed, No = reverted
#
# Example:
#   ./log_exp.sh 048 "F&G + VIX dual filter" \
#     "sharpe=2.21 calmar=5.3 alpha=0.64 maxdd=-18.5% ret=+310% final=$410k cost=$1200 trades=180" \
#     "sharpe=1.95 calmar=4.1 alpha=0.51 maxdd=-22.1% ret=+890% final=$990k cost=$8400 trades=820" \
#     Yes

EXP="$1"
HYPOTHESIS="$2"
VAL_METRICS="$3"
TRAIN_METRICS="$4"
KEPT="$5"

if [[ -z "$EXP" || -z "$HYPOTHESIS" || -z "$VAL_METRICS" || -z "$TRAIN_METRICS" || -z "$KEPT" ]]; then
    echo "Usage: ./log_exp.sh <exp#> \"<hypothesis>\" \"<val metrics>\" \"<train metrics>\" Yes|No"
    exit 1
fi

parse_metrics() {
    local M="$1"
    local SHARPE CALMAR ALPHA MAXDD RET FINAL COST TRADES
    SHARPE=$(echo "$M" | grep -oP 'sharpe=\K[0-9.\-]+');  SHARPE=${SHARPE:-—}
    CALMAR=$(echo "$M" | grep -oP 'calmar=\K[0-9.\-]+');  CALMAR=${CALMAR:-—}
    ALPHA=$(echo "$M"  | grep -oP 'alpha=\K[0-9.\-]+');   ALPHA=${ALPHA:-—}
    MAXDD=$(echo "$M"  | grep -oP 'maxdd=\K[-0-9.%]+');   MAXDD=${MAXDD:-—}
    RET=$(echo "$M"    | grep -oP 'ret=\K[+\-0-9.%]+');   RET=${RET:-—}
    FINAL=$(echo "$M"  | grep -oP 'final=\K\$[0-9kKmM.]+'); FINAL=${FINAL:-—}
    COST=$(echo "$M"   | grep -oP 'cost=\K\$[0-9kKmM.]+');  COST=${COST:-—}
    TRADES=$(echo "$M" | grep -oP 'trades=\K[0-9.]+');    TRADES=${TRADES:-—}
    echo "$SHARPE|$CALMAR|$ALPHA|$MAXDD|$RET|$FINAL|$COST|$TRADES"
}

VAL=$(parse_metrics "$VAL_METRICS")
TRN=$(parse_metrics "$TRAIN_METRICS")

IFS='|' read -r VS VC VA VD VR VF VCO VT <<< "$VAL"
IFS='|' read -r TS TC TA TD TR TF TCO TT <<< "$TRN"

EXP_FMT=$(printf "%03d" "$EXP")

ROW="| $EXP_FMT | $HYPOTHESIS | $VS | $VC | $VA | $VD | $VR | $VF | $VCO | $VT | $TS | $TC | $TD | $TR | $TF | $KEPT |"

echo "$ROW" >> program.md
echo "Logged exp#$EXP_FMT:"
echo "  VAL  : sharpe=$VS calmar=$VC alpha=$VA maxdd=$VD ret=$VR final=$VF cost=$VCO trades=$VT"
echo "  TRAIN: sharpe=$TS calmar=$TC alpha=$TA maxdd=$TD ret=$TR final=$TF cost=$TCO trades=$TT"
echo "  Kept : $KEPT"
