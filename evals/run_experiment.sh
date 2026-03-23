#!/usr/bin/env bash
# run_experiment.sh — autoresearch-style keep/discard experiment runner
#
# Usage:
#   ./evals/run_experiment.sh "description of change"
#   ./evals/run_experiment.sh --baseline    # record initial baseline
#
# Runs the full 155-case eval, extracts metrics, and either:
#   - KEEP: commits changes + appends to experiment_log.tsv
#   - DISCARD: git resets + logs to dead_ends.jsonl
#
# Prerequisites: llama-server in PATH, pip install "openhas[eval]"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/experiment_log.tsv"
DEAD_ENDS="$SCRIPT_DIR/dead_ends.jsonl"
EVAL_CMD="python $SCRIPT_DIR/recall_eval.py --hybrid --cross-encoder"

description="${1:?Usage: $0 \"description of change\"}"
is_baseline=false
[[ "$description" == "--baseline" ]] && { is_baseline=true; description="baseline — hybrid + cross-encoder"; }

echo "═══ Running experiment: $description"
echo "═══ Eval command: $EVAL_CMD"
echo ""

# capture eval output
eval_output=$($EVAL_CMD 2>&1) || { echo "ERROR: eval failed"; echo "$eval_output"; exit 1; }
echo "$eval_output"

# extract MEAN line metrics
mean_line=$(echo "$eval_output" | grep -E '^\s+MEAN' | head -1)
if [[ -z "$mean_line" ]]; then
    echo "ERROR: could not find MEAN line in eval output"
    exit 1
fi

# parse: MEAN  R@1  R@3  R@5  MRR  [AR@5=X.XX]  time
r1=$(echo "$mean_line" | awk '{print $2}')
r3=$(echo "$mean_line" | awk '{print $3}')
r5=$(echo "$mean_line" | awk '{print $4}')
mrr=$(echo "$mean_line" | awk '{print $5}')
ar5=$(echo "$mean_line" | grep -oP 'AR@5=\K[0-9.]+' || echo "1.00")
latency=$(echo "$mean_line" | grep -oP '[0-9.]+s$' || echo "?")

echo ""
echo "═══ Metrics: R@1=$r1 R@3=$r3 R@5=$r5 MRR=$mrr AR@5=$ar5 latency=$latency"

# get previous best R@1 from log
prev_r1=$(tail -1 "$LOG_FILE" | awk -F'\t' 'NR>0 && $2 != "R@1" {print $2}')
prev_r1="${prev_r1:-0.000}"

commit_hash=$(git -C "$SCRIPT_DIR/.." rev-parse --short HEAD)

if $is_baseline; then
    status="baseline"
    echo -e "${commit_hash}\t${r1}\t${r3}\t${r5}\t${mrr}\t${ar5}\t${latency}\t${status}\t${description}" >> "$LOG_FILE"
    echo "═══ Baseline recorded."
else
    # keep/discard decision
    improved=$(python3 -c "print('yes' if float('$r1') > float('$prev_r1') else 'no')")
    ar5_ok=$(python3 -c "print('yes' if float('$ar5') >= 0.99 else 'no')")

    if [[ "$improved" == "yes" && "$ar5_ok" == "yes" ]]; then
        status="keep"
        echo -e "${commit_hash}\t${r1}\t${r3}\t${r5}\t${mrr}\t${ar5}\t${latency}\t${status}\t${description}" >> "$LOG_FILE"
        echo "═══ KEEP — R@1 improved: $prev_r1 → $r1"
    else
        status="discard"
        delta=$(python3 -c "print(f'{float(\"$r1\") - float(\"$prev_r1\"):.3f}')")
        reason="R@1 regressed" && [[ "$ar5_ok" != "yes" ]] && reason="AR@5 dropped below 1.00"
        echo "{\"date\": \"$(date +%Y-%m-%d)\", \"commit\": \"$commit_hash\", \"description\": \"$description\", \"r1\": $r1, \"r1_delta\": $delta, \"ar5\": $ar5, \"reason\": \"$reason\"}" >> "$DEAD_ENDS"
        echo -e "${commit_hash}\t${r1}\t${r3}\t${r5}\t${mrr}\t${ar5}\t${latency}\t${status}\t${description}" >> "$LOG_FILE"
        echo "═══ DISCARD — $reason (R@1: $prev_r1 → $r1, delta: $delta)"
        echo "═══ Run 'git checkout -- .' to revert changes."
    fi
fi
