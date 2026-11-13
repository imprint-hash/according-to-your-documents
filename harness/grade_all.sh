#!/bin/bash
# Keep grading every subject/condition file (resumable) until each reaches its expected size.
cd "$(dirname "$0")/.."
declare -A EXPECT=(
 [gemma-4-31b]=280 [deepseek-v4-flash]=280 [gemini-3.1-flash-lite]=280 [gpt-oss-120b]=280 [qwen3.8-27b]=280 [qwen3.5-397b-adaption]=280
 [gemma-4-31b@typical-t1]=280 [deepseek-v4-flash@typical-t1]=280 [gemini-3.1-flash-lite@typical-t1]=280 [qwen3.5-397b-adaption@typical-t1]=140
 [gemma-4-31b@typical-t2]=140 [deepseek-v4-flash@typical-t2]=140 [gemini-3.1-flash-lite@typical-t2]=140
 [gemma-4-31b@hardened-t2]=140 [deepseek-v4-flash@hardened-t2]=140 [gemini-3.1-flash-lite@hardened-t2]=140 )
while true; do
  left=0
  for s in "${!EXPECT[@]}"; do
    raw=$(grep -vc '"error"' "results/raw/$s.jsonl" 2>/dev/null); raw=${raw:-0}
    graded=$(wc -l < "results/graded/${s}__glm-5.2.jsonl" 2>/dev/null); graded=${graded:-0}
    if [ "$graded" -lt "${EXPECT[$s]}" ]; then
      left=1
      if [ "$graded" -lt "$raw" ] && ! pgrep -f "grade.py $s " >/dev/null; then
        python3 harness/grade.py "$s" --judge glm-5.2 >> "results/logs/grade-$s.log" 2>&1 &
      fi
    fi
  done
  [ $left -eq 0 ] && echo "all graded" && break
  sleep 120
done
