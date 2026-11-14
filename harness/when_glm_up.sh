#!/bin/bash
# Every 10 min: if GLM-5.2 answers, grade every raw file (resumable) and classify flagged claims. Runs until killed.
cd "$(dirname "$0")/.."
while true; do
  if python3 -c "
import sys; sys.path.insert(0,'harness'); from llm import chat
t,_=chat('glm-5.2','Reply OK','OK?',max_tokens=50); sys.exit(0 if t else 1)" >/dev/null 2>&1; then
    echo "$(date +%H:%M) glm up: grading"
    for f in results/raw/*.jsonl; do s=$(basename "$f" .jsonl); python3 harness/grade.py "$s" --judge glm-5.2 >> logs/grade.log 2>&1; done
    python3 harness/classify_items.py --judge glm-5.2 >> logs/items.log 2>&1
    echo "$(date +%H:%M) pass done"
  else
    echo "$(date +%H:%M) glm down"
  fi
  sleep 600
done
