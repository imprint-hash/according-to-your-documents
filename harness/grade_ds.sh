#!/bin/bash
# Grade every raw file with deepseek-v4-flash, 8 files at a time.
cd "$(dirname "$0")/.."
ls results/raw/*.jsonl | xargs -n1 basename | sed 's/\.jsonl$//' | xargs -P 8 -I{} sh -c 'python3 harness/grade.py "{}" --judge deepseek-v4-flash >> logs/grade-ds.log 2>&1'
echo "all done" >> logs/grade-ds.log
