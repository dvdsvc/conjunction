#!/bin/bash
LOG=/root/autodl-tmp/full_pipeline.log
echo "=== $(date) ==="
grep -E "^Epoch|^Best|^Saved|^Computing|^\[|^===|^Data:|^Model" $LOG 2>/dev/null
echo "---"
tail -1 $LOG
echo "---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
