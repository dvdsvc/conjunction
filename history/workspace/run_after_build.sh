#!/bin/bash
# Wait for current build to finish, then run train + eval
set -e
cd /root/autodl-tmp/openalex_data/pencil_data
PYTHON=/root/miniconda3/bin/python

# Wait for build_temporal_data to finish
echo "Waiting for build_temporal_data to finish..."
while ps aux | grep build_temporal_data | grep -v grep > /dev/null 2>&1; do
    sleep 30
done
echo "Build finished at $(date)"

# Check if subgraph files exist
ls -lh /root/autodl-tmp/openalex_data/pencil_data/temporal/fold*_subgraphs.npz
echo ""

echo "=== Step 2: Train all folds ==="
$PYTHON -u train_temporal_lbd.py --train-all

echo "=== Step 3: Eval all folds ==="
$PYTHON -u train_temporal_lbd.py --eval-all

echo "=== PIPELINE COMPLETE ==="
