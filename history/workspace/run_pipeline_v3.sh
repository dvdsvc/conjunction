#!/bin/bash
# Pipeline v3: GPU subgraph build → train → eval
set -e
cd /root/autodl-tmp/openalex_data/pencil_data
PYTHON=/root/miniconda3/bin/python

echo "=== Step 1: Build temporal data (GPU subgraphs) ==="
$PYTHON -u build_temporal_data.py

echo "=== Step 2: Train all folds ==="
$PYTHON -u train_temporal_lbd.py --train-all

echo "=== Step 3: Eval full fold 0 ==="
$PYTHON -u train_temporal_lbd.py --eval-full --fold 0

echo "=== Step 4: Eval full fold 1 ==="
$PYTHON -u train_temporal_lbd.py --eval-full --fold 1

echo "=== PIPELINE COMPLETE ==="
