#!/bin/bash
# Full temporal LBD pipeline: build → train → eval
set -e
cd /root/autodl-tmp/openalex_data/pencil_data
PYTHON=/root/miniconda3/bin/python

echo "=== Step 1: Build temporal data ==="
$PYTHON -u build_temporal_data.py

echo "=== Step 2: Train all folds ==="
$PYTHON -u train_temporal_lbd.py --train-all

echo "=== Step 3: Eval all folds ==="
$PYTHON -u train_temporal_lbd.py --eval-all

echo "=== PIPELINE COMPLETE ==="
