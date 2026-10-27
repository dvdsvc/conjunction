#!/bin/bash
set -e
cd /root/autodl-tmp/openalex_data/pencil_data
PYTHON=/root/miniconda3/bin/python

echo '=== Step 1: Build (pairs + subgraphs) ==='
$PYTHON -u build_temporal_data.py

echo '=== Step 2: Train all folds ==='
$PYTHON -u train_temporal_lbd.py --train-all

echo '=== Step 3: Full eval fold 0 ==='
$PYTHON -u train_temporal_lbd.py --fold 0 --eval-full

echo '=== Step 4: Full eval fold 1 ==='
$PYTHON -u train_temporal_lbd.py --fold 1 --eval-full

echo '=== PIPELINE V2 COMPLETE ==='
