#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
PY=/root/miniconda3/bin/python
D=/root/autodl-tmp/openalex_data/processed
L=/root/autodl-tmp/pipeline_final.log

echo "=== Pipeline Start $(date) ===" > $L

echo "[1/3] GNN $(date)" >> $L
$PY /root/autodl-tmp/train_gnn.py --data-dir $D --epochs 10 >> $L 2>&1
echo "[1/3] GNN done $(date)" >> $L

echo "[2/3] Phoenix $(date)" >> $L
$PY /root/autodl-tmp/train_phoenix.py --data-dir $D --epochs 20 --batch-size 128 --d-model 384 --n-heads 12 --n-layers 10 --save-dir $D/checkpoints >> $L 2>&1
echo "[2/3] Phoenix done $(date)" >> $L

echo "[3/3] LBD eval $(date)" >> $L
$PY /root/autodl-tmp/evaluate_lbd.py --data-dir $D --checkpoint $D/checkpoints/phoenix_best.pt >> $L 2>&1
echo "[3/3] LBD done $(date)" >> $L

echo "=== Pipeline Done $(date) ===" >> $L
