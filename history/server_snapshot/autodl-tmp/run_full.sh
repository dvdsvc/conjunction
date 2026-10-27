#!/bin/bash
set -e
PYTHON="/root/miniconda3/bin/python"
DATA="/root/autodl-tmp/openalex_data/processed"
LOG="/root/autodl-tmp/full_pipeline.log"

echo "=== Phoenix Full Pipeline $(date) ===" | tee $LOG

echo "[1/3] GNN Pre-training..." | tee -a $LOG
$PYTHON /root/autodl-tmp/train_gnn.py --data-dir $DATA --epochs 10 2>&1 | tee -a $LOG
echo "[1/3] GNN done $(date)" | tee -a $LOG

echo "[2/3] Phoenix Transformer..." | tee -a $LOG
$PYTHON /root/autodl-tmp/train_phoenix.py --data-dir $DATA --epochs 20 --batch-size 128 --d-model 384 --n-heads 12 --n-layers 10 --save-dir $DATA/checkpoints 2>&1 | tee -a $LOG
echo "[2/3] Phoenix done $(date)" | tee -a $LOG

echo "[3/3] LBD Evaluation..." | tee -a $LOG
$PYTHON /root/autodl-tmp/evaluate_lbd.py --data-dir $DATA --checkpoint $DATA/checkpoints/phoenix_best.pt 2>&1 | tee -a $LOG
echo "[3/3] LBD done $(date)" | tee -a $LOG

echo "=== Pipeline Complete $(date) ===" | tee -a $LOG
