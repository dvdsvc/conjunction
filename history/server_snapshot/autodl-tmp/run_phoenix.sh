#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
PYTHON="/root/miniconda3/bin/python"
DATA="/root/autodl-tmp/openalex_data/processed"
LOG="/root/autodl-tmp/phoenix_run.log"

echo "=== Phoenix Training $(date) ===" | tee $LOG

echo "[1/2] Training Phoenix Transformer..." | tee -a $LOG
$PYTHON /root/autodl-tmp/train_phoenix.py \
    --data-dir $DATA \
    --epochs 20 \
    --batch-size 128 \
    --d-model 384 \
    --n-heads 12 \
    --n-layers 10 \
    --save-dir $DATA/checkpoints 2>&1 | tee -a $LOG

echo "[2/2] LBD Evaluation..." | tee -a $LOG
$PYTHON /root/autodl-tmp/evaluate_lbd.py \
    --data-dir $DATA \
    --checkpoint $DATA/checkpoints/phoenix_best.pt 2>&1 | tee -a $LOG

echo "=== Done $(date) ===" | tee -a $LOG
