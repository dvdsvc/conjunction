#!/bin/bash
# Full training pipeline for the Phoenix citation model.
# Run on the 5090 server after concepts download is complete.
#
# Usage: bash run_pipeline.sh [--skip-gnn] [--data-dir /path/to/data]

set -euo pipefail

DATA_DIR="${1:-/root/autodl-tmp/openalex_data}"
PROCESSED_DIR="$DATA_DIR/processed"
PYTHON="/root/miniconda3/bin/python"
NPROC=$(nproc)
WORKERS=6  # conservative: leave room for other tasks on this server

echo "=== Phoenix Citation Pipeline ==="
echo "Data dir: $DATA_DIR"
echo "CPU cores: $NPROC, using $WORKERS workers"
echo ""

# Step 1: Install dependencies
echo "[1/4] Installing dependencies..."
pip install torch-geometric torch-scatter torch-sparse 2>/dev/null || \
  echo "WARNING: PyG install failed, GNN will use fallback"

# Step 2: Process data
echo "[2/4] Processing data..."
$PYTHON process_data.py --data-dir "$DATA_DIR" --workers "$WORKERS"
echo ""

# Step 3: GNN pre-training (optional)
if [[ "${2:-}" != "--skip-gnn" ]]; then
    echo "[3/4] Training GNN..."
    $PYTHON train_gnn.py --data-dir "$PROCESSED_DIR" --epochs 10
else
    echo "[3/4] Skipping GNN (--skip-gnn)"
    # Create dummy GNN embeddings
    $PYTHON -c "
import torch, json
with open('$PROCESSED_DIR/config.json') as f:
    cfg = json.load(f)
n = cfg['n_papers']
torch.save(torch.zeros(n, 64), '$PROCESSED_DIR/gnn_embeddings.pt')
print(f'Created dummy GNN embeddings: ({n}, 64)')
"
fi
echo ""

# Step 4: Train Phoenix Transformer
echo "[4/4] Training Phoenix Transformer..."
$PYTHON train_phoenix.py \
    --data-dir "$PROCESSED_DIR" \
    --epochs 20 \
    --batch-size 128 \
    --d-model 384 \
    --n-heads 12 \
    --n-layers 10 \
    --lr 3e-4 \
    --accumulation-steps 1 \
    --mask-ratio 0.2 \
    --n-negatives 5 \
    --max-refs 100 \
    --max-candidates 24 \
    --num-workers 8
echo ""

echo "=== Pipeline complete ==="
echo "Checkpoints: $PROCESSED_DIR/checkpoints/"
