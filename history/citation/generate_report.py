#!/usr/bin/env python3
"""Generate final report after training and evaluation."""
import os, json, time, glob
import torch

DATA_DIR = "/root/autodl-tmp/openalex_data/processed"
REPORT_PATH = "/root/autodl-tmp/openalex_data/report.txt"

lines = []
def log(s=""):
    print(s)
    lines.append(s)

log("=" * 50)
log("Phoenix Pre-training Report")
log("=" * 50)
log()

# 1. Data summary
log("--- Data ---")
fd = torch.load(os.path.join(DATA_DIR, "features.pt"), map_location="cpu", weights_only=False)
splits = torch.load(os.path.join(DATA_DIR, "splits.pt"), map_location="cpu", weights_only=False)
ref_lists = torch.load(os.path.join(DATA_DIR, "ref_lists.pt"), map_location="cpu", weights_only=False)
gnn = torch.load(os.path.join(DATA_DIR, "gnn_embeddings.pt"), map_location="cpu", weights_only=False)

n = fd["features"].size(0)
train_n = len(splits["train"])
val_n = len(splits["val"])
test_n = len(splits["test"])
non_empty = sum(1 for r in ref_lists if len(r) > 0)
avg_refs = sum(len(r) for r in ref_lists) / max(non_empty, 1)

log(f"  Total papers: {n:,}")
log(f"  Train: {train_n:,}  Val: {val_n:,}  Test: {test_n:,}")
log(f"  Non-empty refs: {non_empty:,} ({non_empty/n*100:.1f}%)")
log(f"  Avg refs (non-empty): {avg_refs:.1f}")
log(f"  GNN dim: {gnn.size(1)}")

title_path = os.path.join(DATA_DIR, "title_embeddings.pt")
if os.path.exists(title_path):
    te = torch.load(title_path, map_location="cpu", weights_only=False)
    log(f"  Title embeddings: {te.shape}")
else:
    log(f"  Title embeddings: NOT FOUND")

# 2. Checkpoint
log()
log("--- Checkpoint ---")
ckpt_path = os.path.join(DATA_DIR, "checkpoints/phoenix_best.pt")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    log(f"  Best epoch: {ckpt.get('epoch', '?')}")
    log(f"  Val AUC: {ckpt.get('val_auc', '?'):.4f}")
    log(f"  Val loss: {ckpt.get('val_loss', '?'):.4f}")
    n_params = sum(v.numel() for v in ckpt["model"].values())
    log(f"  Model params (state_dict): {n_params:,}")
else:
    log("  No checkpoint found")

# 3. Config
log()
log("--- Model Config ---")
config_path = os.path.join(DATA_DIR, "config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        log(f"  {k}: {v}")

# 4. LBD bridges
log()
log("--- LBD Bridges ---")
bridges = torch.load(os.path.join(DATA_DIR, "lbd_bridges.pt"), map_location="cpu", weights_only=False)
log(f"  Total bridge pairs: {len(bridges):,}")
year_norm_2021 = (2021 - 1970) / 56.0
test_bridges = [b for b in bridges if fd["features"][b["source_idx"]][0].item() > year_norm_2021]
log(f"  Test bridges (>2021): {len(test_bridges):,}")

log()
log("=" * 50)
log("Report saved to: " + REPORT_PATH)
log("=" * 50)

with open(REPORT_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
