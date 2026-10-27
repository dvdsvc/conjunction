#!/usr/bin/env python3
"""Train Phoenix Citation Transformer.

Usage:
  python train_phoenix.py --data-dir /root/autodl-tmp/openalex_data/processed
  python train_phoenix.py --data-dir ./processed --epochs 20 --batch-size 32
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from phoenix_model import (
    PhoenixCitationTransformer,
    CitationDataset,
    collate_variable_length,
)


def load_data(data_dir):
    """Load all processed data."""
    print("Loading data...")
    features_data = torch.load(os.path.join(data_dir, "features.pt"), weights_only=False)
    splits = torch.load(os.path.join(data_dir, "splits.pt"), weights_only=False)
    ref_lists = torch.load(os.path.join(data_dir, "ref_lists.pt"), weights_only=False)

    # Load adjacency
    adj_data = torch.load(os.path.join(data_dir, "adjacency.pt"), weights_only=False)
    adj = adj_data["adj"]  # dict: idx -> [neighbor indices]

    # Load config
    with open(os.path.join(data_dir, "config.json")) as f:
        config = json.load(f)

    scalar_features = features_data["features"]  # (n, 5)
    concept_features = features_data["concepts"]  # (n, vocab_size)

    # Load GNN embeddings if available
    gnn_path = os.path.join(data_dir, "gnn_embeddings.pt")
    if os.path.exists(gnn_path):
        gnn_embeddings = torch.load(gnn_path, weights_only=False)
        print(f"Loaded GNN embeddings: {gnn_embeddings.shape}")
    else:
        # Use zeros if GNN not trained yet
        n = scalar_features.size(0)
        gnn_embeddings = torch.zeros(n, 64)
        print("WARNING: GNN embeddings not found, using zeros")

    return {
        "features": scalar_features,
        "concepts": concept_features,
        "gnn_embeddings": gnn_embeddings,
        "ref_lists": ref_lists,
        "adj": adj,
        "splits": splits,
        "config": config,
    }


def train_epoch(model, loader, optimizer, device, accumulation_steps=1):
    """Train one epoch with bf16 mixed precision."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_pred = 0

    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        scalar = batch["scalar"].to(device, non_blocking=True)
        concept_idx = batch["concept_indices"].to(device, non_blocking=True)
        gnn_emb = batch["gnn_emb"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(scalar, concept_idx, gnn_emb, attn_mask)
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            loss = F.binary_cross_entropy_with_logits(
                logits[valid_mask], labels[valid_mask]
            )
            loss = loss / accumulation_steps

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            preds = (torch.sigmoid(logits[valid_mask]) > 0.5).float()
            correct = (preds == labels[valid_mask]).sum().item()
            total_correct += correct
            total_pred += valid_mask.sum().item()
            total_loss += loss.item() * accumulation_steps * valid_mask.sum().item()

    avg_loss = total_loss / max(total_pred, 1)
    accuracy = total_correct / max(total_pred, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_pred = 0
    all_scores = []
    all_labels = []

    for batch in loader:
        scalar = batch["scalar"].to(device, non_blocking=True)
        concept_idx = batch["concept_indices"].to(device, non_blocking=True)
        gnn_emb = batch["gnn_emb"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(scalar, concept_idx, gnn_emb, attn_mask)

        valid_mask = labels >= 0
        if valid_mask.sum() == 0:
            continue

        loss = F.binary_cross_entropy_with_logits(
            logits[valid_mask], labels[valid_mask])

        probs = torch.sigmoid(logits[valid_mask])
        preds = (probs > 0.5).float()
        correct = (preds == labels[valid_mask]).sum().item()

        total_correct += correct
        total_pred += valid_mask.sum().item()
        total_loss += loss.item() * valid_mask.sum().item()

        all_scores.append(probs.cpu())
        all_labels.append(labels[valid_mask].cpu())

    avg_loss = total_loss / max(total_pred, 1)
    accuracy = total_correct / max(total_pred, 1)

    # Compute AUC
    if all_scores:
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            # Sample-based AUC to avoid OOM on large validation sets
            max_sample = 10000
            ps = pos_scores[torch.randperm(len(pos_scores))[:max_sample]]
            ns = neg_scores[torch.randperm(len(neg_scores))[:max_sample]]
            auc = (ps.unsqueeze(1) > ns.unsqueeze(0)).float().mean().item()
        else:
            auc = 0.5
    else:
        auc = 0.5

    return avg_loss, accuracy, auc


def evaluate_lbd_bridges(model, data, device, top_k=100):
    """Evaluate on LBD bridge ground truth.

    For each bridge source paper, rank all candidates by model score
    and check if the target paper appears in top-K.
    """
    bridges_path = os.path.join(os.path.dirname(data.get("data_dir", "")), "lbd_bridges.pt")
    # This is handled separately — needs full ranking
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--n-negatives", type=int, default=5)
    parser.add_argument("--max-refs", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default=None, help="checkpoint save directory")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    data = load_data(args.data_dir)

    # Create datasets
    concept_vocab_size = data["config"]["concept_vocab_size"]
    train_dataset = CitationDataset(
        paper_indices=data["splits"]["train"],
        ref_lists=data["ref_lists"],
        features=data["features"],
        concepts=data["concepts"],
        concept_vocab_size=concept_vocab_size,
        gnn_embeddings=data["gnn_embeddings"],
        adj=data["adj"],
        mask_ratio=args.mask_ratio,
        n_negatives=args.n_negatives,
        max_refs=args.max_refs,
        max_candidates=args.max_candidates,
    )
    val_dataset = CitationDataset(
        paper_indices=data["splits"]["val"],
        ref_lists=data["ref_lists"],
        features=data["features"],
        concepts=data["concepts"],
        concept_vocab_size=concept_vocab_size,
        gnn_embeddings=data["gnn_embeddings"],
        adj=data["adj"],
        mask_ratio=args.mask_ratio,
        n_negatives=args.n_negatives,
        max_refs=args.max_refs,
        max_candidates=args.max_candidates,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_variable_length,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_variable_length,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # Model
    torch.backends.cudnn.benchmark = True
    concept_vocab_size = data["config"]["concept_vocab_size"]
    gnn_emb_dim = data["gnn_embeddings"].size(1)

    model = PhoenixCitationTransformer(
        scalar_dim=5,
        concept_vocab_size=concept_vocab_size,
        concept_emb_dim=32,
        gnn_emb_dim=gnn_emb_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_seq_len=args.max_refs + args.max_candidates + 2,
        output_multiplier=0.125,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} params ({trainable:,} trainable)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Save dir
    save_dir = args.save_dir or os.path.join(args.data_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    best_auc = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, args.device,
            accumulation_steps=args.accumulation_steps,
        )
        train_time = time.time() - t0

        t0 = time.time()
        val_loss, val_acc, val_auc = evaluate(model, val_loader, args.device)
        eval_time = time.time() - t0

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:2d}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_auc={val_auc:.4f} "
              f"lr={lr:.2e} train={train_time:.0f}s eval={eval_time:.0f}s")

        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_loss": val_loss,
            }, os.path.join(save_dir, "phoenix_best.pt"))
            print(f"  → Saved best (AUC={val_auc:.4f})")

    print(f"\nBest val AUC: {best_auc:.4f}")
    print(f"Checkpoints: {save_dir}")


if __name__ == "__main__":
    main()
