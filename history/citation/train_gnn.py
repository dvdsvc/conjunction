#!/usr/bin/env python3
"""GraphSAGE pre-training for structural paper embeddings.

Learns 64-dim structural embeddings for 3.4M papers via link prediction
on the citation graph. These embeddings capture 1-2 hop neighborhood
structure ("引火") and feed into the Phoenix Transformer ("点火").

Requires PyTorch Geometric: pip install torch-geometric torch-scatter torch-sparse
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.nn import SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("WARNING: PyTorch Geometric not installed. Install with:")
    print("  pip install torch-geometric torch-scatter torch-sparse")


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE for link prediction."""

    def __init__(self, in_dim, hidden_dim=128, out_dim=64, dropout=0.0):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class LinkPredictor(nn.Module):
    """MLP-based link predictor for edge scoring."""

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, z_src, z_dst):
        h = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(h).squeeze(-1)


def train_epoch(model, predictor, loader, optimizer, device, num_neg=5):
    """One training epoch with mini-batch sampling."""
    model.train()
    predictor.train()
    total_loss = 0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        edge_index = batch.edge_index

        # Get embeddings for this batch
        z = model(x, edge_index)

        # Positive edges (from batch)
        # batch.e_id gives us the original edge indices
        # batch.edge_label_index gives us the (src, dst) pairs to score
        if hasattr(batch, 'edge_label_index'):
            pos_src = batch.edge_label_index[0]
            pos_dst = batch.edge_label_index[1]
        else:
            # Fallback: use first N edges
            n_pos = min(edge_index.size(1) // 2, 512)
            pos_src = edge_index[0, :n_pos]
            pos_dst = edge_index[1, :n_pos]

        n_pos = pos_src.size(0)
        if n_pos == 0:
            continue

        pos_scores = predictor(z[pos_src], z[pos_dst]).clamp(-10, 10)

        # Negative sampling: random dst nodes
        neg_dst = torch.randint(0, z.size(0), (n_pos * num_neg,), device=device)
        neg_src = pos_src.repeat_interleave(num_neg)
        neg_scores = predictor(z[neg_src], z[neg_dst]).clamp(-10, 10)

        # Binary cross-entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(predictor.parameters()), 1.0)
        optimizer.step()

        total_loss += loss.item() * n_pos
        total_examples += n_pos

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  NaN/Inf at step: loss={loss.item()} n_pos={n_pos}")
            optimizer.zero_grad()
            continue

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(model, predictor, loader, device, num_neg=5):
    """Evaluate link prediction (AUC on held-out edges)."""
    model.eval()
    predictor.eval()
    all_scores = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        edge_index = batch.edge_index
        z = model(x, edge_index)

        if hasattr(batch, 'edge_label_index'):
            pos_src = batch.edge_label_index[0]
            pos_dst = batch.edge_label_index[1]
        else:
            n_pos = min(edge_index.size(1) // 2, 512)
            pos_src = edge_index[0, :n_pos]
            pos_dst = edge_index[1, :n_pos]

        n_pos = pos_src.size(0)
        if n_pos == 0:
            continue
        pos_scores = predictor(z[pos_src], z[pos_dst]).clamp(-10, 10)
        neg_dst = torch.randint(0, z.size(0), (n_pos * num_neg,), device=device)
        neg_src = pos_src.repeat_interleave(num_neg)
        neg_scores = predictor(z[neg_src], z[neg_dst]).clamp(-10, 10)

        all_scores.append(pos_scores.cpu())
        all_scores.append(neg_scores.cpu())
        all_labels.append(torch.ones(n_pos))
        all_labels.append(torch.zeros(n_pos * num_neg))

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    # Simple AUC approximation
    # Sample-based AUC to avoid OOM
    max_sample = 10000
    ps = scores[labels == 1]
    ns = scores[labels == 0]
    ps = ps[torch.randperm(len(ps))[:max_sample]]
    ns = ns[torch.randperm(len(ns))[:max_sample]]
    auc = (ps.unsqueeze(1) > ns.unsqueeze(0)).float().mean()
    return auc.item()


def save_embeddings(model, data, device, output_path, batch_size=65536):
    """Compute and save embeddings for all nodes."""
    model.eval()
    n = data.x.size(0)
    embeddings = torch.zeros(n, model.conv2.out_channels)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            # For full-graph inference, we need the full edge_index
            # In practice, use NeighborLoader for large graphs
            z = model(data.x.to(device), data.edge_index.to(device))
            embeddings[start:end] = z[start:end].cpu()

    torch.save(embeddings, output_path)
    print(f"Saved embeddings: {embeddings.shape} -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="NeighborLoader batch size")
    parser.add_argument("--num-neighbors", type=int, nargs="+", default=[15, 10],
                        help="Neighbors per layer for NeighborLoader")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not HAS_PYG:
        return

    print(f"Device: {args.device}")

    # Load data
    print("Loading data...")
    features_data = torch.load(os.path.join(args.data_dir, "features.pt"), weights_only=False)
    adj_data = torch.load(os.path.join(args.data_dir, "adjacency.pt"), weights_only=False)
    splits = torch.load(os.path.join(args.data_dir, "splits.pt"), weights_only=False)

    # Build PyG Data object
    # Features: scalar (5-dim) only — concepts too large to densify
    scalar_features = features_data["features"]  # (n, 5)
    # concept_features = features_data["concepts"]  # sparse, skip for GNN
    x = scalar_features.float()
    # Normalize features to prevent gradient explosion
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
    x = (x - mean) / std
    edge_index = adj_data["edge_index"]

    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index)

    # Add train/val masks
    n = x.size(0)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[splits["train"]] = True
    val_mask[splits["val"]] = True
    data.train_mask = train_mask
    data.val_mask = val_mask

    print(f"Data: {n} nodes, {edge_index.size(1)} edges, {x.size(1)} features")

    # NeighborLoader for mini-batch training
    train_loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        input_nodes=data.train_mask,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        input_nodes=data.val_mask,
    )

    # Models
    model = GraphSAGE(
        in_dim=x.size(1),
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        dropout=args.dropout,
    ).to(args.device)
    predictor = LinkPredictor(args.out_dim).to(args.device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr,
    )

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in predictor.parameters())
    print(f"Model params: {total_params:,}")

    # Training loop
    best_auc = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, predictor, train_loader, optimizer, args.device)
        train_time = time.time() - t0

        t0 = time.time()
        auc = evaluate(model, predictor, val_loader, args.device)
        eval_time = time.time() - t0

        print(f"Epoch {epoch:2d}: loss={loss:.4f} AUC={auc:.4f} "
              f"train={train_time:.1f}s eval={eval_time:.1f}s")

        if auc > best_auc:
            best_auc = auc
            # Save best model
            torch.save({
                "model": model.state_dict(),
                "predictor": predictor.state_dict(),
                "epoch": epoch,
                "auc": auc,
            }, os.path.join(args.data_dir, "gnn_best.pt"))

    # Compute and save final embeddings for all nodes
    print("Computing full-graph embeddings...")
    checkpoint = torch.load(os.path.join(args.data_dir, "gnn_best.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model"])
    save_embeddings(model, data, args.device,
                    os.path.join(args.data_dir, "gnn_embeddings.pt"))

    print(f"Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
