#!/usr/bin/env python3
"""GraphSAGE pre-training on cross-domain edges only.

Optimized: full-graph inference for eval (no NeighborLoader eval loop).
"""
import argparse, os, time, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


class GraphSAGE(nn.Module):
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
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, z_src, z_dst):
        return self.mlp(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)


@torch.no_grad()
def evaluate_edges(model, predictor, x, edge_index, cross_edges, device, n_sample=50000):
    """Evaluate AUC on a sample of cross-domain edges."""
    model.eval()
    predictor.eval()

    # Full graph inference (whole graph at once)
    z = model(x.to(device), edge_index.to(device)).cpu()

    # Sample cross-domain edges
    n_cross = cross_edges.size(1)
    if n_cross > n_sample:
        idx = torch.randperm(n_cross)[:n_sample]
    else:
        idx = torch.arange(n_cross)

    pos_src = cross_edges[0, idx]
    pos_dst = cross_edges[1, idx]
    pos_scores = predictor(z[pos_src].to(device), z[pos_dst].to(device)).cpu()

    # Random negatives
    neg_dst = torch.randint(0, x.size(0), (len(idx),))
    neg_scores = predictor(z[pos_src].to(device), z[neg_dst].to(device)).cpu()

    # AUC
    scores = torch.cat([pos_scores, neg_scores]).numpy()
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    pos_s = scores[labels == 1]
    neg_s = scores[labels == 0]
    auc = (pos_s[:, None] > neg_s[None, :]).mean()
    return auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--out-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--num-neighbors", type=int, nargs="+", default=[15, 10])
    p.add_argument("--num-neg", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Device: {args.device}")

    # Load data
    print("Loading data...")
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), weights_only=False)
    adj = torch.load(os.path.join(args.data_dir, "adjacency.pt"), weights_only=False)
    cross_edges = torch.load(os.path.join(args.data_dir, "cross_domain_edges.pt"), weights_only=False)
    splits = torch.load(os.path.join(args.data_dir, "splits.pt"), weights_only=False)

    x = fd["features"].float()
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-6)
    x = (x - mean) / std
    full_ei = adj["edge_index"]
    n = x.size(0)

    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[splits["train"]] = True
    cross_train = cross_edges[:, train_mask[cross_edges[0]]]

    print(f"Nodes: {n:,}, Full edges: {full_ei.size(1):,}, Cross-domain train edges: {cross_train.size(1):,}")

    data = Data(x=x, edge_index=full_ei)
    train_loader = NeighborLoader(data, num_neighbors=args.num_neighbors,
                                  batch_size=args.batch_size, input_nodes=train_mask)

    model = GraphSAGE(x.size(1), args.hidden_dim, args.out_dim).to(args.device)
    predictor = LinkPredictor(args.out_dim).to(args.device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)
    print(f"Model: {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in predictor.parameters()):,} params")

    # Pre-build source→edge index
    cross_src_map = {}
    for i in range(cross_train.size(1)):
        cross_src_map.setdefault(cross_train[0, i].item(), []).append(i)

    best_auc = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        predictor.train()
        total_loss, n_batches = 0, 0
        t0 = time.time()

        for batch in train_loader:
            batch = batch.to(args.device)
            z = model(batch.x, batch.edge_index)
            nid_to_local = {nid.item(): i for i, nid in enumerate(batch.n_id)}

            pos_pairs = []
            for nid in batch.n_id.cpu().tolist():
                if nid in cross_src_map:
                    for ei in cross_src_map[nid]:
                        dst = cross_train[1, ei].item()
                        if dst in nid_to_local:
                            pos_pairs.append((nid_to_local[nid], nid_to_local[dst]))

            if len(pos_pairs) < 8:
                continue

            pos_src = torch.tensor([p[0] for p in pos_pairs], device=args.device)
            pos_dst = torch.tensor([p[1] for p in pos_pairs], device=args.device)
            n_pos = len(pos_pairs)

            pos_scores = predictor(z[pos_src], z[pos_dst]).clamp(-10, 10)
            neg_dst = torch.randint(0, z.size(0), (n_pos * args.num_neg,), device=args.device)
            neg_src = pos_src.repeat_interleave(args.num_neg)
            neg_scores = predictor(z[neg_src], z[neg_dst]).clamp(-10, 10)

            loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
                   F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(predictor.parameters()), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        train_time = time.time() - t0

        # Eval: full-graph inference (fast, one pass)
        eval_t0 = time.time()
        auc = evaluate_edges(model, predictor, x, full_ei, cross_edges, args.device)
        eval_time = time.time() - eval_t0

        avg_loss = total_loss / max(n_batches, 1)
        marker = " *" if auc > best_auc else ""
        if auc > best_auc:
            best_auc = auc
            torch.save({"model": model.state_dict(), "predictor": predictor.state_dict(),
                        "epoch": epoch, "auc": auc},
                       os.path.join(args.data_dir, "gnn_cross_best.pt"))
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={avg_loss:.4f}  AUC={auc:.4f}  train={train_time:.1f}s eval={eval_time:.1f}s{marker}", flush=True)

    # Save final embeddings
    print("Saving final embeddings...")
    ckpt = torch.load(os.path.join(args.data_dir, "gnn_cross_best.pt"), weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        z = model(data.x.to(args.device), data.edge_index.to(args.device)).cpu()
    torch.save(z, os.path.join(args.data_dir, "gnn_cross_embeddings.pt"))
    print(f"Saved: {z.shape}, Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
