#!/usr/bin/env python3
"""MLP-based cross-domain embedding (no graph sampling needed).

Replaces GraphSAGE with a simple MLP on scalar features.
Trained on cross-domain link prediction.
GPU-friendly: no NeighborLoader, direct batch training.
"""
import argparse, os, time, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np


class CrossDomainMLP(nn.Module):
    """2-layer MLP: scalar(5) → hidden(128) → out(64), L2-normalized."""
    def __init__(self, in_dim=5, hidden_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class LinkPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, z_src, z_dst):
        return self.mlp(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--num-neg", type=int, default=63)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Device: {args.device}")

    # Load data
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), map_location="cpu", weights_only=False)
    cross_edges = torch.load(os.path.join(args.data_dir, "cross_domain_edges.pt"), map_location="cpu", weights_only=False)

    x = fd["features"].float()
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-6)
    x_norm = ((x - mean) / std).to(args.device)

    n = x.size(0)
    print(f"Nodes: {n:,}, Cross-domain edges: {cross_edges.size(1):,}")

    # Split edges
    year_norm = fd["features"][:, 0]  # year_norm is first feature
    year_threshold = (2018 - 1970) / 56.0
    src_year = year_norm[cross_edges[0]]
    train_mask = src_year <= year_threshold
    val_mask = src_year > year_threshold
    cross_train = cross_edges[:, train_mask].to(args.device)
    cross_val = cross_edges[:, val_mask].to(args.device)
    print(f"Cross-domain: train={cross_train.size(1):,}, val={cross_val.size(1):,}")

    # Models
    mlp = CrossDomainMLP(5, 128, 64).to(args.device)
    predictor = LinkPredictor(64).to(args.device)
    optimizer = torch.optim.AdamW(list(mlp.parameters()) + list(predictor.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    total_params = sum(p.numel() for p in mlp.parameters()) + sum(p.numel() for p in predictor.parameters())
    print(f"Model: {total_params:,} params")

    # Precompute all embeddings for eval
    def compute_all_embeddings():
        mlp.eval()
        with torch.no_grad():
            return mlp(x_norm)  # (n, 64)

    best_auc = 0
    for epoch in range(1, args.epochs + 1):
        mlp.train()
        predictor.train()
        total_loss, n_batches = 0, 0
        t0 = time.time()

        # Shuffle train edges
        perm = torch.randperm(cross_train.size(1), device=args.device)

        for start in range(0, cross_train.size(1), args.batch_size):
            end = min(start + args.batch_size, cross_train.size(1))
            idx = perm[start:end]
            pos_src = cross_train[0, idx]
            pos_dst = cross_train[1, idx]
            n_pos = len(idx)

            # Get embeddings
            z = mlp(x_norm)
            pos_scores = predictor(z[pos_src], z[pos_dst]).clamp(-10, 10)

            # Negative sampling: random dst
            neg_dst = torch.randint(0, n, (n_pos * args.num_neg,), device=args.device)
            neg_src = pos_src.repeat_interleave(args.num_neg)
            neg_scores = predictor(z[neg_src], z[neg_dst]).clamp(-10, 10)

            loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
                   F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(mlp.parameters()) + list(predictor.parameters()), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_time = time.time() - t0

        # Eval
        eval_t0 = time.time()
        with torch.no_grad():
            z = compute_all_embeddings()
            n_eval = min(50000, cross_val.size(1))
            eval_idx = torch.randperm(cross_val.size(1), device=args.device)[:n_eval]
            pos_src_eval = cross_val[0, eval_idx]
            pos_dst_eval = cross_val[1, eval_idx]
            pos_scores_eval = predictor(z[pos_src_eval], z[pos_dst_eval]).cpu()
            neg_dst_eval = torch.randint(0, n, (n_eval,), device=args.device)
            neg_scores_eval = predictor(z[pos_src_eval], z[neg_dst_eval]).cpu()
        scores = torch.cat([pos_scores_eval, neg_scores_eval]).numpy()
        labels = np.concatenate([np.ones(n_eval), np.zeros(n_eval)])
        pos_s, neg_s = scores[labels == 1], scores[labels == 0]
        auc = (pos_s[:, None] > neg_s[None, :]).mean()
        eval_time = time.time() - eval_t0

        avg_loss = total_loss / max(n_batches, 1)
        marker = " *" if auc > best_auc else ""
        if auc > best_auc:
            best_auc = auc
            torch.save(mlp.state_dict(), os.path.join(args.data_dir, "cross_mlp_best.pt"))
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={avg_loss:.4f}  AUC={auc:.4f}  train={train_time:.1f}s eval={eval_time:.1f}s{marker}", flush=True)

    # Save final embeddings
    print("Saving embeddings...")
    mlp.load_state_dict(torch.load(os.path.join(args.data_dir, "cross_mlp_best.pt"), weights_only=False))
    z = compute_all_embeddings().cpu()
    torch.save(z, os.path.join(args.data_dir, "cross_mlp_embeddings.pt"))
    print(f"Saved: {z.shape}, Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
