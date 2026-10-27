#!/usr/bin/env python3
"""Train CandidateTower with cross-domain MLP embeddings.

User vectors: Phoenix Transformer (frozen, using original gnn_emb)
Candidate vectors: MLP(cross_mlp_emb 64d → 384d)
"""
import argparse, json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from phoenix_model import PhoenixCitationTransformer
from train_candidate_tower_batch import (
    CandidateTower, load_phoenix, prepare_samples, batch_user_vectors, info_nce_loss,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--n-train", type=int, default=50000)
    p.add_argument("--n-neg", type=int, default=63)
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), map_location="cpu", weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), map_location="cpu", weights_only=False)
    gnn_emb = torch.load(os.path.join(args.data_dir, "gnn_embeddings.pt"), map_location="cpu", weights_only=False)
    cross_emb = torch.load(os.path.join(args.data_dir, "cross_mlp_embeddings.pt"), map_location="cpu", weights_only=False)
    with open(os.path.join(args.data_dir, "config.json")) as f: config = json.load(f)

    phoenix, arch = load_phoenix(
        os.path.join(args.data_dir, "checkpoints/phoenix_best.pt"), config, args.device)
    d_model = arch["d_model"]
    gnn_dim = gnn_emb.size(1)
    cross_dim = cross_emb.size(1)
    concept_dim = config["concept_vocab_size"]
    print(f"Phoenix: d={d_model}, gnn_dim={gnn_dim}, cross_dim={cross_dim}")

    features = fd["features"]
    concepts = fd["concepts"]
    n_total = features.size(0)

    # Load train bridges
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges.pt"), map_location="cpu", weights_only=False)
    year_threshold = 2018
    year_norm_threshold = (year_threshold - 1970) / 56.0
    train_bridges = [b for b in bridges if features[b["source_idx"]][0].item() <= year_norm_threshold]
    print(f"Train bridges (≤2018): {len(train_bridges):,}")
    if len(train_bridges) > args.n_train:
        random.seed(42)
        train_bridges = random.sample(train_bridges, args.n_train)

    # Pre-compute user vectors (using original gnn_emb for Phoenix input)
    print(f"Pre-computing {len(train_bridges)} user vectors...", flush=True)
    samples = prepare_samples(train_bridges, features, concepts, gnn_emb, ref_lists, args.max_history)
    print(f"Valid samples: {len(samples)}")

    t0 = time.time()
    user_vecs = batch_user_vectors(phoenix, samples, features, concepts, gnn_emb,
                                    concept_dim, args.device, batch_size=128)
    print(f"User vectors: {user_vecs.shape} in {time.time()-t0:.0f}s")

    # Collect cross_mlp embeddings for candidates
    pos_cross = torch.stack([cross_emb[s["tgt_idx"]] for s in samples])
    rng = random.Random(42)
    neg_cross = torch.stack([cross_emb[rng.sample(range(n_total), args.n_neg)] for _ in samples])
    print(f"Dataset: user={user_vecs.shape}, pos_cross={pos_cross.shape}, neg_cross={neg_cross.shape}")

    # Train CandidateTower
    class BridgeDataset(Dataset):
        def __init__(self, uv, pc, nc):
            self.uv, self.pc, self.nc = uv, pc, nc
        def __len__(self): return len(self.uv)
        def __getitem__(self, i):
            return {"user_vec": self.uv[i], "pos_gnn": self.pc[i], "neg_gnn": self.nc[i]}

    loader = DataLoader(BridgeDataset(user_vecs, pos_cross, neg_cross),
                        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    cand_tower = CandidateTower(cross_dim, d_model).to(args.device)
    optimizer = torch.optim.AdamW(cand_tower.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining CandidateTower: {sum(p.numel() for p in cand_tower.parameters()):,} params")
    print(f"  epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, neg={args.n_neg}\n")

    best_loss = float('inf')
    for epoch in range(args.epochs):
        cand_tower.train()
        total_loss, n_batches = 0, 0
        t0 = time.time()

        for batch in loader:
            user_vec = batch["user_vec"].to(args.device)
            pos_gnn = batch["pos_gnn"].to(args.device)
            neg_gnn = batch["neg_gnn"].to(args.device)

            pos_cand = cand_tower(pos_gnn)
            neg_cand = cand_tower(neg_gnn.view(-1, cross_dim)).view(-1, args.n_neg, d_model)
            all_cand = torch.cat([pos_cand.unsqueeze(1), neg_cand], dim=1)

            loss = info_nce_loss(user_vec, all_cand)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cand_tower.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        elapsed = time.time() - t0

        cand_tower.eval()
        with torch.no_grad():
            eval_n = min(5000, len(user_vecs))
            uv = user_vecs[:eval_n].to(args.device)
            pg = pos_cross[:eval_n].to(args.device)
            ng = neg_cross[:eval_n].to(args.device)
            pos_sim = (cand_tower(pg) * uv).sum(dim=1)
            neg_sim = (cand_tower(ng.view(-1, cross_dim)).view(eval_n, args.n_neg, d_model) * uv.unsqueeze(1)).sum(dim=2)
            acc = (pos_sim.unsqueeze(1) > neg_sim).float().mean().item()

        marker = " *" if avg_loss < best_loss else ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(cand_tower.state_dict(),
                       os.path.join(args.data_dir, "checkpoints/candidate_tower_cross_best.pt"))
        print(f"Epoch {epoch+1:2d}/{args.epochs}  loss={avg_loss:.4f}  acc={acc:.4f}  {elapsed:.0f}s{marker}", flush=True)

    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Saved: checkpoints/candidate_tower_cross_best.pt")


if __name__ == "__main__":
    main()
