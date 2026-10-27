#!/usr/bin/env python3
"""Train CandidateTower with all bridge samples + in-batch hard negatives.

Key improvements over train_candidate_tower_batch.py:
1. ALL 1M+ train bridges (not 50K subset)
2. In-batch hard negatives (other positives in batch, not random)
3. Pre-compute user vectors once, then fast training loop

In-batch negatives: for each user, the other B-1 positive candidates in the
batch serve as negatives. These are much harder than random papers because
they are all actual cross-domain bridge targets.
"""
import argparse, json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from phoenix_model import PhoenixCitationTransformer
from train_candidate_tower_batch import (
    CandidateTower, load_phoenix, prepare_samples, batch_user_vectors,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--n-train", type=int, default=0, help="0 = use all bridges")
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
    with open(os.path.join(args.data_dir, "config.json")) as f: config = json.load(f)

    phoenix, arch = load_phoenix(
        os.path.join(args.data_dir, "checkpoints/phoenix_best.pt"), config, args.device)
    d_model = arch["d_model"]
    gnn_dim = gnn_emb.size(1)
    concept_dim = config["concept_vocab_size"]
    print(f"Phoenix: d={d_model}, gnn_dim={gnn_dim}")

    features = fd["features"]
    concepts = fd["concepts"]
    n_total = features.size(0)

    # Load train bridges
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges.pt"), map_location="cpu", weights_only=False)
    year_norm_threshold = (2018 - 1970) / 56.0
    train_bridges = [b for b in bridges if features[b["source_idx"]][0].item() <= year_norm_threshold]
    print(f"Train bridges (≤2018): {len(train_bridges):,}")
    if args.n_train > 0 and len(train_bridges) > args.n_train:
        random.seed(42)
        train_bridges = random.sample(train_bridges, args.n_train)

    # Prepare samples
    print(f"Preparing samples...", flush=True)
    samples = prepare_samples(train_bridges, features, concepts, gnn_emb, ref_lists, args.max_history)
    print(f"Valid samples: {len(samples):,}")

    # Pre-compute user vectors
    print(f"Extracting {len(samples):,} user vectors...", flush=True)
    t0 = time.time()
    user_vecs = batch_user_vectors(phoenix, samples, features, concepts, gnn_emb,
                                    concept_dim, args.device, batch_size=128)
    print(f"User vectors: {user_vecs.shape} in {time.time()-t0:.0f}s")

    # Collect candidate GNN embeddings
    pos_gnn = torch.stack([gnn_emb[s["tgt_idx"]] for s in samples])
    print(f"Dataset: user={user_vecs.shape}, pos_gnn={pos_gnn.shape}")

    # Dataset: no pre-sampled negatives — use in-batch hard negatives
    class SimpleDataset(Dataset):
        def __init__(self, uv, pg):
            self.uv, self.pg = uv, pg
        def __len__(self): return len(self.uv)
        def __getitem__(self, i): return self.uv[i], self.pg[i]

    loader = DataLoader(SimpleDataset(user_vecs, pos_gnn),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    cand_tower = CandidateTower(gnn_dim, d_model).to(args.device)
    optimizer = torch.optim.AdamW(cand_tower.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining CandidateTower (in-batch hard negatives)")
    print(f"  {sum(p.numel() for p in cand_tower.parameters()):,} params")
    print(f"  epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    print(f"  negatives per sample = {args.batch_size - 1} (in-batch)\n")

    best_loss = float('inf')
    for epoch in range(args.epochs):
        cand_tower.train()
        total_loss, n_batches = 0, 0
        t0 = time.time()

        for uv_batch, pg_batch in loader:
            uv = uv_batch.to(args.device)       # (B, 384)
            pg = pg_batch.to(args.device)       # (B, 64)

            cand = cand_tower(pg)               # (B, 384) L2-normalized

            # InfoNCE: user_i should match cand_i, but NOT cand_j (j≠i)
            # logits = (B, B) similarity matrix
            logits = torch.mm(uv, cand.t()) / 0.07  # temperature
            labels = torch.arange(logits.size(0), device=logits.device)
            loss = F.cross_entropy(logits, labels)

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
            pg_eval = pos_gnn[:eval_n].to(args.device)
            cand_eval = cand_tower(pg_eval)

            # In-batch accuracy
            sim = torch.mm(uv, cand_eval.t())  # (eval_n, eval_n)
            preds = sim.argmax(dim=1)
            acc = (preds == torch.arange(eval_n, device=preds.device)).float().mean().item()

        marker = " *" if avg_loss < best_loss else ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(cand_tower.state_dict(),
                       os.path.join(args.data_dir, "checkpoints/candidate_tower_inbatch_best.pt"))
        print(f"Epoch {epoch+1:2d}/{args.epochs}  loss={avg_loss:.4f}  acc={acc:.4f}  {elapsed:.0f}s{marker}", flush=True)

    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Saved: checkpoints/candidate_tower_inbatch_best.pt")


if __name__ == "__main__":
    main()
