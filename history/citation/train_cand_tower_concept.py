#!/usr/bin/env python3
"""Train CandidateTower with gnn_emb + 1-hop concept features.

Input: gnn_emb (64d) + 1-hop concept aggregation via Phoenix's concept_proj (32d) = 96d
Training: InfoNCE with in-batch negatives, all 1M+ bridge pairs
"""
import argparse, json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from phoenix_model import PhoenixCitationTransformer
from train_candidate_tower_batch import load_phoenix, prepare_samples, batch_user_vectors


class CandidateTowerConcept(nn.Module):
    """MLP: [gnn_emb(64d) + concept_proj(32d)] → 384d, L2-normalized."""
    def __init__(self, gnn_dim, concept_dim, concept_emb_dim, d_model, concept_proj_weight):
        super().__init__()
        self.concept_proj = nn.Linear(concept_dim, concept_emb_dim, bias=False)
        with torch.no_grad():
            self.concept_proj.weight.copy_(concept_proj_weight)
        self.concept_proj.weight.requires_grad = False  # freeze

        total_dim = gnn_dim + concept_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, gnn_emb, concept_sparse_idx, concept_sparse_val, nnz_per_sample):
        """Forward with sparse concept features built from indices/values."""
        B = gnn_emb.size(0)
        device = gnn_emb.device
        concept_dim = self.concept_proj.in_features

        # Build sparse concept multi-hot
        concept_mh = torch.sparse_coo_tensor(
            concept_sparse_idx.to(device),
            concept_sparse_val.to(device),
            size=(B, concept_dim)
        ).to_dense()

        concept_emb = self.concept_proj(concept_mh)  # (B, 32)
        x = torch.cat([gnn_emb, concept_emb], dim=-1)  # (B, 96)
        return F.normalize(self.mlp(x), p=2, dim=-1)


def build_1hop_concept_sparse(ref_lists, concepts, candidate_indices, concept_vocab_size):
    """Build sparse 1-hop concept multi-hot for a set of candidate papers.

    Returns: indices, values, nnz_per_sample for torch.sparse_coo_tensor.
    """
    all_rows, all_cols, all_vals = [], [], []
    nnz_per = []

    for row, idx in enumerate(candidate_indices):
        unique_concepts = set()
        for r in ref_lists[idx]:
            unique_concepts.update(concepts[r])
        valid = [c for c in unique_concepts if c < concept_vocab_size]
        nnz_per.append(len(valid))
        for c in valid:
            all_rows.append(row)
            all_cols.append(c)
            all_vals.append(1.0)

    if not all_rows:
        indices = torch.zeros(2, 1, dtype=torch.long)
        values = torch.zeros(1)
    else:
        indices = torch.tensor([all_rows, all_cols], dtype=torch.long)
        values = torch.tensor(all_vals, dtype=torch.float32)

    return indices, values, nnz_per


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
    concept_vocab_size = config["concept_vocab_size"]
    concept_emb_dim = phoenix.concept_proj.weight.shape[0]  # 32
    print(f"Phoenix: d={d_model}, gnn={gnn_dim}, concept={concept_vocab_size}→{concept_emb_dim}")

    features = fd["features"]
    concepts = fd["concepts"]

    # Load train bridges
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges.pt"), map_location="cpu", weights_only=False)
    year_norm_threshold = (2018 - 1970) / 56.0
    train_bridges = [b for b in bridges if features[b["source_idx"]][0].item() <= year_norm_threshold]
    print(f"Train bridges (≤2018): {len(train_bridges):,}")
    if args.n_train > 0 and len(train_bridges) > args.n_train:
        random.seed(42)
        train_bridges = random.sample(train_bridges, args.n_train)

    # Prepare samples
    print("Preparing samples...", flush=True)
    samples = prepare_samples(train_bridges, features, concepts, gnn_emb, ref_lists, args.max_history)
    print(f"Valid samples: {len(samples):,}")

    # Pre-compute user vectors
    print(f"Extracting {len(samples):,} user vectors...", flush=True)
    t0 = time.time()
    user_vecs = batch_user_vectors(phoenix, samples, features, concepts, gnn_emb,
                                    concept_vocab_size, args.device, batch_size=128)
    print(f"User vectors: {user_vecs.shape} in {time.time()-t0:.0f}s")

    # Candidate GNN embeddings
    pos_gnn = torch.stack([gnn_emb[s["tgt_idx"]] for s in samples])
    tgt_indices = [s["tgt_idx"] for s in samples]
    print(f"Dataset: user={user_vecs.shape}, pos_gnn={pos_gnn.shape}")

    # Dataset
    class BridgeDataset(Dataset):
        def __init__(self, uv, pg, ti):
            self.uv, self.pg, self.ti = uv, pg, ti
        def __len__(self): return len(self.uv)
        def __getitem__(self, i): return self.uv[i], self.pg[i], self.ti[i]

    def collate_fn(batch):
        uv, pg, ti = zip(*batch)
        uv = torch.stack(uv)
        pg = torch.stack(pg)
        ti = list(ti)
        # Build sparse concept features for this batch
        ci, cv, nnz = build_1hop_concept_sparse(ref_lists, concepts, ti, concept_vocab_size)
        return uv, pg, ti, ci, cv, nnz

    loader = DataLoader(BridgeDataset(user_vecs, pos_gnn, tgt_indices),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_fn)

    # Model: CandidateTower with 1-hop concept
    cand_tower = CandidateTowerConcept(
        gnn_dim, concept_vocab_size, concept_emb_dim, d_model,
        phoenix.concept_proj.weight.detach().clone()
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        [p for p in cand_tower.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in cand_tower.parameters() if p.requires_grad)
    print(f"\nTraining CandidateTower+Concept: {n_params:,} trainable params")
    print(f"  epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    print(f"  negatives = {args.batch_size - 1} (in-batch)\n")

    best_loss = float('inf')
    for epoch in range(args.epochs):
        cand_tower.train()
        total_loss, n_batches = 0, 0
        t0 = time.time()

        for uv_batch, pg_batch, ti_batch, ci, cv, nnz in loader:
            uv = uv_batch.to(args.device)
            pg = pg_batch.to(args.device)

            cand = cand_tower(pg, ci, cv, nnz)

            logits = torch.mm(uv, cand.t()) / 0.07
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

        # Eval
        cand_tower.eval()
        with torch.no_grad():
            eval_n = min(5000, len(user_vecs))
            uv = user_vecs[:eval_n].to(args.device)
            pg = pos_gnn[:eval_n].to(args.device)
            ti_eval = tgt_indices[:eval_n]
            ci, cv, nnz = build_1hop_concept_sparse(ref_lists, concepts, ti_eval, concept_vocab_size)
            cand_eval = cand_tower(pg, ci, cv, nnz)
            sim = torch.mm(uv, cand_eval.t())
            preds = sim.argmax(dim=1)
            acc = (preds == torch.arange(eval_n, device=preds.device)).float().mean().item()

        marker = " *" if avg_loss < best_loss else ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(cand_tower.state_dict(),
                       os.path.join(args.data_dir, "checkpoints/cand_tower_concept_best.pt"))
        print(f"Epoch {epoch+1:2d}/{args.epochs}  loss={avg_loss:.4f}  acc={acc:.4f}  {elapsed:.0f}s{marker}", flush=True)

    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Saved: checkpoints/cand_tower_concept_best.pt")


if __name__ == "__main__":
    main()
