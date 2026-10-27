#!/usr/bin/env python3
"""Batch pre-compute user vectors for CandidateTower training.

Key optimization: batch transformer forward passes, but build concept multi-hot
per-batch to avoid OOM on the full 20K × seq_len × 12785 tensor.
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from phoenix_model import PhoenixCitationTransformer


def infer_model_config(state_dict, scalar_dim):
    d_model = state_dict["feature_proj.0.weight"].shape[0]
    total_in = state_dict["feature_proj.0.weight"].shape[1]
    ce = state_dict["concept_proj.weight"].shape[0]
    ge = total_in - scalar_dim - ce
    n_heads = 12 if d_model == 384 else d_model // 64
    n_layers = len([k for k in state_dict if k.endswith(".attn_norm.weight")])
    return dict(d_model=d_model, n_heads=n_heads, n_kv_heads=n_heads,
                n_layers=n_layers, concept_emb_dim=ce, gnn_emb_dim=ge)


def load_phoenix(ckpt_path, config, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = infer_model_config(ckpt["model"], config["scalar_feature_dim"])
    model = PhoenixCitationTransformer(
        scalar_dim=config["scalar_feature_dim"],
        concept_vocab_size=config["concept_vocab_size"],
        **arch, dropout=0.0, output_multiplier=0.125,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, arch


class CandidateTower(nn.Module):
    def __init__(self, gnn_emb_dim, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(gnn_emb_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
    def forward(self, gnn_emb):
        x = self.proj(gnn_emb)
        return F.normalize(x, p=2, dim=-1)


# ─────────────────────────────────────────────────────
# Prepare per-sample data, then batch forward
# ─────────────────────────────────────────────────────

def prepare_samples(bridge_pairs, features, concepts, gnn_emb, ref_lists, max_history):
    """Prepare index lists (CPU only, no GPU memory)."""
    samples = []
    for pair in bridge_pairs:
        src_idx = pair["source_idx"]
        src_refs = ref_lists[src_idx]
        if len(src_refs) < 2:
            continue
        if len(src_refs) > max_history:
            src_refs = random.sample(src_refs, max_history)
        samples.append({
            "src_idx": src_idx,
            "tgt_idx": pair["target_idx"],
            "ref_idxs": src_refs,
        })
    return samples


@torch.no_grad()
def batch_user_vectors(phoenix, samples, features, concepts, gnn_emb, concept_dim, device, batch_size=128):
    """Extract user vectors in batches, building concept multi-hot per-batch."""
    all_vecs = []
    N = len(samples)
    scalar_dim = features.shape[1]
    gnn_dim = gnn_emb.shape[1]

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_samples = samples[start:end]
        B = len(batch_samples)

        # Find max seq len in this batch
        max_seq = max(1 + len(s["ref_idxs"]) for s in batch_samples)

        # Pre-allocate on CPU, then move to GPU
        b_scalars = torch.zeros(B, max_seq, scalar_dim)
        b_gnns    = torch.zeros(B, max_seq, gnn_dim)
        b_cmh     = torch.zeros(B, max_seq, concept_dim)

        for i, s in enumerate(batch_samples):
            idxs = [s["src_idx"]] + s["ref_idxs"]
            sl = len(idxs)
            b_scalars[i, :sl] = features[idxs]
            b_gnns[i, :sl]    = gnn_emb[idxs]
            for j, idx in enumerate(idxs):
                c = concepts[idx]
                if c:
                    valid = [x for x in c if x < concept_dim]
                    if valid:
                        b_cmh[i, j].scatter_(0, torch.tensor(valid), 1.0)

        # Move to GPU
        b_scalars = b_scalars.to(device)
        b_gnns    = b_gnns.to(device)
        b_cmh     = b_cmh.to(device)

        # Forward
        c_proj = phoenix.concept_proj(b_cmh)            # [B, T, concept_emb_dim]
        x = torch.cat([b_scalars, c_proj, b_gnns], dim=-1)
        x = phoenix.feature_proj(x)

        T = x.size(1)
        mask = torch.ones(B, T, T, device=device)
        for block in phoenix.blocks:
            x = block(x, mask)
        x = phoenix.final_norm(x)

        user_repr = x.mean(dim=1)
        user_repr = F.normalize(user_repr, p=2, dim=-1)
        all_vecs.append(user_repr.cpu())

        if (start // batch_size) % 5 == 0 and start > 0:
            print(f"  [{start}/{N}]", flush=True)

    return torch.cat(all_vecs, dim=0)


def info_nce_loss(user_vecs, cand_vecs, temperature=0.07):
    logits = torch.bmm(cand_vecs, user_vecs.unsqueeze(-1)).squeeze(-1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n-train", type=int, default=100000)
    p.add_argument("--n-neg", type=int, default=63)
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), weights_only=False)
    gnn_emb = torch.load(os.path.join(args.data_dir, "gnn_embeddings.pt"), weights_only=False)
    with open(os.path.join(args.data_dir, "config.json")) as f: config = json.load(f)

    ckpt_path = args.checkpoint or os.path.join(args.data_dir, "checkpoints/phoenix_best.pt")
    phoenix, arch = load_phoenix(ckpt_path, config, args.device)
    d_model = arch["d_model"]
    gnn_dim = gnn_emb.size(1)
    concept_dim = config["concept_vocab_size"]
    print(f"Phoenix frozen: d={d_model}, gnn_dim={gnn_dim}, concept_dim={concept_dim}")

    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges.pt"), weights_only=False)
    year_threshold = 2018
    year_norm_threshold = (year_threshold - 1970) / 56.0
    features = fd["features"]
    concepts = fd["concepts"]
    n_total = features.size(0)

    train_bridges = [b for b in bridges if features[b["source_idx"]][0].item() <= year_norm_threshold]
    print(f"Train bridges (≤2018): {len(train_bridges):,}")
    if len(train_bridges) > args.n_train:
        random.seed(42)
        train_bridges = random.sample(train_bridges, args.n_train)

    # ── Phase 1: Prepare + batch extract user vectors ──
    print(f"Preparing {len(train_bridges)} samples...", flush=True)
    samples = prepare_samples(train_bridges, features, concepts, gnn_emb, ref_lists, args.max_history)
    print(f"Valid samples: {len(samples):,}")

    print(f"Extracting user vectors (batch_size=128)...", flush=True)
    t0 = time.time()
    user_vecs = batch_user_vectors(phoenix, samples, features, concepts, gnn_emb,
                                    concept_dim, args.device, batch_size=128)
    elapsed = time.time() - t0
    print(f"Done: {len(user_vecs)} vectors in {elapsed:.0f}s ({len(user_vecs)/max(elapsed,1):.0f} vec/s)")

    # Collect GNN embeddings
    pos_gnns = torch.stack([gnn_emb[s["tgt_idx"]] for s in samples])
    rng = random.Random(42)
    neg_gnns = torch.stack([gnn_emb[rng.sample(range(n_total), args.n_neg)] for _ in samples])
    print(f"Dataset: user={user_vecs.shape}, pos_gnn={pos_gnns.shape}, neg_gnn={neg_gnns.shape}")

    # ── Phase 2: Train Candidate Tower ──
    class BridgeDataset(Dataset):
        def __init__(self, uv, pg, ng):
            self.uv, self.pg, self.ng = uv, pg, ng
        def __len__(self): return len(self.uv)
        def __getitem__(self, i):
            return {"user_vec": self.uv[i], "pos_gnn": self.pg[i], "neg_gnn": self.ng[i]}

    loader = DataLoader(BridgeDataset(user_vecs, pos_gnns, neg_gnns),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    cand_tower = CandidateTower(gnn_dim, d_model).to(args.device)
    optimizer = torch.optim.AdamW(cand_tower.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining CandidateTower: {sum(p.numel() for p in cand_tower.parameters()):,} params")
    print(f"  epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, neg={args.n_neg}\n")

    best_loss = float('inf')
    for epoch in range(args.epochs):
        cand_tower.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            user_vec = batch["user_vec"].to(args.device)
            pos_gnn  = batch["pos_gnn"].to(args.device)
            neg_gnn  = batch["neg_gnn"].to(args.device)

            pos_cand = cand_tower(pos_gnn)
            neg_cand = cand_tower(neg_gnn.view(-1, gnn_dim)).view(-1, args.n_neg, d_model)
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
            pg = pos_gnns[:eval_n].to(args.device)
            ng = neg_gnns[:eval_n].to(args.device)
            pos_sim = (cand_tower(pg) * uv).sum(dim=1)
            neg_sim = (cand_tower(ng.view(-1, gnn_dim)).view(eval_n, args.n_neg, d_model) * uv.unsqueeze(1)).sum(dim=2)
            acc = (pos_sim.unsqueeze(1) > neg_sim).float().mean().item()

        marker = " *" if avg_loss < best_loss else ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(cand_tower.state_dict(),
                       os.path.join(args.data_dir, "checkpoints/candidate_tower_best.pt"))
        print(f"Epoch {epoch+1:2d}/{args.epochs}  loss={avg_loss:.4f}  acc={acc:.4f}  {elapsed:.0f}s{marker}", flush=True)

    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Saved: checkpoints/candidate_tower_best.pt")


if __name__ == "__main__":
    main()
