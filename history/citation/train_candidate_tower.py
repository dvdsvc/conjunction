#!/usr/bin/env python3
"""Train CandidateTower for two-tower retrieval (LBD bridge discovery).

Follows X's PhoenixRetrievalModel architecture:
  - User Tower: Frozen Phoenix Transformer → user vector (384d, L2 norm)
  - Candidate Tower: MLP → candidate vector (384d, L2 norm)
  - Loss: InfoNCE (contrastive)

Training pairs from train split (source year ≤ 2018).
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from phoenix_model import PhoenixCitationTransformer


# ─────────────────────────────────────────────
# 1. Candidate Tower (MLP, like X's CandidateTower)
# ─────────────────────────────────────────────

class CandidateTower(nn.Module):
    """MLP that projects GNN embeddings to d_model space.
    
    Mirrors X's CandidateTower:
      proj_1: gnn_emb_dim → d_model*2 (SiLU)
      proj_2: d_model*2 → d_model (L2 normalize)
    """
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


# ─────────────────────────────────────────────
# 2. User vector extraction (frozen Phoenix)
# ─────────────────────────────────────────────

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


def pad_concepts(list_of_lists):
    max_k = max((len(c) for c in list_of_lists), default=1)
    max_k = max(max_k, 1)
    out = torch.zeros(len(list_of_lists), max_k, dtype=torch.long)
    for i, cidx in enumerate(list_of_lists):
        for j, c in enumerate(cidx):
            out[i, j] = c
    return out


@torch.no_grad()
def extract_user_vector(phoenix, user_scalar, user_concepts, user_gnn,
                         hist_scalars, hist_concepts, hist_gnns, device):
    """Extract L2-normalized user vector from frozen Phoenix Transformer."""
    scalar = torch.cat([user_scalar, hist_scalars], 0).unsqueeze(0)
    gnn    = torch.cat([user_gnn,    hist_gnns],    0).unsqueeze(0)
    c_idx  = pad_concepts([user_concepts] + hist_concepts).unsqueeze(0).to(device)

    S = scalar.size(1)
    mask = torch.ones(1, S, S, device=device)

    concept_mh = torch.zeros(1, S, phoenix.concept_proj.in_features, device=device)
    concept_mh.scatter_(2, c_idx.clamp(min=0), 1.0)

    x = torch.cat([scalar.to(device), phoenix.concept_proj(concept_mh), gnn.to(device)], dim=-1)
    x = phoenix.feature_proj(x)
    for block in phoenix.blocks:
        x = block(x, mask)
    x = phoenix.final_norm(x)

    user_repr = x.mean(dim=1)
    return F.normalize(user_repr, p=2, dim=-1).squeeze(0)


# ─────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────

class BridgeRetrievalDataset(Dataset):
    """Pre-computed (user_vector, positive_gnn, negative_gnn) triples."""
    
    def __init__(self, user_vectors, pos_gnn_embs, neg_gnn_embs):
        """
        user_vectors: (N, d_model) 
        pos_gnn_embs: (N, gnn_dim)
        neg_gnn_embs: (N, n_neg, gnn_dim)
        """
        self.user_vectors = user_vectors
        self.pos_gnn_embs = pos_gnn_embs
        self.neg_gnn_embs = neg_gnn_embs
    
    def __len__(self):
        return len(self.user_vectors)
    
    def __getitem__(self, idx):
        return {
            "user_vec": self.user_vectors[idx],
            "pos_gnn": self.pos_gnn_embs[idx],
            "neg_gnn": self.neg_gnn_embs[idx],
        }


# ─────────────────────────────────────────────
# 4. InfoNCE loss
# ─────────────────────────────────────────────

def info_nce_loss(user_vecs, cand_vecs, temperature=0.07):
    """InfoNCE: user_vecs (B, D), cand_vecs (B, 1+N, D).
    
    First candidate in dim 1 is positive, rest are negatives.
    """
    # cand_vecs: (B, 1+N, D) already L2-normalized
    # user_vecs: (B, D) already L2-normalized
    # logits = user · cand^T / temperature
    logits = torch.bmm(cand_vecs, user_vecs.unsqueeze(-1)).squeeze(-1) / temperature
    # labels: positive is at index 0
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n-train", type=int, default=100000)
    p.add_argument("--n-neg", type=int, default=63)
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), weights_only=False)
    gnn_emb = torch.load(os.path.join(args.data_dir, "gnn_embeddings.pt"), weights_only=False)
    with open(os.path.join(args.data_dir, "config.json")) as f: config = json.load(f)

    # Load frozen Phoenix
    ckpt_path = args.checkpoint or os.path.join(args.data_dir, "checkpoints/phoenix_best.pt")
    phoenix, arch = load_phoenix(ckpt_path, config, args.device)
    d_model = arch["d_model"]
    gnn_dim = gnn_emb.size(1)
    print(f"Phoenix frozen: d={d_model}, gnn_dim={gnn_dim}")

    # Load train bridge pairs
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges.pt"), weights_only=False)
    year_threshold = 2018  # train split: year ≤ 2018
    year_norm_threshold = (year_threshold - 1970) / 56.0
    features = fd["features"]
    concepts = fd["concepts"]
    n_total = features.size(0)

    train_bridges = [b for b in bridges if features[b["source_idx"]][0].item() <= year_norm_threshold]
    print(f"Train bridges (≤2018): {len(train_bridges):,}")
    if len(train_bridges) > args.n_train:
        random.seed(42)
        train_bridges = random.sample(train_bridges, args.n_train)

    # Pre-compute user vectors + collect GNN embeddings
    print(f"Pre-computing {len(train_bridges)} user vectors...", flush=True)
    user_vecs = []
    pos_gnns = []
    neg_gnns_list = []
    
    t0 = time.time()
    for i, pair in enumerate(train_bridges):
        src_idx, tgt_idx = pair["source_idx"], pair["target_idx"]
        src_refs = ref_lists[src_idx]
        if len(src_refs) < 2:
            continue
        if len(src_refs) > args.max_history:
            src_refs = random.sample(src_refs, args.max_history)

        # User vector (frozen Phoenix)
        uv = extract_user_vector(
            phoenix,
            features[src_idx].unsqueeze(0), concepts[src_idx], gnn_emb[src_idx].unsqueeze(0),
            features[src_refs], [concepts[j] for j in src_refs], gnn_emb[src_refs],
            args.device,
        )
        user_vecs.append(uv.cpu())

        # Positive GNN embedding
        pos_gnns.append(gnn_emb[tgt_idx])

        # Negative GNN embeddings (random)
        neg_indices = random.sample(range(n_total), args.n_neg)
        neg_gnns_list.append(gnn_emb[neg_indices])

        if (i+1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(train_bridges)}] {elapsed:.0f}s", flush=True)

    user_vecs = torch.stack(user_vecs)       # (N, d_model)
    pos_gnns = torch.stack(pos_gnns)         # (N, gnn_dim)
    neg_gnns = torch.stack(neg_gnns_list)    # (N, n_neg, gnn_dim)
    print(f"Pre-computed: user={user_vecs.shape}, pos={pos_gnns.shape}, neg={neg_gnns.shape}")

    # Dataset + DataLoader
    dataset = BridgeRetrievalDataset(user_vecs, pos_gnns, neg_gnns)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                        num_workers=2, pin_memory=True)

    # Candidate Tower
    cand_tower = CandidateTower(gnn_dim, d_model).to(args.device)
    optimizer = torch.optim.AdamW(cand_tower.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining CandidateTower: {sum(p.numel() for p in cand_tower.parameters()):,} params")
    print(f"  epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, neg={args.n_neg}")
    print()

    best_loss = float('inf')
    for epoch in range(args.epochs):
        cand_tower.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            user_vec = batch["user_vec"].to(args.device)      # (B, d_model)
            pos_gnn  = batch["pos_gnn"].to(args.device)       # (B, gnn_dim)
            neg_gnn  = batch["neg_gnn"].to(args.device)       # (B, n_neg, gnn_dim)

            # Project candidates
            pos_cand = cand_tower(pos_gnn)                     # (B, d_model) L2-normed
            neg_cand = cand_tower(neg_gnn.view(-1, gnn_dim))   # (B*n_neg, d_model)
            neg_cand = neg_cand.view(-1, args.n_neg, d_model)  # (B, n_neg, d_model)

            # Concat: [positive, negatives] → (B, 1+N, d_model)
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

        # Retrieval accuracy (how often positive has highest sim)
        cand_tower.eval()
        with torch.no_grad():
            # Quick eval on first 5000 samples
            eval_n = min(5000, len(dataset))
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
