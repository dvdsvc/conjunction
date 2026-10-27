#!/usr/bin/env python3
"""Quick test: compare mean pooling vs token-0 pooling for discrimination."""
import argparse, json, os, random
import torch
import torch.nn.functional as F
import numpy as np

from phoenix_model import PhoenixCitationTransformer
from train_candidate_tower_batch import load_phoenix


@torch.no_grad()
def compute_phoenix_repr(phoenix, samples, features, concepts, gnn_emb,
                          concept_dim, device, pooling="mean"):
    """Compute Phoenix representations with different pooling strategies."""
    scalar_dim = features.shape[1]
    gnn_dim = gnn_emb.shape[1]
    all_vecs = []

    for s in samples:
        idxs = [s] if isinstance(s, int) else [s["src_idx"]] + s.get("ref_idxs", [])[:50]
        seq_len = len(idxs)

        b_scalar = torch.zeros(1, seq_len, scalar_dim)
        b_gnn = torch.zeros(1, seq_len, gnn_dim)
        b_concept = torch.zeros(1, seq_len, concept_dim)

        for j, idx in enumerate(idxs):
            b_scalar[0, j] = features[idx]
            b_gnn[0, j] = gnn_emb[idx]
            c = concepts[idx]
            if c:
                valid = [x for x in c if x < concept_dim]
                if valid:
                    b_concept[0, j].scatter_(0, torch.tensor(valid), 1.0)

        b_scalar = b_scalar.to(device)
        b_gnn = b_gnn.to(device)
        b_concept = b_concept.to(device)

        mask = torch.ones(1, seq_len, seq_len, device=device)
        c_proj = phoenix.concept_proj(b_concept)
        x = torch.cat([b_scalar, c_proj, b_gnn], dim=-1)
        x = phoenix.feature_proj(x)
        for block in phoenix.blocks:
            x = block(x, mask)
        x = phoenix.final_norm(x)

        if pooling == "mean":
            vec = x.mean(dim=1)
        elif pooling == "token0":
            vec = x[:, 0, :]  # only the paper itself
        elif pooling == "last":
            vec = x[:, -1, :]

        vec = F.normalize(vec, p=2, dim=-1)
        all_vecs.append(vec.cpu())

    return torch.cat(all_vecs, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), map_location="cpu", weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), map_location="cpu", weights_only=False)
    gnn_emb = torch.load(os.path.join(args.data_dir, "gnn_embeddings.pt"), map_location="cpu", weights_only=False)
    with open(os.path.join(args.data_dir, "config.json")) as f: config = json.load(f)

    phoenix, arch = load_phoenix(
        os.path.join(args.data_dir, "checkpoints/phoenix_best.pt"), config, args.device)
    concept_dim = config["concept_vocab_size"]
    features = fd["features"]
    concepts = fd["concepts"]

    # Sample papers for quick test
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges.pt"), map_location="cpu", weights_only=False)
    year_norm_threshold = (2021 - 1970) / 56.0
    test_bridges = [b for b in bridges if features[b["source_idx"]][0].item() > year_norm_threshold]
    random.seed(42)
    test_sample = random.sample(test_bridges, 200)

    # Build samples with refs
    samples_with_refs = []
    for b in test_sample:
        src = b["source_idx"]
        refs = ref_lists[src][:50]
        if len(refs) >= 2:
            samples_with_refs.append({"src_idx": src, "ref_idxs": refs})

    N = len(samples_with_refs)
    print(f"Testing with {N} papers")

    # Compute with different pooling strategies
    for pooling in ["mean", "token0"]:
        print(f"\n{'='*50}")
        print(f"Pooling: {pooling}")

        vecs = compute_phoenix_repr(
            phoenix, samples_with_refs, features, concepts, gnn_emb,
            concept_dim, args.device, pooling=pooling)

        # Positive pairs: source → target
        pos_sims = []
        for i, s in enumerate(samples_with_refs):
            tgt = test_sample[i]["target_idx"]
            tgt_refs = ref_lists[tgt][:50]
            if len(tgt_refs) < 2:
                continue
            tgt_vec = compute_phoenix_repr(
                phoenix, [tgt], features, concepts, gnn_emb,
                concept_dim, args.device, pooling=pooling)
            sim = F.cosine_similarity(vecs[i:i+1], tgt_vec).item()
            pos_sims.append(sim)

        # Random pairs
        rng = random.Random(123)
        rand_sims = []
        for _ in range(200):
            a, b = rng.randint(0, N-1), rng.randint(0, N-1)
            sim = F.cosine_similarity(vecs[a:a+1], vecs[b:b+1]).item()
            rand_sims.append(sim)

        # Same paper sim (paper itself vs paper + refs)
        same_sims = []
        for i in range(min(50, N)):
            paper_only = compute_phoenix_repr(
                phoenix, [samples_with_refs[i]["src_idx"]], features, concepts, gnn_emb,
                concept_dim, args.device, pooling=pooling)
            sim = F.cosine_similarity(vecs[i:i+1], paper_only).item()
            same_sims.append(sim)

        pos_sims = np.array(pos_sims)
        rand_sims = np.array(rand_sims)
        same_sims = np.array(same_sims)

        print(f"  Same paper (with refs vs no refs): {same_sims.mean():.4f}")
        print(f"  Positive pairs:                    {pos_sims.mean():.4f}")
        print(f"  Random pairs:                      {rand_sims.mean():.4f}")
        print(f"  Discrimination (pos - rand):       {pos_sims.mean() - rand_sims.mean():.4f}")


if __name__ == "__main__":
    main()
