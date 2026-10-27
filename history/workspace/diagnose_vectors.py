#!/usr/bin/env python3
"""Diagnostic: compare user vectors vs candidate vectors for the SAME papers.

Checks:
1. Are user/target vectors for the SAME paper similar?
2. What's the cosine similarity distribution (positive pairs vs random)?
3. Is the padding/masking causing the mismatch?
"""
import argparse, json, os, random
import torch
import torch.nn.functional as F
import numpy as np

from phoenix_model import PhoenixCitationTransformer
from train_candidate_tower_batch import load_phoenix, prepare_samples, batch_user_vectors
from compute_phoenix_candidates import batch_inference as candidate_batch_inference


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
    d_model = arch["d_model"]
    concept_dim = config["concept_vocab_size"]

    features = fd["features"]
    concepts = fd["concepts"]
    N = features.size(0)

    # Load pre-computed candidate vectors
    cand_vecs_all = torch.load(
        os.path.join(args.data_dir, "phoenix_candidate_vectors.pt"),
        map_location="cpu", weights_only=False)
    print(f"Candidate vectors loaded: {cand_vecs_all.shape}")

    # ── Test 1: Same-paper similarity ──
    # Pick some target papers from test bridges, compute their user vector,
    # compare with their pre-computed candidate vector
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges.pt"), map_location="cpu", weights_only=False)
    year_norm_threshold = (2021 - 1970) / 56.0
    test_bridges = [b for b in bridges if features[b["source_idx"]][0].item() > year_norm_threshold]
    random.seed(42)
    test_sample = random.sample(test_bridges, min(100, len(test_bridges)))

    # Compute user vectors for these papers (treating them as "source")
    fake_samples = []
    for b in test_sample:
        tgt = b["target_idx"]
        refs = ref_lists[tgt]
        if len(refs) < 2:
            continue
        if len(refs) > 50:
            refs = random.sample(refs, 50)
        fake_samples.append({"src_idx": tgt, "tgt_idx": tgt, "ref_idxs": refs})

    print(f"\nComputing user vectors for {len(fake_samples)} target papers...")
    user_vecs = batch_user_vectors(phoenix, fake_samples, features, concepts, gnn_emb,
                                    concept_dim, args.device, batch_size=64)

    # Compare with pre-computed candidate vectors
    same_paper_sims = []
    for i, s in enumerate(fake_samples):
        idx = s["src_idx"]
        uv = user_vecs[i]
        cv = cand_vecs_all[idx]
        sim = F.cosine_similarity(uv.unsqueeze(0), cv.unsqueeze(0)).item()
        same_paper_sims.append(sim)

    same_paper_sims = np.array(same_paper_sims)
    print(f"\n{'='*50}")
    print(f"Test 1: Same paper user_vec vs candidate_vec")
    print(f"  Mean cosine sim: {same_paper_sims.mean():.4f}")
    print(f"  Std:  {same_paper_sims.std():.4f}")
    print(f"  Min:  {same_paper_sims.min():.4f}")
    print(f"  Max:  {same_paper_sims.max():.4f}")
    print(f"  Median: {np.median(same_paper_sims):.4f}")

    # ── Test 2: Random pair similarity (baseline) ──
    random_sims = []
    rng = random.Random(123)
    for i in range(200):
        a = rng.randint(0, N - 1)
        b = rng.randint(0, N - 1)
        sim = F.cosine_similarity(cand_vecs_all[a].unsqueeze(0), cand_vecs_all[b].unsqueeze(0)).item()
        random_sims.append(sim)

    random_sims = np.array(random_sims)
    print(f"\n{'='*50}")
    print(f"Test 2: Random candidate pair similarity (baseline)")
    print(f"  Mean: {random_sims.mean():.4f}")
    print(f"  Std:  {random_sims.std():.4f}")

    # ── Test 3: Positive pair similarity (source user_vec vs target candidate_vec) ──
    bridges_sample = random.sample(test_bridges, min(200, len(test_bridges)))
    pos_samples = prepare_samples(bridges_sample, features, concepts, gnn_emb, ref_lists, 50)
    print(f"\nComputing user vectors for {len(pos_samples)} source papers...")
    src_user_vecs = batch_user_vectors(phoenix, pos_samples, features, concepts, gnn_emb,
                                        concept_dim, args.device, batch_size=64)

    pos_sims = []
    for i, s in enumerate(pos_samples):
        uv = src_user_vecs[i]
        cv = cand_vecs_all[s["tgt_idx"]]
        sim = F.cosine_similarity(uv.unsqueeze(0), cv.unsqueeze(0)).item()
        pos_sims.append(sim)

    pos_sims = np.array(pos_sims)
    print(f"\n{'='*50}")
    print(f"Test 3: Positive pairs (source user_vec vs target candidate_vec)")
    print(f"  Mean: {pos_sims.mean():.4f}")
    print(f"  Std:  {pos_sims.std():.4f}")

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"SUMMARY:")
    print(f"  Same paper sim:     {same_paper_sims.mean():.4f}")
    print(f"  Positive pair sim:  {pos_sims.mean():.4f}")
    print(f"  Random pair sim:    {random_sims.mean():.4f}")
    print(f"")
    if same_paper_sims.mean() > 0.5:
        print(f"  → Same paper sim is HIGH: vectors are consistent, problem is elsewhere")
    elif same_paper_sims.mean() > 0.1:
        print(f"  → Same paper sim is MODERATE: some alignment but significant gap")
    else:
        print(f"  → Same paper sim is LOW: user and candidate vectors are MISALIGNED")
        print(f"    → Likely cause: different processing (padding/masking/pooling)")

    if pos_sims.mean() > random_sims.mean() + 0.1:
        print(f"  → Positive pairs > random: signal exists but weak")
    else:
        print(f"  → Positive pairs ≈ random: NO discriminative signal")


if __name__ == "__main__":
    main()
