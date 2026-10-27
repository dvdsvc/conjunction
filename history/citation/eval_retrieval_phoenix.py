#!/usr/bin/env python3
"""Retrieval evaluation using pre-computed Phoenix candidate vectors.

Both towers now use the same frozen Phoenix Transformer:
  User = Phoenix(source_paper + its_citations) → mean_pool → L2_norm
  Candidate = Phoenix(target_paper + its_citations) → mean_pool → L2_norm

No CandidateTower MLP needed — direct cosine similarity via Faiss.
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
import faiss

from phoenix_model import PhoenixCitationTransformer
from train_candidate_tower_batch import load_phoenix, prepare_samples, batch_user_vectors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--n-eval", type=int, default=5000)
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Load pre-computed candidate vectors
    print("Loading pre-computed candidate vectors...", flush=True)
    candidate_vecs = torch.load(
        os.path.join(args.data_dir, "phoenix_candidate_vectors.pt"),
        map_location="cpu", weights_only=False)
    n_total, d_model = candidate_vecs.shape
    print(f"Candidates: {candidate_vecs.shape}")

    # Faiss index
    corpus = candidate_vecs.numpy().astype(np.float32)
    del candidate_vecs
    print("Building Faiss index...", flush=True)
    index = faiss.IndexFlatIP(d_model)
    index.add(corpus)
    print(f"Index: {index.ntotal:,} vectors\n")

    # Load data for user vectors
    print("Loading data for user vectors...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), map_location="cpu", weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), map_location="cpu", weights_only=False)
    gnn_emb = torch.load(os.path.join(args.data_dir, "gnn_embeddings.pt"), map_location="cpu", weights_only=False)
    with open(os.path.join(args.data_dir, "config.json")) as f: config = json.load(f)

    # Load title embeddings if available
    title_path = os.path.join(args.data_dir, "title_embeddings.pt")
    if os.path.exists(title_path):
        title_embeddings = torch.load(title_path, map_location="cpu", weights_only=False)
        print(f"Loaded title embeddings: {title_embeddings.shape}", flush=True)
    else:
        title_embeddings = None
        print("WARNING: title_embeddings.pt not found", flush=True)

    phoenix, arch = load_phoenix(
        os.path.join(args.data_dir, "checkpoints/phoenix_best.pt"), config, args.device)
    concept_dim = config["concept_vocab_size"]

    features = fd["features"]
    concepts = fd["concepts"]

    # Test queries
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges.pt"), map_location="cpu", weights_only=False)
    year_norm_threshold = (2021 - 1970) / 56.0
    test_bridges = [b for b in bridges if features[b["source_idx"]][0].item() > year_norm_threshold]
    print(f"Test bridges (>2021): {len(test_bridges):,}")
    if len(test_bridges) > args.n_eval:
        random.seed(42)
        test_bridges = random.sample(test_bridges, args.n_eval)

    test_samples = prepare_samples(test_bridges, features, concepts, gnn_emb, ref_lists, args.max_history)
    target_indices = np.array([s["tgt_idx"] for s in test_samples])
    print(f"Valid test samples: {len(test_samples)}")

    # Extract user vectors (using Phoenix Transformer)
    print("Extracting user vectors...", flush=True)
    t0 = time.time()
    user_vecs = batch_user_vectors(phoenix, test_samples, features, concepts, gnn_emb,
                                    concept_dim, args.device, batch_size=64,
                                    title_embeddings=title_embeddings)
    user_vecs_np = user_vecs.numpy().astype(np.float32)
    print(f"User vectors: {user_vecs_np.shape} in {time.time()-t0:.0f}s\n")

    # Search
    max_k = 5000
    print(f"Searching top-{max_k} from {n_total:,}...", flush=True)
    t0 = time.time()
    scores, indices = index.search(user_vecs_np, max_k)
    print(f"Search done in {time.time()-t0:.1f}s\n")

    # Vectorized Recall@K
    n_q = len(target_indices)
    match = (indices == target_indices[:, None])
    ranks = np.where(match.any(axis=1), match.argmax(axis=1), max_k)

    k_values = [1, 5, 10, 20, 50, 100, 500, 1000, 5000]
    print("─── Phoenix Symmetric Two-Tower ───")
    print(f"{'K':>6}  {'Recall':>8}  {'Hits':>8}")
    print("─" * 30)
    for k in k_values:
        recall = (ranks < k).sum() / n_q
        hits = int((ranks < k).sum())
        print(f"{k:>6}  {recall*100:>7.1f}%  {hits:>4}/{n_q}")

    found_mask = ranks < max_k
    mrr = (1.0 / (ranks[found_mask] + 1)).sum() / n_q
    print(f"\nMRR@{max_k}: {mrr:.4f}")
    print(f"Found in top-{max_k}: {found_mask.sum()}/{n_q} ({found_mask.sum()/n_q*100:.1f}%)")


if __name__ == "__main__":
    main()
