#!/usr/bin/env python3
"""Two-tower retrieval evaluation — fast version.

Single Faiss search, vectorized Recall@K and MRR.
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
import faiss

from phoenix_model import PhoenixCitationTransformer
from train_candidate_tower_batch import (
    CandidateTower, load_phoenix, prepare_samples, batch_user_vectors, infer_model_config,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--cand-tower-ckpt", default=None)
    p.add_argument("--n-eval", type=int, default=5000)
    p.add_argument("--max-history", type=int, default=50)
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

    cand_tower_ckpt = args.cand_tower_ckpt or os.path.join(args.data_dir, "checkpoints/candidate_tower_best.pt")
    cand_tower = CandidateTower(gnn_dim, d_model).to(args.device)
    cand_tower.load_state_dict(torch.load(cand_tower_ckpt, map_location=args.device))
    cand_tower.eval()

    features = fd["features"]
    concepts = fd["concepts"]
    n_total = features.size(0)

    # ── Build corpus index ──
    print(f"Projecting {n_total:,} candidates...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        chunk = 200000
        all_vecs = []
        for s in range(0, n_total, chunk):
            e = min(s + chunk, n_total)
            v = cand_tower(gnn_emb[s:e].to(args.device)).cpu().numpy()
            all_vecs.append(v)
        corpus = np.concatenate(all_vecs, axis=0).astype(np.float32)
    print(f"Corpus: {corpus.shape} in {time.time()-t0:.0f}s")

    # Faiss index (CPU, brute-force inner product)
    print("Building Faiss index...", flush=True)
    index = faiss.IndexFlatIP(d_model)
    index.add(corpus)
    print(f"Index: {index.ntotal:,} vectors\n")

    # ── Test queries ──
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges.pt"), weights_only=False)
    year_norm_threshold = (2021 - 1970) / 56.0
    test_bridges = [b for b in bridges if features[b["source_idx"]][0].item() > year_norm_threshold]
    print(f"Test bridges (>2021): {len(test_bridges):,}")
    if len(test_bridges) > args.n_eval:
        random.seed(42)
        test_bridges = random.sample(test_bridges, args.n_eval)

    test_samples = prepare_samples(test_bridges, features, concepts, gnn_emb, ref_lists, args.max_history)
    target_indices = np.array([s["tgt_idx"] for s in test_samples])
    print(f"Valid test samples: {len(test_samples)}")

    # Extract user vectors
    print("Extracting user vectors...", flush=True)
    t0 = time.time()
    user_vecs = batch_user_vectors(phoenix, test_samples, features, concepts, gnn_emb,
                                    concept_dim, args.device, batch_size=64)
    user_vecs_np = user_vecs.numpy().astype(np.float32)
    print(f"User vectors: {user_vecs_np.shape} in {time.time()-t0:.0f}s\n")

    # ── Single Faiss search ──
    max_k = 5000
    print(f"Searching top-{max_k} from {n_total:,}...", flush=True)
    t0 = time.time()
    scores, indices = index.search(user_vecs_np, max_k)
    print(f"Search done in {time.time()-t0:.1f}s\n")

    # ── Vectorized Recall@K ──
    n_q = len(target_indices)
    # For each query, find the rank of the target (if present)
    # target_indices shape: (n_q,), indices shape: (n_q, max_k)
    # Use broadcasting: match = (indices == target[:, None]) → (n_q, max_k)
    match = (indices == target_indices[:, None])  # bool, (n_q, max_k)
    # rank[i] = first position where match is True, or max_k if not found
    ranks = np.where(match.any(axis=1), match.argmax(axis=1), max_k)

    k_values = [1, 5, 10, 20, 50, 100, 500, 1000, 5000]
    print("─── Two-Tower Retrieval Results ───")
    print(f"{'K':>6}  {'Recall':>8}  {'Hits':>8}")
    print("─" * 30)
    for k in k_values:
        recall = (ranks < k).sum() / n_q
        hits = int((ranks < k).sum())
        print(f"{k:>6}  {recall*100:>7.1f}%  {hits:>4}/{n_q}")

    # MRR
    found_mask = ranks < max_k
    mrr = (1.0 / (ranks[found_mask] + 1)).sum() / n_q
    print(f"\nMRR@{max_k}: {mrr:.4f}")
    print(f"Target found in top-{max_k}: {found_mask.sum()}/{n_q} ({found_mask.sum()/n_q*100:.1f}%)")


if __name__ == "__main__":
    main()
