#!/usr/bin/env python3
"""Two-tower retrieval evaluation on full 3.55M candidate pool.

CandidateTower projects GNN embeddings → 384d → L2 norm.
User vectors from frozen Phoenix Transformer.
Faiss ANN index for fast top-K retrieval.

Measures Recall@K on test split (>2021) bridge pairs.
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

from phoenix_model import PhoenixCitationTransformer
from train_candidate_tower_batch import (
    CandidateTower, load_phoenix, prepare_samples, batch_user_vectors, infer_model_config,
)


def build_faiss_index(candidate_vectors, use_gpu=False):
    """Build Faiss inner-product index from (N, D) numpy array."""
    d = candidate_vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(candidate_vectors)
    return index


def evaluate_retrieval(index, user_vectors, target_indices, k_values):
    """Evaluate Recall@K for each K in k_values."""
    max_k = max(k_values)
    _, top_k_indices = index.search(user_vectors, max_k)

    results = {}
    for k in k_values:
        hits = 0
        for i in range(len(target_indices)):
            if target_indices[i] in top_k_indices[i, :k]:
                hits += 1
        results[k] = hits / len(target_indices)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--cand-tower-ckpt", default=None)
    p.add_argument("--n-eval", type=int, default=5000)
    p.add_argument("--n-train", type=int, default=0)
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), weights_only=False)
    gnn_emb = torch.load(os.path.join(args.data_dir, "gnn_embeddings.pt"), weights_only=False)
    with open(os.path.join(args.data_dir, "config.json")) as f: config = json.load(f)

    # Load Phoenix
    ckpt_path = args.checkpoint or os.path.join(args.data_dir, "checkpoints/phoenix_best.pt")
    phoenix, arch = load_phoenix(ckpt_path, config, args.device)
    d_model = arch["d_model"]
    gnn_dim = gnn_emb.size(1)
    concept_dim = config["concept_vocab_size"]

    # Load Candidate Tower
    cand_tower_ckpt = args.cand_tower_ckpt or os.path.join(args.data_dir, "checkpoints/candidate_tower_best.pt")
    cand_tower = CandidateTower(gnn_dim, d_model).to(args.device)
    cand_tower.load_state_dict(torch.load(cand_tower_ckpt, map_location=args.device))
    cand_tower.eval()
    for p in cand_tower.parameters():
        p.requires_grad = False

    features = fd["features"]
    concepts = fd["concepts"]
    n_total = features.size(0)

    # ── Build full corpus index (3.55M papers) ──
    print(f"Projecting {n_total:,} candidate GNN embeddings...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        # Process in chunks to avoid OOM
        chunk_size = 100000
        all_cand_vecs = []
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            chunk = gnn_emb[start:end].to(args.device)
            projected = cand_tower(chunk).cpu().numpy()
            all_cand_vecs.append(projected)
            if start % 500000 == 0 and start > 0:
                print(f"  [{start:,}/{n_total:,}]", flush=True)
        corpus_vectors = np.concatenate(all_cand_vecs, axis=0).astype(np.float32)

    elapsed = time.time() - t0
    print(f"Corpus: {corpus_vectors.shape} in {elapsed:.0f}s")

    # Build Faiss index
    print("Building Faiss index...", flush=True)
    t0 = time.time()
    index = build_faiss_index(corpus_vectors, use_gpu=False)
    print(f"Index: {index.ntotal:,} vectors in {time.time()-t0:.0f}s")

    # ── Prepare test queries ──
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges.pt"), weights_only=False)
    year_threshold = 2021
    year_norm_threshold = (year_threshold - 1970) / 56.0

    test_bridges = [b for b in bridges if features[b["source_idx"]][0].item() > year_norm_threshold]
    print(f"Test bridges (>2021): {len(test_bridges):,}")

    if len(test_bridges) > args.n_eval:
        random.seed(42)
        test_bridges = random.sample(test_bridges, args.n_eval)

    print(f"Preparing {len(test_bridges)} test queries...", flush=True)
    test_samples = prepare_samples(test_bridges, features, concepts, gnn_emb, ref_lists, args.max_history)
    target_indices = [s["tgt_idx"] for s in test_samples]
    print(f"Valid test samples: {len(test_samples)}")

    # Extract user vectors
    print("Extracting user vectors...", flush=True)
    t0 = time.time()
    user_vecs = batch_user_vectors(phoenix, test_samples, features, concepts, gnn_emb,
                                    concept_dim, args.device, batch_size=64)
    elapsed = time.time() - t0
    print(f"User vectors: {user_vecs.shape} in {elapsed:.0f}s")

    # ── Evaluate ──
    user_vecs_np = user_vecs.numpy().astype(np.float32)
    k_values = [1, 5, 10, 20, 50, 100, 500, 1000, 5000]
    print(f"\nEvaluating Recall@K on {len(test_samples)} queries from {n_total:,} candidates...\n")
    results = evaluate_retrieval(index, user_vecs_np, target_indices, k_values)

    print("─── Two-Tower Retrieval Results ───")
    print(f"{'K':>6}  {'Recall':>8}  {'Hit rate':>10}")
    print("─" * 30)
    for k in k_values:
        hits = int(results[k] * len(test_samples))
        print(f"{k:>6}  {results[k]*100:>7.1f}%  {hits:>4}/{len(test_samples)}")

    # MRR
    max_k = max(k_values)
    _, top_k_indices = index.search(user_vecs_np, max_k)
    mrr_sum = 0
    for i in range(len(target_indices)):
        for rank in range(max_k):
            if top_k_indices[i, rank] == target_indices[i]:
                mrr_sum += 1.0 / (rank + 1)
                break
    mrr = mrr_sum / len(target_indices)
    print(f"\nMRR@{max_k}: {mrr:.4f}")


if __name__ == "__main__":
    main()
