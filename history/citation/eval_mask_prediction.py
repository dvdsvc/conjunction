#!/usr/bin/env python3
"""Fast mask prediction eval using pre-computed candidate vectors + Faiss.

For each val paper:
  1. Mask 20% of its references
  2. Compute user vector from source paper + remaining refs
  3. Search Faiss for masked refs
  4. Report recall@K

Uses batch_user_vectors from train_candidate_tower_batch.py for efficiency.
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
import faiss

from train_candidate_tower_batch import load_phoenix, batch_user_vectors


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--n-eval", type=int, default=5000)
    p.add_argument("--mask-ratio", type=float, default=0.2)
    p.add_argument("--max-refs", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Load pre-computed candidate vectors
    print("Loading candidate vectors...", flush=True)
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

    # Load data
    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), map_location="cpu", weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), map_location="cpu", weights_only=False)
    gnn_emb = torch.load(os.path.join(args.data_dir, "gnn_embeddings.pt"), map_location="cpu", weights_only=False)
    with open(os.path.join(args.data_dir, "config.json")) as f:
        config = json.load(f)

    title_path = os.path.join(args.data_dir, "title_embeddings.pt")
    if os.path.exists(title_path):
        title_embeddings = torch.load(title_path, map_location="cpu", weights_only=False)
        print(f"Title embeddings: {title_embeddings.shape}", flush=True)
    else:
        title_embeddings = None

    features = fd["features"]
    concepts = fd["concepts"]
    concept_dim = config["concept_vocab_size"]

    # Load model
    ckpt_path = os.path.join(args.data_dir, "checkpoints/phoenix_best.pt")
    phoenix, arch = load_phoenix(ckpt_path, config, args.device)
    print(f"Phoenix: d_model={arch['d_model']}", flush=True)

    # Prepare val papers
    splits = torch.load(os.path.join(args.data_dir, "splits.pt"), weights_only=False)
    val_papers = [i for i in splits["val"] if len(ref_lists[i]) >= 5]
    rng = random.Random(42)
    val_papers = rng.sample(val_papers, min(args.n_eval, len(val_papers)))
    print(f"Val papers: {len(val_papers)} (≥5 refs)", flush=True)

    # For each val paper, mask refs and collect (paper_idx, masked, history)
    samples = []
    for paper_idx in val_papers:
        refs = ref_lists[paper_idx]
        if len(refs) > args.max_refs:
            refs = rng.sample(refs, args.max_refs)
        n_mask = max(1, int(len(refs) * args.mask_ratio))
        perm = rng.sample(range(len(refs)), len(refs))
        masked = [refs[i] for i in perm[:n_mask]]
        history = [refs[i] for i in perm[n_mask:]]
        samples.append({
            "src_idx": paper_idx,
            "ref_idxs": history,  # only remaining refs, NOT masked
            "masked_idxs": masked,
        })

    # Compute user vectors using batch_user_vectors
    print(f"Computing {len(samples)} user vectors...", flush=True)
    t0 = time.time()
    user_vecs = batch_user_vectors(
        phoenix, samples, features, concepts, gnn_emb,
        concept_dim, args.device, batch_size=128,
        title_embeddings=title_embeddings)
    elapsed = time.time() - t0
    print(f"User vectors: {user_vecs.shape} in {elapsed:.0f}s", flush=True)

    # Search
    max_k = 5000
    user_np = user_vecs.numpy().astype(np.float32)
    print(f"Searching top-{max_k}...", flush=True)
    t0 = time.time()
    scores, indices = index.search(user_np, max_k)
    print(f"Search done in {time.time()-t0:.1f}s\n", flush=True)

    # Compute recall: for each sample, check if any masked ref is in top-K
    n_q = len(samples)
    all_ranks = []
    for i in range(n_q):
        masked_set = set(samples[i]["masked_idxs"])
        found_ranks = []
        for m in masked_set:
            matches = np.where(indices[i] == m)[0]
            if len(matches) > 0:
                found_ranks.append(int(matches[0]))
            else:
                found_ranks.append(max_k)
        all_ranks.extend(found_ranks)

    all_ranks = np.array(all_ranks)
    n_queries = len(all_ranks)

    print(f"─── Citation Mask Prediction Recall ───")
    print(f"  Papers: {n_q}  Queries: {n_queries}  (avg {n_queries/n_q:.1f} masked refs/paper)")
    print(f"{'K':>6}  {'Recall':>8}  {'Hits':>8}")
    print("─" * 30)
    for k in [1, 5, 10, 20, 50, 100, 500, 1000, 5000]:
        recall = (all_ranks < k).sum() / n_queries
        hits = int((all_ranks < k).sum())
        print(f"{k:>6}  {recall*100:>7.1f}%  {hits:>4}/{n_queries}")

    found_mask = all_ranks < max_k
    mrr = (1.0 / (all_ranks[found_mask] + 1)).sum() / n_queries
    print(f"\nMRR@{max_k}: {mrr:.4f}")
    print(f"Found in top-{max_k}: {found_mask.sum()}/{n_queries} ({found_mask.sum()/n_queries*100:.1f}%)")


if __name__ == "__main__":
    main()
