#!/usr/bin/env python3
"""Hybrid retrieval: IDF cosine coarse ranking → Phoenix fine ranking.

Stage 1: IDF 1-hop cosine → top-5000 candidates per query
Stage 2: Phoenix Transformer scores each candidate → re-rank → recall@K
"""
import argparse, json, math, os, random, time
from collections import Counter
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import faiss

from train_candidate_tower_batch import load_phoenix


def build_idf_matrix(features, concepts, ref_lists, n_total, concept_vocab):
    """Build IDF-weighted 1-hop concept sparse matrix."""
    # Concept IDF
    concept_df = Counter()
    for c in concepts:
        for cid in c:
            concept_df[cid] += 1
    idf = np.zeros(concept_vocab, dtype=np.float32)
    for cid, df in concept_df.items():
        idf[cid] = math.log(n_total / max(df, 1))

    # 1-hop concepts
    rows, cols = [], []
    for i in range(n_total):
        s = set()
        for ref in ref_lists[i]:
            s.update(concepts[ref])
        if s:
            rows.extend([i] * len(s))
            cols.extend(s)
    data = np.ones(len(rows), dtype=np.float32)
    S_1hop = sp.csc_matrix(
        (data, (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n_total, concept_vocab), dtype=np.float32)
    S_idf = S_1hop.multiply(idf).tocsc()

    # Norms
    norms_sq = np.asarray((S_idf.multiply(S_idf)).sum(axis=1)).ravel()
    norms = np.sqrt(norms_sq, dtype=np.float32)
    return S_idf, norms


@torch.no_grad()
def phoenix_rerank(phoenix, query_idx, cand_indices, features, concepts,
                   gnn_emb, ref_lists, concept_dim, device, title_embeddings=None):
    """Score candidates using Phoenix Transformer.

    User = source paper + its refs
    Candidates = each candidate as single token (no refs)
    Returns scores for each candidate.
    """
    refs = ref_lists[query_idx][:50]
    src_idxs = [query_idx] + refs
    n_uh = len(src_idxs)

    scalar_dim = features.shape[1]
    gnn_dim = gnn_emb.shape[1]
    has_title = title_embeddings is not None
    title_dim = title_embeddings.shape[1] if has_title else 0
    n_cand = len(cand_indices)
    seq_len = n_uh + n_cand

    all_idxs = src_idxs + list(cand_indices)

    b_scalars = features[all_idxs].unsqueeze(0).to(device)
    b_gnns = gnn_emb[all_idxs].unsqueeze(0).to(device)

    b_cmh = torch.zeros(1, seq_len, concept_dim, device=device)
    for j, idx in enumerate(all_idxs):
        c = concepts[idx]
        if c:
            valid = [x for x in c if x < concept_dim]
            if valid:
                b_cmh[0, j].scatter_(0, torch.tensor(valid, device=device), 1.0)

    b_titles = title_embeddings[all_idxs].unsqueeze(0).to(device) if has_title else None

    # Candidate isolation mask
    mask = torch.zeros(1, seq_len, seq_len, device=device)
    mask[0, :n_uh, :n_uh] = 1.0
    for i in range(n_uh, seq_len):
        mask[0, i, :n_uh] = 1.0
        mask[0, i, i] = 1.0

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = phoenix(b_scalars, b_cmh.long(), b_gnns, mask, title_emb=b_titles)

    return logits[0, n_uh:n_uh+n_cand].cpu().float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--n-eval", type=int, default=5000)
    p.add_argument("--top-k", type=int, default=5000, help="IDF coarse top-K")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), map_location="cpu", weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), map_location="cpu", weights_only=False)
    gnn_emb = torch.load(os.path.join(args.data_dir, "gnn_embeddings.pt"), map_location="cpu", weights_only=False)
    with open(os.path.join(args.data_dir, "config.json")) as f:
        config = json.load(f)

    title_path = os.path.join(args.data_dir, "title_embeddings.pt")
    title_embeddings = None
    if os.path.exists(title_path):
        title_embeddings = torch.load(title_path, map_location="cpu", weights_only=False)
        print(f"Title embeddings: {title_embeddings.shape}", flush=True)

    features = fd["features"]
    concepts = fd["concepts"]
    n_total = features.size(0)
    concept_vocab = int(max(max(c) if c else 0 for c in concepts)) + 1
    concept_dim = config["concept_vocab_size"]
    year_norm_thr = (2021 - 1970) / 56.0

    # ── Stage 1: IDF coarse ranking ──
    print("Building IDF matrix...", flush=True)
    t0 = time.time()
    S_idf, norms = build_idf_matrix(features, concepts, ref_lists, n_total, concept_vocab)
    print(f"  {time.time()-t0:.0f}s", flush=True)

    # Test bridges
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges.pt"),
                         map_location="cpu", weights_only=False)
    test_bridges = [b for b in bridges
                    if features[b["source_idx"]][0].item() > year_norm_thr]
    rng = random.Random(42)
    test_bridges = rng.sample(test_bridges, min(args.n_eval, len(test_bridges)))
    print(f"Test bridges: {len(test_bridges)}", flush=True)

    # IDF retrieval using batch sparse matrix-matrix product
    print("IDF cosine retrieval (batch sparse)...", flush=True)
    t0 = time.time()

    # Convert S_idf to CSR for fast row slicing
    S_csr = S_idf.tocsr()
    inv_norm = np.zeros(n_total, dtype=np.float32)
    mask_arr = norms > 0
    inv_norm[mask_arr] = 1.0 / norms[mask_arr]

    target_indices = []
    idf_topk_indices = []
    n_queries = len(test_bridges)

    # Batch: build query matrix (batch_size × concept_vocab), sparse dot, argpartition
    batch_size = 100
    for batch_start in range(0, n_queries, batch_size):
        batch_end = min(batch_start + batch_size, n_queries)
        batch_bridges = test_bridges[batch_start:batch_end]
        B = len(batch_bridges)

        # Build query vectors (B × concept_vocab, sparse)
        q_rows, q_cols, q_data = [], [], []
        for j, bridge in enumerate(batch_bridges):
            src = bridge["source_idx"]
            target_indices.append(bridge["target_idx"])
            src_vec = S_csr[src]
            src_norm_val = norms[src]
            if src_norm_val > 0:
                nz = src_vec.indices
                q_rows.extend([j] * len(nz))
                q_cols.extend(nz)
                q_data.extend(src_vec.data / src_norm_val)
        Q = sp.csr_matrix(
            (np.array(q_data, dtype=np.float32),
             (np.array(q_rows, dtype=np.int32), np.array(q_cols, dtype=np.int32))),
            shape=(B, concept_vocab))

        # S_csr (n_total × vocab) dot Q.T (vocab × B) → sims (n_total × B)
        sims = S_csr.dot(Q.T.toarray())  # (n_total, B) dense array
        sims = sims * inv_norm[:, None]  # normalize

        for j, bridge in enumerate(batch_bridges):
            src = bridge["source_idx"]
            col = sims[:, j].copy()
            col[src] = -1  # exclude self
            if args.top_k < n_total:
                topk = np.argpartition(-col, args.top_k)[:args.top_k]
                topk = topk[np.argsort(-col[topk])]
            else:
                topk = np.argsort(-col)[:args.top_k]
            idf_topk_indices.append(topk)

        if (batch_start // batch_size) % 5 == 0 and batch_start > 0:
            elapsed = time.time() - t0
            rate = batch_start / elapsed
            eta = (n_queries - batch_start) / rate / 60
            print(f"  [{batch_start}/{n_queries}] {rate:.0f}/s ETA {eta:.1f}m", flush=True)

    print(f"  {time.time()-t0:.0f}s", flush=True)

    # IDF-only recall
    print("\n─── Stage 1: IDF Cosine Only ───")
    idf_ranks = []
    for i, tgt in enumerate(target_indices):
        matches = np.where(idf_topk_indices[i] == tgt)[0]
        idf_ranks.append(int(matches[0]) if len(matches) > 0 else args.top_k)
    idf_ranks = np.array(idf_ranks)
    for k in [1, 5, 10, 20, 50, 100, 500, 1000, 5000]:
        recall = (idf_ranks < k).sum() / len(idf_ranks)
        print(f"  R@{k}: {recall*100:.1f}%")

    # ── Stage 2: Phoenix re-ranking ──
    print("\n─── Stage 2: IDF + Phoenix Re-rank ───", flush=True)
    ckpt_path = os.path.join(args.data_dir, "checkpoints/phoenix_best.pt")
    phoenix, arch = load_phoenix(ckpt_path, config, args.device)
    print(f"Phoenix: d_model={arch['d_model']}", flush=True)

    hybrid_ranks = []
    t0 = time.time()
    for i, bridge in enumerate(test_bridges):
        src = bridge["source_idx"]
        tgt = bridge["target_idx"]
        cand_idxs = idf_topk_indices[i]

        # Re-rank using Phoenix
        scores = phoenix_rerank(
            phoenix, src, cand_idxs, features, concepts, gnn_emb,
            ref_lists, concept_dim, args.device, title_embeddings)

        # Sort by Phoenix score (descending)
        reranked = cand_idxs[np.argsort(-scores.numpy())]

        # Find target rank
        matches = np.where(reranked == tgt)[0]
        hybrid_ranks.append(int(matches[0]) if len(matches) > 0 else args.top_k)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(test_bridges) - i - 1) / rate / 60
            print(f"  [{i+1}/{len(test_bridges)}] {rate:.0f}/s ETA {eta:.1f}m", flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s", flush=True)

    hybrid_ranks = np.array(hybrid_ranks)
    print(f"\n{'K':>6}  {'IDF only':>10}  {'IDF+Phoenix':>12}")
    print("─" * 35)
    for k in [1, 5, 10, 20, 50, 100, 500, 1000, 5000]:
        r1 = (idf_ranks < k).sum() / len(idf_ranks)
        r2 = (hybrid_ranks < k).sum() / len(hybrid_ranks)
        print(f"{k:>6}  {r1*100:>9.1f}%  {r2*100:>11.1f}%")


if __name__ == "__main__":
    main()
