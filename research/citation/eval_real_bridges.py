#!/usr/bin/env python3
"""Evaluate IDF cosine recall on real LBD bridge pairs (2-hop indirect)."""
import argparse, json, math, os, random, time
from collections import Counter
import numpy as np
import scipy.sparse as sp
import torch


def build_idf_matrix(features, concepts, ref_lists, n_total, concept_vocab):
    concept_df = Counter()
    for c in concepts:
        for cid in c:
            concept_df[cid] += 1
    idf = np.zeros(concept_vocab, dtype=np.float32)
    for cid, df in concept_df.items():
        idf[cid] = math.log(n_total / max(df, 1))

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
    norms_sq = np.asarray((S_idf.multiply(S_idf)).sum(axis=1)).ravel()
    norms = np.sqrt(norms_sq, dtype=np.float32)
    return S_idf, norms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--n-eval", type=int, default=5000)
    p.add_argument("--top-k", type=int, default=5000)
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), map_location="cpu", weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), map_location="cpu", weights_only=False)
    with open(os.path.join(args.data_dir, "config.json")) as f:
        config = json.load(f)

    features = fd["features"]
    concepts = fd["concepts"]
    n_total = features.size(0)
    concept_vocab = int(max(max(c) if c else 0 for c in concepts)) + 1

    # Load real bridges
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges_real.pt"),
                         map_location="cpu", weights_only=False)
    src_all = bridges["source_idx"].numpy()
    tgt_all = bridges["target_idx"].numpy()
    print(f"Total real bridges: {len(src_all):,}", flush=True)

    # Filter post-2021
    year_norm_thr = (2021 - 1970) / 56.0
    mask = features[src_all, 0].numpy() > year_norm_thr
    src_test = src_all[mask]
    tgt_test = tgt_all[mask]
    print(f"Post-2021 test bridges: {len(src_test):,}", flush=True)

    # Sample
    rng = random.Random(42)
    idxs = rng.sample(range(len(src_test)), min(args.n_eval, len(src_test)))
    src_eval = src_test[idxs]
    tgt_eval = tgt_test[idxs]
    print(f"Evaluating: {len(idxs)}", flush=True)

    # Build IDF matrix
    print("Building IDF matrix...", flush=True)
    t0 = time.time()
    S_idf, norms = build_idf_matrix(features, concepts, ref_lists, n_total, concept_vocab)
    print(f"  {time.time()-t0:.0f}s", flush=True)

    # IDF retrieval
    print("IDF cosine retrieval...", flush=True)
    t0 = time.time()
    S_csr = S_idf.tocsr()
    inv_norm = np.zeros(n_total, dtype=np.float32)
    mask_arr = norms > 0
    inv_norm[mask_arr] = 1.0 / norms[mask_arr]

    batch_size = 100
    n_queries = len(src_eval)
    idf_topk_indices = []
    idf_ranks = []

    for batch_start in range(0, n_queries, batch_size):
        batch_end = min(batch_start + batch_size, n_queries)
        B = batch_end - batch_start

        q_rows, q_cols, q_data = [], [], []
        for j in range(B):
            src = src_eval[batch_start + j]
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

        sims = S_csr.dot(Q.T.toarray())
        sims = sims * inv_norm[:, None]

        for j in range(B):
            tgt = tgt_eval[batch_start + j]
            src = src_eval[batch_start + j]
            col = sims[:, j].copy()
            col[src] = -1
            if args.top_k < n_total:
                topk = np.argpartition(-col, args.top_k)[:args.top_k]
                topk = topk[np.argsort(-col[topk])]
            else:
                topk = np.argsort(-col)[:args.top_k]
            idf_topk_indices.append(topk)
            matches = np.where(topk == tgt)[0]
            idf_ranks.append(int(matches[0]) if len(matches) > 0 else args.top_k)

        if (batch_start // batch_size) % 10 == 0 and batch_start > 0:
            elapsed = time.time() - t0
            rate = batch_start / elapsed
            eta = (n_queries - batch_start) / rate / 60
            print(f"  [{batch_start}/{n_queries}] {rate:.0f}/s ETA {eta:.1f}m", flush=True)

    print(f"  {time.time()-t0:.0f}s", flush=True)

    # Results
    idf_ranks = np.array(idf_ranks)
    print(f"\n─── IDF Cosine on Real LBD Bridges (2-hop indirect) ───")
    print(f"  Bridges: {len(idf_ranks)} queries")
    print(f"{'K':>6}  {'Recall':>8}  {'Hits':>8}")
    print("─" * 30)
    for k in [1, 5, 10, 20, 50, 100, 500, 1000, 5000]:
        recall = (idf_ranks < k).sum() / len(idf_ranks)
        hits = int((idf_ranks < k).sum())
        print(f"{k:>6}  {recall*100:>7.1f}%  {hits:>4}/{len(idf_ranks)}")

    found = idf_ranks < args.top_k
    mrr = (1.0 / (idf_ranks[found] + 1)).sum() / len(idf_ranks)
    print(f"\nMRR@{args.top_k}: {mrr:.4f}")
    print(f"Found in top-{args.top_k}: {found.sum()}/{len(idf_ranks)} ({found.sum()/len(idf_ranks)*100:.1f}%)")


if __name__ == "__main__":
    main()
