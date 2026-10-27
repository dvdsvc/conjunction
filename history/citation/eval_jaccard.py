#!/usr/bin/env python3
"""IDF-weighted cosine similarity: source 1-hop concepts vs target 1-hop concepts.

Matches the validation setup (validate_2hop_concept.py) that showed +0.185
discrimination for 1hop_concepts mode.
"""
import argparse, math, os, random, time
from collections import Counter
import numpy as np
import scipy.sparse as sp
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--n-eval", type=int, default=5000)
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"),
                    map_location="cpu", weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"),
                           map_location="cpu", weights_only=False)

    features = fd["features"]
    concepts = fd["concepts"]
    n_total  = features.size(0)
    concept_vocab = int(max(max(c) if c else 0 for c in concepts)) + 1
    year_norm_thr = (2021 - 1970) / 56.0
    print(f"Papers: {n_total:,}  concept_vocab: {concept_vocab:,}", flush=True)

    # ── Concept IDF ──
    print("Computing concept IDF...", flush=True)
    concept_df = Counter()
    for c in concepts:
        for cid in c:
            concept_df[cid] += 1
    idf = np.zeros(concept_vocab, dtype=np.float32)
    for cid, df in concept_df.items():
        idf[cid] = math.log(n_total / max(df, 1))

    # ── Build S_1hop CSC (n_total × concept_vocab), then IDF-weighted ──
    print("Building S_1hop CSC (IDF-weighted)...", flush=True)
    t0 = time.time()
    rows, cols = [], []
    for i in range(n_total):
        s = set()
        for ref in ref_lists[i]:
            s.update(concepts[ref])
        rows.extend([i] * len(s))
        cols.extend(s)
    nnz_data = np.ones(len(rows), dtype=np.float32)
    S_1hop = sp.csc_matrix(
        (nnz_data, (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n_total, concept_vocab), dtype=np.float32)
    print(f"  nnz={S_1hop.nnz:,}  build {time.time()-t0:.0f}s", flush=True)

    # Apply IDF weighting: S_idf[i,c] = 1 * idf[c] if paper i has concept c
    print("Applying IDF weights...", flush=True)
    t0 = time.time()
    S_idf = S_1hop.multiply(idf).tocsc()   # sparse × dense broadcast, CSC format
    print(f"  {time.time()-t0:.1f}s", flush=True)

    # Paper norms (for cosine similarity denominator)
    print("Computing paper norms...", flush=True)
    t0 = time.time()
    paper_norms_sq = np.asarray((S_idf.multiply(S_idf)).sum(axis=1)).ravel()
    paper_norms = np.sqrt(paper_norms_sq, dtype=np.float32)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    inv_norm = np.zeros_like(paper_norms)
    mask = paper_norms > 0
    inv_norm[mask] = 1.0 / paper_norms[mask]

    # ── Load test bridges ──
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges.pt"),
                         map_location="cpu", weights_only=False)
    test_bridges = [b for b in bridges
                    if features[b["source_idx"]][0].item() > year_norm_thr]
    print(f"Test bridges (>2021): {len(test_bridges):,}")
    random.seed(42)
    if len(test_bridges) > args.n_eval:
        test_bridges = random.sample(test_bridges, args.n_eval)

    test_pairs = [(b["source_idx"], b["target_idx"]) for b in test_bridges
                  if len(ref_lists[b["source_idx"]]) >= 2]
    print(f"Valid test pairs: {len(test_pairs)}\n")

    # ── Evaluate ──
    k_values = [1, 5, 10, 20, 50, 100, 500, 1000, 5000]
    hits = {k: 0 for k in k_values}
    ranks_arr = []
    t0 = time.time()

    for qi, (src, tgt) in enumerate(test_pairs):
        src_norm = paper_norms[src]
        tgt_norm = paper_norms[tgt]

        if src_norm == 0 or tgt_norm == 0:
            rank = n_total
            ranks_arr.append(rank)
            if (qi + 1) % 500 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (qi + 1) * (len(test_pairs) - qi - 1)
                print(f"  [{qi+1}/{len(test_pairs)}] R@10={hits[10]/(qi+1)*100:.1f}%  "
                      f"R@100={hits[100]/(qi+1)*100:.1f}%  {elapsed:.0f}s  ETA {eta:.0f}s",
                      flush=True)
            continue

        # Source 1-hop IDF-weighted vector: slice columns from S_idf
        src_1hop = set()
        for ref in ref_lists[src]:
            src_1hop.update(concepts[ref])
        src_cols = np.array(list(src_1hop), dtype=np.int32)
        src_vals = idf[src_cols]
        sub = S_idf[:, src_cols]                       # (N, K_src) sparse CSC
        dot_products = (sub @ src_vals).ravel()        # (N,) dense float32

        cos_sim = dot_products * inv_norm / src_norm   # (N,) cosine similarity

        # Rank among non-zero candidates only
        nnz_mask = cos_sim > 0
        if not nnz_mask.any():
            rank = n_total
            ranks_arr.append(rank)
            for k in k_values:
                if rank < k: hits[k] += 1
            if (qi + 1) % 500 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (qi + 1) * (len(test_pairs) - qi - 1)
                print(f"  [{qi+1}/{len(test_pairs)}] R@10={hits[10]/(qi+1)*100:.1f}%  "
                      f"R@100={hits[100]/(qi+1)*100:.1f}%  {elapsed:.0f}s  ETA {eta:.0f}s",
                      flush=True)
            continue

        cos_nz    = cos_sim[nnz_mask]
        cand_idxs = np.where(nnz_mask)[0]
        tgt_cos   = cos_sim[tgt]

        if tgt_cos == 0:
            # Target has no overlap → rank = n_cand + (tgt position among zeros)
            rank = len(cos_nz) + (np.arange(n_total) < tgt)[~nnz_mask].sum()
        else:
            higher    = int((cos_nz > tgt_cos).sum())
            tie_lower = int(((cos_nz == tgt_cos) & (cand_idxs < tgt)).sum())
            rank = higher + tie_lower
        ranks_arr.append(rank)

        for k in k_values:
            if rank < k:
                hits[k] += 1

        if (qi + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (qi + 1) * (len(test_pairs) - qi - 1)
            r10  = hits[10]  / (qi + 1) * 100
            r100 = hits[100] / (qi + 1) * 100
            print(f"  [{qi+1}/{len(test_pairs)}] R@10={r10:.1f}%  "
                  f"R@100={r100:.1f}%  {elapsed:.0f}s  ETA {eta:.0f}s",
                  flush=True)

    n_q = len(test_pairs)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/n_q*1000:.1f}ms/query)")
    print(f"\n─── IDF-weighted cosine(1hop_src, 1hop_tgt) ───")
    print(f"{'K':>6}  {'Recall':>8}  {'Hits':>8}")
    print("─" * 30)
    for k in k_values:
        recall = hits[k] / n_q
        print(f"{k:>6}  {recall*100:>7.1f}%  {hits[k]:>4}/{n_q}")

    ranks_np = np.array(ranks_arr)
    found = ranks_np < 5000
    mrr = (1.0 / (ranks_np[found] + 1)).sum() / n_q
    print(f"\nMRR@5000: {mrr:.4f}")
    print(f"Found in top-5000: {found.sum()}/{n_q} ({found.sum()/n_q*100:.1f}%)")


if __name__ == "__main__":
    main()
