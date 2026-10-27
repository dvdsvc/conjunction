#!/usr/bin/env python3
"""Process OpenAlex data into training tensors for the Phoenix citation model.

Multi-core optimized: orjson + multiprocessing.Pool (fork) for all CPU-heavy loops.
Target: 20+ cores for concept counting and feature computation.

Reads: climate_papers.jsonl, geology_papers.jsonl, concepts.jsonl, lbd_bridges.jsonl
Writes: features.pt, adjacency.pt, ref_lists.pt, splits.pt, vocab.json, config.json
"""

import argparse
import gc
import json
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import orjson
import torch
from tqdm import tqdm


# Cross-domain concept IDs for bridge detection
CLIMATE_CONCEPTS = {"C49204034", "C91586092"}
GEOLOGY_CONCEPTS = {"C1965285", "C17409809", "C199289684"}


# ---------------------------------------------------------------------------
# 1. Load papers — orjson for fast JSONL parsing
# ---------------------------------------------------------------------------

def load_papers(data_dir):
    """Load all papers from climate + geology JSONL files (orjson)."""
    papers = {}
    for name in ["climate_papers.jsonl", "geology_papers.jsonl"]:
        path = os.path.join(data_dir, name)
        print(f"Loading {path}...")
        with open(path, "rb") as f:
            for line in tqdm(f, desc=name):
                d = orjson.loads(line)
                pid = d["id"]
                if pid not in papers:
                    papers[pid] = d
                else:
                    old_c = len(papers[pid].get("concepts", []))
                    new_c = len(d.get("concepts", []))
                    if new_c > old_c:
                        papers[pid] = d
    return papers


# ---------------------------------------------------------------------------
# 2. Build concept vocab — multiprocessing counter
# ---------------------------------------------------------------------------

def _count_concepts_chunk(paper_ids_chunk):
    """Worker: count level 0-2 concept occurrences for a chunk of paper IDs.
    Relies on fork COW to access _GLOBAL_PAPERS without pickling."""
    papers = _GLOBAL_PAPERS
    counter = Counter()
    for pid in paper_ids_chunk:
        p = papers[pid]
        for c in p.get("concepts", []):
            cid = c["id"].split("/")[-1]
            if c.get("level", 0) <= 2:
                counter[cid] += 1
    return counter


def build_concept_vocab(papers, min_count=5, max_level=2, n_workers=16):
    """Build concept vocabulary using multiprocessing."""
    global _GLOBAL_PAPERS
    _GLOBAL_PAPERS = papers

    all_ids = list(papers.keys())
    chunk_size = max(1, len(all_ids) // n_workers)
    chunks = [all_ids[i:i + chunk_size] for i in range(0, len(all_ids), chunk_size)]
    print(f"Counting concepts across {len(chunks)} workers...")

    merged = Counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for partial in tqdm(pool.map(_count_concepts_chunk, chunks),
                            total=len(chunks), desc="Concept chunks"):
            merged.update(partial)

    vocab = {cid: idx for idx, (cid, _) in enumerate(
        sorted([(c, n) for c, n in merged.items() if n >= min_count],
               key=lambda x: -x[1])
    )}
    print(f"Concept vocab: {len(vocab)} concepts (level 0-{max_level}, min_count={min_count})")
    del _GLOBAL_PAPERS
    return vocab


# ---------------------------------------------------------------------------
# 3. Compute paper features — parallel bridge_score + entropy
# ---------------------------------------------------------------------------

def _compute_features_chunk(paper_ids_chunk):
    """Worker: compute scalar features for a chunk of paper IDs.
    Uses fork COW for _GLOBAL_PAPERS and _GLOBAL_PAPER_DOMAIN."""
    papers = _GLOBAL_PAPERS
    domain = _GLOBAL_PAPER_DOMAIN
    in_degree = _GLOBAL_IN_DEGREE
    vocab = _GLOBAL_VOCAB

    results = {}
    for pid in paper_ids_chunk:
        p = papers[pid]
        year = p.get("year", 0)
        refs = p.get("refs", [])

        # Concept indices (sparse)
        cids = set()
        for c in p.get("concepts", []):
            cid = c["id"].split("/")[-1]
            if c.get("level", 0) <= 2:
                cids.add(cid)
        concept_idx = [vocab[cid] for cid in cids if cid in vocab]

        # Bridge score
        is_c, is_g = domain.get(pid, (False, False))
        bridge = 0.0
        if (is_c or is_g) and refs:
            cross = counted = 0
            for r in refs:
                r_c, r_g = domain.get(r, (False, False))
                if not r_c and not r_g:
                    continue
                counted += 1
                if (is_c and r_g) or (is_g and r_c):
                    cross += 1
            bridge = cross / max(counted, 1)

        # Concept entropy
        scores = [c.get("score", 0) for c in p.get("concepts", []) if c.get("score", 0) > 0]
        if scores:
            total = sum(scores)
            probs = [s / total for s in scores]
            entropy = -sum(pr * math.log(pr + 1e-10) for pr in probs)
        else:
            entropy = 0.0

        results[pid] = {
            "year": year,
            "in_degree": in_degree.get(pid, 0),
            "out_degree": len(refs),
            "bridge_score": bridge,
            "entropy": entropy,
            # concept_indices and refs NOT stored here — read from papers in save_tensors
        }
    return results


def compute_paper_features(papers, vocab, n_workers=16):
    """Compute features using multiprocessing (fork COW, zero pickling)."""
    global _GLOBAL_PAPERS, _GLOBAL_PAPER_DOMAIN, _GLOBAL_IN_DEGREE, _GLOBAL_VOCAB

    # Step 1: in-degree (single-threaded, fast)
    print("Counting in-degrees...")
    in_degree = Counter()
    for p in tqdm(papers.values(), desc="Degrees"):
        for r in p.get("refs", []):
            in_degree[r] += 1

    # Step 2: domain classification (single-threaded, fast)
    print("Classifying papers...")
    paper_domain = {}
    for pid, p in tqdm(papers.items(), desc="Domain"):
        cids = set()
        for c in p.get("concepts", []):
            cid = c["id"].split("/")[-1]
            if c.get("level", 0) <= 2:
                cids.add(cid)
        paper_domain[pid] = (
            bool(cids & CLIMATE_CONCEPTS),
            bool(cids & GEOLOGY_CONCEPTS),
        )

    # Step 3: parallel feature computation
    _GLOBAL_PAPERS = papers
    _GLOBAL_PAPER_DOMAIN = paper_domain
    _GLOBAL_IN_DEGREE = in_degree
    _GLOBAL_VOCAB = vocab

    all_ids = list(papers.keys())
    chunk_size = max(1, len(all_ids) // n_workers)
    chunks = [all_ids[i:i + chunk_size] for i in range(0, len(all_ids), chunk_size)]
    print(f"Computing features across {len(chunks)} workers...")

    features = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for partial in tqdm(pool.map(_compute_features_chunk, chunks),
                            total=len(chunks), desc="Feature chunks"):
            features.update(partial)

    del _GLOBAL_PAPERS, _GLOBAL_PAPER_DOMAIN, _GLOBAL_IN_DEGREE, _GLOBAL_VOCAB
    return all_ids, features


# ---------------------------------------------------------------------------
# 4. Build adjacency — parallel
# ---------------------------------------------------------------------------

def _build_adj_chunk(paper_ids_chunk):
    """Worker: build adjacency for a chunk."""
    papers = _GLOBAL_PAPERS
    idx = _GLOBAL_IDX
    adj = defaultdict(list)
    edges = []
    for pid in paper_ids_chunk:
        if pid not in idx:
            continue
        u = idx[pid]
        for ref_id in papers[pid].get("refs", []):
            if ref_id in idx:
                v = idx[ref_id]
                adj[u].append(v)
                edges.append((u, v))
    return dict(adj), edges


def build_adjacency(papers, paper_id_to_idx, n_workers=16):
    """Build graph adjacency list in parallel."""
    global _GLOBAL_PAPERS, _GLOBAL_IDX
    _GLOBAL_PAPERS = papers
    _GLOBAL_IDX = paper_id_to_idx

    all_ids = list(papers.keys())
    chunk_size = max(1, len(all_ids) // n_workers)
    chunks = [all_ids[i:i + chunk_size] for i in range(0, len(all_ids), chunk_size)]
    print(f"Building adjacency across {len(chunks)} workers...")

    merged_adj = defaultdict(list)
    all_edges = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for adj, edges in tqdm(pool.map(_build_adj_chunk, chunks),
                               total=len(chunks), desc="Adj chunks"):
            for k, v in adj.items():
                merged_adj[k].extend(v)
            all_edges.extend(edges)

    del _GLOBAL_PAPERS, _GLOBAL_IDX
    print(f"Adjacency: {len(merged_adj)} nodes, {len(all_edges)} edges (in-graph only)")
    return dict(merged_adj), all_edges


# ---------------------------------------------------------------------------
# 5. Split, save tensors — same as before
# ---------------------------------------------------------------------------

def time_based_split(papers, paper_ids, train_end=2018, val_end=2021):
    train_ids, val_ids, test_ids = [], [], []
    for pid in paper_ids:
        year = papers[pid].get("year", 0)
        if year <= train_end:
            train_ids.append(pid)
        elif year <= val_end:
            val_ids.append(pid)
        else:
            test_ids.append(pid)
    print(f"Split: train={len(train_ids):,} (≤{train_end}), "
          f"val={len(val_ids):,} ({train_end+1}-{val_end}), "
          f"test={len(test_ids):,} (>{val_end})")
    return train_ids, val_ids, test_ids


def save_tensors(papers, paper_ids, features, paper_id_to_idx, adj, edges,
                 train_ids, val_ids, test_ids, vocab, output_dir):
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    n = len(paper_ids)
    dim_vocab = len(vocab)

    years = [features[pid]["year"] for pid in paper_ids if features[pid]["year"] > 0]
    year_min, year_max = min(years), max(years)
    print(f"Year range: {year_min}-{year_max}")

    print("Building feature tensors (parallel save)...")
    scalar_dim = 5
    feature_tensor = torch.zeros(n, scalar_dim)

    concept_indices = []
    ref_lists = []
    for i, pid in enumerate(tqdm(paper_ids, desc="Tensors")):
        f = features[pid]
        year_norm = (f["year"] - year_min) / max(year_max - year_min, 1) if f["year"] > 0 else 0
        feature_tensor[i, 0] = year_norm
        feature_tensor[i, 1] = math.log1p(f["in_degree"])
        feature_tensor[i, 2] = math.log1p(f["out_degree"])
        feature_tensor[i, 3] = f["bridge_score"]
        feature_tensor[i, 4] = f["entropy"]
        # Get concept indices from papers (not stored in features to save memory)
        p = papers[pid]
        cids = set()
        for c in p.get("concepts", []):
            cid = c["id"].split("/")[-1]
            if c.get("level", 0) <= 2:
                cids.add(cid)
        concept_indices.append([vocab[cid] for cid in cids if cid in vocab])
        refs = p.get("refs", [])
        ref_lists.append([paper_id_to_idx[r] for r in refs if r in paper_id_to_idx])

    torch.save({
        "features": feature_tensor,
        "concepts": concept_indices,
        "paper_ids": paper_ids,
        "paper_id_to_idx": paper_id_to_idx,
        "year_range": (year_min, year_max),
    }, os.path.join(output_dir, "features.pt"))
    avg_c = sum(len(c) for c in concept_indices) / max(n, 1)
    print(f"Saved features.pt: features {feature_tensor.shape}, concepts list[{n}] avg={avg_c:.1f} nonzero/paper")

    if edges:
        src, dst = zip(*edges)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    torch.save({"edge_index": edge_index, "adj": adj},
               os.path.join(output_dir, "adjacency.pt"))
    print(f"Saved adjacency.pt: {edge_index.shape[1]} edges")

    torch.save(ref_lists, os.path.join(output_dir, "ref_lists.pt"))
    avg_refs = np.mean([len(r) for r in ref_lists])
    pct_no_refs = sum(1 for r in ref_lists if not r) / len(ref_lists) * 100
    print(f"Saved ref_lists.pt: avg={avg_refs:.1f} refs/paper, {pct_no_refs:.1f}% with 0 in-graph refs")

    torch.save({
        "train": [paper_id_to_idx[pid] for pid in train_ids if pid in paper_id_to_idx],
        "val":   [paper_id_to_idx[pid] for pid in val_ids   if pid in paper_id_to_idx],
        "test":  [paper_id_to_idx[pid] for pid in test_ids  if pid in paper_id_to_idx],
    }, os.path.join(output_dir, "splits.pt"))

    inv_vocab = {v: k for k, v in vocab.items()}
    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump({"concept_to_idx": vocab, "idx_to_concept": inv_vocab}, f)
    print(f"Saved vocab.json: {len(vocab)} concepts")

    config = {
        "n_papers": n,
        "n_edges": len(edges),
        "scalar_feature_dim": scalar_dim,
        "concept_vocab_size": dim_vocab,
        "year_range": [year_min, year_max],
        "train_size": len([x for x in train_ids if x in paper_id_to_idx]),
        "val_size":   len([x for x in val_ids   if x in paper_id_to_idx]),
        "test_size":  len([x for x in test_ids  if x in paper_id_to_idx]),
        "features": ["year_norm", "log_in_degree", "log_out_degree",
                     "bridge_score", "citing_concept_entropy"],
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config.json")

    bridges_path = os.path.join(os.path.dirname(output_dir), "lbd_bridges.jsonl")
    if os.path.exists(bridges_path):
        bridge_pairs = []
        with open(bridges_path, "rb") as f:
            for line in f:
                d = orjson.loads(line)
                src_idx = paper_id_to_idx.get(d["source"])
                tgt_idx = paper_id_to_idx.get(d["target"])
                if src_idx is not None and tgt_idx is not None:
                    bridge_pairs.append({
                        "source_idx": src_idx,
                        "target_idx": tgt_idx,
                        "ranker_score": d.get("ranker_score", 0),
                        "retrieval_score": d.get("retrieval_score", 0),
                    })
        torch.save(bridge_pairs, os.path.join(output_dir, "lbd_bridges.pt"))
        print(f"Saved lbd_bridges.pt: {len(bridge_pairs)} bridge pairs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-concept-count", type=int, default=5)
    parser.add_argument("--max-concept-level", type=int, default=2)
    parser.add_argument("--train-end", type=int, default=2018)
    parser.add_argument("--val-end", type=int, default=2021)
    parser.add_argument("--workers", "-j", type=int, default=6)
    args = parser.parse_args()

    # Use fork for COW memory sharing (Linux default, avoids pickling papers dict)
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

    output_dir = args.output_dir or os.path.join(args.data_dir, "processed")
    data_dir = args.data_dir

    print(f"Workers: {args.workers}")
    papers = load_papers(data_dir)
    print(f"Total papers: {len(papers):,}")

    vocab = build_concept_vocab(papers, min_count=args.min_concept_count,
                                max_level=args.max_concept_level, n_workers=args.workers)
    gc.collect()

    paper_ids, features = compute_paper_features(papers, vocab, n_workers=args.workers)
    paper_id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    gc.collect()

    adj, edges = build_adjacency(papers, paper_id_to_idx, n_workers=args.workers)
    gc.collect()

    train_ids, val_ids, test_ids = time_based_split(
        papers, paper_ids, args.train_end, args.val_end)

    save_tensors(papers, paper_ids, features, paper_id_to_idx, adj, edges,
                 train_ids, val_ids, test_ids, vocab, output_dir)

    bridge_scores = [features[pid]["bridge_score"] for pid in paper_ids]
    nonzero_bridge = sum(1 for s in bridge_scores if s > 0)
    print(f"\nBridge score stats: {nonzero_bridge:,} papers with bridge_score > 0 "
          f"({100*nonzero_bridge/len(paper_ids):.1f}%)")
    print(f"Mean bridge_score (nonzero): {sum(bridge_scores)/max(nonzero_bridge,1):.3f}")
    print(f"\nDone! All data saved to {output_dir}")


if __name__ == "__main__":
    main()
