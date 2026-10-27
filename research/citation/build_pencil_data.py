"""
Build PENCIL data - fast version.
Saves: graph as sparse tensor, triplets as int tensor.
Subgraph construction deferred to DataLoader at training time.
"""

import json, os, time, random, pickle
from collections import defaultdict, Counter
from itertools import combinations
from multiprocessing import Pool
import numpy as np
import scipy.sparse as sp
import torch

MIN_SCORE = 0.3
MAX_LEVEL = 3
N_WORKERS = 60
YEAR_CUTOFF = 2020

DATA_DIR = '/root/autodl-tmp/openalex_data'
OUT_DIR  = '/root/autodl-tmp/openalex_data/pencil_data'
os.makedirs(OUT_DIR, exist_ok=True)

def parse_line(line):
    d = json.loads(line)
    cids = []
    for c in d.get('concepts', []):
        cid = c['id']
        if '/' in cid:
            cid = cid.rsplit('/', 1)[-1]
        if 1 <= c['level'] <= MAX_LEVEL and c['score'] >= MIN_SCORE:
            cids.append(cid)
    return (d['id'], d['year'], cids) if cids else None

def load_papers_parallel(path):
    with open(path) as f:
        lines = f.readlines()
    with Pool(N_WORKERS) as pool:
        results = pool.map(parse_line, lines, chunksize=5000)
    return [r for r in results if r is not None]

def build_edges_chunk(args):
    papers_chunk, year_lo, year_hi = args
    edges = set()
    degree = Counter()
    for pid, year, cids in papers_chunk:
        if year < year_lo or year > year_hi:
            continue
        uniq = sorted(set(cids))
        for c in uniq:
            degree[c] += 1
        for a, b in combinations(uniq, 2):
            edges.add((a, b))
    return edges, degree

def make_sparse_adj(n_concepts, edges, c2i):
    """Build scipy sparse adjacency matrix from edge set."""
    rows, cols = [], []
    for a, b in edges:
        ia, ib = c2i[a], c2i[b]
        rows.extend([ia, ib])
        cols.extend([ib, ia])
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n_concepts, n_concepts))
    return adj

def find_hard_negs_batch(pos_pairs_chunk, neighbors, pre_edges, c2i, n_neg=5):
    """Vectorized-ish hard negative sampling for a chunk of positive pairs."""
    results = []  # list of (anchor, neg_concept)
    for u, v in pos_pairs_chunk:
        cands = set()
        nu = list(neighbors[u])
        if len(nu) > 30:
            random.shuffle(nu)
            nu = nu[:30]
        for n1 in nu:
            for n2 in neighbors[n1]:
                if n2 != u and n2 != v:
                    e = (min(u, n2), max(u, n2))
                    if e not in pre_edges:
                        cands.add(n2)
                if len(cands) >= 200:
                    break
            if len(cands) >= 200:
                break
        if cands:
            cl = list(cands)
            random.shuffle(cl)
            for neg in cl[:n_neg]:
                results.append((u, neg))
        else:
            results.append(None)  # placeholder to keep alignment
    return results

if __name__ == '__main__':
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)

    # ── Load ──
    print("=== Loading papers ===", flush=True)
    climate = load_papers_parallel(f'{DATA_DIR}/climate_papers.jsonl')
    geology = load_papers_parallel(f'{DATA_DIR}/geology_papers.jsonl')
    all_papers = climate + geology
    del climate, geology
    print(f"  total: {len(all_papers)}", flush=True)

    all_concepts = set()
    for _, _, cids in all_papers:
        all_concepts.update(cids)
    concept_list = sorted(all_concepts)
    c2i = {c: i for i, c in enumerate(concept_list)}
    n_concepts = len(concept_list)
    print(f"  concepts: {n_concepts}", flush=True)

    # ── Pre-2021 graph ──
    print("\n=== Pre-2021 graph ===", flush=True)
    cs = (len(all_papers) + N_WORKERS - 1) // N_WORKERS
    chunks = [(all_papers[i:i+cs], 0, YEAR_CUTOFF) for i in range(0, len(all_papers), cs)]
    pre_edges = set()
    concept_degree = Counter()
    with Pool(N_WORKERS) as pool:
        for e, d in pool.imap_unordered(build_edges_chunk, chunks, chunksize=1):
            pre_edges.update(e)
            concept_degree += d
    del chunks
    print(f"  edges: {len(pre_edges)}, time: {time.time()-t0:.1f}s", flush=True)

    # Save sparse adjacency
    print("  building sparse adjacency...", flush=True)
    adj = make_sparse_adj(n_concepts, pre_edges, c2i)
    sp.save_npz(f'{OUT_DIR}/adj_pre2021.npz', adj)
    print(f"  saved adj: {adj.nnz} entries, shape {adj.shape}", flush=True)

    # Neighbor index (for hard negative sampling)
    neighbors = defaultdict(set)
    for a, b in pre_edges:
        neighbors[a].add(b)
        neighbors[b].add(a)

    # Save degree as tensor
    deg_arr = np.zeros(n_concepts, dtype=np.float32)
    for cid, d in concept_degree.items():
        deg_arr[c2i[cid]] = d
    torch.save(torch.from_numpy(deg_arr), f'{OUT_DIR}/degree.pt')

    # ── Post-2021 positives ──
    print("\n=== Post-2021 positives ===", flush=True)
    chunks_p = [(all_papers[i:i+cs], YEAR_CUTOFF+1, 9999) for i in range(0, len(all_papers), cs)]
    del all_papers
    post_edges = set()
    with Pool(N_WORKERS) as pool:
        for e, _ in pool.imap_unordered(build_edges_chunk, chunks_p, chunksize=1):
            post_edges.update(e)
    del chunks_p

    positive_pairs = list(post_edges - pre_edges)
    del post_edges
    random.shuffle(positive_pairs)
    print(f"  positive pairs: {len(positive_pairs)}", flush=True)

    # ── Build triplets (parallel) ──
    # Format: each row = (anchor_idx, pos_idx) or (anchor_idx, neg_idx)
    # Labels: 1 for positive, 0 for negative
    print("\n=== Building triplets (parallel) ===", flush=True)

    n = len(positive_pairs)
    splits = {
        'train': positive_pairs[:int(n*0.8)],
        'val':   positive_pairs[int(n*0.8):int(n*0.9)],
        'test':  positive_pairs[int(n*0.9):],
    }
    del positive_pairs

    for split_name, pos_pairs in splits.items():
        t1 = time.time()
        print(f"\n  {split_name}: {len(pos_pairs)} positives", flush=True)

        # Encode positive pairs as (anchor_idx, candidate_idx, label)
        pos_indices = []
        for u, v in pos_pairs:
            pos_indices.append([c2i[u], c2i[v], 1])

        # Hard negatives in parallel
        chunk_sz = (len(pos_pairs) + N_WORKERS - 1) // N_WORKERS
        neg_chunks = []
        for i in range(0, len(pos_pairs), chunk_sz):
            neg_chunks.append(pos_pairs[i:i+chunk_sz])

        print(f"  sampling hard negatives ({len(neg_chunks)} chunks)...", flush=True)
        with Pool(N_WORKERS) as pool:
            neg_results = pool.starmap(
                find_hard_negs_batch,
                [(chunk, neighbors, pre_edges, c2i, 5) for chunk in neg_chunks],
                chunksize=1
            )

        # Flatten negatives
        neg_indices = []
        neg_fail = 0
        for chunk_result in neg_results:
            for r in chunk_result:
                if r is None:
                    neg_fail += 1
                else:
                    u, neg = r
                    neg_indices.append([c2i[u], c2i[neg], 0])

        # Combine and save
        all_triples = np.array(pos_indices + neg_indices, dtype=np.int32)
        np.random.shuffle(all_triples)

        out_path = f'{OUT_DIR}/{split_name}_triplets.npy'
        np.save(out_path, all_triples)
        dt = time.time() - t1
        print(f"  {split_name}: {len(all_triples)} triplets (pos={len(pos_indices)}, "
              f"neg={len(neg_indices)}, neg_fail={neg_fail}), {dt:.1f}s", flush=True)

    # ── Save metadata ──
    meta = {
        'n_concepts': n_concepts,
        'n_pre2021_edges': len(pre_edges),
        'max_subgraph_nodes': 128,
        'n_hard_neg_per_pos': 5,
        'year_cutoff': YEAR_CUTOFF,
    }
    with open(f'{OUT_DIR}/metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    with open(f'{OUT_DIR}/concept_ids.json', 'w') as f:
        json.dump(concept_list, f)

    print(f"\n=== Done. Total: {time.time()-t0:.1f}s ===", flush=True)
