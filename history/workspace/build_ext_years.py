"""Build graphs and PPR for 2005 and 2015 on server. Uses same concept set as v2 (L<=3, score>=0.7)."""
import os, time, json
from itertools import combinations
import numpy as np
import scipy.sparse as sp

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
TEMP_DIR = f'{DATA_DIR}/temporal_v2'
PAPERS_GEOL = '/root/autodl-tmp/openalex_data/geology_papers.jsonl'
PAPERS_CLIM = '/root/autodl-tmp/openalex_data/climate_papers.jsonl'

MAX_LEVEL = 3
MIN_SCORE = 0.7
ALPHA = 0.15
N_ITER = 30
YEAR_CUTOFFS = [2005, 2015]


def iter_papers():
    prefix = 'https://openalex.org/'
    for path in [PAPERS_GEOL, PAPERS_CLIM]:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                p = json.loads(line)
                if 'year' not in p or 'concepts' not in p:
                    continue
                concepts = []
                for c in p['concepts']:
                    cid = c['id']
                    if cid.startswith(prefix):
                        cid = cid[len(prefix):]
                    concepts.append((cid, c.get('score', 0), c.get('level', 1)))
                yield {'year': int(p['year']), 'concepts': concepts}


def build_graph(cutoff, concept_to_idx, n):
    """Build cumulative graph for papers up to cutoff year."""
    out_path = f'{TEMP_DIR}/graph_v2_{cutoff}.npz'
    if os.path.exists(out_path):
        print(f"  graph_v2_{cutoff} exists, skipping.", flush=True)
        return

    print(f"  Building graph for cutoff {cutoff}...", flush=True)
    edges = set()
    for p in iter_papers():
        if p['year'] > cutoff:
            continue
        valid = []
        for cid, score, level in p['concepts']:
            if level <= MAX_LEVEL and score >= MIN_SCORE and cid in concept_to_idx:
                valid.append(concept_to_idx[cid])
        for a, b in combinations(valid, 2):
            edges.add((a, b) if a < b else (b, a))

    rows = np.array([e[0] for e in edges], dtype=np.int32)
    cols = np.array([e[1] for e in edges], dtype=np.int32)
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj + adj.T
    sp.save_npz(out_path, adj)
    print(f"  graph_v2_{cutoff}: {adj.nnz:,} edges, saved", flush=True)


def compute_ppr(adj_csr):
    n = adj_csr.shape[0]
    deg = np.array(adj_csr.sum(axis=1)).flatten()
    deg_inv = np.zeros(n, dtype=np.float32)
    deg_inv[deg > 0] = 1.0 / deg[deg > 0]
    M = (sp.diags(deg_inv) @ adj_csr).T.tocsr()

    ppr = np.zeros((n, n), dtype=np.float32)
    batch_size = 64
    for bs in range(0, n, batch_size):
        be = min(bs + batch_size, n)
        B = be - bs
        r = np.zeros((n, B), dtype=np.float32)
        for i in range(B):
            r[bs + i, i] = 1.0
        for _ in range(N_ITER):
            r = (1 - ALPHA) * (M @ r) + ALPHA * r
        ppr[:, bs:be] = r
    return ppr


def main():
    # Load existing concept mapping (from v2 build)
    with open(f'{TEMP_DIR}/concept_ids_v2.json') as f:
        concept_ids = json.load(f)
    concept_to_idx = {cid: i for i, cid in enumerate(concept_ids)}
    n = len(concept_ids)
    print(f"Concepts: {n}", flush=True)

    for cutoff in YEAR_CUTOFFS:
        print(f"\n=== Year {cutoff} ===", flush=True)
        build_graph(cutoff, concept_to_idx, n)

        ppr_path = f'{TEMP_DIR}/ppr_v2_{cutoff}.npy'
        if os.path.exists(ppr_path):
            print(f"  PPR {cutoff} exists, skipping.", flush=True)
            continue

        adj = sp.load_npz(f'{TEMP_DIR}/graph_v2_{cutoff}.npz').tocsr()
        print(f"  {adj.shape[0]} nodes, {adj.nnz:,} edges", flush=True)
        t0 = time.time()
        ppr = compute_ppr(adj)
        print(f"  PPR done in {time.time()-t0:.1f}s", flush=True)
        np.save(ppr_path, ppr)
        print(f"  Saved {ppr_path} ({ppr.nbytes/1e9:.2f}GB)", flush=True)
        del ppr

    print("\nAll done.", flush=True)


if __name__ == '__main__':
    main()
