"""
Build temporal LBD data v2: PPR distance + L<=3 score>=0.7.

Changes from v1:
  - MAX_LEVEL=3, MIN_SCORE=0.7
  - PPR distance instead of BFS shortest path
  - PPR change (t2 - t1) as labels (positive = convergence)
  - New concept_ids.json for the new graph

Pipeline:
  1. Build temporal graphs with new params
  2. Compute PPR distance matrices (parallel)
  3. Compute PPR_change for eligible pairs
  4. Save pairs/labels as chunk files
"""
import os, time, json
from itertools import combinations
from multiprocessing import Process
import numpy as np
import scipy.sparse as sp

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
PAPERS_GEOL = '/root/autodl-tmp/openalex_data/geology_papers.jsonl'
PAPERS_CLIM = '/root/autodl-tmp/openalex_data/climate_papers.jsonl'

TEMP_DIR = f'{DATA_DIR}/temporal_v2'
CHUNK_SIZE = 1_000_000

YEAR_CUTOFFS = [2012, 2017, 2021, 2024]
MIN_SCORE = 0.7
MAX_LEVEL = 3

# PPR params
PPR_ALPHA = 0.15
PPR_ITERS = 30
# PPR threshold: pairs with PPR_t1 > this are "already close" → excluded
PPR_DISTANT_THRESH = 0.0001
# PPR change must exceed this to be included (filters noise)
PPR_CHANGE_MIN = 1e-7


# ── 1. Stream papers ────────────────────────────────────────────────────
def iter_papers():
    """Yield papers one at a time (no full load into memory)."""
    prefix = 'https://openalex.org/'
    for path in [PAPERS_GEOL, PAPERS_CLIM]:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
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


# ── 2. Build concept index + temporal graphs ────────────────────────────
def build_concept_index_and_graphs(year_cutoffs):
    """Build concept_to_idx from papers, then build graphs. Streaming (no full load)."""
    prefix = 'https://openalex.org/'
    concept_to_idx = {}
    idx = 0

    # First pass: collect valid concept IDs
    print("  Pass 1: collecting concept IDs...", flush=True)
    for p in iter_papers():
        for cid, score, level in p['concepts']:
            if level <= MAX_LEVEL and score >= MIN_SCORE:
                if cid not in concept_to_idx:
                    concept_to_idx[cid] = idx
                    idx += 1

    n = len(concept_to_idx)
    print(f"  {n} concepts (L<={MAX_LEVEL}, score>={MIN_SCORE})", flush=True)

    idx_to_cid = {v: k for k, v in concept_to_idx.items()}
    concept_ids = [idx_to_cid[i] for i in range(n)]
    cid_path = f'{TEMP_DIR}/concept_ids_v2.json'
    with open(cid_path, 'w') as f:
        json.dump(concept_ids, f)
    print(f"  Saved {cid_path}", flush=True)

    # Second pass: build graphs per cutoff
    sorted_cutoffs = sorted(year_cutoffs)
    needed = [y for y in sorted_cutoffs
              if not os.path.exists(f'{TEMP_DIR}/graph_v2_{y}.npz')]
    if not needed:
        print("  All graphs exist.", flush=True)
        return concept_to_idx, n

    print(f"  Pass 2: building graphs for {needed}...", flush=True)
    edges = set()
    last_cutoff_idx = 0

    for cutoff in needed:
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
        sp.save_npz(f'{TEMP_DIR}/graph_v2_{cutoff}.npz', adj)
        print(f"  graph_v2_{cutoff}: {adj.nnz:,} edges, saved", flush=True)

    return concept_to_idx, n


# ── 3. Compute PPR distance matrices ───────────────────────────────────
def compute_ppr(adj_csr, alpha=PPR_ALPHA, n_iter=PPR_ITERS):
    """Full PPR matrix via power iteration. Returns dense (n, n) float32."""
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
        for _ in range(n_iter):
            r = (1 - alpha) * (M @ r) + alpha * r
        ppr[:, bs:be] = r
    return ppr


def _compute_ppr_worker(adj_path, out_path, year):
    adj = sp.load_npz(adj_path).tocsr()
    n = adj.shape[0]
    print(f"  [{year}] PPR start ({n} nodes, {adj.nnz:,} edges)...", flush=True)
    t0 = time.time()
    ppr = compute_ppr(adj)
    print(f"  [{year}] PPR done: {time.time()-t0:.1f}s", flush=True)
    np.save(out_path, ppr.astype(np.float32))
    print(f"  [{year}] saved {os.path.getsize(out_path)/1e9:.2f}GB", flush=True)


def compute_ppr_matrices_parallel(year_cutoffs):
    to_compute = [y for y in year_cutoffs
                  if not os.path.exists(f'{TEMP_DIR}/ppr_v2_{y}.npy')]
    if not to_compute:
        print("  All PPR matrices exist.", flush=True)
        return
    print(f"  Computing {len(to_compute)} PPR matrices (parallel)...", flush=True)
    t0 = time.time()
    procs = []
    for y in to_compute:
        p = Process(target=_compute_ppr_worker,
                    args=(f'{TEMP_DIR}/graph_v2_{y}.npz', f'{TEMP_DIR}/ppr_v2_{y}.npy', y))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print(f"  All PPR matrices done: {time.time()-t0:.1f}s", flush=True)


# ── 4. Compute pairs from PPR matrices ──────────────────────────────────
def compute_all_pairs(ppr_t1, ppr_t2, rng, neg_ratio=5):
    print(f"    Computing PPR change for eligible pairs...", flush=True)
    t0 = time.time()

    n = ppr_t1.shape[0]
    rows_all, cols_all = np.triu_indices(n, k=1)
    t1_vals = ppr_t1[rows_all, cols_all]

    # Eligible: PPR_t1 is nonzero AND below threshold (not already close)
    eligible = (t1_vals > 0) & (t1_vals < PPR_DISTANT_THRESH)
    rows_e = rows_all[eligible]
    cols_e = cols_all[eligible]
    del rows_all, cols_all, t1_vals

    t1_e = ppr_t1[rows_e, cols_e]
    t2_e = ppr_t2[rows_e, cols_e]
    ppr_change = t2_e - t1_e  # positive = convergence (PPR increased)
    del t1_e, t2_e

    n_eligible = len(ppr_change)
    print(f"    eligible pairs: {n_eligible:,}", flush=True)
    print(f"    ppr_change: min={ppr_change.min():.8f}, max={ppr_change.max():.8f}, "
          f"mean={ppr_change.mean():.8f}", flush=True)

    # Convergent: PPR increased by more than noise threshold
    conv_mask = ppr_change > PPR_CHANGE_MIN
    n_conv = int(conv_mask.sum())
    print(f"    convergent (ppr_change > {PPR_CHANGE_MIN}): {n_conv:,} ({100*n_conv/n_eligible:.2f}%)", flush=True)

    # Non-convergent sample
    nonconv_mask = ~conv_mask
    n_nonconv_target = min(int(nonconv_mask.sum()), n_conv * neg_ratio)
    nonconv_indices = np.where(nonconv_mask)[0]
    rng.shuffle(nonconv_indices)
    nonconv_indices = nonconv_indices[:n_nonconv_target]

    conv_indices = np.where(conv_mask)[0]
    keep_indices = np.concatenate([conv_indices, nonconv_indices])
    rng.shuffle(keep_indices)

    pairs = np.column_stack([rows_e[keep_indices], cols_e[keep_indices]])
    labels = ppr_change[keep_indices].astype(np.float32)
    del rows_e, cols_e, ppr_change

    dt = time.time() - t0
    print(f"    final dataset: {len(pairs):,} pairs ({n_conv:,} conv + {n_nonconv_target:,} non-conv)", flush=True)
    print(f"    label mean={labels.mean():.8f}, computed in {dt:.0f}s", flush=True)
    return pairs, labels


# ── 5. Save chunks ─────────────────────────────────────────────────────
def save_chunks(pairs, labels, chunk_dir, prefix):
    os.makedirs(chunk_dir, exist_ok=True)
    n = len(pairs)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    for ci in range(n_chunks):
        s = ci * CHUNK_SIZE
        e = min(s + CHUNK_SIZE, n)
        np.savez_compressed(f'{chunk_dir}/{prefix}_chunk_{ci:04d}.npz',
                            pairs=pairs[s:e], labels=labels[s:e])
    print(f"    Saved {n_chunks} chunks ({n:,} pairs) to {chunk_dir}", flush=True)
    return n_chunks


# ── 6. Fold processing ─────────────────────────────────────────────────
def process_fold(fold, train_t1, train_t2, val_t1, val_t2):
    print(f"\n{'='*50}", flush=True)
    print(f"  Fold {fold}: train {train_t1}→{train_t2}, val {val_t1}→{val_t2}", flush=True)
    print(f"{'='*50}", flush=True)

    fold_dir = f'{TEMP_DIR}/fold{fold}_v2_chunks'
    if os.path.exists(f'{fold_dir}/manifest.json'):
        print(f"  Fold {fold} v2 chunks exist, skipping.", flush=True)
        return

    rng = np.random.RandomState(42 + fold)

    print(f"  [fold{fold}] Loading ppr_{train_t1} + ppr_{train_t2}...", flush=True)
    ppr_train_t1 = np.load(f'{TEMP_DIR}/ppr_v2_{train_t1}.npy')
    ppr_train_t2 = np.load(f'{TEMP_DIR}/ppr_v2_{train_t2}.npy')
    train_pairs, train_labels = compute_all_pairs(ppr_train_t1, ppr_train_t2, rng)
    del ppr_train_t1, ppr_train_t2

    n_train_chunks = save_chunks(train_pairs, train_labels, fold_dir, 'train')
    del train_pairs, train_labels

    print(f"  [fold{fold}] Loading ppr_{val_t1} + ppr_{val_t2}...", flush=True)
    ppr_val_t1 = np.load(f'{TEMP_DIR}/ppr_v2_{val_t1}.npy')
    ppr_val_t2 = np.load(f'{TEMP_DIR}/ppr_v2_{val_t2}.npy')
    val_pairs, val_labels = compute_all_pairs(ppr_val_t1, ppr_val_t2, rng)
    del ppr_val_t1, ppr_val_t2

    n_val_chunks = save_chunks(val_pairs, val_labels, fold_dir, 'val')
    del val_pairs, val_labels

    manifest = {
        'fold': fold,
        'window_train': f'{train_t1}_{train_t2}',
        'window_val': f'{val_t1}_{val_t2}',
        'n_train_chunks': n_train_chunks,
        'n_val_chunks': n_val_chunks,
        'method': 'ppr',
        'max_level': MAX_LEVEL,
        'min_score': MIN_SCORE,
    }
    with open(f'{fold_dir}/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  [fold{fold}] Done.", flush=True)


# ── Main ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    t0 = time.time()

    print(f"\n=== Step 1: Build graphs (L<={MAX_LEVEL}, score>={MIN_SCORE}) ===", flush=True)
    concept_to_idx, n = build_concept_index_and_graphs(YEAR_CUTOFFS)

    print("\n=== Step 2: PPR distance matrices ===", flush=True)
    compute_ppr_matrices_parallel(YEAR_CUTOFFS)

    folds = [
        (0, 2012, 2024, 2012, 2021),
        (1, 2017, 2024, 2017, 2021),
    ]
    for fold, tt1, tt2, vt1, vt2 in folds:
        print(f"\n=== Processing fold {fold} ===", flush=True)
        process_fold(fold, tt1, tt2, vt1, vt2)

    print(f"\nTotal: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
