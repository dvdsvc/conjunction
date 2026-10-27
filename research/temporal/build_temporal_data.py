"""
Build temporal LBD data: graphs, distances, pairs (chunk files).
Subgraphs built on-the-fly during training (not precomputed here).

Pipeline:
  1. Build temporal concept co-occurrence graphs (year cutoffs 2012/2017/2021/2024)
  2. Compute all-pairs shortest path distances via BFS (parallel)
  3. Compute distance_change for ALL eligible pairs, keep convergent + sample non-convergent
  4. Save pairs/labels as chunk files for training
"""
import os, time, json
from itertools import combinations
from multiprocessing import Process
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
PAPERS_GEOL = '/root/autodl-tmp/openalex_data/geology_papers.jsonl'
PAPERS_CLIM = '/root/autodl-tmp/openalex_data/climate_papers.jsonl'
CONCEPT_IDS_FILE = f'{DATA_DIR}/concept_ids.json'

TEMP_DIR = f'{DATA_DIR}/temporal_v2'
N_DIST_WORKERS = 4
CHUNK_SIZE = 1_000_000

YEAR_CUTOFFS = [2012, 2017, 2021, 2024]
MIN_SCORE = 0.5
MAX_LEVEL = 2


# ── 1. Load papers ──────────────────────────────────────────────────────
def load_papers():
    papers = []
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
                papers.append({'year': int(p['year']), 'concepts': concepts})
    return papers


# ── 2. Build temporal graph ─────────────────────────────────────────────
def build_all_graphs(papers, year_cutoffs, concept_to_idx):
    n = len(concept_to_idx)
    sorted_cutoffs = sorted(year_cutoffs)
    needed = [y for y in sorted_cutoffs
              if not os.path.exists(f'{TEMP_DIR}/graph_{y}.npz')]
    if not needed:
        print("  All graphs exist.", flush=True)
        return

    papers_sorted = sorted(papers, key=lambda p: p['year'])
    edges = set()
    paper_idx = 0

    for cutoff in needed:
        while paper_idx < len(papers_sorted) and papers_sorted[paper_idx]['year'] <= cutoff:
            p = papers_sorted[paper_idx]
            valid = []
            for cid, score, level in p['concepts']:
                if level <= MAX_LEVEL and score >= MIN_SCORE and cid in concept_to_idx:
                    valid.append(concept_to_idx[cid])
            for a, b in combinations(valid, 2):
                edges.add((a, b) if a < b else (b, a))
            paper_idx += 1

        if edges:
            rows = np.array([e[0] for e in edges], dtype=np.int32)
            cols = np.array([e[1] for e in edges], dtype=np.int32)
        else:
            rows = np.array([], dtype=np.int32)
            cols = np.array([], dtype=np.int32)
        data = np.ones(len(rows), dtype=np.float32)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        adj = adj + adj.T
        sp.save_npz(f'{TEMP_DIR}/graph_{cutoff}.npz', adj)
        print(f"  graph_{cutoff}: {adj.nnz:,} edges, saved", flush=True)


# ── 3. Compute distance matrices via BFS ────────────────────────────────
def _compute_dist_worker(adj_path, out_path, year_cutoff):
    adj = sp.load_npz(adj_path).tocsr()
    n = adj.shape[0]
    print(f"  [{year_cutoff}] BFS start ({n} nodes, {adj.nnz:,} edges)...", flush=True)
    t0 = time.time()
    dist = shortest_path(adj, directed=False, method='D')
    dt = time.time() - t0
    print(f"  [{year_cutoff}] BFS done: {dt:.1f}s", flush=True)
    dist[np.isinf(dist)] = 9999
    np.save(out_path, dist.astype(np.int16))
    print(f"  [{year_cutoff}] saved {os.path.getsize(out_path)/1e9:.2f}GB", flush=True)


def compute_dist_matrices_parallel(year_cutoffs):
    TEMP = TEMP_DIR
    to_compute = [y for y in year_cutoffs
                  if not os.path.exists(f'{TEMP}/dist_{y}.npy')]
    if not to_compute:
        print("  All distance matrices exist.", flush=True)
        return

    print(f"  Computing {len(to_compute)} distance matrices via BFS (parallel)...", flush=True)
    t0 = time.time()
    procs = []
    for y in to_compute:
        p = Process(target=_compute_dist_worker,
                    args=(f'{TEMP}/graph_{y}.npz', f'{TEMP}/dist_{y}.npy', y))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print(f"  All distance matrices done: {time.time()-t0:.1f}s", flush=True)


# ── 4. Compute pairs from FULL distance matrices ───────────────────────
def compute_all_pairs(dm_t1, dm_t2, rng, neg_ratio=5):
    print(f"    Computing dist_change for ALL eligible pairs...", flush=True)
    t0 = time.time()

    n = dm_t1.shape[0]
    rows_all, cols_all = np.triu_indices(n, k=1)
    t1_vals = dm_t1[rows_all, cols_all]
    eligible = (t1_vals >= 3) & (t1_vals < 9999)
    rows_e = rows_all[eligible]
    cols_e = cols_all[eligible]
    del rows_all, cols_all, t1_vals

    t1_e = dm_t1[rows_e, cols_e].astype(np.float32)
    t2_e = dm_t2[rows_e, cols_e].astype(np.float32)
    dist_change = t1_e - t2_e
    del t1_e, t2_e

    n_eligible = len(dist_change)
    print(f"    eligible pairs: {n_eligible:,}", flush=True)
    print(f"    dist_change: min={dist_change.min():.0f}, max={dist_change.max():.0f}, "
          f"mean={dist_change.mean():.3f}", flush=True)

    conv_mask = dist_change > 0
    n_conv = int(conv_mask.sum())
    print(f"    convergent (dist_change > 0): {n_conv:,} ({100*n_conv/n_eligible:.2f}%)", flush=True)

    nonconv_mask = dist_change == 0
    n_nonconv_target = min(int(nonconv_mask.sum()), n_conv * neg_ratio)
    nonconv_indices = np.where(nonconv_mask)[0]
    rng.shuffle(nonconv_indices)
    nonconv_indices = nonconv_indices[:n_nonconv_target]

    conv_indices = np.where(conv_mask)[0]
    keep_indices = np.concatenate([conv_indices, nonconv_indices])
    rng.shuffle(keep_indices)

    pairs = np.column_stack([rows_e[keep_indices], cols_e[keep_indices]])
    labels = dist_change[keep_indices]
    del rows_e, cols_e, dist_change, conv_mask, nonconv_mask

    dt = time.time() - t0
    print(f"    final dataset: {len(pairs):,} pairs ({n_conv:,} conv + {n_nonconv_target:,} non-conv)", flush=True)
    print(f"    label mean={labels.mean():.3f}, computed in {dt:.0f}s", flush=True)
    return pairs, labels


# ── 5. Save pairs/labels as chunk files ─────────────────────────────────
def save_chunks(pairs, labels, chunk_dir, prefix):
    """Save pairs and labels as chunk files (no subgraph precomputation)."""
    os.makedirs(chunk_dir, exist_ok=True)
    n = len(pairs)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    for ci in range(n_chunks):
        s = ci * CHUNK_SIZE
        e = min(s + CHUNK_SIZE, n)
        np.savez_compressed(f'{chunk_dir}/{prefix}_chunk_{ci:04d}.npz',
                            pairs=pairs[s:e], labels=labels[s:e].astype(np.float32))
    print(f"    Saved {n_chunks} chunks ({n:,} pairs) to {chunk_dir}", flush=True)
    return n_chunks


# ── 6. Fold processing ─────────────────────────────────────────────────
def process_fold(fold, train_t1, train_t2, val_t1, val_t2):
    print(f"\n{'='*50}", flush=True)
    print(f"  Fold {fold}: train {train_t1}→{train_t2}, val {val_t1}→{val_t2}", flush=True)
    print(f"{'='*50}", flush=True)

    fold_dir = f'{TEMP_DIR}/fold{fold}_chunks'
    if os.path.exists(f'{fold_dir}/manifest.json'):
        print(f"  Fold {fold} chunks exist, skipping.", flush=True)
        return

    rng = np.random.RandomState(42 + fold)

    # Stage 1: train pairs
    print(f"  [fold{fold}] Loading dist_{train_t1} + dist_{train_t2}...", flush=True)
    dm_train_t1 = np.load(f'{TEMP_DIR}/dist_{train_t1}.npy').astype(np.int32)
    dm_train_t2 = np.load(f'{TEMP_DIR}/dist_{train_t2}.npy').astype(np.int32)
    print(f"  [fold{fold}] Computing train pairs...", flush=True)
    train_pairs, train_labels = compute_all_pairs(dm_train_t1, dm_train_t2, rng)
    del dm_train_t1, dm_train_t2

    print(f"  [fold{fold}] Saving train chunks...", flush=True)
    n_train_chunks = save_chunks(train_pairs, train_labels, fold_dir, 'train')
    del train_pairs, train_labels

    # Stage 2: val pairs
    print(f"  [fold{fold}] Loading dist_{val_t1} + dist_{val_t2}...", flush=True)
    dm_val_t1 = np.load(f'{TEMP_DIR}/dist_{val_t1}.npy').astype(np.int32)
    dm_val_t2 = np.load(f'{TEMP_DIR}/dist_{val_t2}.npy').astype(np.int32)
    print(f"  [fold{fold}] Computing val pairs...", flush=True)
    val_pairs, val_labels = compute_all_pairs(dm_val_t1, dm_val_t2, rng)
    del dm_val_t1, dm_val_t2

    print(f"  [fold{fold}] Saving val chunks...", flush=True)
    n_val_chunks = save_chunks(val_pairs, val_labels, fold_dir, 'val')
    del val_pairs, val_labels

    manifest = {
        'fold': fold,
        'window_train': f'{train_t1}_{train_t2}',
        'window_val': f'{val_t1}_{val_t2}',
        'n_train_chunks': n_train_chunks,
        'n_val_chunks': n_val_chunks,
    }
    with open(f'{fold_dir}/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  [fold{fold}] Done. Manifest saved.", flush=True)


# ── Main ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    t0 = time.time()

    print("Loading concept IDs...", flush=True)
    with open(CONCEPT_IDS_FILE) as f:
        concept_ids = json.load(f)
    concept_to_idx = {cid: i for i, cid in enumerate(concept_ids)}
    print(f"  {len(concept_ids)} concepts", flush=True)

    print("Loading papers...", flush=True)
    papers = load_papers()
    print(f"  {len(papers)} papers", flush=True)

    print("\n=== Step 1: Building temporal graphs ===", flush=True)
    build_all_graphs(papers, YEAR_CUTOFFS, concept_to_idx)
    del papers

    print("\n=== Step 2: Distance matrices (BFS, parallel) ===", flush=True)
    compute_dist_matrices_parallel(YEAR_CUTOFFS)

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
