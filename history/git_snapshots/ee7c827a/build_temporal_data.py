"""
Build temporal LBD data: graphs, distances, pairs, precomputed subgraphs.
Run on server CPU (no GPU needed). Checkpoint-safe — re-run skips completed steps.

Pipeline:
  1. Build temporal concept co-occurrence graphs (year cutoffs 2012/2017/2021/2024)
  2. Compute all-pairs shortest path distance matrices via parallel Dijkstra
  3. Compute distance_change for ALL eligible pairs, keep convergent + sample non-convergent
  4. Precompute subgraphs with vectorized numpy (no multiprocessing, fast batch processing)

Output structure:
  {DATA_DIR}/temporal/
    graph_2012.npz, graph_2017.npz, graph_2021.npz, graph_2024.npz
    dist_2012.npy, dist_2017.npy, dist_2021.npy, dist_2024.npy
    fold0_subgraphs.npz  (train: 2012→2017, val: 2017→2021)
    fold1_subgraphs.npz  (train: 2017→2021, val: 2021→2024)
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

TEMP_DIR = f'{DATA_DIR}/temporal'
N_DIST_WORKERS = 4     # Dijkstra parallel processes
BUILD_BATCH = 20000     # pairs per batch for vectorized subgraph building

YEAR_CUTOFFS = [2012, 2017, 2021, 2024]
MIN_SCORE = 0.3
MAX_LEVEL = 3

# Subgraph config (must match train script)
MAX_NODES = 32
EMBED_DIM = 68   # 2*MAX_NODES + 4


# ── 1. Load papers ──────────────────────────────────────────────────────
def load_papers():
    """Load papers from geology + climate JSONL files."""
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
    """Build all temporal graphs in a single pass over papers."""
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

        out_path = f'{TEMP_DIR}/graph_{cutoff}.npz'
        sp.save_npz(out_path, adj)
        print(f"  graph_{cutoff}: {adj.nnz:,} edges, saved", flush=True)


# ── 3. Compute all-pairs shortest path distances ───────────────────────
def _compute_dist_worker(adj_path, out_path, year_cutoff):
    """Worker: compute all-pairs shortest paths for one graph."""
    adj = sp.load_npz(adj_path).tocsr()
    n = adj.shape[0]
    print(f"  [{year_cutoff}] Dijkstra start ({n} nodes, {adj.nnz:,} edges)...", flush=True)
    t0 = time.time()
    dist = shortest_path(adj, directed=False, method='D')
    dt = time.time() - t0
    print(f"  [{year_cutoff}] Dijkstra done: {dt:.1f}s", flush=True)
    dist[np.isinf(dist)] = 9999
    np.save(out_path, dist.astype(np.int16))
    print(f"  [{year_cutoff}] saved {os.path.getsize(out_path)/1e9:.2f}GB", flush=True)


def compute_dist_matrices_parallel(year_cutoffs):
    """Compute all-pairs distance matrices for multiple graphs in parallel."""
    TEMP = TEMP_DIR
    to_compute = []
    for y in year_cutoffs:
        out_path = f'{TEMP}/dist_{y}.npy'
        if os.path.exists(out_path):
            print(f"  dist_{y}: exists", flush=True)
        else:
            to_compute.append(y)

    if not to_compute:
        print("  All distance matrices exist.", flush=True)
        return

    print(f"  Computing {len(to_compute)} distance matrices in parallel...", flush=True)
    t0 = time.time()

    procs = []
    for y in to_compute:
        p = Process(target=_compute_dist_worker,
                    args=(f'{TEMP}/graph_{y}.npz', f'{TEMP}/dist_{y}.npy', y))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    dt = time.time() - t0
    print(f"  All distance matrices done: {dt:.1f}s", flush=True)


# ── 4. Compute pairs from FULL distance matrices ───────────────────────
def compute_all_pairs(dm_t1, dm_t2, rng, neg_ratio=5):
    """Compute distance change for ALL eligible pairs, keep convergent + sample non-convergent.

    Eligible: T1 distance >= 3 and < 9999, upper triangle (i < j).
    Convergent: dist_change > 0 (distance decreased).
    Non-convergent: dist_change == 0, sampled at neg_ratio x convergent count.
    """
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


# ── 5. Vectorized subgraph precomputation ──────────────────────────────
def precompute_subgraphs_vectorized(pairs, labels, dm_t1, adj_csr, fold_name):
    """Build subgraphs using vectorized numpy operations. No multiprocessing.

    For each pair (u, v), extracts 1-hop ego graph, builds features:
      [one_hot(MN) | adjacency(MN) | existence(1) | dist_u(1) | dist_v(1) | placeholder(1)]

    Processes in batches of BUILD_BATCH to manage memory.
    """
    MN = MAX_NODES; DB = 10.0
    n_pairs = len(pairs)
    indptr = adj_csr.indptr
    indices = adj_csr.indices

    # Output arrays (pre-allocated on disk via incremental save)
    all_inp = np.zeros((n_pairs, MN, EMBED_DIM), dtype=np.float16)
    all_nids = np.zeros((n_pairs, MN), dtype=np.int64)

    n_batches = (n_pairs + BUILD_BATCH - 1) // BUILD_BATCH
    t0 = time.time()

    for bi in range(n_batches):
        s = bi * BUILD_BATCH
        e = min(s + BUILD_BATCH, n_pairs)
        bp = pairs[s:e]
        nb = e - s

        # Extract 1-hop neighbors for all u and v in batch
        nu = indptr[bp[:, 0] + 1] - indptr[bp[:, 0]]
        nv = indptr[bp[:, 1] + 1] - indptr[bp[:, 1]]
        u_nbs = np.full((nb, MN - 2), -1, dtype=np.int64)
        v_nbs = np.full((nb, MN - 2), -1, dtype=np.int64)
        u_cnt = np.minimum(nu, MN - 2)
        v_cnt = np.minimum(nv, MN - 2)

        for i in range(nb):
            u_nbs[i, :u_cnt[i]] = indices[indptr[bp[i, 0]]:indptr[bp[i, 0]] + u_cnt[i]]
            v_nbs[i, :v_cnt[i]] = indices[indptr[bp[i, 1]]:indptr[bp[i, 1]] + v_cnt[i]]

        # Build selected node IDs: [u, v, u_neighbors..., v_neighbors...]
        sel = np.zeros((nb, MN), dtype=np.int64)
        sel[:, 0] = bp[:, 0]
        sel[:, 1] = bp[:, 1]
        sel[:, 2:MN - 2 + 2] = np.where(u_nbs >= 0, u_nbs, bp[:, 0:1])

        # Distance lookups (vectorized)
        du = dm_t1[bp[:, 0]]  # [nb, N] — distance from u to all nodes
        dv = dm_t1[bp[:, 1]]  # [nb, N]

        # Build features
        feat = np.zeros((nb, MN, EMBED_DIM), dtype=np.float32)

        # One-hot + existence + dist_u + dist_v for each selected node
        for j in range(MN):
            nodes_j = sel[:, j]
            feat[:, j, j] = 1.0                           # one-hot
            feat[:, j, 2 * MN] = 1.0                      # existence
            feat[:, j, 2 * MN + 1] = np.minimum(du[np.arange(nb), nodes_j], DB) / DB
            feat[:, j, 2 * MN + 2] = np.minimum(dv[np.arange(nb), nodes_j], DB) / DB

        # Adjacency features (set lookup)
        for i in range(nb):
            sel_set = set(sel[i].tolist())
            for j in range(MN):
                a = sel[i, j]
                for k2 in range(indptr[a], indptr[a + 1]):
                    b = int(indices[k2])
                    if b in sel_set:
                        for k3 in range(MN):
                            if sel[i, k3] == b:
                                feat[i, j, MN + k3] = 1.0
                                break

        all_inp[s:e] = feat.astype(np.float16)
        all_nids[s:e] = sel

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            done = e
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_pairs - done) / rate if rate > 0 else 0
            print(f"    [{fold_name}] {done:,}/{n_pairs:,} subgraphs "
                  f"({rate:.0f}/s, ETA {eta:.0f}s)", flush=True)

    dt = time.time() - t0
    print(f"    [{fold_name}] {n_pairs:,} subgraphs built in {dt:.1f}s "
          f"({n_pairs/dt:.0f} samples/s)", flush=True)
    return all_inp, all_nids


# ── 6. Single fold processing ──────────────────────────────────────────
def process_fold(fold, train_t1, train_t2, val_t1, val_t2):
    """Process a single fold: compute pairs, build subgraphs, save."""
    print(f"\n{'='*50}", flush=True)
    print(f"  Fold {fold}: train {train_t1}→{train_t2}, val {val_t1}→{val_t2}", flush=True)
    print(f"{'='*50}", flush=True)

    rng = np.random.RandomState(42 + fold)

    # Load graphs
    graph_t1 = sp.load_npz(f'{TEMP_DIR}/graph_{train_t1}.npz').tocsr()
    graph_v1 = sp.load_npz(f'{TEMP_DIR}/graph_{val_t1}.npz').tocsr()

    # Stage 1: train pairs — full distance matrix
    print(f"  [fold{fold}] Loading dist_{train_t1} + dist_{train_t2}...", flush=True)
    dm_train_t1 = np.load(f'{TEMP_DIR}/dist_{train_t1}.npy').astype(np.int32)
    dm_train_t2 = np.load(f'{TEMP_DIR}/dist_{train_t2}.npy').astype(np.int32)

    print(f"  [fold{fold}] Computing train pairs (full matrix)...", flush=True)
    train_pairs, train_labels = compute_all_pairs(dm_train_t1, dm_train_t2, rng)

    print(f"  [fold{fold}] Precomputing train subgraphs (vectorized)...", flush=True)
    t_inp, t_nids = precompute_subgraphs_vectorized(
        train_pairs, train_labels, dm_train_t1, graph_t1, f'fold{fold}_train')
    del dm_train_t1, dm_train_t2

    # Stage 2: val pairs — full distance matrix
    print(f"  [fold{fold}] Loading dist_{val_t1} + dist_{val_t2}...", flush=True)
    dm_val_t1 = np.load(f'{TEMP_DIR}/dist_{val_t1}.npy').astype(np.int32)
    dm_val_t2 = np.load(f'{TEMP_DIR}/dist_{val_t2}.npy').astype(np.int32)

    print(f"  [fold{fold}] Computing val pairs (full matrix)...", flush=True)
    val_pairs, val_labels = compute_all_pairs(dm_val_t1, dm_val_t2, rng)

    print(f"  [fold{fold}] Precomputing val subgraphs (vectorized)...", flush=True)
    v_inp, v_nids = precompute_subgraphs_vectorized(
        val_pairs, val_labels, dm_val_t1, graph_v1, f'fold{fold}_val')
    del dm_val_t1, dm_val_t2, graph_t1, graph_v1

    out_path = f'{TEMP_DIR}/fold{fold}_subgraphs.npz'
    np.savez_compressed(out_path,
        train_inp=t_inp, train_nids=t_nids,
        train_labels=train_labels.astype(np.float32),
        val_inp=v_inp, val_nids=v_nids,
        val_labels=val_labels.astype(np.float32),
        train_pairs=train_pairs, val_pairs=val_pairs,
        window_train=f'{train_t1}_{train_t2}', window_val=f'{val_t1}_{val_t2}')
    del t_inp, t_nids, v_inp, v_nids
    print(f"  [fold{fold}] Saved {out_path} ({os.path.getsize(out_path)/1e6:.0f}MB)", flush=True)


# ── Main ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    t0 = time.time()

    # Load concept IDs
    print("Loading concept IDs...", flush=True)
    with open(CONCEPT_IDS_FILE) as f:
        concept_ids = json.load(f)
    concept_to_idx = {cid: i for i, cid in enumerate(concept_ids)}
    n_concepts = len(concept_ids)
    print(f"  {n_concepts} concepts", flush=True)

    # Load papers
    print("Loading papers...", flush=True)
    papers = load_papers()
    print(f"  {len(papers)} papers", flush=True)

    # Step 1: Build temporal graphs
    print("\n=== Step 1: Building temporal graphs ===", flush=True)
    build_all_graphs(papers, YEAR_CUTOFFS, concept_to_idx)
    del papers

    # Step 2: Compute BFS distances
    print("\n=== Computing distance matrices (parallel) ===", flush=True)
    compute_dist_matrices_parallel(YEAR_CUTOFFS)

    # Step 3: Process folds (serial — each fold uses ~18GB for distance matrices)
    folds = [
        (0, 2012, 2017, 2017, 2021),
        (1, 2017, 2021, 2021, 2024),
    ]

    folds_needed = []
    for fold, train_t1, train_t2, val_t1, val_t2 in folds:
        out_path = f'{TEMP_DIR}/fold{fold}_subgraphs.npz'
        if os.path.exists(out_path):
            print(f"  fold{fold}_subgraphs: exists", flush=True)
        else:
            folds_needed.append((fold, train_t1, train_t2, val_t1, val_t2))

    if not folds_needed:
        print("  All folds done.", flush=True)
    else:
        # Run folds serially (each loads 4 distance matrices = ~18.4GB)
        for fold, tt1, tt2, vt1, vt2 in folds_needed:
            print(f"\n=== Processing fold {fold} ===", flush=True)
            process_fold(fold, tt1, tt2, vt1, vt2)

    print(f"\nTotal: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
