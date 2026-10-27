"""
Build temporal LBD data: graphs, distances, pairs, precomputed subgraphs.
Run on server CPU (no GPU needed). Checkpoint-safe — re-run skips completed steps.

Pipeline:
  1. Build temporal concept co-occurrence graphs (year cutoffs 2012/2017/2021/2024)
  2. Compute all-pairs shortest path distance matrices via parallel Dijkstra
  3. Sample positive/negative pairs for rolling window training
  4. Precompute subgraphs (adj+distance features, no SciBERT) for fast training

Parallelism:
  - Distance matrices: 4 graphs computed simultaneously (4 cores, ~37GB RAM)
  - Subgraph precomputation: 16 workers via multiprocessing Pool
  - Fold processing: 2 folds run as parallel subprocesses

Output structure:
  {DATA_DIR}/temporal/
    graph_2012.npz, graph_2017.npz, graph_2021.npz, graph_2024.npz
    dist_2012.npy, dist_2017.npy, dist_2021.npy, dist_2024.npy
    fold0_subgraphs.npz  (train: 2012→2017, val: 2017→2021)
    fold1_subgraphs.npz  (train: 2017→2021, val: 2021→2024)
"""
import os, time, json, sys
from itertools import combinations
from multiprocessing import Process, Pool
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
PAPERS_GEOL = '/root/autodl-tmp/openalex_data/geology_papers.jsonl'
PAPERS_CLIM = '/root/autodl-tmp/openalex_data/climate_papers.jsonl'
CONCEPT_IDS_FILE = f'{DATA_DIR}/concept_ids.json'

TEMP_DIR = f'{DATA_DIR}/temporal'
N_WORKERS = 20          # subgraph precomputation workers
N_DIST_WORKERS = 4     # Dijkstra parallel processes

YEAR_CUTOFFS = [2012, 2017, 2021, 2024]
MIN_SCORE = 0.3
MAX_LEVEL = 3

# Pair sampling (ranking: continuous distance change)
N_TRAIN_SAMPLES = 100000
N_VAL_SAMPLES = 10000

# Subgraph precomputation (must match train script)
MAX_NODES = 32
EMBED_DIM = 68   # 2*MAX_NODES + 4
N_HARD_NEG = 19
N_ANCHORS = 5000
G = 1 + N_HARD_NEG  # 20


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


# ── 2. Build temporal graph (optimized: single pass sorted by year) ────
def build_all_graphs(papers, year_cutoffs, concept_to_idx):
    """Build all temporal graphs in a single pass over papers.

    Sort papers by year, then for each year cutoff, accumulate edges from
    papers with year <= cutoff. Each graph is a superset of the previous.
    This is O(E * K) where K=4 cutoffs, vs O(E * K) for separate passes
    but avoids re-reading papers K times.
    """
    n = len(concept_to_idx)
    # Sort cutoffs ascending
    sorted_cutoffs = sorted(year_cutoffs)

    # Check which graphs need building
    needed = [y for y in sorted_cutoffs
              if not os.path.exists(f'{TEMP_DIR}/graph_{y}.npz')]
    if not needed:
        print("  All graphs exist.", flush=True)
        return

    # Sort papers by year for incremental edge accumulation
    papers_sorted = sorted(papers, key=lambda p: p['year'])

    # Current edges set (incrementally grows as year cutoff increases)
    edges = set()
    paper_idx = 0  # pointer into sorted papers

    for cutoff in needed:
        # Add papers with year <= cutoff that weren't added yet
        while paper_idx < len(papers_sorted) and papers_sorted[paper_idx]['year'] <= cutoff:
            p = papers_sorted[paper_idx]
            valid = []
            for cid, score, level in p['concepts']:
                if level <= MAX_LEVEL and score >= MIN_SCORE and cid in concept_to_idx:
                    valid.append(concept_to_idx[cid])
            for a, b in combinations(valid, 2):
                edges.add((a, b) if a < b else (b, a))
            paper_idx += 1

        # Build sparse matrix
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
    """Compute all-pairs distance matrices for multiple graphs in parallel.
    Each shortest_path(method='D') uses 1 CPU core + ~9.2GB RAM.
    4 graphs in parallel: 4 cores, ~37GB RAM, ~12 min total.
    """
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


# ── 4. Sample pairs (ranking: continuous distance change) ─────────────
def sample_pairs(dm_t1, dm_t2, n_samples, rng):
    """Sample pairs with continuous distance change as label.
    Eligible: T1 distance >= 3 (cross-domain), upper triangle (i < j).
    Label: distance_change = dist_T1 - dist_T2 (higher = more convergence).

    Uses rejection sampling: random (i,j) pairs, keep those meeting criteria.
    Avoids np.argwhere on full 34K×34K matrix (~1.16B entries).
    """
    n_nodes = dm_t1.shape[0]
    pairs_list = []
    labels_list = []
    batch = n_samples * 4  # generous over-sample, most pairs are eligible
    attempts = 0
    max_attempts = 50

    while len(pairs_list) < n_samples and attempts < max_attempts:
        rows = rng.randint(0, n_nodes, size=batch)
        cols = rng.randint(0, n_nodes, size=batch)
        # upper triangle only (avoid duplicates)
        mask_upper = rows < cols
        rows, cols = rows[mask_upper], cols[mask_upper]

        t1_vals = dm_t1[rows, cols]
        eligible = (t1_vals >= 3) & (t1_vals < 9999)
        rows, cols = rows[eligible], cols[eligible]

        if len(rows) > 0:
            t1_e = dm_t1[rows, cols].astype(np.float32)
            t2_e = dm_t2[rows, cols].astype(np.float32)
            dc = t1_e - t2_e
            pairs_list.append(np.column_stack([rows, cols]))
            labels_list.append(dc)

        attempts += 1

    if not pairs_list:
        print(f"    WARNING: no eligible pairs found after {max_attempts} attempts")
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float32)

    all_pairs = np.concatenate(pairs_list)
    all_labels = np.concatenate(labels_list)

    # Shuffle and take n_samples
    perm = rng.permutation(len(all_pairs))
    n = min(n_samples, len(all_pairs))
    pairs = all_pairs[perm[:n]]
    labels = all_labels[perm[:n]]

    print(f"    eligible found: {len(all_pairs):,} (after {attempts} batches)")
    print(f"    dist_change: min={labels.min():.0f}, max={labels.max():.0f}, mean={labels.mean():.2f}")
    print(f"    sampled: {n} pairs, label mean={labels.mean():.2f}")
    return pairs, labels


# ── 5. Precompute subgraphs (PARALLEL) ─────────────────────────────────
def _build_subgraph(u, v, indptr, indices, du_row, dv_row, out_feat, out_nids):
    """Build subgraph for pair (u, v): 1-hop neighbors, adj+distance features."""
    MN = MAX_NODES; DB = 10.0

    sel = [u, v]
    sel_set = {u, v}

    for start, end in [(indptr[u], indptr[u + 1]), (indptr[v], indptr[v + 1])]:
        for j in range(start, end):
            if len(sel) >= MN:
                break
            nb = int(indices[j])
            if nb not in sel_set:
                sel.append(nb)
                sel_set.add(nb)

    ns = len(sel)

    # Features: one-hot + existence + distance to u + distance to v
    out_feat[:MN, :] = 0
    out_nids[:MN] = 0
    for i in range(ns):
        out_feat[i, i] = 1.0
        out_feat[i, 2 * MN] = 1.0
        out_feat[i, 2 * MN + 1] = min(float(du_row[sel[i]]), DB) / DB
        out_feat[i, 2 * MN + 2] = min(float(dv_row[sel[i]]), DB) / DB

    # Adjacency: for each selected node, check which neighbors are also selected
    for i in range(ns):
        a = sel[i]
        for j in range(indptr[a], indptr[a + 1]):
            b = int(indices[j])
            if b in sel_set:
                for k in range(ns):
                    if sel[k] == b:
                        out_feat[i, MN + k] = 1.0
                        break

    for i in range(min(ns, MN)):
        out_nids[i] = sel[i]


def _worker_build_batch(args):
    """Worker function for parallel subgraph precomputation.

    Each worker processes a batch of anchors, returning arrays that will be
    assembled by the main process.
    """
    anchor_batch, pos_pairs, neg_pairs, dm_t1, adj_csr, batch_start_idx, rng_seed = args

    rng = np.random.RandomState(rng_seed)
    indptr = adj_csr.indptr.astype(np.int64)
    indices = adj_csr.indices.astype(np.int64)

    # Build neg candidates index: anchor -> list of negative candidates
    neg_by_anchor = {}
    for a, b in neg_pairs:
        neg_by_anchor.setdefault(int(a), []).append(int(b))

    n_batch = len(anchor_batch)
    out_inp = np.zeros((n_batch * G, MAX_NODES, EMBED_DIM), dtype=np.float16)
    out_nids = np.zeros((n_batch * G, MAX_NODES), dtype=np.int64)
    out_labels = np.zeros(n_batch * G, dtype=np.float32)
    out_anchors = np.zeros(n_batch * G, dtype=np.int64)

    tmp_f = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
    tmp_n = np.zeros(MAX_NODES, dtype=np.int64)

    for local_idx in range(n_batch):
        u = int(anchor_batch[local_idx, 0])
        v = int(anchor_batch[local_idx, 1])
        du_row = dm_t1[u]
        dv_row = dm_t1[v]

        # Positive subgraph
        _build_subgraph(u, v, indptr, indices, du_row, dv_row, tmp_f, tmp_n)
        global_idx = batch_start_idx + local_idx
        out_inp[local_idx * G] = tmp_f.astype(np.float16)
        out_nids[local_idx * G] = tmp_n
        out_labels[local_idx * G] = 1.0
        out_anchors[local_idx * G] = u

        # Negative subgraphs
        neg_cands = neg_by_anchor.get(u, [])
        if len(neg_cands) < N_HARD_NEG:
            extra = [int(b) for _, b in neg_pairs[rng.choice(len(neg_pairs),
                     size=N_HARD_NEG - len(neg_cands), replace=False)]]
            neg_cands = neg_cands + extra

        chosen = rng.choice(len(neg_cands), size=N_HARD_NEG, replace=False)
        for ni in range(N_HARD_NEG):
            neg_v = int(neg_cands[chosen[ni]])
            _build_subgraph(u, neg_v, indptr, indices, du_row, dm_t1[neg_v], tmp_f, tmp_n)
            out_inp[local_idx * G + 1 + ni] = tmp_f.astype(np.float16)
            out_nids[local_idx * G + 1 + ni] = tmp_n
            out_anchors[local_idx * G + 1 + ni] = u

    return out_inp, out_nids, out_labels, out_anchors


def precompute_subgraphs_parallel(pairs, labels, dm_t1, dm_t2, adj_t1, fold_name):
    """Precompute subgraphs using multiprocessing Pool.

    Distributes anchors across N_WORKERS workers. Each worker builds its
    batch of subgraphs independently and returns arrays to be assembled.
    """
    n = len(pairs)
    pos_pairs = pairs[labels == 1]
    neg_pairs = pairs[labels == 0]

    n_anchors = min(N_ANCHORS, len(pos_pairs))
    anchor_idx = np.random.RandomState(42).choice(len(pos_pairs), size=n_anchors, replace=False)
    anchors = pos_pairs[anchor_idx]

    print(f"    {n_anchors} anchors, {N_WORKERS} workers, G={G}...", flush=True)

    # Split anchors into batches for workers
    batch_size = max(1, (n_anchors + N_WORKERS - 1) // N_WORKERS)
    batches = []
    for i in range(0, n_anchors, batch_size):
        end = min(i + batch_size, n_anchors)
        batch_anchors = anchors[i:end]
        batches.append((
            batch_anchors,
            pos_pairs,      # shared (read-only)
            neg_pairs,      # shared (read-only)
            dm_t1,          # shared (read-only), COW on Linux fork
            adj_t1,         # shared (read-only), COW on Linux fork
            i,              # batch_start_idx
            42 + i,         # rng_seed (different per batch)
        ))

    t0 = time.time()

    with Pool(N_WORKERS) as pool:
        results = pool.map(_worker_build_batch, batches)

    # Assemble results into final arrays
    total = n_anchors * G
    out_inp = np.zeros((total, MAX_NODES, EMBED_DIM), dtype=np.float16)
    out_nids = np.zeros((total, MAX_NODES), dtype=np.int64)
    out_labels = np.zeros(total, dtype=np.float32)
    out_anchors = np.zeros(total, dtype=np.int64)

    offset = 0
    for out_i, out_n, out_l, out_a in results:
        n_batch = len(out_l)
        out_inp[offset:offset + n_batch] = out_i
        out_nids[offset:offset + n_batch] = out_n
        out_labels[offset:offset + n_batch] = out_l
        out_anchors[offset:offset + n_batch] = out_a
        offset += n_batch

    dt = time.time() - t0
    print(f"    {total} subgraphs built in {dt:.1f}s ({total/dt:.0f} samples/s)", flush=True)

    return out_inp, out_nids, out_labels, out_anchors


# ── 6. Single fold processing (run as subprocess) ──────────────────────
def process_fold(fold, train_t1, train_t2, val_t1, val_t2):
    """Process a single fold: sample pairs, precompute subgraphs, save.
    Designed to run as a subprocess so both folds can run in parallel.
    """
    print(f"\n{'='*50}", flush=True)
    print(f"  Fold {fold}: train {train_t1}→{train_t2}, val {val_t1}→{val_t2}", flush=True)
    print(f"{'='*50}", flush=True)

    rng = np.random.RandomState(42 + fold)

    # Load graphs (keep in memory for subgraph building)
    graph_t1 = sp.load_npz(f'{TEMP_DIR}/graph_{train_t1}.npz').tocsr()
    graph_v1 = sp.load_npz(f'{TEMP_DIR}/graph_{val_t1}.npz').tocsr()

    # Stage 1: train pairs (2 distance matrices, ~4.6GB each)
    print(f"  [fold{fold}] Loading dist_{train_t1} + dist_{train_t2}...", flush=True)
    dm_train_t1 = np.load(f'{TEMP_DIR}/dist_{train_t1}.npy').astype(np.int32)
    dm_train_t2 = np.load(f'{TEMP_DIR}/dist_{train_t2}.npy').astype(np.int32)

    print(f"  [fold{fold}] Sampling train pairs...", flush=True)
    train_pairs, train_labels = sample_pairs(
        dm_train_t1, dm_train_t2, N_TRAIN_SAMPLES, rng)

    print(f"  [fold{fold}] Precomputing train subgraphs (parallel)...", flush=True)
    t_inp, t_nids, t_lab, t_anc = precompute_subgraphs_parallel(
        train_pairs, train_labels, dm_train_t1, dm_train_t2,
        graph_t1, f'fold{fold}_train')
    del dm_train_t1, dm_train_t2  # free ~9.2GB

    # Stage 2: val pairs
    print(f"  [fold{fold}] Loading dist_{val_t1} + dist_{val_t2}...", flush=True)
    dm_val_t1 = np.load(f'{TEMP_DIR}/dist_{val_t1}.npy').astype(np.int32)
    dm_val_t2 = np.load(f'{TEMP_DIR}/dist_{val_t2}.npy').astype(np.int32)

    print(f"  [fold{fold}] Sampling val pairs...", flush=True)
    val_pairs, val_labels = sample_pairs(
        dm_val_t1, dm_val_t2, N_VAL_SAMPLES, rng)

    print(f"  [fold{fold}] Precomputing val subgraphs (parallel)...", flush=True)
    v_inp, v_nids, v_lab, v_anc = precompute_subgraphs_parallel(
        val_pairs, val_labels, dm_val_t1, dm_val_t2,
        graph_v1, f'fold{fold}_val')
    del dm_val_t1, dm_val_t2, graph_t1, graph_v1

    out_path = f'{TEMP_DIR}/fold{fold}_subgraphs.npz'
    np.savez_compressed(out_path,
        train_inp=t_inp, train_nids=t_nids, train_labels=t_lab, train_anchors=t_anc,
        val_inp=v_inp, val_nids=v_nids, val_labels=v_lab, val_anchors=v_anc,
        train_pairs=train_pairs, val_pairs=val_pairs,
        train_sample_labels=train_labels, val_sample_labels=val_labels,
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

    # Step 1: Build temporal graphs (single pass, sorted by year)
    print("\n=== Step 1: Building temporal graphs ===", flush=True)
    build_all_graphs(papers, YEAR_CUTOFFS, concept_to_idx)
    del papers  # free memory

    # Step 2: Compute BFS distances: all 4 graphs in parallel
    print("\n=== Computing distance matrices (parallel) ===", flush=True)
    compute_dist_matrices_parallel(YEAR_CUTOFFS)

    # Step 3+4: Process folds IN PARALLEL
    folds = [
        (0, 2012, 2017, 2017, 2021),
        (1, 2017, 2021, 2021, 2024),
    ]

    # Check which folds need processing
    folds_needed = []
    for fold, train_t1, train_t2, val_t1, val_t2 in folds:
        out_path = f'{TEMP_DIR}/fold{fold}_subgraphs.npz'
        if os.path.exists(out_path):
            print(f"  fold{fold}_subgraphs: exists", flush=True)
        else:
            folds_needed.append((fold, train_t1, train_t2, val_t1, val_t2))

    if not folds_needed:
        print("  All folds done.", flush=True)
    elif len(folds_needed) == 1:
        # Single fold — run directly
        fold, tt1, tt2, vt1, vt2 = folds_needed[0]
        print(f"\n=== Processing fold {fold} ===", flush=True)
        process_fold(fold, tt1, tt2, vt1, vt2)
    else:
        # Both folds needed — run in parallel
        # Peak memory: 4 dist matrices (18.4GB) + 2 graphs (80MB) + subgraph workers (~64MB)
        # Total: ~18.5GB. Server has 754GB, no problem.
        print(f"\n=== Processing {len(folds_needed)} folds in parallel ===", flush=True)
        print(f"  Peak memory estimate: ~20GB (4 dist matrices + workers)", flush=True)

        procs = []
        for fold, tt1, tt2, vt1, vt2 in folds_needed:
            p = Process(target=process_fold, args=(fold, tt1, tt2, vt1, vt2))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

    print(f"\nTotal: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
