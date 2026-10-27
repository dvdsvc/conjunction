"""
Precompute subgraph encodings with DRNL + matched hard negatives.
Uses parallel scipy shortest_path for full distance matrix.
"""
import json, os, time, random
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from multiprocessing import Pool

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32
N_HARD_NEG = 5
N_WORKERS = 80
DIST_BINS = 10

_g_dist_matrix = None
_g_adj_rows = None
_g_adj_cols = None
_g_degree = None


def _init_worker(dm, adj_rows, adj_cols, degree):
    global _g_dist_matrix, _g_adj_rows, _g_adj_cols, _g_degree
    _g_dist_matrix = dm
    _g_adj_rows = adj_rows
    _g_adj_cols = adj_cols
    _g_degree = degree


def _compute_dist_chunk(args):
    """Placeholder - not used, kept for compatibility."""
    pass


def _build_subgraph(u, v):
    rows, cols, deg, dist_mat = _g_adj_rows, _g_adj_cols, _g_degree, _g_dist_matrix
    nu = cols[rows[u]:rows[u + 1]]
    nv = cols[rows[v]:rows[v + 1]]
    nu_set = set(nu.tolist())
    nv_set = set(nv.tolist())
    shared = sorted(nu_set & nv_set, key=lambda x: -deg[x])
    only_u = sorted(nu_set - nv_set - {v}, key=lambda x: -deg[x])
    only_v = sorted(nv_set - nu_set - {u}, key=lambda x: -deg[x])

    nodes = [u, v]
    rem = MAX_NODES - 2
    for lst in [shared, only_u, only_v]:
        for n in lst:
            if rem <= 0:
                break
            nodes.append(n)
            rem -= 1
        if rem <= 0:
            break

    n = len(nodes)
    nset = set(nodes)
    lmap = {c: i for i, c in enumerate(nodes)}

    FEATURE_DIM = 2 * MAX_NODES + 2 + 2  # 70
    x = np.zeros((MAX_NODES, FEATURE_DIM), dtype=np.float16)
    for i in range(n):
        if i < MAX_NODES:
            x[i, i] = 1.0
        x[i, 2 * MAX_NODES] = 1.0
        node_id = nodes[i]
        du = int(dist_mat[u, node_id])
        dv = int(dist_mat[v, node_id])
        x[i, 2 * MAX_NODES + 1] = min(du, DIST_BINS) / DIST_BINS
        x[i, 2 * MAX_NODES + 2] = min(dv, DIST_BINS) / DIST_BINS

    for a in nodes:
        for b in cols[rows[a]:rows[a + 1]]:
            if b in nset:
                ib = lmap[b]
                if ib < MAX_NODES:
                    x[lmap[a], MAX_NODES + ib] = 1.0

    task = np.zeros((2, FEATURE_DIM), dtype=np.float16)
    task[0] = x[0]
    task[0, 2 * MAX_NODES + 1] = 0.0
    task[0, 2 * MAX_NODES + 2] = 1.0
    task[1] = x[1]
    task[1, 2 * MAX_NODES + 1] = 1.0
    task[1, 2 * MAX_NODES + 2] = 0.0
    return x, task, nodes[:MAX_NODES]


def _sample_hard_neg(u, pos_v):
    """Hard negative: shares neighbors with u, not directly connected,
    matched shared-neighbor count to positive."""
    rows, cols = _g_adj_rows, _g_adj_cols
    u_neighbors = set(cols[rows[u]:rows[u + 1]].tolist())
    v_neighbors = set(cols[rows[pos_v]:rows[pos_v + 1]].tolist())
    n_shared_pos = len(u_neighbors & v_neighbors)

    candidates = set()
    nu = cols[rows[u]:rows[u + 1]]
    limit = min(len(nu), 20)
    for i in range(limit):
        n1 = int(nu[i])
        n2s = cols[rows[n1]:rows[n1 + 1]]
        for j in range(len(n2s)):
            n2 = int(n2s[j])
            if n2 != u and n2 != pos_v and n2 not in u_neighbors:
                candidates.add(n2)
            if len(candidates) >= 200:
                break
        if len(candidates) >= 200:
            break

    if not candidates:
        for _ in range(50):
            neg = np.random.randint(0, len(rows) - 1)
            if neg != u and neg != pos_v and neg not in u_neighbors:
                return neg
        return np.random.randint(0, len(rows) - 1)

    cl = list(candidates)
    np.random.shuffle(cl)
    best, best_diff = cl[0], 9999
    for neg in cl[:30]:
        neg_neighbors = set(cols[rows[neg]:rows[neg + 1]].tolist())
        n_shared_neg = len(u_neighbors & neg_neighbors)
        diff = abs(n_shared_neg - n_shared_pos)
        if diff < best_diff:
            best, best_diff = neg, diff
            if diff == 0:
                break
    return best


def _process_positive(args):
    u, v = args
    inp_list, task_list, node_ids_list = [], [], []
    x, t, nids = _build_subgraph(u, v)
    inp_list.append(x); task_list.append(t); node_ids_list.append(nids)
    for _ in range(N_HARD_NEG):
        neg_v = _sample_hard_neg(u, v)
        x, t, nids = _build_subgraph(u, neg_v)
        inp_list.append(x); task_list.append(t); node_ids_list.append(nids)
    padded = np.zeros((len(node_ids_list), MAX_NODES), dtype=np.int64)
    for i, nids in enumerate(node_ids_list):
        padded[i, :len(nids)] = nids
    return np.stack(inp_list), np.stack(task_list), padded


if __name__ == '__main__':
    t0 = time.time()
    np.random.seed(42)
    random.seed(42)

    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    adj_rows = adj.indptr.astype(np.int64)
    adj_cols = adj.indices.astype(np.int64)
    degree = np.array(adj.sum(axis=1)).flatten().astype(np.float32)
    n_nodes = adj.shape[0]
    print(f"  concepts: {n_nodes}, edges: {adj.nnz}", flush=True)

    dist_cache = f'{DATA_DIR}/dist_matrix.npy'
    if os.path.exists(dist_cache):
        print("Loading cached distance matrix...", flush=True)
        dist_matrix = np.load(dist_cache)
        print(f"  loaded: shape={dist_matrix.shape}", flush=True)
    else:
        print("Computing all-pairs shortest paths (scipy, ~27min)...", flush=True)
        t_sp = time.time()
        dist_matrix = shortest_path(adj, method='D', directed=False)
        dist_matrix[np.isinf(dist_matrix)] = DIST_BINS + 1
        dist_matrix = np.minimum(dist_matrix, DIST_BINS + 1).astype(np.uint8)
        dt = time.time() - t_sp
        print(f"  done in {dt:.1f}s, shape={dist_matrix.shape}, mem={dist_matrix.nbytes/1e6:.1f}MB", flush=True)
        np.save(dist_cache, dist_matrix)
        print(f"  cached to {dist_cache}", flush=True)

    print("Loading positive pairs...", flush=True)
    pos_pairs = []
    with open(f'{DATA_DIR}/train.jsonl') as f:
        for line in f:
            d = json.loads(line)
            if d['label'] == 1:
                pos_pairs.append((d['node_ids'][d['src_idx']], d['node_ids'][d['dst_idx']]))
    print(f"  positives: {len(pos_pairs)}", flush=True)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(pos_pairs))
    n_train = int(len(pos_pairs) * 0.8)
    train_pairs = [pos_pairs[i] for i in perm[:n_train]]
    val_pairs = [pos_pairs[i] for i in perm[n_train:]]
    print(f"  train: {len(train_pairs)}, val: {len(val_pairs)}", flush=True)

    for split_name, pairs in [('train', train_pairs), ('val', val_pairs)]:
        print(f"\n=== {split_name}: {len(pairs)} positives, {N_WORKERS} workers ===", flush=True)
        t1 = time.time()
        with Pool(N_WORKERS, initializer=_init_worker,
                  initargs=(dist_matrix, adj_rows, adj_cols, degree)) as pool:
            results = pool.map(_process_positive, pairs, chunksize=100)

        all_inp = np.concatenate([r[0] for r in results], axis=0)
        all_task = np.concatenate([r[1] for r in results], axis=0)
        all_node_ids = np.concatenate([r[2] for r in results], axis=0)

        n = len(pairs)
        labels = np.zeros(n * (1 + N_HARD_NEG), dtype=np.float32)
        for i in range(n):
            labels[i * (1 + N_HARD_NEG)] = 1.0

        dt = time.time() - t1
        print(f"  {split_name}: {len(labels)} samples, {dt:.1f}s, "
              f"{len(labels)/dt:.0f} samples/s", flush=True)

        out_path = f'{DATA_DIR}/{split_name}_precomp_v2.npz'
        np.savez_compressed(out_path, inp=all_inp, task=all_task, label=labels,
                            node_ids=all_node_ids)
        sz = os.path.getsize(out_path) / 1e6
        print(f"  saved {out_path} ({sz:.1f}MB)", flush=True)
        del results, all_inp, all_task, labels, all_node_ids

    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
    print("Done.", flush=True)
