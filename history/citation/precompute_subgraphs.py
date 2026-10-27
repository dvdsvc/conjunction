"""
Precompute all subgraph encodings (positive + hard negatives).
Save as numpy for fast GPU training.
"""

import json, os, time, random
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 128
EMBED_DIM = 2 * MAX_NODES + 2  # 258
N_HARD_NEG = 5
N_WORKERS = 80

# Shared globals for workers
_g_adj_rows = None
_g_adj_cols = None
_g_degree = None

def _init_worker(adj_rows, adj_cols, degree):
    global _g_adj_rows, _g_adj_cols, _g_degree
    _g_adj_rows = adj_rows
    _g_adj_cols = adj_cols
    _g_degree = degree

def _build_subgraph(u, v):
    """Build encoding for (u, v) pair."""
    rows, cols, deg = _g_adj_rows, _g_adj_cols, _g_degree
    nu = cols[rows[u]:rows[u+1]]
    nv = cols[rows[v]:rows[v+1]]

    nu_set = set(nu.tolist())
    nv_set = set(nv.tolist())
    shared = sorted(nu_set & nv_set, key=lambda x: -deg[x])
    only_u = sorted(nu_set - nv_set - {v}, key=lambda x: -deg[x])
    only_v = sorted(nv_set - nu_set - {u}, key=lambda x: -deg[x])

    nodes = [u, v]
    rem = MAX_NODES - 2
    for lst in [shared, only_u, only_v]:
        for n in lst:
            if rem <= 0: break
            nodes.append(n)
            rem -= 1
        if rem <= 0: break

    n = len(nodes)
    nset = set(nodes)
    lmap = {c: i for i, c in enumerate(nodes)}

    x = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float16)
    for i in range(n):
        if i < MAX_NODES:
            x[i, i] = 1.0
        x[i, 2*MAX_NODES] = 1.0

    for a in nodes:
        for b in cols[rows[a]:rows[a+1]]:
            if b in nset:
                ib = lmap[b]
                if ib < MAX_NODES:
                    x[lmap[a], MAX_NODES + ib] = 1.0

    task = np.zeros((2, EMBED_DIM), dtype=np.float16)
    task[0] = x[0]
    task[0, 2*MAX_NODES + 1] = 1.0
    task[1] = x[1]
    task[1, 2*MAX_NODES + 1] = 1.0

    return x, task

def _sample_hard_neg(u, pos_v):
    """Sample one hard negative: 2-hop disconnected from u."""
    rows, cols = _g_adj_rows, _g_adj_cols
    nu = cols[rows[u]:rows[u+1]]
    # Randomly pick 15 neighbors to expand
    if len(nu) > 15:
        idx = np.random.choice(len(nu), 15, replace=False)
        nu = nu[idx]
    u_neighbors = set(cols[rows[u]:rows[u+1]].tolist())
    for n1 in nu:
        n2s = cols[rows[n1]:rows[n1+1]]
        if len(n2s) > 0:
            neg = np.random.choice(n2s)
            if neg != u and neg != pos_v and neg not in u_neighbors:
                return int(neg)
    # Fallback: random concept not in u's neighbors
    for _ in range(20):
        neg = np.random.randint(0, len(rows)-1)
        if neg != u and neg != pos_v and neg not in u_neighbors:
            return neg
    return np.random.randint(0, len(rows)-1)


def _process_positive(args):
    """Build all subgraphs for one positive pair: 1 pos + N_HARD_NEG negatives."""
    u, v = args
    inp_list = []
    task_list = []

    # Positive
    x, t = _build_subgraph(u, v)
    inp_list.append(x)
    task_list.append(t)

    # Hard negatives
    for _ in range(N_HARD_NEG):
        neg_v = _sample_hard_neg(u, v)
        x, t = _build_subgraph(u, neg_v)
        inp_list.append(x)
        task_list.append(t)

    return np.stack(inp_list), np.stack(task_list)


if __name__ == '__main__':
    t0 = time.time()
    np.random.seed(42)
    random.seed(42)

    # Load graph
    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    adj_rows = adj.indptr.astype(np.int64)
    adj_cols = adj.indices.astype(np.int64)
    degree = np.array(adj.sum(axis=1)).flatten().astype(np.float32)
    print(f"  concepts: {adj.shape[0]}, edges: {adj.nnz}", flush=True)

    # Load positive pairs
    print("Loading positive pairs...", flush=True)
    pos_pairs = []
    with open(f'{DATA_DIR}/train.jsonl') as f:
        for line in f:
            d = json.loads(line)
            if d['label'] == 1:
                pos_pairs.append((d['node_ids'][d['src_idx']], d['node_ids'][d['dst_idx']]))
    print(f"  positives: {len(pos_pairs)}", flush=True)

    # Split
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(pos_pairs))
    n_train = int(len(pos_pairs) * 0.8)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_pairs = [pos_pairs[i] for i in train_idx]
    val_pairs = [pos_pairs[i] for i in val_idx]
    print(f"  train: {len(train_pairs)}, val: {len(val_pairs)}", flush=True)

    # Process train split
    for split_name, pairs in [('train', train_pairs), ('val', val_pairs)]:
        print(f"\n=== {split_name}: {len(pairs)} positives, {N_WORKERS} workers ===", flush=True)
        t1 = time.time()

        with Pool(N_WORKERS, initializer=_init_worker,
                  initargs=(adj_rows, adj_cols, degree)) as pool:
            results = pool.map(_process_positive, pairs, chunksize=50)

        # Stack results: each result is (1+N_HARD_NEG, MAX_NODES, EMBED_DIM) for inp and task
        all_inp = np.concatenate([r[0] for r in results], axis=0)
        all_task = np.concatenate([r[1] for r in results], axis=0)

        # Labels: 1 for positive (index 0 in each group), 0 for negatives
        n = len(pairs)
        labels = np.zeros(n * (1 + N_HARD_NEG), dtype=np.float32)
        for i in range(n):
            labels[i * (1 + N_HARD_NEG)] = 1.0

        dt = time.time() - t1
        print(f"  {split_name}: {len(labels)} samples, {dt:.1f}s, "
              f"{len(labels)/dt:.0f} samples/s", flush=True)

        out_path = f'{DATA_DIR}/{split_name}_precomp.npz'
        np.savez_compressed(out_path, inp=all_inp, task=all_task, label=labels)
        sz = os.path.getsize(out_path) / 1e9
        print(f"  saved {out_path} ({sz:.2f}GB)", flush=True)

        del results, all_inp, all_task, labels

    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
    print("Done.", flush=True)
