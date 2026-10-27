"""
Precompute subgraphs for 3-hop LBD training.
v4: Pure Python + ThreadPoolExecutor. 1-hop subgraph. No Numba.
"""
import os, time
import numpy as np
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 48
EMBED_DIM = 2 * MAX_NODES + 3  # 99
DIST_BINS = 10
N_WORKERS = 16

N_HARD_NEG = 12
STRATA = [(0, 100, 4), (100, 1000, 4), (1000, 5000, 4)]
N_TRAIN = 50000
N_VAL = 5000
G = 1 + N_HARD_NEG  # 13


def _in_sorted_set(s, val):
    return val in s


def _build_subgraph(u, c, indptr, indices, dist_mat, out_feat, out_nids):
    """Build subgraph for one (u, c) pair. 1-hop neighbors."""
    MN = MAX_NODES; ED = EMBED_DIM; DB = float(DIST_BINS)

    u_nbs = indices[indptr[u]:indptr[u+1]]
    c_nbs = indices[indptr[c]:indptr[c+1]]

    sel = [u, c]
    sel_set = {u, c}

    for nb in u_nbs:
        if len(sel) >= MN: break
        if nb not in sel_set:
            sel.append(int(nb))
            sel_set.add(int(nb))

    for nb in c_nbs:
        if len(sel) >= MN: break
        if nb not in sel_set:
            sel.append(int(nb))
            sel_set.add(int(nb))

    ns = len(sel)

    # Clear
    out_feat[:] = 0
    out_nids[:] = 0

    # Features
    for i in range(ns):
        out_feat[i, i] = 1.0
        out_feat[i, 2 * MN] = 1.0
        nd = sel[i]
        out_feat[i, 2*MN+1] = min(float(dist_mat[u, nd]), DB) / DB
        out_feat[i, 2*MN+2] = min(float(dist_mat[c, nd]), DB) / DB

    # Adjacency
    sel_sorted = sorted(sel)
    sel_to_idx = {s: i for i, s in enumerate(sel)}
    for i, a in enumerate(sel):
        for j in range(indptr[a], indptr[a + 1]):
            b = int(indices[j])
            if b in sel_to_idx:
                lb = sel_to_idx[b]
                if lb < MN:
                    out_feat[i, MN + lb] = 1.0

    for i in range(min(ns, MN)):
        out_nids[i] = sel[i]


def _build_anchor(u, v, neg_cands, indptr, indices, dist_mat, out_all_inp, out_all_nids):
    """Build subgraphs for positive + negatives."""
    tmp_f = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
    tmp_n = np.zeros(MAX_NODES, dtype=np.int64)

    # Positive
    _build_subgraph(u, v, indptr, indices, dist_mat, tmp_f, tmp_n)
    out_all_inp[0] = tmp_f
    out_all_nids[0] = tmp_n

    # Negatives
    for ni, neg in enumerate(neg_cands):
        _build_subgraph(u, neg, indptr, indices, dist_mat, tmp_f, tmp_n)
        out_all_inp[ni + 1] = tmp_f
        out_all_nids[ni + 1] = tmp_n


def main():
    global N_TRAIN, N_VAL
    t0 = time.time()
    rng = np.random.RandomState(42)

    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    indptr = adj.indptr.astype(np.int64)
    indices = adj.indices.astype(np.int64)
    n_nodes = adj.shape[0]
    print(f"  {n_nodes} nodes, {adj.nnz} edges", flush=True)

    print("Loading dist_matrix...", flush=True)
    dist_matrix = np.load(f'{DATA_DIR}/dist_matrix.npy')

    # Sample 3-hop pairs
    print("Sampling 3-hop pairs...", flush=True)
    d3_rows, d3_cols = np.where(dist_matrix == 3)
    n_total = len(d3_rows)
    print(f"  total 3-hop pairs: {n_total}", flush=True)
    idx = rng.choice(n_total, size=min(N_TRAIN + N_VAL, n_total), replace=False)
    all_pairs = list(zip(d3_rows[idx].tolist(), d3_cols[idx].tolist()))
    del d3_rows, d3_cols
    train_pairs = all_pairs[:N_TRAIN]
    val_pairs = all_pairs[N_TRAIN:N_TRAIN + N_VAL]
    print(f"  train: {len(train_pairs)}, val: {len(val_pairs)}", flush=True)

    # Index 3-hop candidates
    print("Indexing 3-hop candidates...", flush=True)
    anchor_nodes = set(u for u, v in train_pairs + val_pairs)
    anchor_3hop = {}
    for u in anchor_nodes:
        d3 = np.where(dist_matrix[u] == 3)[0]
        d3 = d3[d3 != u].tolist()
        if d3:
            anchor_3hop[u] = d3
    print(f"  {len(anchor_3hop)} anchors indexed", flush=True)

    def select_negatives(u, v):
        cands = anchor_3hop.get(u, [])
        valid = [c for c in cands if c != v]
        negs = []
        for start, end, count in STRATA:
            stratum = cands[start:end] if len(cands) > start else []
            stratum_valid = [c for c in stratum if c != v]
            if len(stratum_valid) >= count:
                chosen = list(np.random.choice(stratum_valid, size=count, replace=False))
            elif len(stratum_valid) > 0:
                chosen = stratum_valid + list(np.random.choice(stratum_valid,
                    size=count - len(stratum_valid), replace=True))
            else:
                fallback = valid if valid else list(np.random.choice(
                    np.arange(n_nodes), size=20, replace=False))
                chosen = list(np.random.choice(fallback, size=count, replace=True))
            negs.extend(chosen)
        return negs

    def build_split(pairs, split_name):
        print(f"\n=== {split_name}: {len(pairs)} pairs, {N_WORKERS} threads ===", flush=True)
        t1 = time.time()
        n = len(pairs)

        # Pre-allocate output
        out_inp = np.zeros((n * G, MAX_NODES, EMBED_DIM), dtype=np.float16)
        out_nids = np.zeros((n * G, MAX_NODES), dtype=np.int64)
        labels = np.zeros(n * G, dtype=np.float32)
        for i in range(n):
            labels[i * G] = 1.0

        def process_one(args):
            idx, (u, v) = args
            negs = select_negatives(u, v)
            buf_inp = np.zeros((G, MAX_NODES, EMBED_DIM), dtype=np.float32)
            buf_nids = np.zeros((G, MAX_NODES), dtype=np.int64)
            _build_anchor(u, v, negs, indptr, indices, dist_matrix, buf_inp, buf_nids)
            return idx, buf_inp, buf_nids

        # Submit all tasks
        tasks = [(i, pairs[i]) for i in range(n)]

        done = 0
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            for idx, inp, nids in ex.map(process_one, tasks):
                out_inp[idx*G:(idx+1)*G] = inp.astype(np.float16)
                out_nids[idx*G:(idx+1)*G] = nids
                done += 1
                if done % 1000 == 0:
                    elapsed = time.time() - t1
                    eta = elapsed / done * (n - done)
                    print(f"  {done}/{n} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        dt = time.time() - t1
        print(f"  {n*G} samples, {dt:.1f}s, {n*G/dt:.0f} samples/s", flush=True)

        out_path = f'{DATA_DIR}/{split_name}_precomp_3hop.npz'
        np.savez_compressed(out_path, inp=out_inp, label=labels, node_ids=out_nids)
        sz = os.path.getsize(out_path) / 1e6
        print(f"  saved {out_path} ({sz:.1f}MB)", flush=True)

    build_split(train_pairs, 'train')
    build_split(val_pairs, 'val')

    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
