"""
Fast precompute v3: regex-extract pos pairs, Numba subgraph build,
ThreadPoolExecutor parallelism, A²-hard negatives.
"""
import os, time, random, re
import numpy as np
import scipy.sparse as sp
from numba import njit
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32
N_HARD_NEG = 5
EMBED_DIM = 2 * MAX_NODES + 2 + 2  # 68
A2_TOP_K = 1000
N_WORKERS = 80
DIST_BINS = 10


@njit(cache=True)
def _in_sorted(arr, val):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == val: return True
        elif arr[mid] < val: lo = mid + 1
        else: hi = mid - 1
    return False


@njit(cache=True)
def _sort_by_degree_desc(nodes, deg):
    n = len(nodes)
    for i in range(1, n):
        key = nodes[i]; key_deg = deg[key]; j = i - 1
        while j >= 0 and deg[nodes[j]] < key_deg:
            nodes[j+1] = nodes[j]; j -= 1
        nodes[j+1] = key


@njit(cache=True)
def _sorted_intersect(a, b):
    res = np.empty(min(len(a), len(b)), dtype=np.int64)
    cnt = 0; j = 0
    for i in range(len(a)):
        while j < len(b) and b[j] < a[i]: j += 1
        if j < len(b) and b[j] == a[i]:
            res[cnt] = a[i]; cnt += 1
    return res[:cnt]


@njit(nogil=True, cache=True)
def _build_subgraph(u, v, indptr, indices, deg, dist_mat, out_feat, out_nids):
    """Build subgraph features AND node ids."""
    nu = indices[indptr[u]:indptr[u+1]]
    nv = indices[indptr[v]:indptr[v+1]]
    nu_s = nu.copy(); nv_s = nv.copy()
    _sort_by_degree_desc(nu_s, deg); _sort_by_degree_desc(nv_s, deg)
    nu_set = np.sort(nu); nv_set = np.sort(nv)
    shared = _sorted_intersect(nu_set, nv_set)
    _sort_by_degree_desc(shared, deg)

    sel = np.empty(MAX_NODES, dtype=np.int64); ns = 0
    sel[ns] = u; ns += 1; sel[ns] = v; ns += 1
    for i in range(len(shared)):
        if ns >= MAX_NODES: break
        sel[ns] = shared[i]; ns += 1
    if ns < MAX_NODES:
        for i in range(len(nu_s)):
            if ns >= MAX_NODES: break
            nd = nu_s[i]
            if nd == v or _in_sorted(shared, nd): continue
            sel[ns] = nd; ns += 1
    if ns < MAX_NODES:
        for i in range(len(nv_s)):
            if ns >= MAX_NODES: break
            nd = nv_s[i]
            if nd == u or _in_sorted(shared, nd): continue
            already = False
            for k in range(ns):
                if sel[k] == nd: already = True; break
            if already: continue
            sel[ns] = nd; ns += 1

    # Features
    for i in range(MAX_NODES):
        for j in range(EMBED_DIM): out_feat[i, j] = 0.0
    for i in range(ns):
        if i < MAX_NODES: out_feat[i, i] = 1.0
        out_feat[i, 2*MAX_NODES] = 1.0
        nid = sel[i]
        out_feat[i, 2*MAX_NODES+1] = min(float(dist_mat[u, nid]), float(DIST_BINS)) / float(DIST_BINS)
        out_feat[i, 2*MAX_NODES+2] = min(float(dist_mat[v, nid]), float(DIST_BINS)) / float(DIST_BINS)
    sel_sorted = sel[:ns].copy(); sel_sorted.sort()
    for i in range(ns):
        a = sel[i]
        for j in range(indptr[a], indptr[a+1]):
            b = indices[j]
            if _in_sorted(sel_sorted, b):
                for lb in range(ns):
                    if sel[lb] == b:
                        if lb < MAX_NODES: out_feat[i, MAX_NODES+lb] = 1.0
                        break

    # Node ids
    for i in range(MAX_NODES): out_nids[i] = 0
    for i in range(min(ns, MAX_NODES)):
        out_nids[i] = sel[i]


def main():
    t0 = time.time()
    np.random.seed(42); random.seed(42)

    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    indptr = adj.indptr.astype(np.int64)
    indices = adj.indices.astype(np.int64)
    degree = np.array(adj.sum(axis=1)).flatten().astype(np.float32)
    n_nodes = adj.shape[0]
    print(f"  {n_nodes} nodes, {adj.nnz} edges", flush=True)

    # Fast positive pair extraction via regex (skip full JSON parse of 8.7GB)
    print("Extracting positive pairs (regex)...", flush=True)
    t1 = time.time()
    pos_pairs = []
    with open(f'{DATA_DIR}/train.jsonl') as f:
        for line in f:
            if '"label": 1' not in line and '"label":1' not in line:
                continue
            src_m = re.search(r'"src_idx"\s*:\s*(\d+)', line)
            dst_m = re.search(r'"dst_idx"\s*:\s*(\d+)', line)
            if not src_m or not dst_m: continue
            src_idx = int(src_m.group(1))
            dst_idx = int(dst_m.group(1))
            nid_m = re.search(r'"node_ids"\s*:\s*\[([^\]]+)\]', line)
            if not nid_m: continue
            nids = [int(x) for x in nid_m.group(1).split(',')]
            pos_pairs.append((nids[src_idx], nids[dst_idx]))
    print(f"  {len(pos_pairs)} pairs in {time.time()-t1:.1f}s", flush=True)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(pos_pairs))
    n_train = int(len(pos_pairs) * 0.8)
    train_pairs = [pos_pairs[i] for i in perm[:n_train]]
    val_pairs = [pos_pairs[i] for i in perm[n_train:]]
    print(f"  train: {len(train_pairs)}, val: {len(val_pairs)}", flush=True)

    # A² top-K
    print("Computing A² top-K...", flush=True)
    t_a2 = time.time()
    adj2 = adj.dot(adj)
    adj2_lil = adj2.tolil(); del adj2
    print(f"  A²: {time.time()-t_a2:.1f}s", flush=True)

    a2_topk = {}
    for u in range(n_nodes):
        row = adj2_lil[u]
        cands = np.array(row.rows[0]); scores = np.array(row.data[0])
        if len(cands) == 0: continue
        order = np.argsort(-scores)
        a2_topk[u] = cands[order[:A2_TOP_K]].tolist()
    del adj2_lil
    print(f"  indexed {len(a2_topk)} nodes", flush=True)

    dist_matrix = np.load(f'{DATA_DIR}/dist_matrix.npy')

    # JIT warmup
    print("Compiling Numba...", flush=True)
    tmp_f = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
    tmp_n = np.zeros(MAX_NODES, dtype=np.int64)
    _build_subgraph(0, 1, indptr, indices, degree, dist_matrix, tmp_f, tmp_n)
    print("  done", flush=True)

    def _build_anchor(pair):
        u, v = pair
        G = 1 + N_HARD_NEG
        all_inp = np.zeros((G, MAX_NODES, EMBED_DIM), dtype=np.float16)
        all_nids = np.zeros((G, MAX_NODES), dtype=np.int64)
        tmp_f = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
        tmp_n = np.zeros(MAX_NODES, dtype=np.int64)

        _build_subgraph(u, v, indptr, indices, degree, dist_matrix, tmp_f, tmp_n)
        all_inp[0] = tmp_f.astype(np.float16)
        all_nids[0] = tmp_n.copy()

        u_neighbors = set(indices[indptr[u]:indptr[u+1]].tolist())
        cands = a2_topk.get(u, [])
        valid = [c for c in cands if c != v and c not in u_neighbors]
        if len(valid) < N_HARD_NEG:
            valid2 = [c for c in cands if c != v]
            valid = valid2 if valid2 else [np.random.randint(0, n_nodes) for _ in range(20)]
        chosen = list(np.random.choice(valid, size=N_HARD_NEG, replace=(len(valid) < N_HARD_NEG)))
        for i, neg in enumerate(chosen):
            _build_subgraph(u, neg, indptr, indices, degree, dist_matrix, tmp_f, tmp_n)
            all_inp[i+1] = tmp_f.astype(np.float16)
            all_nids[i+1] = tmp_n.copy()

        # Task features (for model compat, not actually used in forward)
        task = np.zeros((G, 2, EMBED_DIM), dtype=np.float16)
        return all_inp, task, all_nids

    for split_name, pairs in [('train', train_pairs), ('val', val_pairs)]:
        print(f"\n=== {split_name}: {len(pairs)} pairs, {N_WORKERS} threads ===", flush=True)
        t1 = time.time()
        G = 1 + N_HARD_NEG
        n = len(pairs)
        all_inp = np.zeros((n * G, MAX_NODES, EMBED_DIM), dtype=np.float16)
        all_nids = np.zeros((n * G, MAX_NODES), dtype=np.int64)
        all_task = np.zeros((n * G, 2, EMBED_DIM), dtype=np.float16)
        labels = np.zeros(n * G, dtype=np.float32)
        for i in range(n): labels[i * G] = 1.0

        done = 0
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            for idx, (inp, task, nids) in enumerate(ex.map(_build_anchor, pairs)):
                all_inp[idx*G:(idx+1)*G] = inp
                all_nids[idx*G:(idx+1)*G] = nids
                all_task[idx*G:(idx+1)*G] = task
                done += 1
                if done % 1000 == 0:
                    print(f"  {done}/{n} ({time.time()-t1:.1f}s)", flush=True)

        dt = time.time() - t1
        print(f"  {n*G} samples, {dt:.1f}s, {n*G/dt:.0f} samples/s", flush=True)

        out_path = f'{DATA_DIR}/{split_name}_precomp_v3.npz'
        np.savez_compressed(out_path, inp=all_inp, task=all_task, label=labels,
                            node_ids=all_nids)
        sz = os.path.getsize(out_path) / 1e6
        print(f"  saved {out_path} ({sz:.1f}MB)", flush=True)
        del all_inp, all_nids, all_task, labels

    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
