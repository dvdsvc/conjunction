"""
Precompute PENCIL v5 training data:
  - 16533 train + 4134 val anchors
  - Each anchor: 1 positive + 200 hard negatives from A² top-1000
  - Store node_ids only (features reconstructed on GPU during training)
  - ~424MB total

Output:
  - train_nids_v5.npy: (16533, 201, 32) int64
  - val_nids_v5.npy: (4134, 101, 32) int64
  - val_cands_v5.npy: (4134, 101) int64 — candidate node IDs for eval
"""
import os, time, random, re
import numpy as np
import scipy.sparse as sp
from numba import njit
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32
EMBED_DIM = 68
A2_TOP_K = 1000       # Hard negatives from top-1000 (matches v3 distribution)
N_TRAIN_NEG = 200     # Pool of negatives for training (sample 99 per step)
N_VAL_CANDS = 100     # 1 positive + 99 negatives for validation
N_WORKERS = 16
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
def _sorted_intersect_sorted(a, b):
    result = np.empty(min(len(a), len(b)), dtype=np.int64)
    cnt = 0; j = 0
    for i in range(len(a)):
        while j < len(b) and b[j] < a[i]: j += 1
        if j < len(b) and b[j] == a[i]:
            result[cnt] = a[i]; cnt += 1
    return result[:cnt]


@njit(cache=True)
def _sort_by_degree_desc(nodes, deg):
    n = len(nodes)
    for i in range(1, n):
        key = nodes[i]; key_deg = deg[key]; j = i - 1
        while j >= 0 and deg[nodes[j]] < key_deg:
            nodes[j+1] = nodes[j]; j -= 1
        nodes[j+1] = key


@njit(nogil=True, cache=True)
def _get_selected_nodes(u, c, indptr, indices, deg, out_nids):
    nu = indices[indptr[u]:indptr[u + 1]]
    nv = indices[indptr[c]:indptr[c + 1]]
    nu_sorted = nu.copy(); nv_sorted = nv.copy()
    _sort_by_degree_desc(nu_sorted, deg); _sort_by_degree_desc(nv_sorted, deg)
    nu_for_set = np.sort(nu); nv_for_set = np.sort(nv)
    shared = _sorted_intersect_sorted(nu_for_set, nv_for_set)
    _sort_by_degree_desc(shared, deg)
    n_sel = 0
    out_nids[n_sel] = u; n_sel += 1; out_nids[n_sel] = c; n_sel += 1
    for i in range(len(shared)):
        if n_sel >= MAX_NODES: break
        out_nids[n_sel] = shared[i]; n_sel += 1
    if n_sel < MAX_NODES:
        for i in range(len(nu_sorted)):
            if n_sel >= MAX_NODES: break
            node = nu_sorted[i]
            if node == c or _in_sorted(shared, node): continue
            out_nids[n_sel] = node; n_sel += 1
    if n_sel < MAX_NODES:
        for i in range(len(nv_sorted)):
            if n_sel >= MAX_NODES: break
            node = nv_sorted[i]
            if node == u or _in_sorted(shared, node): continue
            already = False
            for k in range(n_sel):
                if out_nids[k] == node: already = True; break
            if already: continue
            out_nids[n_sel] = node; n_sel += 1
    for i in range(n_sel, MAX_NODES):
        out_nids[i] = 0


def _extract_val_pairs(data_dir):
    pos_pairs = []
    with open(f'{data_dir}/train.jsonl') as f:
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
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(pos_pairs))
    n_train = int(len(pos_pairs) * 0.8)
    return ([pos_pairs[i] for i in perm[:n_train]],
            [pos_pairs[i] for i in perm[n_train:]])


def _build_anchor(args):
    """Build node_ids for one anchor: positive + neg_pool_size negatives."""
    ai, u, v, a2_cands, neg_pool_size, indptr, indices, deg = args
    u_neighbors = set(indices[indptr[u]:indptr[u + 1]].tolist())
    valid = [c for c in a2_cands if c != v and c not in u_neighbors]
    if len(valid) < neg_pool_size:
        valid = valid + [c for c in a2_cands if c != v]
    if len(valid) == 0:
        return ai, None

    rng = np.random.RandomState(ai)
    chosen = list(rng.choice(valid, size=neg_pool_size,
                              replace=(len(valid) < neg_pool_size)))

    n_total = 1 + neg_pool_size
    nids = np.zeros((n_total, MAX_NODES), dtype=np.int64)
    tmp = np.zeros(MAX_NODES, dtype=np.int64)

    _get_selected_nodes(u, v, indptr, indices, deg, tmp)
    nids[0] = tmp

    for j, neg in enumerate(chosen):
        _get_selected_nodes(u, neg, indptr, indices, deg, tmp)
        nids[1 + j] = tmp

    return ai, nids


def main():
    t0 = time.time()
    np.random.seed(42); random.seed(42)

    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    indptr = adj.indptr.astype(np.int64)
    indices = adj.indices.astype(np.int64)
    degree = np.array(adj.sum(axis=1)).flatten().astype(np.float32)

    print("Computing A² top-1000...", flush=True)
    t_a2 = time.time()
    adj2 = adj.dot(adj)
    adj2_lil = adj2.tolil(); del adj2
    print(f"  {time.time()-t_a2:.1f}s", flush=True)

    a2_topk = {}
    for u in range(adj.shape[0]):
        row = adj2_lil[u]
        cands = np.array(row.rows[0]); scores = np.array(row.data[0])
        if len(cands) == 0: continue
        order = np.argsort(-scores)
        a2_topk[u] = cands[order[:A2_TOP_K]].tolist()
    del adj2_lil
    print(f"  indexed {len(a2_topk)} nodes", flush=True)

    train_pairs, val_pairs = _extract_val_pairs(DATA_DIR)
    print(f"  train: {len(train_pairs)}, val: {len(val_pairs)}", flush=True)

    # JIT warmup
    print("Compiling Numba...", flush=True)
    tmp = np.zeros(MAX_NODES, dtype=np.int64)
    _get_selected_nodes(0, 1, indptr, indices, degree, tmp)
    print("  done", flush=True)

    # Build train data: 1 positive + N_TRAIN_NEG negatives per anchor
    for split_name, pairs, neg_pool_size in [
        ('train', train_pairs, N_TRAIN_NEG),
        ('val', val_pairs, N_VAL_CANDS),
    ]:
        n = len(pairs)
        n_total = 1 + neg_pool_size
        print(f"\n=== {split_name}: {n} anchors, {n_total} candidates each, {N_WORKERS} threads ===", flush=True)
        t1 = time.time()

        all_nids = np.zeros((n, n_total, MAX_NODES), dtype=np.int64)

        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            args_list = [
                (ai, pairs[ai][0], pairs[ai][1],
                 a2_topk.get(pairs[ai][0], []),
                 neg_pool_size, indptr, indices, degree)
                for ai in range(n)
            ]
            done = 0
            skipped = 0
            for ai, nids in ex.map(_build_anchor, args_list):
                if nids is None:
                    skipped += 1
                    continue
                all_nids[ai] = nids
                done += 1
                if done % 1000 == 0:
                    elapsed = time.time() - t1
                    eta = elapsed / done * (n - done)
                    print(f"  {done}/{n} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        dt = time.time() - t1
        print(f"  done: {dt:.0f}s ({done} anchors, {skipped} skipped)", flush=True)

        out_path = f'{DATA_DIR}/{split_name}_nids_v5.npy'
        np.save(out_path, all_nids)
        sz = os.path.getsize(out_path) / 1e6
        print(f"  saved {out_path} ({sz:.1f}MB)", flush=True)
        del all_nids

    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
