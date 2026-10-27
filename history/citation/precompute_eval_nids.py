"""
Precompute eval subgraphs: store only node_ids (32 int64 per subgraph).
Later eval reconstructs features from node_ids in ~0.05ms (vs 2.6ms full build).

This script computes:
  1. A² top-5000 candidates for each val anchor
  2. Subgraph node_ids for all (anchor, candidate) pairs
  3. Saves as binary file: eval_subgraph_nids.npy (4134 × 5001 × 32, int64, ~5GB)
"""
import os, time, re
import numpy as np
import scipy.sparse as sp
from numba import njit
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32
EMBED_DIM = 68
A2_TOP_K = 5000
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
    """Build subgraph and return only selected node ids."""
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
    # Zero-pad
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
    return [pos_pairs[i] for i in perm[n_train:]]


def _build_anchor_nids(args):
    """Build subgraph node_ids for all candidates of one anchor."""
    ai, u, v, cands, indptr, indices, deg = args
    NC = len(cands)
    nids = np.zeros((NC, MAX_NODES), dtype=np.int64)
    tmp = np.zeros(MAX_NODES, dtype=np.int64)
    for j, c in enumerate(cands):
        _get_selected_nodes(u, c, indptr, indices, deg, tmp)
        nids[j] = tmp
    return ai, nids


if __name__ == '__main__':
    t0 = time.time()
    out_path = f'{DATA_DIR}/eval_subgraph_nids.npy'

    if os.path.exists(out_path):
        existing = np.load(out_path, mmap_mode='r')
        print(f"File exists: {out_path} shape={existing.shape}", flush=True)
        print("Delete it first if you want to recompute. Exiting.", flush=True)
        exit(0)

    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    indptr = adj.indptr.astype(np.int64)
    indices = adj.indices.astype(np.int64)
    degree = np.array(adj.sum(axis=1)).flatten().astype(np.float32)

    print("Computing A²...", flush=True)
    t_a2 = time.time()
    adj2 = adj.dot(adj)
    adj2_lil = adj2.tolil(); del adj2
    print(f"  A² done: {time.time()-t_a2:.1f}s, nnz={adj2_lil.nnz}", flush=True)

    print("Extracting val pairs...", flush=True)
    val_pairs = _extract_val_pairs(DATA_DIR)
    N_ANCHORS = len(val_pairs)
    print(f"  {N_ANCHORS} val pairs", flush=True)

    # Build candidate lists
    print(f"Preparing A² top-{A2_TOP_K} candidates...", flush=True)
    anchor_cands = []
    for u, v in val_pairs:
        row = adj2_lil[u]
        cands = np.array(row.rows[0])
        scores_a2 = np.array(row.data[0])
        order = np.argsort(-scores_a2)
        top = list(cands[order[:A2_TOP_K]])
        if v not in set(top):
            top.append(v)
        anchor_cands.append(top)
    del adj2_lil

    # JIT warmup
    print("Compiling Numba...", flush=True)
    tmp_n = np.zeros(MAX_NODES, dtype=np.int64)
    _get_selected_nodes(0, 1, indptr, indices, degree, tmp_n)
    print("  done", flush=True)

    # Build all node_ids with thread pool
    print(f"Building subgraph node_ids ({N_WORKERS} workers)...", flush=True)
    # Determine max candidates
    max_cands = max(len(c) for c in anchor_cands)
    print(f"  max candidates per anchor: {max_cands}", flush=True)

    all_nids = np.zeros((N_ANCHORS, max_cands, MAX_NODES), dtype=np.int64)
    t1 = time.time()

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        args_list = [
            (ai, val_pairs[ai][0], val_pairs[ai][1], anchor_cands[ai],
             indptr, indices, degree)
            for ai in range(N_ANCHORS)
        ]
        done = 0
        for ai, nids in ex.map(_build_anchor_nids, args_list):
            NC = nids.shape[0]
            all_nids[ai, :NC] = nids
            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t1
                eta = elapsed / done * (N_ANCHORS - done)
                print(f"  {done}/{N_ANCHORS} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    dt = time.time() - t1
    print(f"  done: {dt:.0f}s ({dt/N_ANCHORS:.2f}s/anchor)", flush=True)

    # Save
    print(f"Saving {out_path}...", flush=True)
    np.save(out_path, all_nids)
    sz = os.path.getsize(out_path) / 1e9
    print(f"  saved {sz:.2f}GB", flush=True)

    # Also save candidate lengths for eval
    cands_len = np.array([len(c) for c in anchor_cands], dtype=np.int32)
    np.save(f'{DATA_DIR}/eval_cands_len.npy', cands_len)

    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
    print("Done.", flush=True)
