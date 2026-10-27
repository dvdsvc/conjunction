"""
Recall@K: A² baseline vs PENCIL model.
Numba-accelerated subgraph building - no Python objects in hot path.
"""
import os, time, sys
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
from numba import njit

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
TMP_DIR = '/dev/shm/eval_recall'
MAX_NODES = 32
EMBED_DIM = 68  # 2*MAX_NODES + 2 + 2
A2_TOP_K = 1000
TOP_K_EVAL = [1, 3, 5, 10, 50, 100, 500, 1000]
N_WORKERS = 80
DIST_BINS = 10

# Worker globals (COW)
_g_adj_indptr = None
_g_adj_indices = None
_g_degree = None
_g_dist_matrix = None


def _init_worker():
    pass


@njit(cache=True)
def _in_sorted(arr, val):
    """Binary search in sorted array."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == val:
            return True
        elif arr[mid] < val:
            lo = mid + 1
        else:
            hi = mid - 1
    return False


@njit(cache=True)
def _sorted_intersect_sorted(a, b):
    """Intersection of two sorted arrays, preserving a's order."""
    result = np.empty(min(len(a), len(b)), dtype=np.int64)
    cnt = 0
    j = 0
    for i in range(len(a)):
        while j < len(b) and b[j] < a[i]:
            j += 1
        if j < len(b) and b[j] == a[i]:
            result[cnt] = a[i]
            cnt += 1
    return result[:cnt]


@njit(cache=True)
def _sort_by_degree_desc(nodes, deg):
    """Insertion sort by degree descending (small arrays, ~30 elements)."""
    n = len(nodes)
    for i in range(1, n):
        key = nodes[i]
        key_deg = deg[key]
        j = i - 1
        while j >= 0 and deg[nodes[j]] < key_deg:
            nodes[j + 1] = nodes[j]
            j -= 1
        nodes[j + 1] = key


@njit(cache=True)
def _build_subgraph(u, c, indptr, indices, deg, dist_mat):
    """Build one subgraph for pair (u, c). Returns (inp, nids)."""
    nu = indices[indptr[u]:indptr[u + 1]]
    nv = indices[indptr[c]:indptr[c + 1]]

    # Sort neighbors by degree descending
    nu_sorted = nu.copy()
    nv_sorted = nv.copy()
    _sort_by_degree_desc(nu_sorted, deg)
    _sort_by_degree_desc(nv_sorted, deg)

    # Sort for set operations
    nu_for_set = np.sort(nu)
    nv_for_set = np.sort(nv)

    # Shared neighbors
    shared = _sorted_intersect_sorted(nu_for_set, nv_for_set)
    _sort_by_degree_desc(shared, deg)

    # Select nodes: u, c, shared, only_u, only_v
    selected = np.empty(MAX_NODES, dtype=np.int64)
    n_sel = 0
    selected[n_sel] = u; n_sel += 1
    selected[n_sel] = c; n_sel += 1

    for i in range(len(shared)):
        if n_sel >= MAX_NODES: break
        selected[n_sel] = shared[i]; n_sel += 1

    if n_sel < MAX_NODES:
        for i in range(len(nu_sorted)):
            if n_sel >= MAX_NODES: break
            node = nu_sorted[i]
            if node == c: continue
            if _in_sorted(shared, node): continue
            selected[n_sel] = node; n_sel += 1

    if n_sel < MAX_NODES:
        for i in range(len(nv_sorted)):
            if n_sel >= MAX_NODES: break
            node = nv_sorted[i]
            if node == u: continue
            if _in_sorted(shared, node): continue
            already = False
            for k in range(n_sel):
                if selected[k] == node:
                    already = True; break
            if already: continue
            selected[n_sel] = node; n_sel += 1

    # Build features
    inp = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
    for i in range(n_sel):
        if i < MAX_NODES:
            inp[i, i] = 1.0
        inp[i, 2 * MAX_NODES] = 1.0
        node_id = selected[i]
        inp[i, 2 * MAX_NODES + 1] = min(float(dist_mat[u, node_id]), float(DIST_BINS)) / float(DIST_BINS)
        inp[i, 2 * MAX_NODES + 2] = min(float(dist_mat[c, node_id]), float(DIST_BINS)) / float(DIST_BINS)

    # Adjacency: sort selected for binary search lookup
    sel_sorted = selected[:n_sel].copy()
    sel_sorted.sort()
    for i in range(n_sel):
        a = selected[i]
        for j in range(indptr[a], indptr[a + 1]):
            b = indices[j]
            if _in_sorted(sel_sorted, b):
                # Find local index via linear scan (rare, only when found)
                for lb in range(n_sel):
                    if selected[lb] == b:
                        if lb < MAX_NODES:
                            inp[i, MAX_NODES + lb] = 1.0
                        break

    nids = np.zeros(MAX_NODES, dtype=np.int64)
    for i in range(min(n_sel, MAX_NODES)):
        nids[i] = selected[i]

    return inp, nids


def _process_anchor(args):
    """Build subgraphs for one anchor. Write to memmap. Return (anchor_idx, pos_idx)."""
    anchor_idx, u, v, top_cands = args

    NC = A2_TOP_K + 1
    inp_mm = np.memmap(f'{TMP_DIR}/inp.npy', dtype='float16', mode='r+',
                       shape=(NC, MAX_NODES, EMBED_DIM),
                       offset=anchor_idx * NC * MAX_NODES * EMBED_DIM * 2)
    nids_mm = np.memmap(f'{TMP_DIR}/nids.npy', dtype='int64', mode='r+',
                        shape=(NC, MAX_NODES),
                        offset=anchor_idx * NC * MAX_NODES * 8)

    indptr, indices, deg, dist_mat = _g_adj_indptr, _g_adj_indices, _g_degree, _g_dist_matrix
    pos_idx = -1

    # Batch: collect all subgraphs, write once
    all_inp = np.zeros((NC, MAX_NODES, EMBED_DIM), dtype=np.float16)
    all_nids = np.zeros((NC, MAX_NODES), dtype=np.int64)

    for j, c in enumerate(top_cands):
        if c == v:
            pos_idx = j
        inp, nids = _build_subgraph(u, c, indptr, indices, deg, dist_mat)
        all_inp[j] = inp.astype(np.float16)
        all_nids[j] = nids

    # Single memmap write per anchor
    inp_mm[:] = all_inp
    nids_mm[:] = all_nids

    if pos_idx == -1:
        # fallback
        for j, c in enumerate(top_cands):
            if c == v:
                pos_idx = j
                break
        if pos_idx == -1:
            pos_idx = 0
    return anchor_idx, pos_idx


if __name__ == '__main__':
    t0 = time.time()
    os.makedirs(TMP_DIR, exist_ok=True)

    # ---- Phase 1: Data loading ----
    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    adj_rows = adj.indptr.astype(np.int64)
    adj_cols = adj.indices.astype(np.int64)
    degree = np.array(adj.sum(axis=1)).flatten().astype(np.float32)

    print("Computing A²...", flush=True)
    sys.stdout.flush()
    t_a2 = time.time()
    adj2 = adj.dot(adj)
    adj2_lil = adj2.tolil()
    del adj2
    print(f"  A² done: {time.time()-t_a2:.1f}s, nnz={adj2_lil.nnz}", flush=True)

    print("Loading distance matrix...", flush=True)
    dist_matrix = np.load(f'{DATA_DIR}/dist_matrix.npy')

    print("Loading val positives...", flush=True)
    val_data = np.load(f'{DATA_DIR}/val_precomp_v2.npz')
    val_node_ids = val_data['node_ids']
    val_labels = val_data['label']
    G = 6
    n_val = len(val_labels) // G
    val_pairs = []
    for i in range(n_val):
        val_pairs.append((int(val_node_ids[i * G][0]), int(val_node_ids[i * G][1])))
    print(f"  val pairs: {len(val_pairs)}, setup: {time.time()-t0:.1f}s", flush=True)

    # A² baseline
    print(f"\n=== A² baseline ===", flush=True)
    a2_ranks = []
    for u, v in val_pairs:
        row = adj2_lil[u]
        cands = np.array(row.rows[0])
        scores = np.array(row.data[0])
        order = np.argsort(-scores)
        sorted_cands = cands[order]
        pos = np.where(sorted_cands == v)[0]
        if len(pos) > 0:
            a2_ranks.append(int(pos[0]) + 1)
        else:
            a2_ranks.append(len(sorted_cands) + 1)
    a2_ranks = np.array(a2_ranks)
    for k in TOP_K_EVAL:
        print(f"  A² Recall@{k}: {(a2_ranks <= k).mean():.4f}", flush=True)
    print(f"  A² Median: {np.median(a2_ranks):.0f}, MRR: {(1.0/a2_ranks).mean():.4f}", flush=True)

    # ---- Phase 2: Subgraph building ----
    print(f"\n=== PENCIL (Numba, {N_WORKERS} workers) ===", flush=True)

    N_ANCHORS = len(val_pairs)
    NC = A2_TOP_K + 1
    inp_size = N_ANCHORS * NC * MAX_NODES * EMBED_DIM * 2
    nids_size = N_ANCHORS * NC * MAX_NODES * 8
    print(f"  pre-allocating memmap ({inp_size/1e9:.1f}GB + {nids_size/1e9:.1f}GB)...", flush=True)

    for fname, sz in [('inp.npy', inp_size), ('nids.npy', nids_size)]:
        path = f'{TMP_DIR}/{fname}'
        with open(path, 'wb') as f:
            f.truncate(sz)

    # Set globals BEFORE Pool (COW)
    _g_adj_indptr = adj_rows
    _g_adj_indices = adj_cols
    _g_degree = degree
    _g_dist_matrix = dist_matrix

    # Build work_items
    work_items = []
    for idx, (u, v) in enumerate(val_pairs):
        row = adj2_lil[u]
        cands = np.array(row.rows[0])
        scores_a2 = np.array(row.data[0])
        order = np.argsort(-scores_a2)
        top_cands = list(cands[order[:A2_TOP_K]])
        if v not in set(top_cands):
            top_cands.append(v)
        work_items.append((idx, u, v, top_cands))

    del adj, adj2_lil, dist_matrix

    # Numba JIT warmup (compile before forking)
    print("  compiling Numba JIT...", flush=True)
    t_jit = time.time()
    _build_subgraph(work_items[0][1], work_items[0][3][0],
                    adj_rows, adj_cols, degree, _g_dist_matrix)
    print(f"  JIT compiled: {time.time()-t_jit:.1f}s", flush=True)

    t_build = time.time()
    pos_idx_map = {}
    with Pool(N_WORKERS, initializer=_init_worker) as pool:
        for anchor_idx, pos_idx in pool.imap_unordered(
                _process_anchor, work_items, chunksize=10):
            pos_idx_map[anchor_idx] = pos_idx
    print(f"  subgraph build: {time.time()-t_build:.1f}s", flush=True)

    # ---- Phase 3: GPU scoring ----
    import torch
    import torch.nn as nn

    class PencilModel(nn.Module):
        def __init__(self, scibert_embeddings):
            super().__init__()
            self.register_buffer('scibert_table', scibert_embeddings)
            self.scibert_proj = nn.Linear(768, 128)
            self.struct_proj = nn.Linear(68, 128)
            self.dist_embed = nn.Embedding(12 * 12, 32)
            self.combine = nn.Linear(128 + 128 + 32, 256)
            self.layers = nn.ModuleList()
            for _ in range(4):
                self.layers.append(nn.ModuleDict({
                    'attn': nn.MultiheadAttention(256, 8, batch_first=True, dropout=0.1),
                    'ln1': nn.LayerNorm(256),
                    'ffn': nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256), nn.Dropout(0.1)),
                    'ln2': nn.LayerNorm(256),
                    'mp': nn.Sequential(nn.Linear(256, 256), nn.GELU()),
                }))
            self.ln_f = nn.LayerNorm(256)
            self.head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))

        def forward(self, inp, node_ids):
            B, N, _ = inp.shape
            adj = inp[:, :N, 32:32 + N]
            deg = adj.sum(dim=2, keepdim=True).clamp(min=1)
            adj_norm = adj / deg
            dist_u_bin = (inp[:, :, 65] * 10).long().clamp(0, 11)
            dist_v_bin = (inp[:, :, 66] * 10).long().clamp(0, 11)
            dist_emb = self.dist_embed(dist_u_bin * 12 + dist_v_bin)
            x_raw = inp.clone()
            x_raw[:, :, 65] = 0
            x_raw[:, :, 66] = 0
            struct_emb = self.struct_proj(x_raw)
            nids_flat = node_ids.reshape(-1).clamp(0, self.scibert_table.shape[0] - 1)
            scibert_emb = self.scibert_proj(self.scibert_table[nids_flat].reshape(B, N, 768))
            pad_mask = (inp[:, :, :65].abs().sum(dim=2) == 0)
            pm3 = pad_mask.unsqueeze(-1)
            scibert_emb = scibert_emb.masked_fill(pm3, 0.0)
            dist_emb = dist_emb.masked_fill(pm3, 0.0)
            struct_emb = struct_emb.masked_fill(pm3, 0.0)
            x = self.combine(torch.cat([struct_emb, scibert_emb, dist_emb], dim=-1))
            for layer in self.layers:
                h = layer['ln1'](x)
                h, _ = layer['attn'](h, h, h, key_padding_mask=pad_mask)
                x = x + h
                x[:, :N] = x[:, :N] + layer['mp'](torch.bmm(adj_norm, x[:, :N]))
                h = layer['ln2'](x)
                x = x + layer['ffn'](h)
            x = self.ln_f(x)
            return self.head(torch.cat([x[:, 0], x[:, 1]], -1)).squeeze(-1)

    inp_all = np.memmap(f'{TMP_DIR}/inp.npy', dtype='float16', mode='r',
                        shape=(N_ANCHORS, NC, MAX_NODES, EMBED_DIM))
    nids_all = np.memmap(f'{TMP_DIR}/nids.npy', dtype='int64', mode='r',
                         shape=(N_ANCHORS, NC, MAX_NODES))

    device = torch.device('cuda')
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilModel(scibert).to(device)
    model.load_state_dict(torch.load(f'{DATA_DIR}/pencil_v2_best.pt', weights_only=True))
    model.eval()

    pencil_ranks = []
    t_score = time.time()
    with torch.inference_mode():
        for idx in range(N_ANCHORS):
            n_cands = len(work_items[idx][3])
            inp_t = torch.from_numpy(np.array(inp_all[idx, :n_cands]).astype(np.float32)).to(device)
            nids_t = torch.from_numpy(np.array(nids_all[idx, :n_cands])).to(device)
            scores = model(inp_t, nids_t).cpu().numpy()
            pos_score = scores[pos_idx_map[idx]]
            rank = (scores > pos_score).sum() + 1
            pencil_ranks.append(rank)
            if (idx + 1) % 500 == 0:
                print(f"  scored {idx+1}/{N_ANCHORS}, {time.time()-t_score:.1f}s", flush=True)

    pencil_ranks = np.array(pencil_ranks)
    print(f"  scoring: {time.time()-t_score:.1f}s", flush=True)

    print(f"\n=== PENCIL results ===", flush=True)
    for k in TOP_K_EVAL:
        print(f"  PENCIL Recall@{k}: {(pencil_ranks <= k).mean():.4f}", flush=True)
    print(f"  PENCIL Median: {np.median(pencil_ranks):.0f}, MRR: {(1.0/pencil_ranks).mean():.4f}", flush=True)

    os.remove(f'{TMP_DIR}/inp.npy')
    os.remove(f'{TMP_DIR}/nids.npy')

    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
    print("Done.", flush=True)
