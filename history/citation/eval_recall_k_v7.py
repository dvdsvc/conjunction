"""
Recall@K: A² baseline vs PENCIL model.
Workers write to shared memmap files (zero IPC overhead).
Pool is spawned ONLY after all CPU-heavy data loading is done.
dist_matrix is loaded via COW (not pickled to workers).
"""
import os, time, sys
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
TMP_DIR = '/dev/shm/eval_recall'
MAX_NODES = 32
EMBED_DIM = 2 * MAX_NODES + 2 + 2  # 68
A2_TOP_K = 1000
TOP_K_EVAL = [1, 3, 5, 10, 50, 100, 500, 1000]
N_WORKERS = 80

# Worker globals (set via COW before fork, NOT pickled)
_g_adj_rows = None
_g_adj_cols = None
_g_degree = None
_g_dist_matrix = None


def _init_worker():
    pass  # Globals already set via COW


def _process_anchor(args):
    """Build subgraphs for one anchor. Write to memmap. Return (anchor_idx, pos_idx)."""
    anchor_idx, u, v, top_cands = args

    NC = A2_TOP_K + 1
    inp = np.memmap(f'{TMP_DIR}/inp.npy', dtype='float16', mode='r+',
                    shape=(NC, MAX_NODES, EMBED_DIM),
                    offset=anchor_idx * NC * MAX_NODES * EMBED_DIM * 2)
    nids = np.memmap(f'{TMP_DIR}/nids.npy', dtype='int64', mode='r+',
                     shape=(NC, MAX_NODES),
                     offset=anchor_idx * NC * MAX_NODES * 8)

    rows, cols, deg, dist_mat = _g_adj_rows, _g_adj_cols, _g_degree, _g_dist_matrix
    pos_idx = -1

    for j, c in enumerate(top_cands):
        if c == v:
            pos_idx = j
        nu = cols[rows[u]:rows[u + 1]]
        nv = cols[rows[c]:rows[c + 1]]
        nu_set = set(nu.tolist())
        nv_set = set(nv.tolist())
        shared = sorted(nu_set & nv_set, key=lambda x: -deg[x])
        only_u = sorted(nu_set - nv_set - {c}, key=lambda x: -deg[x])
        only_v = sorted(nv_set - nu_set - {u}, key=lambda x: -deg[x])
        nodes = [u, c]
        rem = MAX_NODES - 2
        for lst in [shared, only_u, only_v]:
            for n in lst:
                if rem <= 0:
                    break
                nodes.append(n)
                rem -= 1
            if rem <= 0:
                break
        nset = set(nodes)
        lmap = {nd: i for i, nd in enumerate(nodes)}
        for i_nd, node_id in enumerate(nodes):
            if i_nd < MAX_NODES:
                inp[j, i_nd, i_nd] = 1.0
            inp[j, i_nd, 2 * MAX_NODES] = 1.0
            inp[j, i_nd, 2 * MAX_NODES + 1] = min(int(dist_mat[u, node_id]), 10) / 10
            inp[j, i_nd, 2 * MAX_NODES + 2] = min(int(dist_mat[c, node_id]), 10) / 10
        for a in nodes:
            for b in cols[rows[a]:rows[a + 1]]:
                if b in nset:
                    ib = lmap[b]
                    if ib < MAX_NODES:
                        inp[j, lmap[a], MAX_NODES + ib] = 1.0
        nids[j, :min(len(nodes), MAX_NODES)] = nodes[:MAX_NODES]

    if pos_idx == -1:
        pos_idx = top_cands.index(v) if v in top_cands else 0
    return anchor_idx, pos_idx


if __name__ == '__main__':
    t0 = time.time()
    os.makedirs(TMP_DIR, exist_ok=True)

    # ---- Phase 1: CPU-only data loading (no torch, no Pool) ----
    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    adj_rows = adj.indptr.astype(np.int64)
    adj_cols = adj.indices.astype(np.int64)
    degree = np.array(adj.sum(axis=1)).flatten().astype(np.float32)

    # Cache A² to disk to avoid recomputing every run
    a2_cache = f'{DATA_DIR}/adj2_lil_cache.npz'
    if os.path.exists(a2_cache):
        print("Loading cached A²...", flush=True)
        adj2_lil = sp.load_npz(a2_cache).tolil()
        print(f"  loaded, nnz={adj2_lil.nnz}", flush=True)
    else:
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

    # A² baseline (numpy-optimized)
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

    # ---- Phase 2: Parallel subgraph building (Pool starts HERE) ----
    print(f"\n=== PENCIL (memmap, {N_WORKERS} workers) ===", flush=True)

    N_ANCHORS = len(val_pairs)
    NC = A2_TOP_K + 1
    inp_size = N_ANCHORS * NC * MAX_NODES * EMBED_DIM * 2  # float16
    nids_size = N_ANCHORS * NC * MAX_NODES * 8  # int64
    print(f"  pre-allocating memmap ({inp_size/1e9:.1f}GB + {nids_size/1e9:.1f}GB)...", flush=True)

    # Fast pre-allocation: truncate files to size (sparse, no actual zeroing)
    for fname, sz in [('inp.npy', inp_size), ('nids.npy', nids_size)]:
        path = f'{TMP_DIR}/{fname}'
        with open(path, 'wb') as f:
            f.truncate(sz)

    # Set globals BEFORE Pool (COW, no pickle overhead)
    _g_adj_rows = adj_rows
    _g_adj_cols = adj_cols
    _g_degree = degree
    _g_dist_matrix = dist_matrix

    # Build work_items BEFORE freeing adj2_lil
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

    # Free large objects before spawning workers
    del adj, adj2_lil, dist_matrix

    t_build = time.time()
    pos_idx_map = {}
    with Pool(N_WORKERS, initializer=_init_worker) as pool:
        for anchor_idx, pos_idx in pool.imap_unordered(
                _process_anchor, work_items, chunksize=10):
            pos_idx_map[anchor_idx] = pos_idx
    print(f"  subgraph build: {time.time()-t_build:.1f}s", flush=True)

    # ---- Phase 3: GPU scoring (import torch HERE, workers already done) ----
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
