"""
Recall@K: A² baseline vs PENCIL model.
Workers write to shared memmap files (zero IPC overhead).
"""
import os, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from multiprocessing import Pool

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
TMP_DIR = '/tmp/eval_recall'
MAX_NODES = 32
EMBED_DIM = 2 * MAX_NODES + 2 + 2  # 68
SCIBERT_DIM = 768
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
DIST_BINS = 10
DIST_EMBED_DIM = 32
A2_TOP_K = 1000
TOP_K_EVAL = [1, 3, 5, 10, 50, 100, 500, 1000]
N_WORKERS = 80

_g_adj_rows = None
_g_adj_cols = None
_g_degree = None
_g_dist_matrix = None


def _init_worker(adj_rows, adj_cols, degree, dm):
    global _g_adj_rows, _g_adj_cols, _g_degree, _g_dist_matrix
    _g_adj_rows = adj_rows
    _g_adj_cols = adj_cols
    _g_degree = degree
    _g_dist_matrix = dm


def _process_anchor(args):
    """Build subgraphs for one anchor. Write to memmap. Return nothing."""
    anchor_idx, u, v, top_cands = args
    n_cands = len(top_cands)

    inp = np.memmap(f'{TMP_DIR}/inp.npy', dtype='float16', mode='r+',
                    shape=(4134, A2_TOP_K + 1, MAX_NODES, EMBED_DIM),
                    offset=anchor_idx * (A2_TOP_K + 1) * MAX_NODES * EMBED_DIM * 2)
    nids = np.memmap(f'{TMP_DIR}/nids.npy', dtype='int64', mode='r+',
                     shape=(4134, A2_TOP_K + 1, MAX_NODES),
                     offset=anchor_idx * (A2_TOP_K + 1) * MAX_NODES * 8)

    rows, cols, deg, dist_mat = _g_adj_rows, _g_adj_cols, _g_degree, _g_dist_matrix

    for j, c in enumerate(top_cands):
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
                if rem <= 0: break
                nodes.append(n); rem -= 1
            if rem <= 0: break
        nset = set(nodes); lmap = {nd: i for i, nd in enumerate(nodes)}
        for i_nd, node_id in enumerate(nodes):
            if i_nd < MAX_NODES: inp[j, i_nd, i_nd] = 1.0
            inp[j, i_nd, 2 * MAX_NODES] = 1.0
            inp[j, i_nd, 2 * MAX_NODES + 1] = min(int(dist_mat[u, node_id]), DIST_BINS) / DIST_BINS
            inp[j, i_nd, 2 * MAX_NODES + 2] = min(int(dist_mat[c, node_id]), DIST_BINS) / DIST_BINS
        for a in nodes:
            for b in cols[rows[a]:rows[a + 1]]:
                if b in nset:
                    ib = lmap[b]
                    if ib < MAX_NODES: inp[j, lmap[a], MAX_NODES + ib] = 1.0
        nids[j, :min(len(nodes), MAX_NODES)] = nodes[:MAX_NODES]

    return anchor_idx, top_cands.index(v)


class PencilModel(nn.Module):
    def __init__(self, scibert_embeddings):
        super().__init__()
        self.register_buffer('scibert_table', scibert_embeddings)
        self.scibert_proj = nn.Linear(SCIBERT_DIM, HIDDEN // 2)
        self.struct_proj = nn.Linear(EMBED_DIM, HIDDEN // 2)
        self.dist_embed = nn.Embedding((DIST_BINS + 2) * (DIST_BINS + 2), DIST_EMBED_DIM)
        self.combine = nn.Linear(HIDDEN // 2 + HIDDEN // 2 + DIST_EMBED_DIM, HIDDEN)
        self.layers = nn.ModuleList()
        for _ in range(N_LAYERS):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(HIDDEN, N_HEADS, batch_first=True, dropout=0.1),
                'ln1': nn.LayerNorm(HIDDEN),
                'ffn': nn.Sequential(nn.Linear(HIDDEN, HIDDEN * 4), nn.GELU(), nn.Linear(HIDDEN * 4, HIDDEN), nn.Dropout(0.1)),
                'ln2': nn.LayerNorm(HIDDEN),
                'mp': nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU()),
            }))
        self.ln_f = nn.LayerNorm(HIDDEN)
        self.head = nn.Sequential(nn.Linear(HIDDEN * 2, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

    def forward(self, inp, node_ids):
        B, N, _ = inp.shape
        adj = inp[:, :N, MAX_NODES:MAX_NODES + N]
        deg = adj.sum(dim=2, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        dist_u_bin = (inp[:, :, 2 * MAX_NODES + 1] * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        dist_v_bin = (inp[:, :, 2 * MAX_NODES + 2] * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        dist_emb = self.dist_embed(dist_u_bin * (DIST_BINS + 2) + dist_v_bin)
        x_raw = inp.clone()
        x_raw[:, :, 2 * MAX_NODES + 1] = 0; x_raw[:, :, 2 * MAX_NODES + 2] = 0
        struct_emb = self.struct_proj(x_raw)
        nids_flat = node_ids.reshape(-1).clamp(0, self.scibert_table.shape[0] - 1)
        scibert_emb = self.scibert_proj(self.scibert_table[nids_flat].reshape(B, N, SCIBERT_DIM))
        pad_mask = (inp[:, :, :2 * MAX_NODES + 1].abs().sum(dim=2) == 0)
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


if __name__ == '__main__':
    t0 = time.time()
    os.makedirs(TMP_DIR, exist_ok=True)

    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    adj_rows = adj.indptr.astype(np.int64)
    adj_cols = adj.indices.astype(np.int64)
    degree = np.array(adj.sum(axis=1)).flatten().astype(np.float32)

    print("Computing A²...", flush=True)
    adj2 = adj.dot(adj)
    adj2_lil = adj2.tolil()

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
        cands = list(row.rows[0])
        scores = list(row.data[0])
        if v not in cands:
            cands.append(v); scores.append(0)
        sorted_cands = [c for _, c in sorted(zip(scores, cands), key=lambda x: -x[0])]
        a2_ranks.append(sorted_cands.index(v) + 1)
    a2_ranks = np.array(a2_ranks)
    for k in TOP_K_EVAL:
        print(f"  A² Recall@{k}: {(a2_ranks <= k).mean():.4f}", flush=True)
    print(f"  A² Median: {np.median(a2_ranks):.0f}, MRR: {(1.0/a2_ranks).mean():.4f}", flush=True)

    # PENCIL evaluation
    print(f"\n=== PENCIL (memmap, {N_WORKERS} workers) ===", flush=True)

    # Pre-allocate memmap files
    N_ANCHORS = len(val_pairs)
    NC = A2_TOP_K + 1  # max candidates per anchor
    print(f"  pre-allocating memmap ({N_ANCHORS}×{NC}×{MAX_NODES}×{EMBED_DIM})...", flush=True)
    inp_mm = np.memmap(f'{TMP_DIR}/inp.npy', dtype='float16', mode='w+',
                       shape=(N_ANCHORS, NC, MAX_NODES, EMBED_DIM))
    nids_mm = np.memmap(f'{TMP_DIR}/nids.npy', dtype='int64', mode='w+',
                        shape=(N_ANCHORS, NC, MAX_NODES))
    inp_mm[:] = 0
    nids_mm[:] = 0
    del inp_mm, nids_mm

    # Prepare work items
    work_items = []
    pos_idx_map = {}
    for idx, (u, v) in enumerate(val_pairs):
        row = adj2_lil[u]
        cands = list(row.rows[0])
        scores_a2 = list(row.data[0])
        paired = sorted(zip(scores_a2, cands), key=lambda x: -x[0])
        top_cands = [c for _, c in paired[:A2_TOP_K]]
        if v not in top_cands:
            top_cands.append(v)
        work_items.append((idx, u, v, top_cands))

    # Build subgraphs in parallel (writes to memmap, returns only pos_idx)
    t_build = time.time()
    with Pool(N_WORKERS, initializer=_init_worker,
              initargs=(adj_rows, adj_cols, degree, dist_matrix)) as pool:
        for anchor_idx, pos_idx in pool.imap_unordered(
                _process_anchor, work_items, chunksize=10):
            pos_idx_map[anchor_idx] = pos_idx
    print(f"  subgraph build: {time.time()-t_build:.1f}s", flush=True)

    # Load memmap for reading
    inp_all = np.memmap(f'{TMP_DIR}/inp.npy', dtype='float16', mode='r',
                        shape=(N_ANCHORS, NC, MAX_NODES, EMBED_DIM))
    nids_all = np.memmap(f'{TMP_DIR}/nids.npy', dtype='int64', mode='r',
                         shape=(N_ANCHORS, NC, MAX_NODES))

    # Load model and score
    device = torch.device('cuda')
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilModel(scibert).to(device)
    model.load_state_dict(torch.load(f'{DATA_DIR}/pencil_v2_best.pt', weights_only=True))
    model.eval()

    pencil_ranks = []
    t_score = time.time()
    for idx in range(N_ANCHORS):
        n_cands = len(work_items[idx][3])
        inp_t = torch.from_numpy(np.array(inp_all[idx, :n_cands]).astype(np.float32)).to(device)
        nids_t = torch.from_numpy(np.array(nids_all[idx, :n_cands])).to(device)
        with torch.no_grad():
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

    # Cleanup
    os.remove(f'{TMP_DIR}/inp.npy')
    os.remove(f'{TMP_DIR}/nids.npy')

    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
    print("Done.", flush=True)
