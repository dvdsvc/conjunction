"""
Recall@K eval v4: A² top-5000 + PENCIL v4 re-rank.
Key optimizations vs v3:
  1. Thread-parallel anchor processing (16 workers, each builds all subgraphs for one anchor)
  2. Single GPU forward pass per anchor (all ~5000 candidates at once) instead of 1-at-a-time
  3. Self-contained val pair extraction from train.jsonl
"""
import os, time, re, sys
import numpy as np
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor
from numba import njit

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32
EMBED_DIM = 68
A2_TOP_K = 5000
TOP_K_EVAL = [1, 3, 5, 10, 50, 100, 500, 1000, 2000, 3000, 5000]
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
def _build_subgraph(u, c, indptr, indices, deg, dist_mat, out_inp, out_nids):
    nu = indices[indptr[u]:indptr[u + 1]]
    nv = indices[indptr[c]:indptr[c + 1]]
    nu_sorted = nu.copy(); nv_sorted = nv.copy()
    _sort_by_degree_desc(nu_sorted, deg); _sort_by_degree_desc(nv_sorted, deg)
    nu_for_set = np.sort(nu); nv_for_set = np.sort(nv)
    shared = _sorted_intersect_sorted(nu_for_set, nv_for_set)
    _sort_by_degree_desc(shared, deg)
    selected = np.empty(MAX_NODES, dtype=np.int64); n_sel = 0
    selected[n_sel] = u; n_sel += 1; selected[n_sel] = c; n_sel += 1
    for i in range(len(shared)):
        if n_sel >= MAX_NODES: break
        selected[n_sel] = shared[i]; n_sel += 1
    if n_sel < MAX_NODES:
        for i in range(len(nu_sorted)):
            if n_sel >= MAX_NODES: break
            node = nu_sorted[i]
            if node == c or _in_sorted(shared, node): continue
            selected[n_sel] = node; n_sel += 1
    if n_sel < MAX_NODES:
        for i in range(len(nv_sorted)):
            if n_sel >= MAX_NODES: break
            node = nv_sorted[i]
            if node == u or _in_sorted(shared, node): continue
            already = False
            for k in range(n_sel):
                if selected[k] == node: already = True; break
            if already: continue
            selected[n_sel] = node; n_sel += 1
    for i in range(MAX_NODES):
        for j in range(EMBED_DIM): out_inp[i, j] = 0.0
        out_nids[i] = 0
    for i in range(n_sel):
        if i < MAX_NODES: out_inp[i, i] = 1.0
        out_inp[i, 2 * MAX_NODES] = 1.0
        node_id = selected[i]
        out_inp[i, 2 * MAX_NODES + 1] = min(float(dist_mat[u, node_id]), float(DIST_BINS)) / float(DIST_BINS)
        out_inp[i, 2 * MAX_NODES + 2] = min(float(dist_mat[c, node_id]), float(DIST_BINS)) / float(DIST_BINS)
    sel_sorted = selected[:n_sel].copy(); sel_sorted.sort()
    for i in range(n_sel):
        a = selected[i]
        for j in range(indptr[a], indptr[a + 1]):
            b = indices[j]
            if _in_sorted(sel_sorted, b):
                for lb in range(n_sel):
                    if selected[lb] == b:
                        if lb < MAX_NODES: out_inp[i, MAX_NODES + lb] = 1.0
                        break
    for i in range(min(n_sel, MAX_NODES)):
        out_nids[i] = selected[i]


def _score_anchor(args):
    """Build all subgraphs for one anchor, score on GPU, return rank."""
    ai, u, v, cands, indptr, indices, deg, dist_mat, model, device = args
    NC = len(cands)

    # Build all subgraphs
    inp = np.zeros((NC, MAX_NODES, EMBED_DIM), dtype=np.float16)
    nids = np.zeros((NC, MAX_NODES), dtype=np.int64)
    tmp_inp = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
    tmp_nids = np.zeros(MAX_NODES, dtype=np.int64)
    pos_idx = 0
    for j, c in enumerate(cands):
        if c == v: pos_idx = j
        _build_subgraph(u, c, indptr, indices, deg, dist_mat, tmp_inp, tmp_nids)
        inp[j] = tmp_inp.astype(np.float16)
        nids[j] = tmp_nids

    # Single GPU pass: score all candidates at once
    inp_t = torch.from_numpy(inp.astype(np.float32)).to(device)
    nids_t = torch.from_numpy(nids).to(device)
    with torch.inference_mode():
        scores = model(inp_t, nids_t).cpu().numpy()

    pos_score = scores[pos_idx]
    rank = int((scores > pos_score).sum() + 1)
    return ai, rank


def _extract_val_pairs(data_dir):
    """Extract val pairs from train.jsonl using same 80/20 split as precompute."""
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


if __name__ == '__main__':
    t0 = time.time()

    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    adj_rows = adj.indptr.astype(np.int64)
    adj_cols = adj.indices.astype(np.int64)
    degree = np.array(adj.sum(axis=1)).flatten().astype(np.float32)

    print("Computing A²...", flush=True)
    t_a2 = time.time()
    adj2 = adj.dot(adj)
    adj2_lil = adj2.tolil(); del adj2
    print(f"  A² done: {time.time()-t_a2:.1f}s, nnz={adj2_lil.nnz}", flush=True)

    print("Loading distance matrix...", flush=True)
    dist_matrix = np.load(f'{DATA_DIR}/dist_matrix.npy')

    print("Extracting val pairs from train.jsonl...", flush=True)
    val_pairs = _extract_val_pairs(DATA_DIR)
    N_ANCHORS = len(val_pairs)
    print(f"  val pairs: {N_ANCHORS}, setup: {time.time()-t0:.1f}s", flush=True)

    # A² baseline
    print(f"\n=== A² baseline (top-{A2_TOP_K}) ===", flush=True)
    a2_ranks = []
    a2_in_topk = 0
    for u, v in val_pairs:
        row = adj2_lil[u]
        cands = np.array(row.rows[0])
        scores = np.array(row.data[0])
        order = np.argsort(-scores)
        sorted_cands = cands[order]
        pos = np.where(sorted_cands == v)[0]
        if len(pos) > 0:
            rank = int(pos[0]) + 1
            a2_ranks.append(rank)
            if rank <= A2_TOP_K: a2_in_topk += 1
        else:
            a2_ranks.append(len(sorted_cands) + 1)
    a2_ranks = np.array(a2_ranks)
    print(f"  A² positive in top-{A2_TOP_K}: {a2_in_topk}/{N_ANCHORS} ({a2_in_topk/N_ANCHORS:.4f})", flush=True)
    for k in TOP_K_EVAL:
        print(f"  A² Recall@{k}: {(a2_ranks <= k).mean():.4f}", flush=True)
    print(f"  A² Median: {np.median(a2_ranks):.0f}, MRR: {(1.0/a2_ranks).mean():.4f}", flush=True)

    # Build candidate lists
    print(f"\n=== Preparing A² top-{A2_TOP_K} candidates ===", flush=True)
    anchor_cands = []
    for idx, (u, v) in enumerate(val_pairs):
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
    print("  compiling Numba...", flush=True)
    t_jit = time.time()
    tmp_f = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
    tmp_n = np.zeros(MAX_NODES, dtype=np.int64)
    _build_subgraph(0, 1, adj_rows, adj_cols, degree, dist_matrix, tmp_f, tmp_n)
    print(f"  JIT: {time.time()-t_jit:.1f}s", flush=True)

    # Load PENCIL model
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
            x_raw = inp.clone(); x_raw[:, :, 65] = 0; x_raw[:, :, 66] = 0
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

    device = torch.device('cuda')
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilModel(scibert).to(device)
    ckpt_path = f'{DATA_DIR}/pencil_v4_best.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = f'{DATA_DIR}/pencil_v3_best.pt'
        print(f"  v4 checkpoint not found, using v3", flush=True)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    print(f"  model on {torch.cuda.get_device_name(0)}, ckpt={ckpt_path}", flush=True)

    # PENCIL re-rank: thread-parallel anchors, single GPU pass per anchor
    print(f"\n=== PENCIL v4 re-rank (top-{A2_TOP_K}, {N_WORKERS} workers) ===", flush=True)
    pencil_ranks = np.zeros(N_ANCHORS, dtype=np.int64)
    done = 0
    t_re = time.time()

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        args_list = [
            (ai, val_pairs[ai][0], val_pairs[ai][1], anchor_cands[ai],
             adj_rows, adj_cols, degree, dist_matrix, model, device)
            for ai in range(N_ANCHORS)
        ]
        for ai, rank in ex.map(_score_anchor, args_list):
            pencil_ranks[ai] = rank
            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t_re
                eta = elapsed / done * (N_ANCHORS - done)
                print(f"  {done}/{N_ANCHORS} done ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    dt_re = time.time() - t_re
    print(f"  re-rank total: {dt_re:.0f}s ({dt_re/N_ANCHORS:.2f}s/anchor)", flush=True)

    # End-to-end Recall@K
    e2e_ranks = np.where(a2_ranks <= A2_TOP_K, pencil_ranks, a2_ranks)

    print(f"\n=== E2E (A² top-{A2_TOP_K} + PENCIL v4 re-rank) ===", flush=True)
    for k in TOP_K_EVAL:
        print(f"  E2E Recall@{k}: {(e2e_ranks <= k).mean():.4f}", flush=True)
    print(f"  E2E Median: {np.median(e2e_ranks):.0f}, MRR: {(1.0/e2e_ranks).mean():.4f}", flush=True)

    # PENCIL-only (within A² top-5000 subset)
    mask = a2_ranks <= A2_TOP_K
    pencil_sub = pencil_ranks[mask]
    print(f"\n=== PENCIL v4 (within A² top-{A2_TOP_K}, n={len(pencil_sub)}) ===", flush=True)
    for k in [1, 3, 5, 10, 50, 100, 500, 1000, 2000, 3000, 5000]:
        print(f"  PENCIL Recall@{k}: {(pencil_sub <= k).mean():.4f}", flush=True)
    print(f"  PENCIL Median: {np.median(pencil_sub):.0f}, MRR: {(1.0/pencil_sub).mean():.4f}", flush=True)

    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
    print("Done.", flush=True)
