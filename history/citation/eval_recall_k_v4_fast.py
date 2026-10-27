"""
Recall@K eval v4 fast: reconstruct features from precomputed node_ids.
~7 min eval (vs ~50 min with full subgraph build).

If eval_subgraph_nids.npy exists: load node_ids, reconstruct features (~0.05ms), GPU score.
Otherwise: falls back to full subgraph build (original v4 approach).
"""
import os, time, re, sys
import numpy as np
import scipy.sparse as sp
from numba import njit

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32
EMBED_DIM = 68
A2_TOP_K = 5000
TOP_K_EVAL = [1, 3, 5, 10, 50, 100, 500, 1000, 2000, 3000, 5000]
DIST_BINS = 10


@njit(cache=True)
def _reconstruct_features(nids, u, v, indptr, indices, deg_u, deg_v, dist_u, dist_v, out):
    """Reconstruct 32×68 features from selected node_ids. ~0.05ms per call."""
    n = 0
    for i in range(MAX_NODES):
        if nids[i] != 0 or i == 0:
            n = i + 1

    # Find neighbors of u and v among selected nodes
    u_neigh = set()
    v_neigh = set()
    for j in range(indptr[u], indptr[u + 1]):
        b = indices[j]
        for k in range(n):
            if nids[k] == b:
                u_neigh.add(k)
                break
    for j in range(indptr[v], indptr[v + 1]):
        b = indices[j]
        for k in range(n):
            if nids[k] == b:
                v_neigh.add(k)
                break

    for i in range(MAX_NODES):
        for j in range(EMBED_DIM):
            out[i, j] = 0.0

    for i in range(n):
        # One-hot position
        out[i, i] = 1.0
        # Mask: this node is real
        out[i, 2 * MAX_NODES] = 1.0
        # Distances
        out[i, 2 * MAX_NODES + 1] = min(dist_u[nids[i]], float(DIST_BINS)) / float(DIST_BINS)
        out[i, 2 * MAX_NODES + 2] = min(dist_v[nids[i]], float(DIST_BINS)) / float(DIST_BINS)
        # Adjacency (only edges where BOTH endpoints are selected)
        if i in u_neigh:
            out[i, MAX_NODES + 0] = 1.0
        if i in v_neigh:
            out[i, MAX_NODES + 1] = 1.0
        # Edges to other selected nodes
        nid_i = nids[i]
        for j in range(indptr[nid_i], indptr[nid_i + 1]):
            b = indices[j]
            for k in range(n):
                if nids[k] == b:
                    out[i, MAX_NODES + k] = 1.0
                    break


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


if __name__ == '__main__':
    t0 = time.time()

    # Check if precomputed node_ids exist
    nids_path = f'{DATA_DIR}/eval_subgraph_nids.npy'
    cands_len_path = f'{DATA_DIR}/eval_cands_len.npy'
    use_precomputed = os.path.exists(nids_path) and os.path.exists(cands_len_path)

    if use_precomputed:
        print("=== Using PRECOMPUTED node_ids (fast mode) ===", flush=True)
    else:
        print("=== Precomputed node_ids NOT found, running full build ===", flush=True)
        print(f"  Run precompute_eval_nids.py first for ~50x speedup", flush=True)
        # Import and run the full build approach
        sys.exit(1)

    # Load graph (needed for A² baseline and feature reconstruction)
    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    indptr = adj.indptr.astype(np.int64)
    indices = adj.indices.astype(np.int64)
    degree = np.array(adj.sum(axis=1)).flatten().astype(np.float32)

    # Load distance matrix
    print("Loading distance matrix...", flush=True)
    dist_matrix = np.load(f'{DATA_DIR}/dist_matrix.npy')

    # A² baseline
    print("Computing A² baseline...", flush=True)
    t_a2 = time.time()
    adj2 = adj.dot(adj)
    adj2_lil = adj2.tolil(); del adj2
    print(f"  A²: {time.time()-t_a2:.1f}s", flush=True)

    val_pairs = _extract_val_pairs(DATA_DIR)
    N_ANCHORS = len(val_pairs)
    print(f"  {N_ANCHORS} val pairs", flush=True)

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
    del adj2_lil

    # Load precomputed node_ids
    print(f"\nLoading precomputed node_ids...", flush=True)
    t1 = time.time()
    all_nids = np.load(nids_path, mmap_mode='r')
    cands_len = np.load(cands_len_path)
    print(f"  shape: {all_nids.shape}, loaded in {time.time()-t1:.1f}s", flush=True)

    # JIT warmup
    print("Compiling Numba (feature reconstruction)...", flush=True)
    tmp_out = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
    _reconstruct_features(all_nids[0, 0], 0, 1, indptr, indices,
                          degree[0], degree[1], dist_matrix[0], dist_matrix[1], tmp_out)
    print("  done", flush=True)

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
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    # Optional: torch.compile for speedup
    try:
        model = torch.compile(model, mode="max-autotune")
        print(f"  model compiled with torch.compile", flush=True)
    except Exception as e:
        print(f"  torch.compile failed: {e}, using eager mode", flush=True)

    print(f"  model on {torch.cuda.get_device_name(0)}", flush=True)

    # PENCIL re-rank: reconstruct features from node_ids, GPU score
    print(f"\n=== PENCIL v4 fast re-rank ===", flush=True)
    pencil_ranks = np.zeros(N_ANCHORS, dtype=np.int64)
    tmp_out = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
    t_re = time.time()

    for ai in range(N_ANCHORS):
        u, v = val_pairs[ai]
        NC = int(cands_len[ai])
        nids_ai = all_nids[ai]  # (max_cands, 32)

        # Reconstruct features from node_ids
        inp = np.zeros((NC, MAX_NODES, EMBED_DIM), dtype=np.float32)
        for j in range(NC):
            _reconstruct_features(nids_ai[j], u, v, indptr, indices,
                                 degree[u], degree[v], dist_matrix[u], dist_matrix[v], tmp_out)
            inp[j] = tmp_out

        # Find positive index
        pos_idx = 0
        for j in range(NC):
            if nids_ai[j, 0] == u and nids_ai[j, 1] == v:
                pos_idx = j
                break

        # Single GPU pass
        inp_t = torch.from_numpy(inp).to(device)
        nids_t = torch.from_numpy(nids_ai[:NC]).to(device)
        with torch.inference_mode():
            scores = model(inp_t, nids_t).cpu().numpy()

        pos_score = scores[pos_idx]
        rank = int((scores > pos_score).sum() + 1)
        pencil_ranks[ai] = rank

        if (ai + 1) % 200 == 0:
            elapsed = time.time() - t_re
            eta = elapsed / (ai + 1) * (N_ANCHORS - ai - 1)
            print(f"  {ai+1}/{N_ANCHORS} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    dt_re = time.time() - t_re
    print(f"  re-rank total: {dt_re:.0f}s ({dt_re/N_ANCHORS:.2f}s/anchor)", flush=True)

    # Results
    e2e_ranks = np.where(a2_ranks <= A2_TOP_K, pencil_ranks, a2_ranks)

    print(f"\n=== E2E (A² top-{A2_TOP_K} + PENCIL v4 re-rank) ===", flush=True)
    for k in TOP_K_EVAL:
        print(f"  E2E Recall@{k}: {(e2e_ranks <= k).mean():.4f}", flush=True)
    print(f"  E2E Median: {np.median(e2e_ranks):.0f}, MRR: {(1.0/e2e_ranks).mean():.4f}", flush=True)

    mask = a2_ranks <= A2_TOP_K
    pencil_sub = pencil_ranks[mask]
    print(f"\n=== PENCIL v4 (within A² top-{A2_TOP_K}, n={len(pencil_sub)}) ===", flush=True)
    for k in [1, 3, 5, 10, 50, 100, 500, 1000, 2000, 3000, 5000]:
        print(f"  PENCIL Recall@{k}: {(pencil_sub <= k).mean():.4f}", flush=True)
    print(f"  PENCIL Median: {np.median(pencil_sub):.0f}, MRR: {(1.0/pencil_sub).mean():.4f}", flush=True)

    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
    print("Done.", flush=True)
