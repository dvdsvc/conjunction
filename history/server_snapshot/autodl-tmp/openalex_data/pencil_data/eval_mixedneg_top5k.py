"""
Recall@K eval for PENCIL v5 mixedneg on A² top-5000.
Optimized: multi-process Numba batch build + pre-sorted neighbors + batched GPU scoring.
"""
import os, time
import numpy as np
import scipy.sparse as sp
from numba import njit
import torch
import torch.nn as nn
from multiprocessing import Pool

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32
EMBED_DIM = 68
SCIBERT_DIM = 768
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
DIST_BINS = 10
DIST_EMBED_DIM = 32
A2_TOP_K = 5000
TOP_K_EVAL = [1, 3, 5, 10, 50, 100, 500, 1000, 2000, 3000, 5000]
N_WORKERS = 4
OUTER_BATCH = 100
INNER_BATCH = 1000


# ── Numba helpers ──────────────────────────────────────────────────────
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
def _sorted_intersect(a, b):
    r = np.empty(min(len(a), len(b)), dtype=np.int64)
    c = 0; j = 0
    for i in range(len(a)):
        while j < len(b) and b[j] < a[i]: j += 1
        if j < len(b) and b[j] == a[i]:
            r[c] = a[i]; c += 1
    return r[:c]


@njit(nogil=True, cache=True)
def _build_batch(us, vs, offsets, cand_flat,
                 indptr, indices, deg, dist_mat,
                 nb_flat, nb_off,
                 out_inp, out_nids):
    """Build subgraphs for a chunk of anchors.
    Pre-sorted neighbors passed via nb_flat/nb_off (computed once per worker).
    """
    n = len(us)
    MN = MAX_NODES; ED = EMBED_DIM; DB = float(DIST_BINS)

    for ai in range(n):
        u = us[ai]; v = vs[ai]
        s = offsets[ai]; e = offsets[ai + 1]

        # u's neighbors: sorted by degree (precomputed)
        u_s = nb_off[u]; u_e = nb_off[u + 1]
        nu = nb_flat[u_s:u_e]
        # Also plain sorted for intersection
        nu_plain = nu.copy(); nu_plain.sort()

        for ci in range(e - s):
            c = cand_flat[s + ci]

            # c's neighbors: sorted by degree (precomputed)
            c_s = nb_off[c]; c_e = nb_off[c + 1]
            nv = nb_flat[c_s:c_e]
            nv_plain = nv.copy(); nv_plain.sort()

            # Shared neighbors (sorted by degree after intersection)
            shared = _sorted_intersect(nu_plain, nv_plain)

            # Node selection
            sel = np.empty(MN, dtype=np.int64); ns = 0
            sel[ns] = u; ns += 1
            sel[ns] = c; ns += 1
            for i in range(len(shared)):
                if ns >= MN: break
                sel[ns] = shared[i]; ns += 1
            if ns < MN:
                for i in range(len(nu)):
                    if ns >= MN: break
                    nd = nu[i]
                    if nd == c or _in_sorted(shared, nd): continue
                    sel[ns] = nd; ns += 1
            if ns < MN:
                for i in range(len(nv)):
                    if ns >= MN: break
                    nd = nv[i]
                    if nd == u or _in_sorted(shared, nd): continue
                    dup = False
                    for k in range(ns):
                        if sel[k] == nd: dup = True; break
                    if dup: continue
                    sel[ns] = nd; ns += 1

            # Fill features
            idx = s + ci
            for i in range(MN):
                for j in range(ED):
                    out_inp[idx, i, j] = 0.0
                out_nids[idx, i] = 0

            for i in range(ns):
                if i < MN:
                    out_inp[idx, i, i] = 1.0
                out_inp[idx, i, 2 * MN] = 1.0
                nd = sel[i]
                out_inp[idx, i, 2*MN+1] = min(float(dist_mat[u, nd]), DB) / DB
                out_inp[idx, i, 2*MN+2] = min(float(dist_mat[c, nd]), DB) / DB

            ss = sel[:ns].copy(); ss.sort()
            for i in range(ns):
                a = sel[i]
                for j in range(indptr[a], indptr[a + 1]):
                    b = indices[j]
                    if _in_sorted(ss, b):
                        for lb in range(ns):
                            if sel[lb] == b:
                                if lb < MN:
                                    out_inp[idx, i, MN + lb] = 1.0
                                break

            for i in range(min(ns, MN)):
                out_nids[idx, i] = sel[i]


# ── Worker process ─────────────────────────────────────────────────────
def _worker_init():
    """Each worker loads heavy arrays from disk once."""
    global _dist, _indptr, _indices, _deg, _nb_flat, _nb_off
    _dist = np.load(f'{DATA_DIR}/dist_matrix.npy')
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    _indptr = adj.indptr.astype(np.int64)
    _indices = adj.indices.astype(np.int64)
    _deg = np.array(adj.sum(axis=1)).flatten().astype(np.float32)
    # Pre-sort each node's neighbors by degree (descending)
    n_nodes = len(_deg)
    lists = []
    for i in range(n_nodes):
        nb = _indices[_indptr[i]:_indptr[i + 1]].copy()
        # insertion sort by degree desc
        for a in range(1, len(nb)):
            key = nb[a]; kd = _deg[key]; b = a - 1
            while b >= 0 and _deg[nb[b]] < kd:
                nb[b + 1] = nb[b]; b -= 1
            nb[b + 1] = key
        lists.append(nb)
    lens = np.array([len(x) for x in lists], dtype=np.int64)
    _nb_off = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(lens, out=_nb_off[1:])
    _nb_flat = np.empty(_nb_off[-1], dtype=np.int64)
    for i in range(n_nodes):
        _nb_flat[_nb_off[i]:_nb_off[i + 1]] = lists[i]


def _worker_process(chunk):
    """Process a chunk of anchors. Writes results to temp file, returns (path, pos_indices)."""
    import tempfile
    anchors, cand_lists = chunk
    n = len(anchors)
    us = np.array([a[0] for a in anchors], dtype=np.int64)
    vs = np.array([a[1] for a in anchors], dtype=np.int64)

    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, c in enumerate(cand_lists):
        offsets[i + 1] = offsets[i] + len(c)
    flat = np.empty(offsets[-1], dtype=np.int64)
    for i, c in enumerate(cand_lists):
        flat[offsets[i]:offsets[i + 1]] = c

    pos_indices = []
    for i in range(n):
        try:
            pos_indices.append(cand_lists[i].index(vs[i]))
        except ValueError:
            pos_indices.append(0)

    total_c = offsets[-1]
    out_inp = np.zeros((total_c, MAX_NODES, EMBED_DIM), dtype=np.float32)
    out_nids = np.zeros((total_c, MAX_NODES), dtype=np.int64)

    _build_batch(us, vs, offsets, flat,
                 _indptr, _indices, _deg, _dist,
                 _nb_flat, _nb_off,
                 out_inp, out_nids)

    # Write to temp file to avoid pickling large arrays
    path = tempfile.mktemp(suffix='.npz', prefix='/root/autodl-tmp/eval_chunk_')
    np.savez(path, inp=out_inp, nids=out_nids, pos=np.array(pos_indices, dtype=np.int64))
    del out_inp, out_nids
    return path


# ── Model ──────────────────────────────────────────────────────────────
class PencilV5(nn.Module):
    def __init__(self, scibert_embeddings):
        super().__init__()
        self.register_buffer('scibert_table', scibert_embeddings)
        self.scibert_proj = nn.Linear(SCIBERT_DIM, HIDDEN // 2)
        self.struct_proj = nn.Linear(EMBED_DIM, HIDDEN // 2)
        self.dist_embed = nn.Embedding((DIST_BINS + 2) * (DIST_BINS + 2), DIST_EMBED_DIM)
        self.combine = nn.Linear(HIDDEN // 2 + HIDDEN // 2 + DIST_EMBED_DIM, HIDDEN)
        self.cand_pos_embed = nn.Embedding(64, HIDDEN)
        self.layers = nn.ModuleList()
        for _ in range(N_LAYERS):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(HIDDEN, N_HEADS, batch_first=True, dropout=0.1),
                'ln1': nn.LayerNorm(HIDDEN),
                'ffn': nn.Sequential(nn.Linear(HIDDEN, HIDDEN * 4), nn.GELU(),
                                     nn.Linear(HIDDEN * 4, HIDDEN), nn.Dropout(0.1)),
                'ln2': nn.LayerNorm(HIDDEN),
                'mp': nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU()),
            }))
        self.ln_f = nn.LayerNorm(HIDDEN)
        self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

    @torch.no_grad()
    def score_batch(self, inp, node_ids, inner_bs=INNER_BATCH):
        """inp: (N, 32, 68), node_ids: (N, 32)"""
        n = inp.shape[0]
        device = next(self.parameters()).device
        scores = []
        attn_mask = torch.zeros(MAX_NODES, MAX_NODES, device=device)

        for st in range(0, n, inner_bs):
            ed = min(st + inner_bs, n)
            bi = inp[st:ed].unsqueeze(1).to(device)   # (B,1,32,68)
            bn = node_ids[st:ed].unsqueeze(1).to(device)
            B, G, N, _ = bi.shape
            fi = bi.reshape(B*G*N, -1)
            fn = bn.reshape(B*G*N)

            xr = fi.clone()
            xr[:, 2*MAX_NODES+1] = 0; xr[:, 2*MAX_NODES+2] = 0
            se = self.struct_proj(xr)
            sc = self.scibert_proj(self.scibert_table[fn.clamp(0, self.scibert_table.shape[0]-1)])
            du = (fi[:, 2*MAX_NODES+1] * DIST_BINS).long().clamp(0, DIST_BINS+1)
            dv = (fi[:, 2*MAX_NODES+2] * DIST_BINS).long().clamp(0, DIST_BINS+1)
            de = self.dist_embed(du * (DIST_BINS+2) + dv)
            x = self.combine(torch.cat([se, sc, de], -1))
            x = x + self.cand_pos_embed(torch.zeros(B*G*N, dtype=torch.long, device=device))
            x = x.reshape(B, G*N, -1)

            adj = bi[:, :, :, MAX_NODES:MAX_NODES+MAX_NODES]
            an = adj / adj.sum(3, keepdim=True).clamp(min=1)

            for layer in self.layers:
                h = layer['ln1'](x)
                h, _ = layer['attn'](h, h, h, attn_mask=attn_mask)
                x = x + h
                xc = x.reshape(B, G, MAX_NODES, -1)
                x = x + layer['mp'](torch.einsum('bgnh,bgnm->bgmh', xc, an)).reshape(B, G*MAX_NODES, -1)
                h = layer['ln2'](x)
                x = x + layer['ffn'](h)

            x = self.ln_f(x)
            pooled = x.reshape(B, G, MAX_NODES, -1).mean(2)
            scores.append(self.head(pooled).squeeze(-1).squeeze(-1).cpu())

        return torch.cat(scores).numpy()


# ── Main ───────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    # Load graph + A²
    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    print("Computing A²...", flush=True)
    t0 = time.time()
    adj2_lil = adj.dot(adj).tolil()
    print(f"  A² done: {time.time()-t0:.1f}s", flush=True)

    # Val pairs
    val_data = np.load(f'{DATA_DIR}/val_precomp_v2.npz')
    val_nids = val_data['node_ids']
    val_labels = val_data['label']
    G = 6
    n_val = len(val_labels) // G
    val_pairs = [(int(val_nids[i*G][0]), int(val_nids[i*G][1])) for i in range(n_val)]
    N_ANCHORS = len(val_pairs)
    print(f"  val pairs: {N_ANCHORS}", flush=True)

    # A² baseline
    print(f"\n=== A² baseline (top-{A2_TOP_K}) ===", flush=True)
    a2_ranks = []
    for u, v in val_pairs:
        row = adj2_lil[u]
        cands = np.array(row.rows[0]); sc = np.array(row.data[0])
        order = np.argsort(-sc)
        pos = np.where(cands[order] == v)[0]
        a2_ranks.append(int(pos[0])+1 if len(pos) > 0 else len(cands)+1)
    a2_ranks = np.array(a2_ranks)
    for k in TOP_K_EVAL:
        print(f"  A² Recall@{k}: {(a2_ranks <= k).mean():.4f}", flush=True)
    print(f"  A² Median: {np.median(a2_ranks):.0f}, MRR: {(1.0/a2_ranks).mean():.4f}", flush=True)

    # Candidate lists
    print(f"\n=== Preparing candidates ===", flush=True)
    cand_lists = []
    for u, v in val_pairs:
        row = adj2_lil[u]
        cands = np.array(row.rows[0]); sc = np.array(row.data[0])
        order = np.argsort(-sc)
        top = list(cands[order[:A2_TOP_K]])
        if v not in set(top): top.append(v)
        cand_lists.append(top)
    del adj2_lil

    # Split into chunks for workers
    chunk_size = (N_ANCHORS + N_WORKERS - 1) // N_WORKERS
    chunks = []
    for i in range(0, N_ANCHORS, chunk_size):
        j = min(i + chunk_size, N_ANCHORS)
        chunks.append((val_pairs[i:j], cand_lists[i:j]))
    print(f"  {len(chunks)} chunks × ~{chunk_size} anchors, workers={N_WORKERS}", flush=True)

    # Build subgraphs with multiprocessing
    print("  Building subgraphs (multi-process)...", flush=True)
    t0 = time.time()
    pencil_ranks = np.empty(N_ANCHORS, dtype=np.int32)

    with Pool(N_WORKERS, initializer=_worker_init) as pool:
        chunk_paths = pool.map(_worker_process, chunks)

    t_build = time.time() - t0
    print(f"  Subgraphs built: {t_build:.1f}s", flush=True)

    # Load model
    print("Loading model...", flush=True)
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilV5(scibert).to(device)
    model.load_state_dict(torch.load(f'{DATA_DIR}/pencil_v5_mixedneg_best.pt', weights_only=True))
    model.eval()
    _ = model.score_batch(torch.zeros(2, MAX_NODES, EMBED_DIM), torch.zeros(2, MAX_NODES, dtype=torch.long))
    torch.cuda.synchronize()

    # Score each chunk on GPU (load from temp files)
    RESULT_FILE = f'{DATA_DIR}/eval_mixedneg_top5k_ranks.npy'
    print(f"\n=== Scoring (inner_bs={INNER_BATCH}) ===", flush=True)
    t0 = time.time()
    offset = 0
    for chunk_idx, path in enumerate(chunk_paths):
        data = np.load(path)
        out_inp = data['inp']; out_nids = data['nids']; pos_indices = data['pos']
        os.remove(path)

        scores = model.score_batch(
            torch.from_numpy(out_inp), torch.from_numpy(out_nids))
        ci = 0
        for ai_in_chunk, pi in enumerate(pos_indices):
            nc = len(cand_lists[offset + ai_in_chunk])
            sc = scores[ci:ci+nc]
            pencil_ranks[offset + ai_in_chunk] = int((sc > sc[pi]).sum()) + 1
            ci += nc
        offset += len(pos_indices)
        # Save checkpoint after each chunk
        np.save(RESULT_FILE, pencil_ranks)
        print(f"  chunk {chunk_idx+1}/{len(chunk_paths)} scored ({time.time()-t0:.1f}s elapsed)", flush=True)

    t_score = time.time() - t0
    print(f"\nTotal: build={t_build:.1f}s score={t_score:.1f}s", flush=True)

    # Results
    print(f"\n=== PENCIL v5 mixedneg (top-{A2_TOP_K}) ===", flush=True)
    for k in TOP_K_EVAL:
        print(f"  PENCIL Recall@{k}: {(pencil_ranks <= k).mean():.4f}", flush=True)
    print(f"  PENCIL Median: {np.median(pencil_ranks):.0f}, MRR: {(1.0/pencil_ranks).mean():.4f}", flush=True)

    for k in [1000, 5000]:
        e2e = np.minimum(pencil_ranks.astype(np.int64), k)
        print(f"\n  E2E top-{k} Recall@1: {(e2e <= 1).mean():.4f}", flush=True)
        print(f"  E2E top-{k} Recall@10: {(e2e <= 10).mean():.4f}", flush=True)
        print(f"  E2E top-{k} Recall@100: {(e2e <= 100).mean():.4f}", flush=True)


if __name__ == '__main__':
    main()
