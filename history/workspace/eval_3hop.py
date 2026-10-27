"""
3-hop LBD eval: find pairs connected by exactly 3 hops (no shorter path).
Subgraph uses 2-hop BFS with path-priority node selection. MAX_NODES=48.
Requires retrained model (EMBED_DIM changed from 68 to 99).
"""
import os, time
from collections import deque
import numpy as np
import scipy.sparse as sp
from numba import njit, prange
import torch
import torch.nn as nn

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 48
EMBED_DIM = 2 * MAX_NODES + 3  # 99
SCIBERT_DIM = 768
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
DIST_BINS = 10
DIST_EMBED_DIM = 32
TOP_K_EVAL = [1, 3, 5, 10, 50, 100, 500, 1000]
INNER_BATCH = 500
N_VAL = 2000  # number of 3-hop val pairs to sample
N_CANDS = 1000  # candidates per anchor


# ── BFS helpers ────────────────────────────────────────────────────────
def bfs_k_hop(adj, src, max_hops=3):
    """BFS from src, return dict {node: distance} for nodes within max_hops."""
    dist = {src: 0}
    queue = deque([(src, 0)])
    while queue:
        node, d = queue.popleft()
        if d >= max_hops:
            continue
        for nb in adj[node].indices:
            if nb not in dist:
                dist[nb] = d + 1
                queue.append((nb, d + 1))
    return dist


def find_shortest_path(adj, src, dst, max_hops=3):
    """BFS to find shortest path from src to dst. Returns list of nodes or None."""
    if src == dst:
        return [src]
    parent = {src: -1}
    queue = deque([src])
    while queue:
        node = queue.popleft()
        d_node = 0
        tmp = node
        while parent[tmp] != -1:
            d_node += 1
            tmp = parent[tmp]
        if d_node >= max_hops:
            continue
        for nb in adj[node].indices:
            if nb not in parent:
                parent[nb] = node
                if nb == dst:
                    path = [dst]
                    while path[-1] != src:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                queue.append(nb)
    return None


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


@njit(cache=True)
def _collect_2hop_neighbors(node, indptr, indices, out, exclude):
    """Collect 2-hop neighbors of node into out array. Return count.
    Prioritize 1-hop neighbors first, then 2-hop. Skip nodes in exclude."""
    nc = 0
    n_out = len(out)
    for i in range(indptr[node], indptr[node + 1]):
        nb = indices[i]
        if not _in_sorted(exclude, nb):
            if nc < n_out: out[nc] = nb; nc += 1
    for i in range(indptr[node], indptr[node + 1]):
        nb1 = indices[i]
        for j in range(indptr[nb1], indptr[nb1 + 1]):
            nb2 = indices[j]
            if nb2 != node and not _in_sorted(exclude, nb2):
                found = False
                for k in range(nc):
                    if out[k] == nb2:
                        found = True; break
                if not found:
                    if nc < n_out: out[nc] = nb2; nc += 1
    return nc


@njit(parallel=True, nogil=True, cache=True)
def _build_anchor_3hop(u, v, cands, n_cands,
                       indptr, indices, deg, dist_mat,
                       path_nodes, n_path,
                       nb_flat, nb_off,
                       out_inp, out_nids, out_pos):
    """Build subgraphs for 3-hop eval. Path nodes get priority in selection."""
    MN = MAX_NODES; ED = EMBED_DIM; DB = float(DIST_BINS)

    # u's 2-hop neighbors (for filling remaining slots)
    u_exclude = np.empty(1, dtype=np.int64); u_exclude[0] = u
    u_2hop = np.empty(35000, dtype=np.int64)
    u_n = _collect_2hop_neighbors(u, indptr, indices, u_2hop, u_exclude)

    for ci in prange(n_cands):
        c = cands[ci]

        # c's 2-hop neighbors
        c_exclude = np.empty(1, dtype=np.int64); c_exclude[0] = c
        c_2hop = np.empty(35000, dtype=np.int64)
        c_n = _collect_2hop_neighbors(c, indptr, indices, c_2hop, c_exclude)

        # Node selection: path nodes first, then fill with neighbors
        sel = np.empty(MN, dtype=np.int64); ns = 0
        # Add path nodes (u, w1, w2, ..., c)
        for i in range(n_path):
            if ns >= MN: break
            sel[ns] = path_nodes[i]; ns += 1
        # Fill with u's 2-hop neighbors
        for i in range(u_n):
            if ns >= MN: break
            nd = u_2hop[i]
            dup = False
            for k in range(ns):
                if sel[k] == nd: dup = True; break
            if not dup:
                sel[ns] = nd; ns += 1
        # Fill with c's 2-hop neighbors
        for i in range(c_n):
            if ns >= MN: break
            nd = c_2hop[i]
            dup = False
            for k in range(ns):
                if sel[k] == nd: dup = True; break
            if not dup:
                sel[ns] = nd; ns += 1

        # Clear features
        for i in range(MN):
            for j in range(ED):
                out_inp[ci, i, j] = 0.0
            out_nids[ci, i] = 0

        # Fill features
        for i in range(ns):
            if i < MN:
                out_inp[ci, i, i] = 1.0
            out_inp[ci, i, 2 * MN] = 1.0
            nd = sel[i]
            out_inp[ci, i, 2*MN+1] = min(float(dist_mat[u, nd]), DB) / DB
            out_inp[ci, i, 2*MN+2] = min(float(dist_mat[c, nd]), DB) / DB

        # Adjacency
        ss = sel[:ns].copy(); ss.sort()
        for i in range(ns):
            a = sel[i]
            for j in range(indptr[a], indptr[a + 1]):
                b = indices[j]
                if _in_sorted(ss, b):
                    for lb in range(ns):
                        if sel[lb] == b:
                            if lb < MN:
                                out_inp[ci, i, MN + lb] = 1.0
                            break

        for i in range(min(ns, MN)):
            out_nids[ci, i] = sel[i]

        if c == v:
            out_pos[0] = ci


# ── Model (same architecture, different EMBED_DIM) ─────────────────────
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
        n = inp.shape[0]
        device = next(self.parameters()).device
        scores = []
        attn_mask = torch.zeros(MAX_NODES, MAX_NODES, device=device)

        for st in range(0, n, inner_bs):
            ed = min(st + inner_bs, n)
            bi = inp[st:ed].unsqueeze(1).to(device)
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

            adj_m = bi[:, :, :, MAX_NODES:MAX_NODES+MAX_NODES]
            an = adj_m / adj_m.sum(3, keepdim=True).clamp(min=1)

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
    print(f"MAX_NODES={MAX_NODES}, EMBED_DIM={EMBED_DIM}", flush=True)

    # Load graph
    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    n_nodes = adj.shape[0]
    print(f"  nodes: {n_nodes}, edges: {adj.nnz}", flush=True)

    # Find 3-hop val pairs via BFS
    print(f"Finding 3-hop val pairs (N_VAL={N_VAL})...", flush=True)
    rng = np.random.RandomState(42)
    val_pairs_3hop = []
    attempts = 0
    max_attempts = N_VAL * 20

    while len(val_pairs_3hop) < N_VAL and attempts < max_attempts:
        src = rng.randint(0, n_nodes)
        dist = bfs_k_hop(adj, src, max_hops=3)
        # Find nodes at exactly distance 3
        d3_nodes = [n for n, d in dist.items() if d == 3]
        if d3_nodes:
            dst = d3_nodes[rng.randint(0, len(d3_nodes))]
            val_pairs_3hop.append((src, dst))
        attempts += 1

    print(f"  found {len(val_pairs_3hop)} 3-hop pairs in {attempts} attempts", flush=True)
    if len(val_pairs_3hop) == 0:
        print("ERROR: no 3-hop pairs found!", flush=True)
        return

    N_ANCHORS = len(val_pairs_3hop)

    # Build 3-hop candidate lists
    print("Building 3-hop candidate lists...", flush=True)
    cand_lists = []
    cand_ranks_a3 = []  # A³ baseline ranks
    t0 = time.time()
    for ai, (u, v) in enumerate(val_pairs_3hop):
        dist = bfs_k_hop(adj, u, max_hops=3)
        # 3-hop candidates: exactly distance 3
        d3_cands = [(n, 1.0/(d+1)) for n, d in dist.items() if d == 3 and n != u]
        # Sort by... just BFS order for now (no A³ score available cheaply)
        cands = [n for n, _ in d3_cands]
        # Ensure v is in candidates
        if v not in cands:
            cands.append(v)
        # Take top N_CANDS
        if len(cands) > N_CANDS:
            # Keep v and sample the rest
            cands_no_v = [c for c in cands if c != v]
            rng.shuffle(cands_no_v)
            cands = cands_no_v[:N_CANDS-1] + [v]
        cand_lists.append(cands)

        # A³ baseline rank: v's position in the 3-hop candidate list
        v_pos = cands.index(v) if v in cands else len(cands)
        cand_ranks_a3.append(v_pos + 1)

        if (ai + 1) % 200 == 0:
            print(f"  [{ai+1}/{N_ANCHORS}] candidates built ({time.time()-t0:.0f}s)", flush=True)

    print(f"  max cands: {max(len(c) for c in cand_lists)}", flush=True)

    # A³ baseline (rank by BFS order - crude)
    print(f"\n=== A³ baseline (BFS order) ===", flush=True)
    a3_ranks = np.array(cand_ranks_a3)
    for k in TOP_K_EVAL:
        print(f"  A³ Recall@{k}: {(a3_ranks <= k).mean():.4f}", flush=True)
    print(f"  A³ Median: {np.median(a3_ranks):.0f}, MRR: {(1.0/a3_ranks).mean():.4f}", flush=True)

    # Pre-compute neighbor lists
    print("Pre-computing neighbor lists...", flush=True)
    _indptr = adj.indptr.astype(np.int64)
    _indices = adj.indices.astype(np.int64)
    _deg = np.array(adj.sum(axis=1)).flatten().astype(np.float32)

    # Load dist_matrix
    print("Loading dist_matrix...", flush=True)
    dist_mat = np.load(f'{DATA_DIR}/dist_matrix.npy')

    # Load model
    print("Loading model...", flush=True)
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilV5(scibert).to(device)
    MODEL_FILE = f'{DATA_DIR}/pencil_3hop_best.pt'
    if not os.path.exists(MODEL_FILE):
        print(f"WARNING: {MODEL_FILE} not found. Using random weights for testing.", flush=True)
    else:
        model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    model.eval()
    _ = model.score_batch(torch.zeros(2, MAX_NODES, EMBED_DIM), torch.zeros(2, MAX_NODES, dtype=torch.long))
    torch.cuda.synchronize()

    # Pre-allocate buffers
    max_cands = max(len(c) for c in cand_lists)
    inp_buf = np.zeros((max_cands, MAX_NODES, EMBED_DIM), dtype=np.float32)
    nids_buf = np.zeros((max_cands, MAX_NODES), dtype=np.int64)
    pos_buf = np.zeros(1, dtype=np.int64)

    # Per-anchor eval
    RESULT_FILE = f'{DATA_DIR}/eval_3hop_ranks.npy'
    print(f"\n=== Per-anchor 3-hop eval ===", flush=True)
    pencil_ranks = np.zeros(N_ANCHORS, dtype=np.int32)
    t0 = time.time()

    for ai in range(N_ANCHORS):
        u, v = val_pairs_3hop[ai]
        cands = np.array(cand_lists[ai], dtype=np.int64)
        n_cands = len(cands)
        pos_buf[0] = -1

        # Find shortest path u → v (for path-priority node selection)
        path = find_shortest_path(adj, u, v, max_hops=3)
        if path is None:
            path = [u, v]  # fallback
        path_nodes = np.array(path, dtype=np.int64)
        n_path = len(path)

        # Build subgraphs
        _build_anchor_3hop(u, v, cands, n_cands,
                           _indptr, _indices, _deg, dist_mat,
                           path_nodes, n_path,
                           nb_flat=np.empty(0, dtype=np.int64),  # not used in 3hop version
                           nb_off=np.empty(0, dtype=np.int64),
                           out_inp=inp_buf, out_nids=nids_buf, out_pos=pos_buf)

        # Score on GPU
        scores = model.score_batch(
            torch.from_numpy(inp_buf[:n_cands]),
            torch.from_numpy(nids_buf[:n_cands]))
        pi = int(pos_buf[0])
        if pi < 0:
            pencil_ranks[ai] = n_cands + 1
        else:
            pencil_ranks[ai] = int((scores > scores[pi]).sum()) + 1

        if (ai + 1) % 100 == 0 or ai == N_ANCHORS - 1:
            elapsed = time.time() - t0
            rate = (ai + 1) / elapsed
            eta = (N_ANCHORS - ai - 1) / rate
            np.save(RESULT_FILE, pencil_ranks)
            print(f"  [{ai+1}/{N_ANCHORS}] {elapsed:.0f}s, {rate:.1f} anc/s, ETA {eta:.0f}s", flush=True)

    t_total = time.time() - t0
    np.save(RESULT_FILE, pencil_ranks)

    print(f"\nTotal: {t_total:.1f}s ({t_total/60:.1f} min)", flush=True)

    # Results
    print(f"\n=== PENCIL 3-hop results ===", flush=True)
    for k in TOP_K_EVAL:
        print(f"  PENCIL Recall@{k}: {(pencil_ranks <= k).mean():.4f}", flush=True)
    print(f"  PENCIL Median: {np.median(pencil_ranks):.0f}, MRR: {(1.0/pencil_ranks).mean():.4f}", flush=True)


if __name__ == '__main__':
    main()
