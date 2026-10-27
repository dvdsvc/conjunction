"""
Recall@K eval for PENCIL v5 with candidate isolation mask.

Scoring strategy for large K:
  - Encode source u's 32 nodes once → cache
  - For each batch of candidates: [source_cached, cand_batch] with isolation mask
  - Pool candidate embeddings → score
"""
import os, time, re
import numpy as np
import scipy.sparse as sp
from numba import njit
import torch
import torch.nn as nn

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32
EMBED_DIM = 68
SCIBERT_DIM = 768
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
DIST_BINS = 10
DIST_EMBED_DIM = 32
A2_TOP_K = 1000
TOP_K_EVAL = [1, 3, 5, 10, 50, 100, 500, 1000]


@njit(cache=True)
def _reconstruct_features(nids, u, v, dist_matrix, dist_u, dist_v, out):
    n = 0
    for i in range(MAX_NODES):
        if nids[i] != 0 or i == 0: n = i + 1
    for i in range(MAX_NODES):
        for j in range(EMBED_DIM): out[i, j] = 0.0
    for i in range(n):
        out[i, i] = 1.0
        out[i, 2 * MAX_NODES] = 1.0
        out[i, 2 * MAX_NODES + 1] = min(dist_u[nids[i]], float(DIST_BINS)) / float(DIST_BINS)
        out[i, 2 * MAX_NODES + 2] = min(dist_v[nids[i]], float(DIST_BINS)) / float(DIST_BINS)
        for j in range(n):
            if dist_matrix[nids[i], nids[j]] == 1:
                out[i, MAX_NODES + j] = 1.0


def _extract_val_pairs(data_dir):
    pos_pairs = []
    with open(f'{data_dir}/train.jsonl') as f:
        for line in f:
            if '"label": 1' not in line and '"label":1' not in line: continue
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
                'ffn': nn.Sequential(
                    nn.Linear(HIDDEN, HIDDEN * 4), nn.GELU(),
                    nn.Linear(HIDDEN * 4, HIDDEN), nn.Dropout(0.1)),
                'ln2': nn.LayerNorm(HIDDEN),
                'mp': nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU()),
            }))
        self.ln_f = nn.LayerNorm(HIDDEN)
        self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

    def _build_node_features(self, inp, node_ids):
        """Build node embeddings from features. Returns (B*G*N, H)."""
        B, G, N, _ = inp.shape
        flat_inp = inp.reshape(B * G * N, -1)
        flat_nids = node_ids.reshape(B * G * N)

        x_raw = flat_inp.clone()
        x_raw[:, 2 * MAX_NODES + 1] = 0
        x_raw[:, 2 * MAX_NODES + 2] = 0
        struct_emb = self.struct_proj(x_raw)

        nids_flat = flat_nids.clamp(0, self.scibert_table.shape[0] - 1)
        scibert_emb = self.scibert_proj(self.scibert_table[nids_flat])

        dist_u_bin = (flat_inp[:, 2 * MAX_NODES + 1] * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        dist_v_bin = (flat_inp[:, 2 * MAX_NODES + 2] * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        dist_idx = dist_u_bin * (DIST_BINS + 2) + dist_v_bin
        dist_emb = self.dist_embed(dist_idx)

        x = self.combine(torch.cat([struct_emb, scibert_emb, dist_emb], dim=-1))

        cand_idx = torch.arange(G, device=inp.device).unsqueeze(1).expand(-1, N).reshape(-1)
        cand_idx = cand_idx.unsqueeze(0).expand(B, -1).reshape(-1)
        x = x + self.cand_pos_embed(cand_idx)
        return x.reshape(B, G * N, -1)

    def _build_iso_mask(self, n_cands, n_nodes, device):
        S = (1 + n_cands) * n_nodes
        mask = torch.full((S, S), -10000.0, device=device)
        src_end = n_nodes
        mask[:src_end, :] = 0.0
        for k in range(1 + n_cands):
            start = k * n_nodes
            end = (k + 1) * n_nodes
            mask[start:end, :src_end] = 0.0
            mask[start:end, start:end] = 0.0
        return mask

    def forward(self, inp, node_ids):
        B, G, N, _ = inp.shape
        device = inp.device

        x = self._build_node_features(inp, node_ids)

        adj = inp[:, :, :, MAX_NODES:MAX_NODES + N]
        deg = adj.sum(dim=3, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        pad_mask = (inp[:, :, :, :2 * MAX_NODES + 1].abs().sum(dim=3) == 0)
        pad_mask_flat = pad_mask.reshape(B, G * N)
        iso_mask = self._build_iso_mask(G - 1, N, device)

        for layer in self.layers:
            h = layer['ln1'](x)
            h, _ = layer['attn'](h, h, h, attn_mask=iso_mask, key_padding_mask=pad_mask_flat)
            x = x + h
            x_cand = x.reshape(B, G, N, -1)
            h_mp = layer['mp'](torch.einsum('bgnh,bgnm->bgmh', x_cand, adj_norm))
            x_cand = x_cand + h_mp
            x = x.reshape(B, G * N, -1)
            x[:, :] = x_cand.reshape(B, G * N, -1)
            h = layer['ln2'](x)
            x = x + layer['ffn'](h)

        x = self.ln_f(x)
        x_cand = x.reshape(B, G, N, -1)
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (x_cand * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)
        scores = self.head(pooled).squeeze(-1)
        return scores

    def score_batched(self, src_inp, src_nids, cand_inp, cand_nids, batch_size=50):
        """Score candidates with isolation mask, batched for memory.

        Args:
            src_inp: (1, 1, 32, 68) — source subgraph features
            src_nids: (1, 1, 32) — source node IDs
            cand_inp: (K, 32, 68) — candidate subgraph features
            cand_nids: (K, 32) — candidate node IDs
            batch_size: candidates per batch
        Returns:
            (K,) — scores
        """
        K = cand_inp.shape[0]
        all_scores = []

        for start in range(0, K, batch_size):
            end = min(start + batch_size, K)
            bs = end - start

            # Concatenate: [source, cand_start: cand_end]
            batch_inp = torch.cat([src_inp, cand_inp[start:end].unsqueeze(0)], dim=1)  # (1, 1+bs, 32, 68)
            batch_nids = torch.cat([src_nids, cand_nids[start:end].unsqueeze(0)], dim=1)  # (1, 1+bs, 32)

            scores = self.forward(batch_inp, batch_nids)  # (1, 1+bs)
            all_scores.append(scores[0, 1:])  # skip source score

        return torch.cat(all_scores)  # (K,)


if __name__ == '__main__':
    t0 = time.time()

    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()

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
    for u, v in val_pairs:
        row = adj2_lil[u]
        cands = np.array(row.rows[0])
        scores_a2 = np.array(row.data[0])
        order = np.argsort(-scores_a2)
        sorted_cands = cands[order]
        pos = np.where(sorted_cands == v)[0]
        a2_ranks.append(int(pos[0]) + 1 if len(pos) > 0 else len(sorted_cands) + 1)
    a2_ranks = np.array(a2_ranks)
    for k in TOP_K_EVAL:
        print(f"  A² Recall@{k}: {(a2_ranks <= k).mean():.4f}", flush=True)
    print(f"  A² Median: {np.median(a2_ranks):.0f}, MRR: {(1.0/a2_ranks).mean():.4f}", flush=True)
    del adj2_lil

    # Load precomputed node_ids (top-5000) and build top-1000 + positive
    nids_path = f'{DATA_DIR}/eval_subgraph_nids.npy'
    cands_len_path = f'{DATA_DIR}/eval_cands_len.npy'
    print(f"\nLoading precomputed node_ids (top-5000)...", flush=True)
    all_nids = np.load(nids_path, mmap_mode='r')
    cands_len_full = np.load(cands_len_path)

    print("Building top-1000 + positive candidate lists...", flush=True)
    max_nc = 1001
    all_nids_1k = np.zeros((N_ANCHORS, max_nc, MAX_NODES), dtype=np.int64)
    cands_len_1k = np.zeros(N_ANCHORS, dtype=np.int32)
    for ai in range(N_ANCHORS):
        u, v = val_pairs[ai]
        NC_full = int(cands_len_full[ai])
        nids_ai = all_nids[ai]
        pos_idx = -1
        for j in range(NC_full):
            if nids_ai[j, 0] == u and nids_ai[j, 1] == v:
                pos_idx = j; break
        nc = min(1000, NC_full)
        all_nids_1k[ai, :nc] = nids_ai[:nc]
        if pos_idx >= nc:
            all_nids_1k[ai, nc] = nids_ai[pos_idx]
            cands_len_1k[ai] = nc + 1
        else:
            cands_len_1k[ai] = nc

    # Numba warmup
    print("Compiling Numba...", flush=True)
    tmp_out = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
    _reconstruct_features(all_nids_1k[0, 0], 0, 1, dist_matrix,
                          dist_matrix[0], dist_matrix[1], tmp_out)
    print("  done", flush=True)

    # Load model
    device = torch.device('cuda')
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilV5(scibert).to(device)
    ckpt_path = f'{DATA_DIR}/pencil_v5_iso_best.pt'
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    print(f"  model on {torch.cuda.get_device_name(0)}", flush=True)

    # PENCIL v5 re-rank (top-1000) with batched isolation mask scoring
    print(f"\n=== PENCIL v5 iso re-rank (top-{A2_TOP_K}) ===", flush=True)
    pencil_ranks = np.zeros(N_ANCHORS, dtype=np.int64)
    t_re = time.time()
    CAND_BATCH = 50  # candidates per batch (1 source + 50 cands = 51*32 = 1632 tokens)

    for ai in range(N_ANCHORS):
        u, v = val_pairs[ai]
        NC = int(cands_len_1k[ai])
        nids_ai = all_nids_1k[ai]

        # Reconstruct features for all candidates
        all_feats = np.zeros((NC, MAX_NODES, EMBED_DIM), dtype=np.float32)
        for j in range(NC):
            _reconstruct_features(nids_ai[j], u, v, dist_matrix,
                                 dist_matrix[u], dist_matrix[v], tmp_out)
            all_feats[j] = tmp_out

        # Source is the positive (find it)
        pos_idx = -1
        for j in range(NC):
            if nids_ai[j, 0] == u and nids_ai[j, 1] == v:
                pos_idx = j; break
        if pos_idx < 0:
            print(f"  WARNING: anchor {ai} positive not found, skipping", flush=True)
            pencil_ranks[ai] = NC + 1
            continue

        # Source subgraph
        src_inp = torch.from_numpy(all_feats[pos_idx:pos_idx+1]).unsqueeze(0).to(device)  # (1,1,32,68)
        src_nids = torch.from_numpy(nids_ai[pos_idx:pos_idx+1]).unsqueeze(0).to(device)  # (1,1,32)

        # Candidates (all except source)
        nids_nc = nids_ai[:NC]
        mask = np.ones(NC, dtype=bool)
        mask[pos_idx] = False
        n_cands = int(mask.sum())

        if n_cands == 0:
            # Only the positive exists, trivially rank 1
            pencil_ranks[ai] = 1
            continue

        cand_feats = torch.from_numpy(all_feats[mask]).to(device)  # (n_cands, 32, 68)
        cand_nids_t = torch.from_numpy(nids_nc[mask]).to(device)   # (n_cands, 32)

        # Score with batched isolation mask
        with torch.inference_mode():
            scores = model.score_batched(src_inp, src_nids, cand_feats, cand_nids_t,
                                          batch_size=CAND_BATCH).cpu().numpy()

        # Rank: source's score vs all candidates
        # Actually, we need the source score too. Let me compute it.
        # The source gets scored when it's in the candidate position.
        # But in score_batched, source is always at position 0 (special).
        # We need to compare: does any candidate score higher than source?
        # Let me compute source score separately.
        with torch.inference_mode():
            # Score source vs itself (degenerate, but gives source score)
            src_score = model.forward(src_inp, src_nids)[0, 0].item()

        rank = int((scores > src_score).sum() + 1)
        pencil_ranks[ai] = rank

        if (ai + 1) % 200 == 0:
            elapsed = time.time() - t_re
            eta = elapsed / (ai + 1) * (N_ANCHORS - ai - 1)
            print(f"  {ai+1}/{N_ANCHORS} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    dt_re = time.time() - t_re
    print(f"  re-rank total: {dt_re:.0f}s ({dt_re/N_ANCHORS:.2f}s/anchor)", flush=True)

    # Results
    e2e_ranks = np.where(a2_ranks <= A2_TOP_K, pencil_ranks, a2_ranks)

    print(f"\n=== E2E (A² top-{A2_TOP_K} + PENCIL v5 iso re-rank) ===", flush=True)
    for k in TOP_K_EVAL:
        print(f"  E2E Recall@{k}: {(e2e_ranks <= k).mean():.4f}", flush=True)
    print(f"  E2E Median: {np.median(e2e_ranks):.0f}, MRR: {(1.0/e2e_ranks).mean():.4f}", flush=True)

    mask = a2_ranks <= A2_TOP_K
    pencil_sub = pencil_ranks[mask]
    print(f"\n=== PENCIL v5 iso (within A² top-{A2_TOP_K}, n={len(pencil_sub)}) ===", flush=True)
    for k in TOP_K_EVAL:
        print(f"  PENCIL Recall@{k}: {(pencil_sub <= k).mean():.4f}", flush=True)
    print(f"  PENCIL Median: {np.median(pencil_sub):.0f}, MRR: {(1.0/pencil_sub).mean():.4f}", flush=True)

    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
    print("Done.", flush=True)
