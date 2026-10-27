import os, re, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import json

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32
EMBED_DIM = 68
SCIBERT_DIM = 768
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
DIST_BINS = 10
DIST_EMBED_DIM = 32

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
                'ffn': nn.Sequential(nn.Linear(HIDDEN, HIDDEN * 4), nn.GELU(), nn.Linear(HIDDEN * 4, HIDDEN), nn.Dropout(0.1)),
                'ln2': nn.LayerNorm(HIDDEN),
                'mp': nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU()),
            }))
        self.ln_f = nn.LayerNorm(HIDDEN)
        self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

    def _build_node_features(self, inp, node_ids):
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


def count_nn(feat):
    return int(np.greater(np.abs(feat[:, :1]).sum(axis=1), 0).sum())


def get_uv_idx(feat, nn):
    u_idx = v_idx = None
    for j in range(nn):
        if feat[j, 65] == 0: u_idx = j
        if feat[j, 66] == 0: v_idx = j
    return u_idx, v_idx


if __name__ == '__main__':
    device = torch.device('cuda')
    print("Loading...", flush=True)
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilV5(scibert).to(device)
    model.load_state_dict(torch.load(f'{DATA_DIR}/pencil_v5_iso_best.pt', weights_only=True))
    model.eval()

    concept_ids = json.load(open(f'{DATA_DIR}/concept_ids.json'))
    concept_names = json.load(open(f'{DATA_DIR}/concept_names.json'))
    def get_name(idx):
        if idx < 0 or idx >= len(concept_ids): return f"?{idx}"
        return concept_names.get(concept_ids[idx], concept_ids[idx])

    val_pairs = _extract_val_pairs(DATA_DIR)
    val_data = np.load(f'{DATA_DIR}/val_precomp_v3.npz')
    val_inp_t = torch.from_numpy(val_data['inp'].astype(np.float32)).reshape(-1, 6, 32, 68)
    val_nids_t = torch.from_numpy(val_data['node_ids']).reshape(-1, 6, 32)
    val_inp = val_inp_t.numpy()
    val_nids = val_nids_t.numpy()

    # ===== Analysis 1: Structure comparison =====
    print(f"\n{'='*60}", flush=True)
    print(f"Analysis 1: Positive vs Negative Subgraph Statistics", flush=True)
    print(f"{'='*60}", flush=True)

    pos_nn_l, neg_nn_l = [], []
    pos_adj_l, neg_adj_l = [], []
    pos_du_l, pos_dv_l, neg_du_l, neg_dv_l = [], [], [], []

    for i in range(len(val_pairs)):
        for feat, nn_l, adj_l, du_l, dv_l in [
            (val_inp[i, 0], pos_nn_l, pos_adj_l, pos_du_l, pos_dv_l),
            (val_inp[i, 1], neg_nn_l, neg_adj_l, neg_du_l, neg_dv_l)]:
            nn = count_nn(feat)
            nn_l.append(nn)
            adj_l.append(float(feat[:nn, 32:32+nn].sum()))
            d_u = feat[:, 65]; d_v = feat[:, 66]
            du_l.append(float(d_u[d_u > 0].mean()) if np.any(d_u > 0) else 0)
            dv_l.append(float(d_v[d_v > 0].mean()) if np.any(d_v > 0) else 0)

    print(f"  Subgraph nodes:     pos={np.mean(pos_nn_l):.1f}, neg={np.mean(neg_nn_l):.1f}", flush=True)
    print(f"  Edges (adj sum):    pos={np.mean(pos_adj_l):.1f}, neg={np.mean(neg_adj_l):.1f}", flush=True)
    print(f"  Mean dist(u, node): pos={np.mean(pos_du_l):.4f}, neg={np.mean(neg_du_l):.4f}", flush=True)
    print(f"  Mean dist(v, node): pos={np.mean(pos_dv_l):.4f}, neg={np.mean(neg_dv_l):.4f}", flush=True)

    # ===== Analysis 2: Score distribution =====
    print(f"\n{'='*60}", flush=True)
    print(f"Analysis 2: Score Distribution (200 anchors)", flush=True)
    print(f"{'='*60}", flush=True)
    with torch.inference_mode():
        scores = model(val_inp_t[:200].to(device), val_nids_t[:200].to(device)).cpu().numpy()
    pos_s = scores[:, 0]; neg_s = scores[:, 1:]
    print(f"  Pos score:  mean={pos_s.mean():.4f}  std={pos_s.std():.4f}  [{pos_s.min():.4f}, {pos_s.max():.4f}]", flush=True)
    print(f"  Neg score:  mean={neg_s.mean():.4f}  std={neg_s.std():.4f}  [{neg_s.min():.4f}, {neg_s.max():.4f}]", flush=True)
    print(f"  Margin(pos-max_neg): mean={(pos_s - neg_s.max(axis=1)).mean():.4f}", flush=True)
    print(f"  Pos>0: {(pos_s > 0).mean():.4f}  Neg>0: {(neg_s > 0).mean():.4f}", flush=True)

    # ===== Analysis 3: Feature ablation =====
    print(f"\n{'='*60}", flush=True)
    print(f"Analysis 3: Feature Ablation (50 anchors)", flush=True)
    print(f"{'='*60}", flush=True)
    N_ABL = 50
    inp_50 = val_inp_t[:N_ABL].to(device)
    nids_50 = val_nids_t[:N_ABL].to(device)

    def score_pos_margin(inp_mod):
        with torch.inference_mode():
            s = model(inp_mod, nids_50).cpu().numpy()
        return s[:, 0].mean(), (s[:, 0] - s[:, 1:].max(axis=1)).mean()

    base_p, base_m = score_pos_margin(inp_50)
    inp_nd = inp_50.clone(); inp_nd[:, :, :, 65] = 0; inp_nd[:, :, :, 66] = 0
    nd_p, nd_m = score_pos_margin(inp_nd)
    inp_na = inp_50.clone(); inp_na[:, :, :, 32:64] = 0
    na_p, na_m = score_pos_margin(inp_na)
    inp_np = inp_50.clone(); inp_np[:, :, :, 0:32] = 0
    np_p, np_m = score_pos_margin(inp_np)
    inp_od = inp_50.clone(); inp_od[:, :, :, 0:65] = 0
    od_p, od_m = score_pos_margin(inp_od)

    print(f"  {'Ablation':<25s} {'pos_score':>10s} {'margin':>10s}", flush=True)
    print(f"  {'Base':<25s} {base_p:>10.4f} {base_m:>10.4f}", flush=True)
    print(f"  {'No distance (65,66)':<25s} {nd_p:>10.4f} {nd_m:>10.4f}", flush=True)
    print(f"  {'No adjacency (32:64)':<25s} {na_p:>10.4f} {na_m:>10.4f}", flush=True)
    print(f"  {'No position (0:32)':<25s} {np_p:>10.4f} {np_m:>10.4f}", flush=True)
    print(f"  {'Only distance':<25s} {od_p:>10.4f} {od_m:>10.4f}", flush=True)

    # ===== Analysis 4: Concrete examples =====
    print(f"\n{'='*60}", flush=True)
    print(f"Analysis 4: Concrete Examples (10 val anchors)", flush=True)
    print(f"{'='*60}", flush=True)

    with torch.inference_mode():
        all_scores = model(val_inp_t[:50].to(device), val_nids_t[:50].to(device)).cpu().numpy()

    for ai in range(10):
        u, v = val_pairs[ai]
        u_name, v_name = get_name(u), get_name(v)
        pos_score = all_scores[ai, 0]
        neg_scores = all_scores[ai, 1:]
        neg_max = neg_scores.max()

        pos_feat = val_inp[ai, 0]
        pos_nids_arr = val_nids[ai, 0]
        nn = count_nn(pos_feat)
        u_idx, v_idx = get_uv_idx(pos_feat, nn)

        u_nbs = set()
        v_nbs = set()
        if u_idx is not None:
            u_nbs = {j for j in range(nn) if j != u_idx and pos_feat[u_idx, 32+j] > 0}
        if v_idx is not None:
            v_nbs = {j for j in range(nn) if j != v_idx and pos_feat[v_idx, 32+j] > 0}
        cross = sum(1 for a in u_nbs for b in v_nbs if pos_feat[a, 32+b] > 0)

        u_nb_names = [get_name(int(pos_nids_arr[j])) for j in sorted(u_nbs)[:6]]
        v_nb_names = [get_name(int(pos_nids_arr[j])) for j in sorted(v_nbs)[:6]]

        # Neg candidate
        neg_feat = val_inp[ai, 1]
        neg_nids_arr = val_nids[ai, 1]
        neg_cand_name = get_name(int(neg_nids_arr[1]))

        print(f"\n  [{ai}] {u_name} ——bridge——> {v_name}", flush=True)
        print(f"      score: pos={pos_score:.3f}  neg_max={neg_max:.3f}  margin={pos_score-neg_max:.3f}", flush=True)
        print(f"      u@node[{u_idx}]  v@node[{v_idx}]  {nn} nodes", flush=True)
        print(f"      u neighbors ({len(u_nbs)}): {', '.join(u_nb_names)}", flush=True)
        print(f"      v neighbors ({len(v_nbs)}): {', '.join(v_nb_names)}", flush=True)
        print(f"      cross-links (u-nb <-> v-nb): {cross}", flush=True)
        print(f"      neg candidate: {neg_cand_name} (score={neg_scores[0]:.3f})", flush=True)

    # ===== Analysis 5: Global structural patterns =====
    print(f"\n{'='*60}", flush=True)
    print(f"Analysis 5: Global Structural Patterns (all val)", flush=True)
    print(f"{'='*60}", flush=True)

    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()

    cross_pos, cross_neg = [], []
    u_deg_pos, u_deg_neg = [], []
    v_deg_pos, v_deg_neg = [], []
    shared_nb = []
    u_dist2_pos, u_dist2_neg = [], []  # how many nodes at dist=2 from u

    for i in range(len(val_pairs)):
        u, v = val_pairs[i]
        shared_nb.append(len(set(adj[u].indices) & set(adj[v].indices)))

        for feat, cl, ud, vd, d2 in [
            (val_inp[i, 0], cross_pos, u_deg_pos, v_deg_pos, u_dist2_pos),
            (val_inp[i, 1], cross_neg, u_deg_neg, v_deg_neg, u_dist2_neg)]:
            nn = count_nn(feat)
            ui, vi = get_uv_idx(feat, nn)
            if ui is not None and vi is not None:
                un = {j for j in range(nn) if j != ui and feat[ui, 32+j] > 0}
                vn = {j for j in range(nn) if j != vi and feat[vi, 32+j] > 0}
                ud.append(len(un)); vd.append(len(vn))
                cl.append(sum(1 for a in un for b in vn if feat[a, 32+b] > 0))
                # Nodes at dist=2 from u (dist_u > 0.19 and <= 0.3, i.e. dist=2/10)
                d2.append(int(np.sum((feat[:, 65] > 0.15) & (feat[:, 65] <= 0.3))))
            else:
                ud.append(0); vd.append(0); cl.append(0); d2.append(0)

    print(f"  Shared neighbors (full graph): mean={np.mean(shared_nb):.1f}", flush=True)
    print(f"  u degree in subgraph:  pos={np.mean(u_deg_pos):.1f}  neg={np.mean(u_deg_neg):.1f}", flush=True)
    print(f"  v degree in subgraph:  pos={np.mean(v_deg_pos):.1f}  neg={np.mean(v_deg_neg):.1f}", flush=True)
    print(f"  Cross-links u-nb<->v-nb: pos={np.mean(cross_pos):.1f}  neg={np.mean(cross_neg):.1f}", flush=True)
    print(f"  Nodes at dist=2 from u: pos={np.mean(u_dist2_pos):.1f}  neg={np.mean(u_dist2_neg):.1f}", flush=True)

    # Correlation: does margin correlate with cross-links?
    with torch.inference_mode():
        all_s = model(val_inp_t.to(device), val_nids_t.to(device)).cpu().numpy()
    margins = all_s[:, 0] - all_s[:, 1:].max(axis=1)
    print(f"\n  Correlation with margin:", flush=True)
    print(f"    cross_links: r={np.corrcoef(cross_pos, margins)[0,1]:.4f}", flush=True)
    print(f"    u_degree:    r={np.corrcoef(u_deg_pos, margins)[0,1]:.4f}", flush=True)
    print(f"    v_degree:    r={np.corrcoef(v_deg_pos, margins)[0,1]:.4f}", flush=True)
    print(f"    shared_nb:   r={np.corrcoef(shared_nb, margins)[0,1]:.4f}", flush=True)

    print(f"\nDone.", flush=True)
