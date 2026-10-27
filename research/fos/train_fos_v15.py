"""
Temporal LBD: FOS v15 = topology features + relevance grades + LambdaRank loss.

Adapted from train_temporal_v15.py for the FOS Art_Business temporal graph.
Pair features: 35-dim (21 original + 6 gravity + 8 topology)

Usage:
  python train_fos_v15.py --train
  python train_fos_v15.py --eval
"""
import os, time, argparse, json
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = '/root/autodl-tmp/openalex_data/fos_data'

MAX_NODES = 64
MAX_MERGED = 256
EMBED_DIM = 2 * MAX_NODES + 1
SCIBERT_DIM = 768
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8

GNN_HIDDEN = 128
PAIR_DIM = 35
FUSION_DIM = 256

BATCH_SIZE = 8
N_EPOCHS = 30
LR = 3e-4
N_CANDIDATES = 20
PATIENCE = 5

CONCEPT_IDS_FILE = f'{DATA_DIR}/concept_ids.json'


def load_scibert_fos(n_concepts):
    """Load FOS SciBERT embeddings directly (no remapping needed)."""
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    print(f"  SciBERT: loaded {scibert.shape[0]} x {scibert.shape[1]}", flush=True)
    assert scibert.shape[0] == n_concepts, \
        f"SciBERT shape {scibert.shape[0]} != n_concepts {n_concepts}"
    return scibert


# ── LambdaRank loss ───────────────────────────────────────────────────

def lambdarank_loss(scores, labels):
    B, K = scores.shape
    device = scores.device
    sorted_rel = torch.sort(labels, descending=True).values
    discounts = torch.log2(torch.arange(2, K + 2, device=device, dtype=torch.float32))
    gains = 2.0 ** sorted_rel - 1.0
    idcg = (gains / discounts).sum(dim=1).clamp(min=1e-10)
    s_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    rel_diff = labels.unsqueeze(2) - labels.unsqueeze(1)
    pair_mask = (rel_diff > 0).float()
    g_i = 2.0 ** labels - 1.0
    g_j = g_i
    delta_gain = torch.abs(g_i.unsqueeze(2) / discounts[:K].unsqueeze(0).unsqueeze(2)
                           - g_j.unsqueeze(1) / discounts[:K].unsqueeze(0).unsqueeze(1))
    delta_ndcg = delta_gain / idcg.unsqueeze(1).unsqueeze(2)
    rho = torch.sigmoid(-s_diff)
    loss_matrix = delta_ndcg * rho * pair_mask
    total_pairs = pair_mask.sum().clamp(min=1)
    return loss_matrix.sum() / total_pairs


# ── Model ──────────────────────────────────────────────────────────────

class PairMLP(nn.Module):
    def __init__(self, in_dim=PAIR_DIM, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, out_dim), nn.GELU(),
        )

    def forward(self, pair_feats):
        B, K, D = pair_feats.shape
        return self.net(pair_feats.reshape(B * K, D)).reshape(B, K, -1)


class SubgraphGNN(nn.Module):
    def __init__(self, scibert_dim=SCIBERT_DIM, hidden=GNN_HIDDEN):
        super().__init__()
        self.hidden = hidden
        self.node_proj = nn.Linear(scibert_dim + 1, hidden)
        self.w_self = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.w_neigh = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.lns = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.readout = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(2 * hidden, hidden), nn.GELU(),
        )

    def forward(self, scibert_table, merged_nodes, edge_idx, edge_val, u_local, v_local):
        BK, M = merged_nodes.shape
        device = merged_nodes.device
        B = edge_idx.shape[0]
        K_per_sample = BK // B
        batch_offsets = torch.arange(B, device=device).unsqueeze(1).unsqueeze(2) * (K_per_sample * M)
        edge_idx_global = edge_idx + batch_offsets
        flat_idx = edge_idx_global.reshape(2, -1)
        flat_val = edge_val.reshape(-1)
        keep = flat_val != 0
        adj_sparse = torch.sparse_coo_tensor(
            flat_idx[:, keep], flat_val[keep],
            size=(BK * M, BK * M), device=device)
        deg = torch.sparse.sum(adj_sparse, dim=1).to_dense().unsqueeze(1).clamp(min=1)
        row_deg = deg[flat_idx[0, keep]]
        adj_norm = torch.sparse_coo_tensor(
            flat_idx[:, keep], flat_val[keep] / row_deg.squeeze(),
            size=(BK * M, BK * M), device=device)
        nids = merged_nodes.clamp(0, scibert_table.shape[0] - 1)
        emb = scibert_table[nids]
        log_deg = torch.log1p(deg.reshape(BK, M, 1))
        node_feat = torch.cat([emb, log_deg], dim=2)
        x = self.node_proj(node_feat)
        x_flat = x.reshape(BK * M, -1)
        pad_mask = (merged_nodes == 0)
        for i in range(3):
            x_neigh = torch.sparse.mm(adj_norm, x_flat)
            x_self = self.w_self[i](x_flat) + self.w_neigh[i](x_neigh)
            x_flat = self.lns[i](x_flat + F.gelu(x_self))
        x = x_flat.reshape(BK, M, -1)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        BKM = torch.arange(BK, device=device)
        u_emb = x[BKM, u_local]
        v_emb = x[BKM, v_local]
        valid = (~pad_mask).unsqueeze(-1).float()
        mean_pool = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        readout_in = torch.cat([u_emb, v_emb, mean_pool], dim=1)
        return self.readout(readout_in)


class PencilEncoder(nn.Module):
    def __init__(self, scibert_embeddings):
        super().__init__()
        self.register_buffer('scibert_table', scibert_embeddings)
        self.scibert_proj = nn.Linear(SCIBERT_DIM, HIDDEN // 2)
        self.struct_proj = nn.Linear(EMBED_DIM, HIDDEN // 2)
        self.combine = nn.Linear(HIDDEN // 2 + HIDDEN // 2, HIDDEN)
        self.cand_pos_embed = nn.Embedding(128, HIDDEN)
        self.layers = nn.ModuleList()
        for _ in range(N_LAYERS):
            self.layers.append(nn.ModuleDict({
                'qkv': nn.Linear(HIDDEN, HIDDEN * 3),
                'out_proj': nn.Linear(HIDDEN, HIDDEN),
                'ln1': nn.LayerNorm(HIDDEN),
                'ffn': nn.Sequential(
                    nn.Linear(HIDDEN, HIDDEN * 4), nn.GELU(),
                    nn.Linear(HIDDEN * 4, HIDDEN), nn.Dropout(0.1)),
                'ln2': nn.LayerNorm(HIDDEN),
                'mp': nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU()),
            }))
        self.ln_f = nn.LayerNorm(HIDDEN)

    def _build_node_features(self, inp, node_ids):
        B, G, N, _ = inp.shape
        flat_inp = inp.reshape(B * G * N, -1)
        flat_nids = node_ids.reshape(B * G * N)
        struct_emb = self.struct_proj(flat_inp)
        nids_flat = flat_nids.clamp(0, self.scibert_table.shape[0] - 1)
        scibert_emb = self.scibert_proj(self.scibert_table[nids_flat])
        x = self.combine(torch.cat([struct_emb, scibert_emb], dim=-1))
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
        pad_mask = (inp[:, :, :, :2 * MAX_NODES].abs().sum(dim=3) == 0)
        pad_mask_flat = pad_mask.reshape(B, G * N)
        iso_mask = self._build_iso_mask(G - 1, N, device)
        pad_expand = pad_mask_flat.unsqueeze(1)
        attn_allowed = (~pad_expand).expand(B, G * N, G * N)
        iso_allowed = (iso_mask == 0.0).unsqueeze(0)
        bool_mask = (attn_allowed & iso_allowed).unsqueeze(1)
        S = G * N
        for layer in self.layers:
            h = layer['ln1'](x)
            qkv = layer['qkv'](h)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, S, N_HEADS, HIDDEN // N_HEADS).transpose(1, 2)
            k = k.view(B, S, N_HEADS, HIDDEN // N_HEADS).transpose(1, 2)
            v = v.view(B, S, N_HEADS, HIDDEN // N_HEADS).transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bool_mask,
                dropout_p=0.1 if self.training else 0.0)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, HIDDEN)
            h = layer['out_proj'](attn_out)
            x = x + h
            x_cand = x.reshape(B, -1, MAX_NODES, x.shape[-1])
            h_mp = layer['mp'](torch.einsum('bgnh,bgnm->bgmh', x_cand, adj_norm))
            x_cand = x_cand + h_mp
            x = x_cand.reshape(x.shape)
            h = layer['ln2'](x)
            x = x + layer['ffn'](h)
        x = self.ln_f(x)
        x_cand = x.reshape(B, G, N, -1)
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (x_cand * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)
        return pooled


PENCIL_GROUP = 20

class PencilV15(nn.Module):
    def __init__(self, scibert_embeddings):
        super().__init__()
        self.register_buffer('scibert_table', scibert_embeddings)
        self.pair_mlp = PairMLP(PAIR_DIM, 64)
        self.pencil = PencilEncoder(scibert_embeddings)
        self.gnn = SubgraphGNN(SCIBERT_DIM, GNN_HIDDEN)
        fusion_in = 64 + HIDDEN + GNN_HIDDEN
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, FUSION_DIM), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(FUSION_DIM, FUSION_DIM), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(FUSION_DIM, FUSION_DIM), nn.GELU(),
            nn.Linear(FUSION_DIM, 1),
        )

    def forward(self, inp, node_ids, pair_feats, merged_nodes, edge_idx, edge_val,
                u_local, v_local):
        B, G = inp.shape[:2]
        pair_vec = self.pair_mlp(pair_feats)
        pencil_parts = []
        anchor_inp = inp[:, :1]
        anchor_nid = node_ids[:, :1]
        cand_inp = inp[:, 1:]
        cand_nid = node_ids[:, 1:]
        K = cand_inp.shape[1]
        for start in range(0, K, PENCIL_GROUP):
            end = min(start + PENCIL_GROUP, K)
            g_inp = torch.cat([anchor_inp, cand_inp[:, start:end]], dim=1)
            g_nid = torch.cat([anchor_nid, cand_nid[:, start:end]], dim=1)
            pencil_parts.append(self.pencil(g_inp, g_nid))
        anchor_vec = pencil_parts[0][:, :1]
        cand_vecs = [p[:, 1:] for p in pencil_parts]
        pencil_vec = torch.cat([anchor_vec] + cand_vecs, dim=1)
        mn_flat = merged_nodes.reshape(B * G, MAX_MERGED)
        ul_flat = u_local.reshape(B * G)
        vl_flat = v_local.reshape(B * G)
        gnn_vec = self.gnn(self.scibert_table, mn_flat, edge_idx, edge_val,
                           ul_flat, vl_flat)
        gnn_vec = gnn_vec.reshape(B, G, GNN_HIDDEN)
        fused = torch.cat([pair_vec, pencil_vec, gnn_vec], dim=-1)
        fused = self.fusion(fused)
        scores = self.head(fused).squeeze(-1)
        return scores


# ── Dataset ────────────────────────────────────────────────────────────

class AnchorDatasetV15(torch.utils.data.Dataset):
    def __init__(self, pairs_by_src, concept_feats, pair_feats, merged_nodes,
                 deltas, u_local, v_local, adj_csr, n_candidates, rng_seed=42):
        self.anchors = [u for u in sorted(pairs_by_src.keys())
                        if len(pairs_by_src[u]) >= 2]
        self.pairs_by_src = pairs_by_src
        self.concept_feats = concept_feats
        self.pair_feats = pair_feats
        self.merged_nodes = merged_nodes
        self.deltas = deltas
        self.u_local = u_local
        self.v_local = v_local
        self.adj_csr = adj_csr.tocsr().astype(np.float32)
        self.K = n_candidates
        self.rng = np.random.RandomState(rng_seed)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        u = self.anchors[idx]
        cands = self.pairs_by_src[u]
        if len(cands) >= self.K:
            idxs = self.rng.choice(len(cands), self.K, replace=False)
        else:
            idxs = self.rng.choice(len(cands), self.K, replace=True)
        pair_is = [cands[i][0] for i in idxs]
        vs = [cands[i][1] for i in idxs]
        dcs = np.array([self.deltas[pi] for pi in pair_is], dtype=np.float32)
        feats = self.concept_feats[vs]
        nids = np.zeros((len(vs), MAX_NODES), dtype=np.int64)
        for ci, v in enumerate(vs):
            nids[ci, 0] = u
            nids[ci, 1] = v
        pf = np.array(self.pair_feats[pair_is], dtype=np.float32)
        mn = np.array(self.merged_nodes[pair_is], dtype=np.int64)
        ul = self.u_local[pair_is].astype(np.int64)
        vl = self.v_local[pair_is].astype(np.int64)
        edge_rows = []
        edge_cols = []
        edge_vals_list = []
        adj = self.adj_csr
        uniq_set = set()
        nn_per_cand = []
        for ci in range(self.K):
            nodes = mn[ci]
            nn = int(np.count_nonzero(nodes))
            nn_per_cand.append(nn)
            if nn >= 2:
                uniq_set.update(int(x) for x in nodes[:nn])
        if uniq_set:
            uniq = np.array(sorted(uniq_set), dtype=np.int64)
            n2i = {int(n): i for i, n in enumerate(uniq)}
            sub = adj[uniq][:, uniq].toarray()
            for ci in range(self.K):
                nn = nn_per_cand[ci]
                if nn < 2:
                    continue
                sel = mn[ci, :nn]
                idx = np.array([n2i[int(x)] for x in sel], dtype=np.int64)
                local_adj = sub[idx][:, idx]
                r, c = np.where(local_adj != 0)
                if len(r) > 0:
                    edge_rows.append(ci * MAX_MERGED + r.astype(np.int64))
                    edge_cols.append(ci * MAX_MERGED + c.astype(np.int64))
                    edge_vals_list.append(local_adj[r, c].astype(np.float32))
        if edge_rows:
            edge_idx = np.stack([np.concatenate(edge_rows),
                                 np.concatenate(edge_cols)])
            edge_val = np.concatenate(edge_vals_list)
        else:
            edge_idx = np.zeros((2, 0), dtype=np.int64)
            edge_val = np.zeros(0, dtype=np.float32)
        G = 1 + self.K
        out_feats = np.zeros((G, MAX_NODES, EMBED_DIM), dtype=np.float32)
        out_feats[1:] = feats
        out_nids = np.zeros((G, MAX_NODES), dtype=np.int64)
        out_nids[0, 0] = u
        out_nids[1:] = nids
        out_dcs = np.zeros(G, dtype=np.float32)
        out_dcs[1:] = dcs
        out_pf = np.zeros((G, pf.shape[1]), dtype=np.float32)
        out_pf[1:] = pf
        out_mn = np.zeros((G, MAX_MERGED), dtype=np.int32)
        out_mn[0, 0] = u
        out_mn[1:] = mn
        out_ul = np.zeros(G, dtype=np.int64)
        out_ul[1:] = ul
        out_vl = np.zeros(G, dtype=np.int64)
        out_vl[1:] = vl
        return (torch.from_numpy(out_feats),
                torch.from_numpy(out_nids),
                torch.from_numpy(out_dcs),
                torch.from_numpy(out_pf),
                torch.from_numpy(out_mn),
                torch.from_numpy(edge_idx),
                torch.from_numpy(edge_val),
                torch.from_numpy(out_ul),
                torch.from_numpy(out_vl))


def anchor_collate_v15(batch):
    max_e = max(b[5].shape[1] for b in batch)
    B = len(batch)
    edge_idx = torch.zeros(B, 2, max_e, dtype=torch.int64)
    edge_val = torch.zeros(B, max_e, dtype=torch.float32)
    for i, b in enumerate(batch):
        e = b[5].shape[1]
        if e > 0:
            edge_idx[i, :, :e] = b[5]
            edge_val[i, :e] = b[6]
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]),
            torch.stack([b[4] for b in batch]),
            edge_idx, edge_val,
            torch.stack([b[7] for b in batch]),
            torch.stack([b[8] for b in batch]))


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumlen = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumlen.append(total)

    def __len__(self):
        return self.cumlen[-1] if self.cumlen else 0

    def __getitem__(self, idx):
        for i, cum in enumerate(self.cumlen):
            if idx < cum:
                offset = self.cumlen[i - 1] if i > 0 else 0
                return self.datasets[i][idx - offset]
        raise IndexError


# ── Data loading ───────────────────────────────────────────────────────

def load_precomputed_v15(fold, split='train'):
    base = f'{DATA_DIR}/precomputed_fos_v15/fold{fold}'
    t0 = time.time()
    pf = np.load(f'{base}/{split}_pair_feats.npy', mmap_mode='r')
    mn = np.load(f'{base}/{split}_merged_nodes.npy', mmap_mode='r')
    deltas = np.load(f'{base}/{split}_deltas.npy')
    dist_t2 = np.load(f'{base}/{split}_dist_t2.npy')
    u_loc = np.load(f'{base}/{split}_u_local.npy')
    v_loc = np.load(f'{base}/{split}_v_local.npy')
    print(f"    npy loaded: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    with open(f'{base}/{split}_pairs_by_src.json') as f:
        pairs_by_src = json.load(f)
    pairs_by_src = {int(k): v for k, v in pairs_by_src.items()}
    print(f"    json loaded: {time.time()-t0:.1f}s", flush=True)
    print(f"    {split}: {len(deltas):,} pairs, {len(pairs_by_src):,} anchors", flush=True)
    return pairs_by_src, pf, mn, deltas, dist_t2, u_loc, v_loc


# ── NDCG computation ──────────────────────────────────────────────────

def compute_ndcg_at_k(scores, labels, k=10):
    ranked = np.argsort(-scores)[:k]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum((2.0 ** labels[ranked] - 1.0) / discounts)
    ideal = np.sort(labels)[::-1][:k]
    idcg = np.sum((2.0 ** ideal - 1.0) / discounts)
    return dcg / max(idcg, 1e-10)


# ── Train ─────────────────────────────────────────────────────────────

def train_rolling(train_datasets, val_dataset, scibert_table):
    n_train = sum(len(ds) for ds in train_datasets)
    n_val = len(val_dataset)
    print(f"\n{'='*60}\n  Rolling train (FOS v15 LambdaRank): {len(train_datasets)} windows, "
          f"{n_train:,} train, {n_val:,} val\n{'='*60}", flush=True)
    device = torch.device('cuda')

    train_ds = ConcatDataset(train_datasets)
    train_loader = torch.utils.data.DataLoader(
        train_ds, BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=anchor_collate_v15, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=anchor_collate_v15, pin_memory=True)

    model = PencilV15(scibert_table).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)

    # Warmup
    G = 1 + N_CANDIDATES
    dummy_inp = torch.zeros(2, G, MAX_NODES, EMBED_DIM, device=device)
    dummy_nid = torch.zeros(2, G, MAX_NODES, dtype=torch.long, device=device)
    dummy_pf = torch.zeros(2, G, PAIR_DIM, device=device)
    dummy_mn = torch.zeros(2, G, MAX_MERGED, dtype=torch.long, device=device)
    dummy_ei = torch.zeros(2, 2, 10, dtype=torch.long, device=device)
    dummy_ev = torch.zeros(2, 10, device=device)
    dummy_ul = torch.zeros(2, G, dtype=torch.long, device=device)
    dummy_vl = torch.ones(2, G, dtype=torch.long, device=device)
    print("  Warmup starting...", flush=True)
    try:
        model.train()
        _ = model(dummy_inp, dummy_nid, dummy_pf, dummy_mn, dummy_ei, dummy_ev, dummy_ul, dummy_vl)
        torch.cuda.synchronize()
        print("  Warmup done.", flush=True)
    except Exception as e:
        print(f"  WARMUP ERROR: {e}", flush=True)
        import traceback; traceback.print_exc()
        return

    model_dir = f'{DATA_DIR}/model_fos_v15'
    os.makedirs(model_dir, exist_ok=True)
    best_ndcg = -1
    patience_counter = 0

    print(f"\n--- Training (FOS v15 LambdaRank, {N_CANDIDATES} cands, bs={BATCH_SIZE}) ---",
          flush=True)

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, nb = 0.0, 0

        for batch_idx, (inp, node_ids, dcs, pf, mn, edge_idx, edge_val, ul, vl) in enumerate(train_loader):
            inp = inp.to(device, non_blocking=True)
            node_ids = node_ids.to(device, non_blocking=True)
            dcs = dcs.to(device, non_blocking=True)
            pf = pf.to(device, non_blocking=True)
            mn = mn.to(device, non_blocking=True)
            edge_idx = edge_idx.to(device, non_blocking=True)
            edge_val = edge_val.to(device, non_blocking=True)
            ul = ul.to(device, non_blocking=True)
            vl = vl.to(device, non_blocking=True)

            try:
                scores = model(inp, node_ids, pf, mn, edge_idx, edge_val, ul, vl)
                loss = lambdarank_loss(scores, dcs)
            except Exception as e:
                print(f"  BATCH {batch_idx} ERROR: {e}", flush=True)
                import traceback; traceback.print_exc()
                return

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1
            if nb % 50 == 0:
                print(f"    batch {nb} ({time.time()-t0:.0f}s elapsed)", flush=True)

        sched.step()
        dt_train = time.time() - t0

        # Validation
        model.eval()
        all_ndcg = []
        all_mrr = []
        recall_dict = {k: [] for k in [10, 15, 20, 30, 50, 100]}
        prec_dict = {k: [] for k in [10, 15, 20, 30, 50, 100]}
        with torch.no_grad():
            for inp, node_ids, dcs, pf, mn, edge_idx, edge_val, ul, vl in val_loader:
                inp = inp.to(device, non_blocking=True)
                node_ids = node_ids.to(device, non_blocking=True)
                pf = pf.to(device, non_blocking=True)
                mn = mn.to(device, non_blocking=True)
                edge_idx = edge_idx.to(device, non_blocking=True)
                edge_val = edge_val.to(device, non_blocking=True)
                ul = ul.to(device, non_blocking=True)
                vl = vl.to(device, non_blocking=True)

                scores = model(inp, node_ids, pf, mn, edge_idx, edge_val, ul, vl)
                scores_np = scores.float().cpu().numpy()
                rel_np = dcs.numpy()

                for b in range(scores_np.shape[0]):
                    ranked = np.argsort(-scores_np[b])
                    ndcg = compute_ndcg_at_k(scores_np[b], rel_np[b], k=10)
                    all_ndcg.append(ndcg)

                    pos_mask = rel_np[b] >= 4
                    n_pos = int(pos_mask.sum())
                    if n_pos == 0:
                        continue

                    for k_val in [10, 15, 20, 30, 50, 100]:
                        k_actual = min(k_val, len(ranked))
                        hits_k = int(pos_mask[ranked[:k_actual]].sum())
                        recall_dict[k_val].append(hits_k / n_pos)
                        prec_dict[k_val].append(hits_k / k_actual)

                    for rank, idx in enumerate(ranked):
                        if pos_mask[idx]:
                            all_mrr.append(1.0 / (rank + 1))
                            break

        ndcg_mean = np.mean(all_ndcg) if all_ndcg else 0
        mrr = np.mean(all_mrr) if all_mrr else 0
        n_with_pos = len(all_mrr)
        recall_means = {k: np.mean(v) if v else 0 for k, v in recall_dict.items()}
        prec_means = {k: np.mean(v) if v else 0 for k, v in prec_dict.items()}
        dt_eval = time.time() - t0 - dt_train

        marker = ""
        if ndcg_mean > best_ndcg:
            best_ndcg = ndcg_mean
            patience_counter = 0
            torch.save(model.state_dict(), f'{model_dir}/best.pt')
            marker = " *"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1:2d}/{N_EPOCHS}  loss={total_loss/max(nb,1):.6f}  "
              f"NDCG@10={ndcg_mean:.4f}  MRR={mrr:.4f}  "
              f"R@10={recall_means[10]:.3f}  R@20={recall_means[20]:.3f}  "
              f"R@50={recall_means[50]:.3f}  "
              f"P@10={prec_means[10]:.3f}  P@20={prec_means[20]:.3f}  "
              f"({n_with_pos} pos)  "
              f"train={dt_train:.0f}s  eval={dt_eval:.0f}s{marker}", flush=True)
        torch.save(model.state_dict(), f'{model_dir}/last.pt')

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (patience={PATIENCE})", flush=True)
            break

    print(f"\n  Best NDCG@10: {best_ndcg:.4f}", flush=True)
    return f'{model_dir}/best.pt'


# ── PPR ego-graph builders ────────────────────────────────────────────

def build_concept_feature_table(ppr_mat, n_concepts, device):
    MN = MAX_NODES
    ppr_work = ppr_mat.copy()
    np.fill_diagonal(ppr_work, 0)
    concept_nids = np.zeros((n_concepts, MN), dtype=np.int64)
    concept_nids[:, 0] = np.arange(n_concepts)
    for c in range(n_concepts):
        n_neigh = min(n_concepts - 1, MN - 1)
        if n_neigh < n_concepts - 1:
            top_idx = np.argpartition(ppr_work[c], -n_neigh)[-n_neigh:]
        else:
            top_idx = np.argsort(ppr_work[c])[-n_neigh:]
        concept_nids[c, 1:1 + n_neigh] = top_idx
    concept_feats = np.zeros((n_concepts, MN, EMBED_DIM), dtype=np.float32)
    pos_idx = np.arange(MN)
    concept_feats[:, pos_idx, pos_idx] = 1.0
    nids_t = torch.from_numpy(concept_nids).to(device)
    ppr_gpu = torch.from_numpy(ppr_work.astype(np.float32)).to(device)
    ppr_vals = ppr_gpu[torch.arange(n_concepts, device=device)[:, None], nids_t]
    ppr_log = torch.log10(ppr_vals.clamp(min=1e-10))
    ppr_min = ppr_log.min(dim=1, keepdim=True).values
    ppr_max = ppr_log.max(dim=1, keepdim=True).values
    ppr_norm = (ppr_log - ppr_min) / (ppr_max - ppr_min).clamp(min=1e-10)
    concept_feats[:, :, 2 * MN] = ppr_norm.cpu().numpy()
    del ppr_gpu, ppr_work
    return concept_feats, concept_nids


def fill_adjacency_features(concept_feats, concept_nids, adj):
    MN = MAX_NODES
    n_concepts = len(concept_nids)
    adj_csc = adj.tocsc()
    for c in range(n_concepts):
        nids_c = concept_nids[c]
        cols = adj_csc[:, nids_c].toarray()
        for i in range(MN):
            col_i = cols[:, i]
            for j in range(MN):
                if i != j and col_i[nids_c[j]]:
                    concept_feats[c, i, MN + j] = 1.0


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    with open(CONCEPT_IDS_FILE) as f:
        concept_ids = json.load(f)
    n_concepts = len(concept_ids)
    print(f"  {n_concepts} concepts", flush=True)

    scibert_table = load_scibert_fos(n_concepts)

    rolling_folds = [
        (0, 2005, 2010),
        (1, 2010, 2015),
        (2, 2015, 2020),
        (3, 2020, 2024),
    ]

    def build_or_cache_feats(year, adj, ppr_mat, n_concepts, device):
        cache_path = f'{DATA_DIR}/concept_feats_fos_{year}.npy'
        if os.path.exists(cache_path):
            print(f"  Loading cached concept_feats {year}...", flush=True)
            return np.load(cache_path)
        print(f"  Building concept_feats {year}...", flush=True)
        t0 = time.time()
        feats, nids = build_concept_feature_table(ppr_mat, n_concepts, device)
        fill_adjacency_features(feats, nids, adj)
        np.save(cache_path, feats)
        print(f"  Cached ({time.time()-t0:.1f}s)", flush=True)
        return feats

    if args.train:
        years_needed = sorted(set(t1 for _, t1, _ in rolling_folds))
        concept_feats = {}
        adjs = {}
        for year in years_needed:
            print(f"Loading graph_fos_{year}...", flush=True)
            adj = sp.load_npz(f'{DATA_DIR}/graph_fos_{year}.npz').tocsr()
            adjs[year] = adj
            ppr = np.load(f'{DATA_DIR}/ppr_fos_{year}.npy').astype(np.float32)
            concept_feats[year] = build_or_cache_feats(year, adj, ppr, n_concepts, device)
            del ppr

        train_datasets = []
        for fold_idx, t1, t2 in rolling_folds:
            print(f"\n  Building dataset for fold {fold_idx} ({t1}->{t2})...", flush=True)
            src, pf, mn, deltas, dist_t2, ul, vl = load_precomputed_v15(fold_idx, 'train')
            ds = AnchorDatasetV15(src, concept_feats[t1], pf, mn, deltas,
                                  ul, vl, adjs[t1], N_CANDIDATES,
                                  rng_seed=42 + fold_idx)
            print(f"    {len(ds):,} anchors", flush=True)
            if fold_idx < 3:
                train_datasets.append(ds)
            else:
                val_dataset = ds

        train_rolling(train_datasets, val_dataset, scibert_table)

    elif args.eval:
        t1_val = 2020
        adj_val = sp.load_npz(f'{DATA_DIR}/graph_fos_{t1_val}.npz').tocsr()
        ppr_val = np.load(f'{DATA_DIR}/ppr_fos_{t1_val}.npy').astype(np.float32)
        concept_feats_val = build_or_cache_feats(t1_val, adj_val, ppr_val, n_concepts, device)
        del ppr_val

        src_val, pf_val, mn_val, deltas_val, dist_t2_val, ul_val, vl_val = load_precomputed_v15(3, 'train')
        val_ds = AnchorDatasetV15(src_val, concept_feats_val, pf_val,
                                  mn_val, deltas_val, ul_val, vl_val,
                                  adj_val, N_CANDIDATES, rng_seed=99)

        model = PencilV15(scibert_table).to(device)
        model_file = f'{DATA_DIR}/model_fos_v15/best.pt'
        if os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file, weights_only=True))
        model.eval()

        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
            collate_fn=anchor_collate_v15, pin_memory=True, persistent_workers=True,
            prefetch_factor=4)

        all_ndcg = []
        all_mrr = []
        recall_dict = {k: [] for k in [10, 15, 20, 30, 50, 100]}
        prec_dict = {k: [] for k in [10, 15, 20, 30, 50, 100]}
        with torch.no_grad():
            for inp, node_ids, dcs, pf, mn, edge_idx, edge_val, ul, vl in val_loader:
                inp = inp.to(device)
                node_ids = node_ids.to(device)
                pf = pf.to(device)
                mn = mn.to(device)
                edge_idx = edge_idx.to(device)
                edge_val = edge_val.to(device)
                ul = ul.to(device)
                vl = vl.to(device)
                scores = model(inp, node_ids, pf, mn, edge_idx, edge_val, ul, vl)
                scores_np = scores.float().cpu().numpy()
                dcs_np = dcs.numpy()
                for b in range(scores_np.shape[0]):
                    ranked = np.argsort(-scores_np[b])
                    ndcg = compute_ndcg_at_k(scores_np[b], dcs_np[b], k=10)
                    all_ndcg.append(ndcg)
                    pos_mask = dcs_np[b] >= 4
                    n_pos = int(pos_mask.sum())
                    if n_pos == 0:
                        continue
                    for k_val in [10, 15, 20, 30, 50, 100]:
                        k_actual = min(k_val, len(ranked))
                        hits_k = int(pos_mask[ranked[:k_actual]].sum())
                        recall_dict[k_val].append(hits_k / n_pos)
                        prec_dict[k_val].append(hits_k / k_actual)
                    for rank, idx in enumerate(ranked):
                        if pos_mask[idx]:
                            all_mrr.append(1.0 / (rank + 1))
                            break

        print(f"\n=== Eval fold 3 (FOS v15 LambdaRank) ===", flush=True)
        print(f"  NDCG@10={np.mean(all_ndcg):.4f}  MRR={np.mean(all_mrr):.4f}", flush=True)
        for k_val in [10, 15, 20, 30, 50, 100]:
            r = np.mean(recall_dict[k_val]) if recall_dict[k_val] else 0
            p = np.mean(prec_dict[k_val]) if prec_dict[k_val] else 0
            print(f"  R@{k_val}={r:.3f}  P@{k_val}={p:.3f}", flush=True)

    else:
        print("Use --train or --eval", flush=True)


if __name__ == '__main__':
    main()
