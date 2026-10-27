"""
Temporal LBD: v9 = BFS labels + pair features + SubgraphGNN + PPR ego-graph.

Architecture (three-path fusion):
  Path 1 (PairMLP): 15-dim hand-crafted pair features → 64-dim
  Path 2 (PencilEncoder): PPR ego-graph 64 nodes + attention + MP → 256-dim
  Path 3 (SubgraphGNN): merged 2-hop subgraph 512 nodes + GraphSAGE → 128-dim
  Fusion: 64 + 256 + 128 = 448 → 256 → 1 (pairwise ranking loss)

Labels: BFS distance change (delta>=2 = positive, delta=0 = negative)

Requires precomputed data from precompute_v9.py:
  /root/autodl-tmp/openalex_data/pencil_data/temporal_v2/precomputed_v9/fold{fold}/

Usage:
  python train_temporal_v9.py --fold 0
  python train_temporal_v9.py --train-all
  python train_temporal_v9.py --eval --fold 0
"""
import os, time, argparse, json
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

HAS_SDPA = hasattr(F, 'scaled_dot_product_attention')

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
TEMP_DIR = f'{DATA_DIR}/temporal_l3'

MAX_NODES = 64          # PENCIL ego-graph nodes
MAX_MERGED = 256        # GNN merged subgraph nodes
EMBED_DIM = 2 * MAX_NODES + 1   # 129
SCIBERT_DIM = 768
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8

GNN_HIDDEN = 128
PAIR_DIM = 15
FUSION_DIM = 256

BATCH_SIZE = 8
N_EPOCHS = 30
LR = 3e-4
N_CANDIDATES = 100
PATIENCE = 5

CONCEPT_IDS_FILE = f'{DATA_DIR}/concept_ids_l3.json'
OLD_CONCEPT_IDS_FILE = f'{DATA_DIR}/concept_ids.json'


def load_scibert_remapped(concept_ids, n_concepts):
    scibert_old = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    with open(OLD_CONCEPT_IDS_FILE) as f:
        old_ids = json.load(f)
    old_id_to_idx = {cid: i for i, cid in enumerate(old_ids)}
    scibert = torch.zeros(n_concepts, SCIBERT_DIM)
    mapped = 0
    for new_idx, cid in enumerate(concept_ids):
        if cid in old_id_to_idx:
            scibert[new_idx] = scibert_old[old_id_to_idx[cid]]
            mapped += 1
    print(f"  SciBERT: mapped {mapped}/{n_concepts} concepts", flush=True)
    return scibert


# ── Pairwise ranking loss ──────────────────────────────────────────────

def pairwise_ranking_loss(scores, targets):
    """All-pairs ranking loss: push positive (delta>=2) above negative (delta=0)."""
    B, K = scores.shape
    total_loss = scores.sum() * 0.0
    total_pairs = torch.tensor(0, device=scores.device)
    for b in range(B):
        s = scores[b]
        r = targets[b]
        r_diff = r.unsqueeze(0) - r.unsqueeze(1)
        s_diff = s.unsqueeze(0) - s.unsqueeze(1)
        pair_mask = (r_diff > 0).float()
        npairs = pair_mask.sum()
        if npairs == 0:
            continue
        total_loss = total_loss + (F.relu(1.0 - s_diff) * pair_mask).sum()
        total_pairs = total_pairs + npairs
    return total_loss / total_pairs.clamp(min=1)


# ── Path 1: PairMLP ────────────────────────────────────────────────────

class PairMLP(nn.Module):
    """Encode 15-dim pair features into 64-dim vector."""
    def __init__(self, in_dim=PAIR_DIM, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, out_dim), nn.GELU(),
        )

    def forward(self, pair_feats):
        """pair_feats: [B, K, PAIR_DIM] → [B, K, out_dim]"""
        B, K, D = pair_feats.shape
        return self.net(pair_feats.reshape(B * K, D)).reshape(B, K, -1)


# ── Path 3: SubgraphGNN ────────────────────────────────────────────────

class SubgraphGNN(nn.Module):
    """3-layer GraphSAGE on merged 2-hop subgraph.

    Input: SciBERT embeddings for nodes in merged subgraph + degree features.
    Readout: cat(node_emb[u], node_emb[v], mean_pool) → 128-dim.
    """
    def __init__(self, scibert_dim=SCIBERT_DIM, hidden=GNN_HIDDEN):
        super().__init__()
        self.hidden = hidden
        # Node feature projection: SciBERT(768) + degree(1) → hidden
        self.node_proj = nn.Linear(scibert_dim + 1, hidden)
        # 3 GraphSAGE-style layers
        self.w_self = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.w_neigh = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.lns = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        # Readout: 3 * hidden → hidden
        self.readout = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(2 * hidden, hidden), nn.GELU(),
        )

    def forward(self, scibert_table, merged_nodes, edge_idx, edge_val, u_local, v_local):
        """
        Args:
            scibert_table: [n_concepts, 768] global embeddings
            merged_nodes: [B*K, MAX_MERGED] int32 node indices
            edge_idx: [B, 2, E] int64 sparse edge indices (flattened)
            edge_val: [B, E] float32 sparse edge values
            u_local: [B*K] int32 local index of u
            v_local: [B*K] int32 local index of v
        Returns:
            [B*K, hidden] GNN vector per pair
        """
        BK, M = merged_nodes.shape
        device = merged_nodes.device

        # Build sparse adjacency [BK*M, BK*M] from edge lists
        B = edge_idx.shape[0]
        K_per_sample = BK // B
        batch_offsets = torch.arange(B, device=device).unsqueeze(1).unsqueeze(2) * (K_per_sample * M)
        edge_idx_global = edge_idx + batch_offsets  # [B, 2, E]
        flat_idx = edge_idx_global.reshape(2, -1)    # [2, B*E]
        flat_val = edge_val.reshape(-1)              # [B*E]
        # Remove padding (idx == 0 AND val == 0) — all-zero rows stay out
        keep = flat_val != 0
        adj_sparse = torch.sparse_coo_tensor(
            flat_idx[:, keep], flat_val[keep],
            size=(BK * M, BK * M), device=device)

        # Normalize: D^{-1} A
        deg = torch.sparse.sum(adj_sparse, dim=1).to_dense().unsqueeze(1).clamp(min=1)
        # Scale each edge value by 1/deg(row)
        row_deg = deg[flat_idx[0, keep]]
        adj_norm = torch.sparse_coo_tensor(
            flat_idx[:, keep], flat_val[keep] / row_deg.squeeze(),
            size=(BK * M, BK * M), device=device)

        # Node features: SciBERT + log(degree)
        nids = merged_nodes.clamp(0, scibert_table.shape[0] - 1)
        emb = scibert_table[nids]  # [BK, M, 768]
        log_deg = torch.log1p(deg.reshape(BK, M, 1))
        node_feat = torch.cat([emb, log_deg], dim=2)  # [BK, M, 769]
        x = self.node_proj(node_feat)  # [BK, M, hidden]
        x_flat = x.reshape(BK * M, -1)  # [BK*M, hidden]

        # Pad mask
        pad_mask = (merged_nodes == 0)

        # 3-layer sparse message passing
        for i in range(3):
            x_neigh = torch.sparse.mm(adj_norm, x_flat)  # [BK*M, hidden]
            x_self = self.w_self[i](x_flat) + self.w_neigh[i](x_neigh)
            x_flat = self.lns[i](x_flat + F.gelu(x_self))

        x = x_flat.reshape(BK, M, -1)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # Readout: cat(u_emb, v_emb, mean_pool)
        BKM = torch.arange(BK, device=device)
        u_emb = x[BKM, u_local]
        v_emb = x[BKM, v_local]
        valid = (~pad_mask).unsqueeze(-1).float()
        mean_pool = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        readout_in = torch.cat([u_emb, v_emb, mean_pool], dim=1)
        return self.readout(readout_in)


# ── Path 2: PencilEncoder (modified from v7) ───────────────────────────

class PencilEncoder(nn.Module):
    """PPR ego-graph encoder: 64 nodes + Flash Attention + graph MP → 256-dim."""
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
        """Returns pooled representation [B, G, HIDDEN]."""
        B, G, N, _ = inp.shape
        device = inp.device
        x = self._build_node_features(inp, node_ids)
        adj = inp[:, :, :, MAX_NODES:MAX_NODES + N]
        deg = adj.sum(dim=3, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        pad_mask = (inp[:, :, :, :2 * MAX_NODES].abs().sum(dim=3) == 0)
        pad_mask_flat = pad_mask.reshape(B, G * N)
        iso_mask = self._build_iso_mask(G - 1, N, device)

        # Attention mask
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
        return pooled  # [B, G, HIDDEN]


# ── V9 Model: Fusion of 3 paths ────────────────────────────────────────

PENCIL_GROUP = 20   # candidates per PencilEncoder call (keeps S=704, enables FlashAttn)

class PencilV9(nn.Module):
    """Fusion model: PairMLP + PencilEncoder + SubgraphGNN → scoring head."""
    def __init__(self, scibert_embeddings):
        super().__init__()
        self.register_buffer('scibert_table', scibert_embeddings)

        self.pair_mlp = PairMLP(PAIR_DIM, 64)
        self.pencil = PencilEncoder(scibert_embeddings)
        self.gnn = SubgraphGNN(SCIBERT_DIM, GNN_HIDDEN)

        fusion_in = 64 + HIDDEN + GNN_HIDDEN  # 448
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
        """
        Args:
            inp: [B, G, N, EMBED_DIM] PENCIL ego-graph features
            node_ids: [B, G, N] node IDs in ego-graph
            pair_feats: [B, G, PAIR_DIM] pair features
            merged_nodes: [B, G, MAX_MERGED] merged subgraph node indices
            edge_idx: [B, 2, E] sparse edge indices
            edge_val: [B, E] sparse edge values
            u_local: [B, G] local index of u in merged subgraph
            v_local: [B, G] local index of v in merged subgraph
        Returns:
            scores: [B, G] per-candidate scores
        """
        B, G = inp.shape[:2]
        device = inp.device

        # Path 1: Pair features
        pair_vec = self.pair_mlp(pair_feats)  # [B, G, 64]

        # Path 2: PENCIL ego-graph (grouped to enable Flash Attention)
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

        # Path 3: SubgraphGNN (sparse)
        mn_flat = merged_nodes.reshape(B * G, MAX_MERGED)
        ul_flat = u_local.reshape(B * G)
        vl_flat = v_local.reshape(B * G)
        gnn_vec = self.gnn(self.scibert_table, mn_flat, edge_idx, edge_val,
                           ul_flat, vl_flat)
        gnn_vec = gnn_vec.reshape(B, G, GNN_HIDDEN)

        # Fusion
        fused = torch.cat([pair_vec, pencil_vec, gnn_vec], dim=-1)
        fused = self.fusion(fused)
        scores = self.head(fused).squeeze(-1)
        return scores


# ── Dataset (v9: with pair features + merged subgraph) ─────────────────



class AnchorDatasetV9(torch.utils.data.Dataset):
    def __init__(self, pairs_by_src, concept_feats, pair_feats, merged_nodes,
                 deltas, u_local, v_local, adj_csr, n_candidates, rng_seed=42):
        # Filter: need >=2 candidates AND at least 1 positive (delta >= 2)
        self.anchors = [u for u in sorted(pairs_by_src.keys())
                        if len(pairs_by_src[u]) >= 2
                        and any(deltas[p[0]] >= 2 for p in pairs_by_src[u])]
        self.pairs_by_src = pairs_by_src
        self.concept_feats = concept_feats
        self.pair_feats = pair_feats
        self.merged_nodes = merged_nodes
        self.deltas = deltas
        self.u_local = u_local
        self.v_local = v_local
        self.adj_csr = adj_csr.tocsr().astype(np.float32)  # sparse, ~10MB
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

        pf = self.pair_feats[pair_is]
        mn = self.merged_nodes[pair_is]
        ul = self.u_local[pair_is].astype(np.int64)
        vl = self.v_local[pair_is].astype(np.int64)

        # Build sparse edge list (much smaller than dense [K, 512, 512])
        edge_rows = []
        edge_cols = []
        edge_vals_list = []
        adj = self.adj_csr
        # Collect unique nodes from all candidates
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
            sub = adj[uniq][:, uniq].toarray()  # [U, U] dense
            for ci in range(self.K):
                nn = nn_per_cand[ci]
                if nn < 2:
                    continue
                sel = mn[ci, :nn]
                idx = np.array([n2i[int(x)] for x in sel], dtype=np.int64)
                local_adj = sub[idx][:, idx]  # [nn, nn]
                r, c = np.where(local_adj != 0)
                if len(r) > 0:
                    edge_rows.append(ci * MAX_MERGED + r.astype(np.int64))
                    edge_cols.append(ci * MAX_MERGED + c.astype(np.int64))
                    edge_vals_list.append(local_adj[r, c].astype(np.float32))

        # Build sparse edge tensors
        if edge_rows:
            edge_idx = np.stack([np.concatenate(edge_rows),
                                 np.concatenate(edge_cols)])  # [2, E]
            edge_val = np.concatenate(edge_vals_list)  # [E]
        else:
            edge_idx = np.zeros((2, 0), dtype=np.int64)
            edge_val = np.zeros(0, dtype=np.float32)

        # Prepend anchor (index 0 = query node u, zero features)
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
        # edge_idx/edge_val already computed above (sparse format)
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


def anchor_collate_v9(batch):
    # Edge lists are variable length → pad to max in batch
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


# ── Precomputed data loading ───────────────────────────────────────────

def load_precomputed_v9(fold, split):
    """Load precomputed v9 data. Uses mmap for large arrays."""
    base = f'{TEMP_DIR}/precomputed_v9/fold{fold}'
    t0 = time.time()
    pf = np.load(f'{base}/{split}_pair_feats.npy', mmap_mode='r')
    mn = np.load(f'{base}/{split}_merged_nodes.npy', mmap_mode='r')
    deltas = np.load(f'{base}/{split}_deltas.npy')
    u_loc = np.load(f'{base}/{split}_u_local.npy')
    v_loc = np.load(f'{base}/{split}_v_local.npy')
    dt = time.time() - t0
    print(f"    npy loaded: {dt:.1f}s", flush=True)

    t0 = time.time()
    with open(f'{base}/{split}_pairs_by_src.json') as f:
        pairs_by_src = json.load(f)
    pairs_by_src = {int(k): v for k, v in pairs_by_src.items()}
    dt = time.time() - t0
    print(f"    json loaded: {dt:.1f}s", flush=True)
    print(f"    {split}: {len(deltas):,} pairs, {len(pairs_by_src):,} anchors", flush=True)
    return pairs_by_src, pf, mn, deltas, u_loc, v_loc


# ── Train ──────────────────────────────────────────────────────────────

def train_fold(fold, concept_feats_train, concept_feats_val,
               adj_train, adj_val, scibert_table):
    print(f"\n{'='*60}\n  Fold {fold} (v9: BFS label + pair features + SubgraphGNN)\n{'='*60}",
          flush=True)
    device = torch.device('cuda')

    # Load precomputed v9 data
    print("  Loading precomputed train data...", flush=True)
    (src_train, pf_train, mn_train,
     deltas_train, ul_train, vl_train) = load_precomputed_v9(fold, 'train')

    print("  Loading precomputed val data...", flush=True)
    (src_val, pf_val, mn_val,
     deltas_val, ul_val, vl_val) = load_precomputed_v9(fold, 'val')

    train_ds = AnchorDatasetV9(src_train, concept_feats_train, pf_train,
                               mn_train, deltas_train, ul_train, vl_train,
                               adj_train, N_CANDIDATES)
    train_loader = torch.utils.data.DataLoader(
        train_ds, BATCH_SIZE, shuffle=True, num_workers=4,
        collate_fn=anchor_collate_v9, pin_memory=True, persistent_workers=True)

    val_ds = AnchorDatasetV9(src_val, concept_feats_val, pf_val,
                             mn_val, deltas_val, ul_val, vl_val,
                             adj_val, N_CANDIDATES, rng_seed=99)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        collate_fn=anchor_collate_v9, pin_memory=True, persistent_workers=True)

    model = PencilV9(scibert_table).to(device)
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
    model.train()
    _ = model(dummy_inp, dummy_nid, dummy_pf, dummy_mn, dummy_ei, dummy_ev, dummy_ul, dummy_vl)
    torch.cuda.synchronize()

    model_dir = f'{TEMP_DIR}/model_fold{fold}_v9'
    os.makedirs(model_dir, exist_ok=True)
    best_disc = -1
    patience_counter = 0

    print(f"\n--- Training (v9, {N_CANDIDATES} cands, bs={BATCH_SIZE}, patience={PATIENCE}) ---",
          flush=True)

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, nb = 0.0, 0

        for inp, node_ids, dcs, pf, mn, edge_idx, edge_val, ul, vl in train_loader:
            inp = inp.to(device)
            node_ids = node_ids.to(device)
            dcs = dcs.to(device)
            pf = pf.to(device)
            mn = mn.to(device)
            edge_idx = edge_idx.to(device)
            edge_val = edge_val.to(device)
            ul = ul.to(device)
            vl = vl.to(device)

            scores = model(inp, node_ids, pf, mn, edge_idx, edge_val, ul, vl)
            loss = pairwise_ranking_loss(scores, dcs)

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
        all_top_means = []
        all_bot_means = []
        with torch.no_grad():
            for vi, (inp, node_ids, dcs, pf, mn, edge_idx, edge_val, ul, vl) in enumerate(val_loader):
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
                    K = len(ranked)
                    top_k = max(1, K // 10)
                    all_top_means.append(dcs_np[b, ranked[:top_k]].mean())
                    all_bot_means.append(dcs_np[b, ranked[top_k:]].mean())
                if (vi + 1) % 100 == 0:
                    print(f"    val batch {vi+1} ({time.time()-t0:.0f}s)", flush=True)

        top_mean = np.mean(all_top_means)
        bot_mean = np.mean(all_bot_means)
        disc = top_mean / max(abs(bot_mean), 1e-8)
        dt_eval = time.time() - t0 - dt_train

        marker = ""
        if disc > best_disc:
            best_disc = disc
            patience_counter = 0
            torch.save(model.state_dict(), f'{model_dir}/best.pt')
            marker = " *"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1:2d}/{N_EPOCHS}  loss={total_loss/max(nb,1):.6f}  "
              f"disc={disc:.3f}  top={top_mean:.6f}  bot={bot_mean:.6f}  "
              f"train={dt_train:.1f}s  eval={dt_eval:.1f}s{marker}", flush=True)
        torch.save(model.state_dict(), f'{model_dir}/last.pt')

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (patience={PATIENCE})", flush=True)
            break

    print(f"\n  Best disc: {best_disc:.3f}", flush=True)
    return f'{model_dir}/best.pt'


# ── Eval ───────────────────────────────────────────────────────────────

def eval_fold(fold, concept_feats_val, adj_val, scibert_table):
    print(f"\n{'='*60}\n  Eval fold {fold} (v9)\n{'='*60}", flush=True)
    device = torch.device('cuda')

    (src_val, pf_val, mn_val,
     deltas_val, ul_val, vl_val) = load_precomputed_v9(fold, 'val')

    val_ds = AnchorDatasetV9(src_val, concept_feats_val, pf_val,
                             mn_val, deltas_val, ul_val, vl_val,
                             adj_val, N_CANDIDATES, rng_seed=99)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        collate_fn=anchor_collate_v9, pin_memory=True, persistent_workers=True)

    model = PencilV9(scibert_table).to(device)
    model_file = f'{TEMP_DIR}/model_fold{fold}_v9/best.pt'
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, weights_only=True))
        print(f"  Loaded {model_file}", flush=True)
    else:
        print(f"  WARNING: {model_file} not found, using random weights", flush=True)
    model.eval()

    from scipy.stats import spearmanr
    all_top_means = []
    all_bot_means = []
    all_scores_list = []
    all_labels_list = []

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
                K = len(ranked)
                top_k = max(1, K // 10)
                all_top_means.append(dcs_np[b, ranked[:top_k]].mean())
                all_bot_means.append(dcs_np[b, ranked[top_k:]].mean())
            all_scores_list.append(scores_np.flatten())
            all_labels_list.append(dcs_np.flatten())

    top_mean = np.mean(all_top_means)
    bot_mean = np.mean(all_bot_means)
    disc = top_mean / max(abs(bot_mean), 1e-8)

    all_scores = np.concatenate(all_scores_list)
    all_labels = np.concatenate(all_labels_list)
    sp_corr, _ = spearmanr(all_scores, all_labels)

    print(f"\n=== Eval fold {fold} (v9) ===", flush=True)
    print(f"  Disc ratio: {disc:.3f}", flush=True)
    print(f"  Spearman: {sp_corr:.4f}", flush=True)
    print(f"  top_mean: {top_mean:.6f}, bot_mean: {bot_mean:.6f}", flush=True)
    return disc


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--train-all', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train-graph-year', type=int, default=2005)
    parser.add_argument('--val-graph-year', type=int, default=2015)
    args = parser.parse_args()

    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    print("Loading v2 concept IDs...", flush=True)
    with open(CONCEPT_IDS_FILE) as f:
        concept_ids = json.load(f)
    n_concepts = len(concept_ids)
    print(f"  {n_concepts} concepts", flush=True)

    print("Loading SciBERT (remapped)...", flush=True)
    scibert_table = load_scibert_remapped(concept_ids, n_concepts)

    # Build or load cached PPR ego-graph features
    def build_or_cache_feats(year, adj, ppr_mat, n_concepts, device):
        cache_path = f'{TEMP_DIR}/concept_feats_v9_{year}.npy'
        if os.path.exists(cache_path):
            print(f"  Loading cached concept_feats {year}...", flush=True)
            t0 = time.time()
            feats = np.load(cache_path)
            print(f"    Done: {time.time()-t0:.1f}s", flush=True)
            return feats
        print(f"  Building concept_feats {year} (first run, will cache)...", flush=True)
        t0 = time.time()
        feats, nids = build_concept_feature_table(ppr_mat, n_concepts, device)
        fill_adjacency_features(feats, nids, adj)
        np.save(cache_path, feats)
        print(f"  Cached → {cache_path} ({time.time()-t0:.1f}s)", flush=True)
        return feats

    print(f"Loading graph_v2_{args.train_graph_year}...", flush=True)
    adj_train = sp.load_npz(f'{TEMP_DIR}/graph_v2_{args.train_graph_year}.npz').tocsr()
    print(f"  {adj_train.shape[0]} nodes, {adj_train.nnz:,} edges", flush=True)

    ppr_train = np.load(f'{TEMP_DIR}/ppr_v2_{args.train_graph_year}.npy').astype(np.float32)
    concept_feats_train = build_or_cache_feats(
        args.train_graph_year, adj_train, ppr_train, n_concepts, device)
    del ppr_train

    print(f"Loading graph_v2_{args.val_graph_year}...", flush=True)
    adj_val = sp.load_npz(f'{TEMP_DIR}/graph_v2_{args.val_graph_year}.npz').tocsr()

    ppr_val = np.load(f'{TEMP_DIR}/ppr_v2_{args.val_graph_year}.npy').astype(np.float32)
    concept_feats_val = build_or_cache_feats(
        args.val_graph_year, adj_val, ppr_val, n_concepts, device)
    del ppr_val

    if args.eval:
        eval_fold(args.fold, concept_feats_val, adj_val, scibert_table)
    elif args.train_all:
        for fold in [0]:
            train_fold(fold, concept_feats_train, concept_feats_val,
                       adj_train, adj_val, scibert_table)
    else:
        train_fold(args.fold, concept_feats_train, concept_feats_val,
                   adj_train, adj_val, scibert_table)


# ── Reuse PPR ego-graph builders from v7 ───────────────────────────────

def build_concept_feature_table(ppr_mat, n_concepts, device):
    """Build ego-graph features using top-PPR nodes (same as v7)."""
    MN = MAX_NODES
    ppr_work = ppr_mat.copy()
    np.fill_diagonal(ppr_work, 0)

    concept_nids = np.zeros((n_concepts, MN), dtype=np.int64)
    concept_nids[:, 0] = np.arange(n_concepts)

    t0 = time.time()
    for c in range(n_concepts):
        n_neigh = min(n_concepts - 1, MN - 1)
        if n_neigh < n_concepts - 1:
            top_idx = np.argpartition(ppr_work[c], -n_neigh)[-n_neigh:]
        else:
            top_idx = np.argsort(ppr_work[c])[-n_neigh:]
        concept_nids[c, 1:1 + n_neigh] = top_idx
    print(f"    Top-PPR selection: {time.time()-t0:.1f}s", flush=True)

    concept_feats = np.zeros((n_concepts, MN, EMBED_DIM), dtype=np.float32)
    pos_idx = np.arange(MN)
    concept_feats[:, pos_idx, pos_idx] = 1.0

    t0 = time.time()
    nids_t = torch.from_numpy(concept_nids).to(device)
    ppr_gpu = torch.from_numpy(ppr_work.astype(np.float32)).to(device)
    ppr_vals = ppr_gpu[torch.arange(n_concepts, device=device)[:, None], nids_t]
    ppr_log = torch.log10(ppr_vals.clamp(min=1e-10))
    ppr_min = ppr_log.min(dim=1, keepdim=True).values
    ppr_max = ppr_log.max(dim=1, keepdim=True).values
    ppr_norm = (ppr_log - ppr_min) / (ppr_max - ppr_min).clamp(min=1e-10)
    concept_feats[:, :, 2 * MN] = ppr_norm.cpu().numpy()
    del ppr_gpu, ppr_work
    print(f"    PPR features: {time.time()-t0:.1f}s", flush=True)
    return concept_feats, concept_nids


def fill_adjacency_features(concept_feats, concept_nids, adj):
    """Fill adjacency channel (same as v7)."""
    MN = MAX_NODES
    n_concepts = len(concept_nids)
    t0 = time.time()
    adj_csc = adj.tocsc()
    for c in range(n_concepts):
        nids_c = concept_nids[c]
        cols = adj_csc[:, nids_c].toarray()
        for i in range(MN):
            col_i = cols[:, i]
            for j in range(MN):
                if i != j and col_i[nids_c[j]]:
                    concept_feats[c, i, MN + j] = 1.0
    print(f"    Adjacency fill: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
