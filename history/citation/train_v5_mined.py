"""
PENCIL v5 + Model-Mined Hard Negatives with Clustering

Core idea: Let the MODEL discover its own hard negatives by scoring all 32K
concepts for each anchor, then picking the ones it's most confused about.

Pipeline every MINE_EVERY epochs:
  1. Encode all 32K concepts → embeddings matrix (32K × 256)
  2. For each anchor (u, v):
     - Encode source subgraph → source embedding
     - Score = source_emb @ concept_embeddings.T → 32K scores
     - Exclude: positive v + same-cluster concepts
     - Top-K highest scores = model's hardest cross-domain negatives
  3. Retrain with these mined negatives

This way negatives are truly MODEL-DISCOVERED, not human-defined.
"""
import os, time
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32
EMBED_DIM = 68
SCIBERT_DIM = 768
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
N_EPOCHS = 30
LR = 3e-4
BATCH_SIZE = 16
N_CANDIDATES = 50      # negatives per anchor
DIST_BINS = 10
DIST_EMBED_DIM = 32
N_CLUSTERS = 100
MINE_EVERY = 5         # re-mine negatives every N epochs
MINE_TOP_K = 100       # mine top-100 negatives per anchor
CONCEPT_ENCODE_BS = 200


# ─────────────────────── Data construction ───────────────────────

def run_leiden_clustering(adj):
    """Run Leiden community detection on concept co-occurrence graph.

    Returns:
        labels: (n_concepts,) int32 — community ID per concept
    """
    import leidenalg
    import igraph as ig
    print("  Building igraph from adjacency...", flush=True)
    t0 = time.time()
    # igraph expects edge list with weights
    adj_coo = adj.tocoo()
    edges = list(zip(adj_coo.row.tolist(), adj_coo.col.tolist()))
    weights = adj_coo.data.tolist()
    g = ig.Graph(edges=edges, directed=False)
    g.es['weight'] = weights
    print(f"  igraph built in {time.time()-t0:.1f}s, {g.vcount()} nodes, {g.ecount()} edges", flush=True)

    print("  Running Leiden...", flush=True)
    t0 = time.time()
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                         resolution_parameter=1.0, weights='weight')
    labels = np.array(partition.membership, dtype=np.int32)
    n_communities = len(set(labels))
    print(f"  Leiden done in {time.time()-t0:.1f}s, {n_communities} communities", flush=True)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  size: min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}", flush=True)
    return labels


def compute_a2_candidates(adj, u_ids, top_k=5000):
    """Precompute A² top-K candidate pool per anchor u.

    Returns:
        cand_pool: dict[int, np.ndarray] — u_id → array of candidate concept IDs (sorted by A² score desc)
    """
    print("  Computing A² = adj @ adj...", flush=True)
    t0 = time.time()
    adj2 = adj.dot(adj).tocsr()  # co-citation matrix
    print(f"  A² done in {time.time()-t0:.1f}s, nnz={adj2.nnz}", flush=True)

    unique_u = np.unique(u_ids)
    print(f"  Building candidate pool for {len(unique_u)} unique anchors (top-{top_k})...", flush=True)
    t0 = time.time()
    cand_pool = {}
    for u in unique_u:
        start, end = adj2.indptr[u], adj2.indptr[u + 1]
        if start == end:
            cand_pool[u] = np.array([u], dtype=np.int64)
            continue
        ids = adj2.indices[start:end]
        cands = adj2.data[start:end].astype(np.float64)
        if len(ids) > top_k:
            top_idx = np.argpartition(cands, -top_k)[-top_k:]
            ids = ids[top_idx]
            cands = cands[top_idx]
        order = np.argsort(-cands)
        cand_pool[u] = ids[order]
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    return cand_pool


def build_concept_feature_table(adj, dist_matrix, scibert, max_nodes=32):
    """Precompute 68-dim features for all concepts' ego subgraphs.

    Returns:
        concept_feats: (n_concepts, 32, 68) float32
        concept_nids:  (n_concepts, 32)      int64
    """
    n = adj.shape[0]
    degrees = np.array(adj.sum(axis=1)).flatten()
    concept_feats = np.zeros((n, max_nodes, EMBED_DIM), dtype=np.float32)
    concept_nids = np.zeros((n, max_nodes), dtype=np.int64)

    for c in range(n):
        neighbors = adj[c].indices
        n_neigh = min(len(neighbors), max_nodes - 1)
        if n_neigh > 0:
            top_idx = np.argsort(degrees[neighbors])[-n_neigh:]
            nids = [c] + neighbors[top_idx].tolist()
        else:
            nids = [c]
        nids = nids[:max_nodes]
        nn = len(nids)
        concept_nids[c, :nn] = nids

        for i in range(nn):
            concept_feats[c, i, i] = 1.0
            concept_feats[c, i, 2 * MAX_NODES] = 1.0
            concept_feats[c, i, 2 * MAX_NODES + 1] = min(dist_matrix[c, nids[i]], DIST_BINS) / DIST_BINS
            concept_feats[c, i, 2 * MAX_NODES + 2] = 0.0  # no v
            for j in range(nn):
                if dist_matrix[nids[i], nids[j]] == 1:
                    concept_feats[c, i, MAX_NODES + j] = 1.0

    return concept_feats, concept_nids


def load_train_anchors(data_dir):
    """Load training anchor pairs and their positive subgraph from precomputed data.

    Returns:
        pos_inp:  (N, 32, 68) float32 — positive subgraph features (relative to u,v)
        pos_nids: (N, 32)      int64  — positive subgraph node IDs
        u_ids:    (N,)          int64  — anchor concept u
        v_ids:    (N,)          int64  — positive bridge concept v
    """
    raw = np.load(f'{data_dir}/train_precomp_v3.npz')
    inp = raw['inp'].astype(np.float32).reshape(-1, 6, MAX_NODES, EMBED_DIM)
    nids = raw['node_ids'].reshape(-1, 6, MAX_NODES)
    # Positive is at index 0 in the 6 candidates
    pos_inp = inp[:, 0]       # (N, 32, 68)
    pos_nids = nids[:, 0]     # (N, 32)
    u_ids = nids[:, 0, 0]     # node 0 of pos subgraph = u
    v_ids = nids[:, 0, 1]     # node 1 of pos subgraph = v
    return pos_inp, pos_nids, u_ids, v_ids


def load_val_data(data_dir):
    """Load validation data (keep original mixed negatives for consistent eval)."""
    raw = np.load(f'{data_dir}/val_precomp_v5_hardneg.npz')
    inp = torch.from_numpy(raw['inp'].astype(np.float32))
    nids = torch.from_numpy(raw['node_ids'])
    return inp, nids


# ─────────────────────── Model ───────────────────────

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
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

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

    def _forward_transformer(self, x, adj_norm, pad_mask_flat, iso_mask):
        """Run Transformer layers. x: (B, S, H)"""
        for layer in self.layers:
            h = layer['ln1'](x)
            h, _ = layer['attn'](h, h, h, attn_mask=iso_mask, key_padding_mask=pad_mask_flat)
            x = x + h
            B = x.shape[0]
            x_cand = x.reshape(B, -1, MAX_NODES, x.shape[-1])
            h_mp = layer['mp'](torch.einsum('bgnh,bgnm->bgmh', x_cand, adj_norm))
            x_cand = x_cand + h_mp
            x = x_cand.reshape(x.shape)
            h = layer['ln2'](x)
            x = x + layer['ffn'](h)
        return self.ln_f(x)

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
        x = self._forward_transformer(x, adj_norm, pad_mask_flat, iso_mask)
        x_cand = x.reshape(B, G, N, -1)
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (x_cand * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)
        scores = self.head(pooled).squeeze(-1)
        return scores

    def encode_anchor(self, inp, node_ids):
        """Encode anchor subgraph, return pooled embedding before head.

        Args:
            inp: (B, 1, 32, 68) — source subgraph features
            node_ids: (B, 1, 32) — source node IDs
        Returns:
            (B, 256) — pooled source embedding
        """
        B, G, N, _ = inp.shape
        device = inp.device
        x = self._build_node_features(inp, node_ids)
        adj = inp[:, :, :, MAX_NODES:MAX_NODES + N]
        deg = adj.sum(dim=3, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        pad_mask = (inp[:, :, :, :2 * MAX_NODES + 1].abs().sum(dim=3) == 0)
        pad_mask_flat = pad_mask.reshape(B, G * N)
        iso_mask = self._build_iso_mask(G - 1, N, device)
        x = self._forward_transformer(x, adj_norm, pad_mask_flat, iso_mask)
        x_cand = x.reshape(B, G, N, -1)
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (x_cand * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)
        return pooled.squeeze(1)  # (B, H)
    @torch.no_grad()
    def encode_concepts(self, inp, node_ids, batch_size=CONCEPT_ENCODE_BS):
        """Encode concept subgraphs with Transformer self-attention."""
        n = inp.shape[0]
        device = next(self.parameters()).device
        all_pooled = []
        N = MAX_NODES
        attn_mask = torch.zeros(N, N, device=device)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b_inp = inp[start:end].to(device)
            b_nids = node_ids[start:end].to(device)

            b_inp_4d = b_inp.unsqueeze(1)
            b_nids_4d = b_nids.unsqueeze(1)
            x = self._build_node_features(b_inp_4d, b_nids_4d)

            adj = b_inp[:, :, N:N + N]
            deg = adj.sum(dim=2, keepdim=True).clamp(min=1)
            adj_norm = adj / deg

            for layer in self.layers:
                h = layer['ln1'](x)
                h, _ = layer['attn'](h, h, h, attn_mask=attn_mask)
                x = x + h
                h_mp = layer['mp'](torch.einsum('bnh,bnm->bmh', x, adj_norm))
                x = x + h_mp
                h = layer['ln2'](x)
                x = x + layer['ffn'](h)

            x = self.ln_f(x)
            pooled = x.mean(dim=1)
            all_pooled.append(pooled.cpu())

        return torch.cat(all_pooled, dim=0)


# ─────────────────────── Mining ───────────────────────

@torch.no_grad()
def mine_hard_negatives(model, concept_feats_t, concept_nids_t,
                        pos_inp_t, pos_nids_np, u_ids, v_ids,
                        cand_pool=None, cluster_labels=None,
                        top_k=MINE_TOP_K, n_neg=N_CANDIDATES,
                        easy_ratio=0.2):
    """Mine hard negatives with Leiden filtering + easy negative mixing.

    For each anchor (u, v):
      1. Get A² candidates, score with model
      2. Filter out same-cluster (likely false negatives)
      3. Take top-(n_neg*(1-easy_ratio)) hardest cross-cluster
      4. Add n_neg*easy_ratio random easy negatives from distant clusters

    Returns:
        mined_neg_ids: (N_anchors, n_neg) int64
    """
    device = next(model.parameters()).device
    N_anchors = len(u_ids)
    N_concepts = len(concept_feats_t)
    n_hard = int(n_neg * (1 - easy_ratio))
    n_easy = n_neg - n_hard

    # Encode all concepts
    print(f"  Encoding {N_concepts} concepts...", flush=True)
    t0 = time.time()
    concept_embs = model.encode_concepts(concept_feats_t, concept_nids_t)
    concept_embs_np = concept_embs.numpy()
    print(f"  encoded in {time.time()-t0:.1f}s", flush=True)

    # Precompute: for each cluster, which concepts belong to it
    cluster_to_concepts = {}
    if cluster_labels is not None:
        for cid in np.unique(cluster_labels):
            cluster_to_concepts[cid] = np.where(cluster_labels == cid)[0]

    print(f"  Mining: {n_hard} hard + {n_easy} easy per anchor", flush=True)
    t0 = time.time()
    mined = np.zeros((N_anchors, n_neg), dtype=np.int64)
    ANCHOR_BS = 64

    for start in range(0, N_anchors, ANCHOR_BS):
        end = min(start + ANCHOR_BS, N_anchors)
        bs = end - start

        src_inp = pos_inp_t[start:end].unsqueeze(1).to(device)
        src_nids_real = torch.from_numpy(pos_nids_np[start:end]).unsqueeze(1).to(device)
        src_emb = model.encode_anchor(src_inp, src_nids_real)
        src_emb_np = src_emb.cpu().numpy()

        for i in range(bs):
            ai = start + i
            u = u_ids[ai]
            v = v_ids[ai]

            # Get A² candidates
            if cand_pool is not None and u in cand_pool:
                cands = cand_pool[u]
            else:
                cands = np.arange(N_concepts)

            # Score candidates
            cand_embs = concept_embs_np[cands]
            scores = src_emb_np[i] @ cand_embs.T

            # Exclude positive v
            scores[cands == v] = -1e9

            # Leiden filter: exclude same-cluster
            if cluster_labels is not None:
                v_cluster = cluster_labels[v]
                same_mask = (cluster_labels[cands] == v_cluster)
                scores[same_mask] = -1e9

            # Top-n_hard hardest cross-cluster
            k = min(top_k, len(scores))
            if k > 0 and n_hard > 0:
                topk_idx = np.argpartition(scores, -k)[-k:]
                topk_scores = scores[topk_idx]
                order = np.argsort(-topk_scores)
                hard_negs = cands[topk_idx[order][:n_hard]]
            else:
                hard_negs = np.array([], dtype=np.int64)

            # Easy negatives: random from distant clusters
            easy_negs = np.array([], dtype=np.int64)
            if n_easy > 0 and cluster_labels is not None:
                v_cluster = cluster_labels[v]
                # Pick from different clusters randomly
                diff_cluster_mask = (cluster_labels != v_cluster)
                diff_concepts = np.where(diff_cluster_mask)[0]
                if len(diff_concepts) > 0:
                    easy_idx = np.random.choice(len(diff_concepts), size=min(n_easy, len(diff_concepts)), replace=False)
                    easy_negs = diff_concepts[easy_idx]

            # Combine
            all_negs = np.concatenate([hard_negs, easy_negs])
            mined[ai, :min(n_neg, len(all_negs))] = all_negs[:n_neg]

        if (start // ANCHOR_BS) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (start + bs) * (N_anchors - start - bs) if start + bs < N_anchors else 0
            print(f"    {start+bs}/{N_anchors} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    print(f"  mining done in {time.time()-t0:.1f}s", flush=True)
    return mined


# ─────────────────────── Dataset ───────────────────────

class ModelMinedDataset(Dataset):
    """Dataset supporting both random and model-mined negatives.

    When mined_neg_ids is None, samples random negatives from concept pool.
    """
    def __init__(self, pos_inp, pos_nids, concept_feats, concept_nids,
                 mined_neg_ids, n_candidates, n_concepts):
        self.pos_inp = pos_inp          # (N, 32, 68)
        self.pos_nids = pos_nids        # (N, 32)
        self.concept_feats = concept_feats  # (n_concepts, 32, 68)
        self.concept_nids = concept_nids    # (n_concepts, 32)
        self.mined_neg_ids = mined_neg_ids  # (N, K) or None for random
        self.K = n_candidates
        self.n_concepts = n_concepts

    def update_mined_negatives(self, mined_neg_ids):
        self.mined_neg_ids = mined_neg_ids

    def __len__(self):
        return len(self.pos_inp)

    def __getitem__(self, idx):
        pos_feat = self.pos_inp[idx]       # (32, 68)
        pos_nid = self.pos_nids[idx]       # (32,)

        if self.mined_neg_ids is not None:
            neg_ids = self.mined_neg_ids[idx]  # (K,)
        else:
            # Random negatives from concept pool
            neg_ids = np.random.randint(0, self.n_concepts, size=self.K)

        neg_feats = self.concept_feats[neg_ids]  # (K, 32, 68)
        neg_nids = self.concept_nids[neg_ids]    # (K, 32)

        all_inp = np.concatenate([pos_feat[np.newaxis], neg_feats], axis=0)
        all_nid = np.concatenate([pos_nid[np.newaxis], neg_nids], axis=0)
        G = 1 + self.K
        perm = np.random.permutation(G)
        pos_idx = int(np.where(perm == 0)[0][0])
        return (torch.from_numpy(all_inp[perm].copy()),
                torch.from_numpy(all_nid[perm].copy()),
                pos_idx)


def anchor_collate(batch):
    inp = torch.stack([b[0] for b in batch])
    nids = torch.stack([b[1] for b in batch])
    pos_idx = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return inp, nids, pos_idx


# ─────────────────────── Validation (standard) ───────────────────────

class ValDataset(Dataset):
    def __init__(self, inp, node_ids, n_candidates):
        self.inp = inp
        self.node_ids = node_ids
        self.K = n_candidates

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inp = self.inp[idx]
        nids = self.node_ids[idx]
        G = 1 + self.K
        perm = torch.randperm(G)
        pos_idx = (perm == 0).nonzero(as_tuple=False).item()
        return inp[perm], nids[perm], pos_idx


def val_collate(batch):
    inp = torch.stack([b[0] for b in batch])
    nids = torch.stack([b[1] for b in batch])
    pos_idx = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return inp, nids, pos_idx


# ─────────────────────── Clustering ───────────────────────

def cluster_concepts(model, concept_feats_t, concept_nids_t, n_clusters):
    """Encode concepts and K-Means cluster."""
    print(f"  Encoding {len(concept_feats_t)} concepts for clustering...", flush=True)
    t0 = time.time()
    emb = model.encode_concepts(concept_feats_t, concept_nids_t).numpy()
    # Remove mean + L2 normalize
    emb = emb - emb.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
    emb = emb / norms
    print(f"  encoded+normalized in {time.time()-t0:.1f}s, sample sim: {np.dot(emb[0], emb[100]):.4f}", flush=True)

    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048, n_init=3)
    labels = km.fit_predict(emb)

    unique, counts = np.unique(labels, return_counts=True)
    print(f"  clusters: min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}", flush=True)
    return labels, emb


# ─────────────────────── Main ───────────────────────

def mem_log(tag):
    print(f"  [MEM] {tag}: {torch.cuda.memory_allocated()/1e6:.0f}MB allocated, {torch.cuda.mem_get_info()[0]/1e9:.1f}GB free", flush=True)

def main():
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)
    mem_log("start")

    # Load graph + distance matrix
    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    n_concepts = adj.shape[0]
    print(f"  {n_concepts} concepts", flush=True)
    dist_matrix = np.load(f'{DATA_DIR}/dist_matrix.npy')
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    mem_log("after data load")

    # Build concept feature table
    print("Building concept feature table...", flush=True)
    t0 = time.time()
    concept_feats_np, concept_nids_np = build_concept_feature_table(adj, dist_matrix, scibert)
    concept_feats_t = torch.from_numpy(concept_feats_np)
    concept_nids_t = torch.from_numpy(concept_nids_np)
    print(f"  done in {time.time()-t0:.1f}s, shape={concept_feats_np.shape}", flush=True)
    mem_log("after concept table")

    # Load concept names
    import json
    concept_ids_list = json.load(open(f'{DATA_DIR}/concept_ids.json'))
    concept_names = json.load(open(f'{DATA_DIR}/concept_names.json'))

    # Load training anchors
    print("Loading training anchors...", flush=True)
    pos_inp, pos_nids, u_ids, v_ids = load_train_anchors(DATA_DIR)
    N_anchors = len(u_ids)
    print(f"  {N_anchors} anchors", flush=True)

    # Load validation data (standard mixedneg format)
    val_inp, val_nids = load_val_data(DATA_DIR)
    print(f"  val: {val_inp.shape}", flush=True)
    mem_log("after all data load")

    # Build model
    print("Loading model...", flush=True)
    model = PencilV5(scibert).to(device)
    del scibert
    ckpt_path = f'{DATA_DIR}/pencil_v5_mixedneg_best.pt'
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(f"  loaded {ckpt_path}", flush=True)

    # Compute A² candidate pool
    print("\n=== Computing A² candidate pool ===", flush=True)
    cand_pool = compute_a2_candidates(adj, u_ids, top_k=5000)

    # Leiden clustering
    print("\n=== Leiden clustering ===", flush=True)
    cluster_labels = run_leiden_clustering(adj)
    del adj, dist_matrix  # free memory

    # Initial mining: Leiden-filtered hard + easy mixing
    print("\n=== Initial mining (Leiden-filtered) ===", flush=True)
    model.eval()
    torch.cuda.empty_cache()
    mined_neg_ids = mine_hard_negatives(
        model, concept_feats_t, concept_nids_t,
        torch.from_numpy(pos_inp), pos_nids, u_ids, v_ids,
        cand_pool=cand_pool, cluster_labels=cluster_labels)

    # Build datasets
    n_concepts = concept_feats_np.shape[0]
    pos_inp_t = torch.from_numpy(pos_inp)
    train_ds = ModelMinedDataset(pos_inp_t, pos_nids, concept_feats_np, concept_nids_np,
                                  mined_neg_ids, N_CANDIDATES, n_concepts)
    val_ds = ValDataset(val_inp, val_nids, 20)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=anchor_collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            num_workers=0, collate_fn=val_collate, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)

    # Warmup
    dummy_inp = torch.zeros(2, 1 + N_CANDIDATES, MAX_NODES, EMBED_DIM, device=device)
    dummy_nid = torch.zeros(2, 1 + N_CANDIDATES, MAX_NODES, dtype=torch.long, device=device)
    model.train()
    _ = model(dummy_inp, dummy_nid)
    torch.cuda.synchronize()
    print(f"  GPU warmup done", flush=True)

    print(f"\n=== Training: Leiden-filtered A² mining, re-mine every {MINE_EVERY} epochs ===", flush=True)
    best_mrr = 0
    t_total = time.time()

    for epoch in range(N_EPOCHS):
        # === Re-mine with current model ===
        if epoch > 0 and epoch % MINE_EVERY == 0:
            print(f"\n--- Re-mining at epoch {epoch+1} ---", flush=True)
            model.eval()
            torch.cuda.empty_cache()
            mined_neg_ids = mine_hard_negatives(
                model, concept_feats_t, concept_nids_t,
                pos_inp_t, pos_nids, u_ids, v_ids,
                cand_pool=cand_pool, cluster_labels=cluster_labels)
            train_ds.update_mined_negatives(mined_neg_ids)
            train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                                      num_workers=0, collate_fn=anchor_collate, pin_memory=True)

        # === Training ===
        t0 = time.time()
        model.train()
        total_loss, nb = 0, 0

        for inp, node_ids, pos_idx in train_loader:
            inp = inp.to(device)
            node_ids = node_ids.to(device)
            pos_idx = pos_idx.to(device)

            scores = model(inp, node_ids)

            # InfoNCE loss: (1+K)-way classification
            temperature = 1.0
            logits = scores / temperature
            targets = pos_idx
            loss = torch.nn.functional.cross_entropy(logits, targets)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            nb += 1

        sched.step()
        dt_train = time.time() - t0

        # === Validation ===
        model.eval()
        val_ranks = []
        with torch.no_grad():
            for inp, node_ids, pos_idx in val_loader:
                inp = inp.to(device)
                node_ids = node_ids.to(device)
                pos_idx = pos_idx.to(device)
                scores = model(inp, node_ids)
                B = scores.shape[0]
                pos_scores = scores.gather(1, pos_idx.unsqueeze(1))
                neg_mask = torch.ones_like(scores, dtype=torch.bool)
                neg_mask.scatter_(1, pos_idx.unsqueeze(1), False)
                neg_scores = scores[neg_mask].reshape(B, -1)
                higher = (neg_scores > pos_scores).sum(dim=1)
                rank = higher + 1
                val_ranks.append(rank.cpu())

        ranks = torch.cat(val_ranks).float()
        mrr = (1.0 / ranks).mean().item()
        hit1 = (ranks <= 1).float().mean().item()
        hit5 = (ranks <= 5).float().mean().item()
        dt_eval = time.time() - t0

        marker = ""
        if mrr > best_mrr:
            best_mrr = mrr
            torch.save(model.state_dict(), f'{DATA_DIR}/pencil_v5_mined_best.pt')
            marker = " *"

        mine_info = " [re-mine]" if (epoch > 0 and epoch % MINE_EVERY == 0) else ""
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}  loss={total_loss/nb:.4f}  "
              f"MRR={mrr:.4f}  Hit@1={hit1:.4f}  Hit@5={hit5:.4f}  "
              f"train={dt_train:.1f}s  eval={dt_eval:.1f}s{mine_info}{marker}", flush=True)

    print(f"\nBest MRR: {best_mrr:.4f}", flush=True)
    print(f"Total: {time.time()-t_total:.1f}s", flush=True)


if __name__ == '__main__':
    main()
