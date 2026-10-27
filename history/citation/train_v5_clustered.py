"""
PENCIL v5 + Unsupervised Clustering for Adaptive Negative Sampling

Key idea: Use the model's own Transformer-encoded concept representations
to cluster concepts via K-Means, then sample negatives from DIFFERENT clusters
than the positive. This lets the model discover its own "hard negatives"
without human-defined rules.

Pipeline:
  1. Precompute 32-node ego subgraphs for all ~32K concepts
  2. Every CLUSTER_EVERY epochs:
     a. Encode all concept subgraphs through frozen model → 256-dim embeddings
     b. K-Means clustering on embeddings
     c. Log cross-cluster concept pairs (potential discoveries)
  3. Sample negatives: 3 hard (cross-cluster) + 1 (same-cluster) + 1 random
  4. Position randomization + BPR loss (same as v5 iso)
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
N_CANDIDATES = 5      # 1 positive + 5 negatives
DIST_BINS = 10
DIST_EMBED_DIM = 32
N_CLUSTERS = 100
CLUSTER_EVERY = 5     # re-cluster every N epochs
N_HARD_CROSS = 3      # negatives from different clusters
N_HARD_SAME = 1       # negatives from same cluster
N_RANDOM = 1          # random negatives
CONCEPT_ENCODE_BS = 500  # batch size for encoding concepts


def _extract_concept_ids(data_dir):
    """Extract all unique concept IDs from the graph."""
    adj = sp.load_npz(f'{data_dir}/adj_pre2021.npz').tocsr()
    return adj, np.arange(adj.shape[0], dtype=np.int64)


def build_concept_subgraphs(adj, dist_matrix, max_nodes=32):
    """Build ego subgraphs for all concepts: self + top-(max_nodes-1) neighbors by degree."""
    n_concepts = adj.shape[0]
    degrees = np.array(adj.sum(axis=1)).flatten()
    all_nids = np.zeros((n_concepts, max_nodes), dtype=np.int64)
    concept_n_nodes = np.ones(n_concepts, dtype=np.int32)

    for c in range(n_concepts):
        neighbors = adj[c].indices
        if len(neighbors) == 0:
            all_nids[c, 0] = c
            concept_n_nodes[c] = 1
            continue
        n_neigh = min(len(neighbors), max_nodes - 1)
        top_idx = np.argsort(degrees[neighbors])[-n_neigh:]
        top_neighbors = neighbors[top_idx]
        all_nids[c, 0] = c
        all_nids[c, 1:n_neigh+1] = top_neighbors
        concept_n_nodes[c] = n_neigh + 1

    return all_nids, concept_n_nodes


class ClusteredNegDataset(Dataset):
    """Dataset with cluster-aware negative sampling.

    Precomputed: anchor positive subgraph features + A² candidate pool.
    Dynamic: selects which candidates based on cluster assignments.
    """
    def __init__(self, inp, node_ids, pos_nids, a2_neg_nids, n_candidates):
        """
        Args:
            inp: (N, 1+20, 32, 68) — precomputed features (pos + 20 A² hard negs)
            node_ids: (N, 1+20, 32) — precomputed node IDs
            pos_nids: (N,) int64 — positive concept ID for each anchor
            a2_neg_nids: (N, 20) int64 — A² negative concept IDs
            n_candidates: K (total negatives to sample per anchor)
        """
        self.inp = inp
        self.node_ids = node_ids
        self.pos_nids = pos_nids
        self.a2_neg_nids = a2_neg_nids
        self.K = n_candidates
        self.cluster_labels = None
        self.anchor_clusters = None
        self.a2_neg_clusters = None

    def update_clusters(self, cluster_labels):
        """Update cluster assignments for all concepts."""
        self.cluster_labels = cluster_labels
        self.anchor_clusters = cluster_labels[self.pos_nids]
        self.a2_neg_clusters = cluster_labels[self.a2_neg_nids]

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        pos = self.inp[idx, 0:1]       # (1, 32, 68)
        pos_nid = self.node_ids[idx, 0:1]  # (1, 32)
        inp_neg = self.inp[idx, 1:]     # (20, 32, 68)
        nid_neg = self.node_ids[idx, 1:]  # (20, 32)

        # Select K negatives from the 20 A² candidates
        if self.cluster_labels is not None and self.anchor_clusters is not None:
            neg_clus = self.a2_neg_clusters[idx]  # (20,)
            anc_clus = self.anchor_clusters[idx]
            cross_mask = neg_clus != anc_clus
            same_mask = neg_clus == anc_clus
            cross_indices = np.where(cross_mask)[0]
            same_indices = np.where(same_mask)[0]

            chosen = []
            # Cross-cluster hard negatives
            if len(cross_indices) >= N_HARD_CROSS:
                chosen.extend(np.random.choice(cross_indices, N_HARD_CROSS, replace=False).tolist())
            else:
                chosen.extend(cross_indices.tolist())
            # Same-cluster hard negatives
            if len(same_indices) >= N_HARD_SAME:
                chosen.extend(np.random.choice(same_indices, N_HARD_SAME, replace=False).tolist())
            else:
                chosen.extend(same_indices.tolist())
            # Fill remaining with random from pool
            remaining = self.K - len(chosen)
            if remaining > 0:
                available = [i for i in range(20) if i not in chosen]
                if len(available) >= remaining:
                    chosen.extend(np.random.choice(available, remaining, replace=False).tolist())
                else:
                    chosen.extend(available.tolist())
            chosen = chosen[:self.K]
        else:
            # No clustering yet: random from pool
            chosen = np.random.choice(20, self.K, replace=False).tolist()

        sel_inp = inp_neg[chosen]    # (K, 32, 68)
        sel_nid = nid_neg[chosen]    # (K, 32)

        # Combine: [positive, negatives] then shuffle
        all_inp = torch.cat([pos, sel_inp], dim=0)    # (1+K, 32, 68)
        all_nid = torch.cat([pos_nid, sel_nid], dim=0)  # (1+K, 32)
        G = 1 + self.K
        perm = torch.randperm(G)
        pos_idx = (perm == 0).nonzero(as_tuple=False).item()
        return all_inp[perm], all_nid[perm], pos_idx


def anchor_collate(batch):
    inp = torch.stack([b[0] for b in batch])
    nids = torch.stack([b[1] for b in batch])
    pos_idx = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return inp, nids, pos_idx


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
        x = self._build_base_features(inp, node_ids)  # (B, G*N, H)
        # x is already reshaped to (B, G*N, H). cand_pos_embed needs (B*G*N,)
        x_flat = x.reshape(B * G * N, -1)
        cand_idx = torch.arange(G, device=inp.device).unsqueeze(1).expand(-1, N).reshape(-1)
        cand_idx = cand_idx.unsqueeze(0).expand(B, -1).reshape(-1)
        x_flat = x_flat + self.cand_pos_embed(cand_idx)
        return x_flat.reshape(B, G * N, -1)

    def _build_base_features(self, inp, node_ids):
        """Build node features WITHOUT cand_pos_embed (for concept encoding)."""
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

    @torch.no_grad()
    def encode_concepts(self, inp, node_ids, batch_size=1000):
        """Encode concept subgraphs with FULL Transformer self-attention.

        Each concept is an independent batch item (32 nodes self-attend).
        MultiheadAttention applies the same (32,32) mask to all items in the batch.
        """
        n = inp.shape[0]
        device = next(self.parameters()).device
        all_pooled = []
        N = MAX_NODES

        # Self-attention mask: 32x32, all zeros (full self-attention within each concept)
        attn_mask = torch.zeros(N, N, device=device)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bs = end - start
            b_inp = inp[start:end].to(device)      # (bs, 32, 68)
            b_nids = node_ids[start:end].to(device)  # (bs, 32)

            # Build node features WITH cand_pos_embed (matching training)
            b_inp_4d = b_inp.unsqueeze(1)
            b_nids_4d = b_nids.unsqueeze(1)
            x = self._build_node_features(b_inp_4d, b_nids_4d)  # (bs, 32, H)

            # Graph message passing adjacency
            adj = b_inp[:, :, N:N + N]  # (bs, 32, 32)
            deg = adj.sum(dim=2, keepdim=True).clamp(min=1)
            adj_norm = adj / deg

            for layer in self.layers:
                h = layer['ln1'](x)  # (bs, 32, H)
                h, _ = layer['attn'](h, h, h, attn_mask=attn_mask)  # self-attention per concept
                x = x + h

                # Graph message passing
                h_mp = layer['mp'](torch.einsum('bnh,bnm->bmh', x, adj_norm))
                x = x + h_mp

                h = layer['ln2'](x)
                x = x + layer['ffn'](h)

            x = self.ln_f(x)  # (bs, 32, H)
            pooled = x.mean(dim=1)  # (bs, H)
            all_pooled.append(pooled.cpu())

        return torch.cat(all_pooled, dim=0)  # (n_concepts, H)


def cluster_concepts(model, concept_feats, concept_nids, n_clusters):
    """Encode all concepts and run K-Means clustering."""
    print(f"  Encoding {len(concept_feats)} concepts...", flush=True)
    t0 = time.time()
    embeddings = model.encode_concepts(concept_feats, concept_nids).numpy()
    print(f"  encoded in {time.time()-t0:.1f}s, shape={embeddings.shape}", flush=True)

    # Remove mean (cancels constant cand_pos_embed component) + L2 normalize
    embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    embeddings = embeddings / norms
    print(f"  normalized, checking pairwise sim: {np.dot(embeddings[0], embeddings[100]):.4f}", flush=True)

    print(f"  K-Means (K={n_clusters})...", flush=True)
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048, n_init=3)
    labels = km.fit_predict(embeddings)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # Cluster size distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}", flush=True)

    return labels, embeddings


def log_cross_cluster_discoveries(embeddings, labels, concept_names, concept_ids, top_k=10):
    """Find and log cross-cluster concept pairs with high cosine similarity."""
    from sklearn.metrics.pairwise import cosine_similarity
    n_clusters = labels.max() + 1
    pairs_found = []
    for ci in range(n_clusters):
        for cj in range(ci + 1, min(ci + 5, n_clusters)):
            ci_mask = labels == ci
            cj_mask = labels == cj
            if ci_mask.sum() == 0 or cj_mask.sum() == 0:
                continue
            sim = cosine_similarity(embeddings[ci_mask], embeddings[cj_mask])
            best_i = np.argmax(sim.max(axis=1))
            best_j = np.argmax(sim.max(axis=0))
            global_i = np.where(ci_mask)[0][best_i]
            global_j = np.where(cj_mask)[0][best_j]
            score = sim[best_i, best_j]
            pairs_found.append((score, global_i, global_j, ci, cj))

    pairs_found.sort(reverse=True)
    print(f"\n  === Top {top_k} Cross-Cluster Discoveries ===", flush=True)
    for score, i, j, ci, cj in pairs_found[:top_k]:
        name_i = concept_names.get(concept_ids[i], str(concept_ids[i]))
        name_j = concept_names.get(concept_ids[j], str(concept_ids[j]))
        print(f"  [{ci}↔{cj}] sim={score:.3f}  {name_i} — {name_j}", flush=True)


def main():
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # Load graph + distance matrix
    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()
    n_concepts = adj.shape[0]
    print(f"  {n_concepts} concepts", flush=True)

    print("Loading distance matrix...", flush=True)
    dist_matrix = np.load(f'{DATA_DIR}/dist_matrix.npy')

    # Build concept ego subgraphs
    print("Building concept subgraphs...", flush=True)
    t0 = time.time()
    concept_nids, concept_n_nodes = build_concept_subgraphs(adj, dist_matrix)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # Reconstruct features for all concepts (distances from self = dist_matrix[concept])
    print("Reconstructing concept features...", flush=True)
    t0 = time.time()
    concept_feats = np.zeros((n_concepts, MAX_NODES, EMBED_DIM), dtype=np.float32)
    for c in range(n_concepts):
        n = int(concept_n_nodes[c])
        nids_c = concept_nids[c]
        for i in range(n):
            concept_feats[c, i, i] = 1.0                          # one-hot position
            concept_feats[c, i, 2 * MAX_NODES] = 1.0              # existence flag
            concept_feats[c, i, 2 * MAX_NODES + 1] = min(dist_matrix[c, nids_c[i]], DIST_BINS) / DIST_BINS
            concept_feats[c, i, 2 * MAX_NODES + 2] = 0.0          # no v for concept-level
            for j in range(n):
                if dist_matrix[nids_c[i], nids_c[j]] == 1:
                    concept_feats[c, i, MAX_NODES + j] = 1.0      # adjacency
    concept_feats_t = torch.from_numpy(concept_feats)
    concept_nids_t = torch.from_numpy(concept_nids)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # Load concept names for logging
    import json
    concept_ids = json.load(open(f'{DATA_DIR}/concept_ids.json'))
    concept_names = json.load(open(f'{DATA_DIR}/concept_names.json'))

    # Load SciBERT + model
    print("Loading model...", flush=True)
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilV5(scibert).to(device)
    # Load pre-trained v5 mixedneg checkpoint as starting point
    ckpt_path = f'{DATA_DIR}/pencil_v5_mixedneg_best.pt'
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(f"  loaded checkpoint: {ckpt_path}", flush=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}", flush=True)

    # Load training data (v5 hardneg: 1 pos + 20 A² candidates)
    print("Loading training data...", flush=True)
    t0 = time.time()
    train_raw = np.load(f'{DATA_DIR}/train_precomp_v5_hardneg.npz')
    val_raw = np.load(f'{DATA_DIR}/val_precomp_v5_hardneg.npz')

    train_inp = torch.from_numpy(train_raw['inp'].astype(np.float32))
    train_nids = torch.from_numpy(train_raw['node_ids'])
    val_inp = torch.from_numpy(val_raw['inp'].astype(np.float32))
    val_nids = torch.from_numpy(val_raw['node_ids'])
    print(f"  train: {train_inp.shape}, val: {val_inp.shape}, loaded in {time.time()-t0:.1f}s", flush=True)

    # Extract positive and negative concept IDs for cluster-aware sampling
    # Positive concept = node_ids[anchor, 0, 1] (second node in pos subgraph = the bridge concept v)
    # A² negative concepts = node_ids[anchor, 1:21, 1] (second node in each neg subgraph)
    train_pos_nids = train_nids[:, 0, 1].numpy()   # (N_train,)
    train_neg_nids = train_nids[:, 1:, 1].numpy()   # (N_train, 20)
    val_pos_nids = val_nids[:, 0, 1].numpy()
    val_neg_nids = val_nids[:, 1:, 1].numpy()

    train_ds = ClusteredNegDataset(train_inp, train_nids, train_pos_nids, train_neg_nids, N_CANDIDATES)
    val_ds = ClusteredNegDataset(val_inp, val_nids, val_pos_nids, val_neg_nids, N_CANDIDATES)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)

    # Warmup
    dummy_inp = torch.zeros(2, 1 + N_CANDIDATES, MAX_NODES, EMBED_DIM, device=device)
    dummy_nid = torch.zeros(2, 1 + N_CANDIDATES, MAX_NODES, dtype=torch.long, device=device)
    _ = model(dummy_inp, dummy_nid)
    torch.cuda.synchronize()
    print(f"  GPU warmup done", flush=True)

    print(f"\n=== Training PENCIL v5 + Clustering ({N_CLUSTERS} clusters, every {CLUSTER_EVERY} epochs) ===", flush=True)
    best_mrr = 0
    t_total = time.time()
    current_labels = None

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=anchor_collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            num_workers=0, collate_fn=anchor_collate, pin_memory=True)

    for epoch in range(N_EPOCHS):
        # === Clustering phase ===
        if epoch % CLUSTER_EVERY == 0:
            print(f"\n--- Clustering at epoch {epoch+1} ---", flush=True)
            model.eval()
            current_labels, embeddings = cluster_concepts(
                model, concept_feats_t, concept_nids_t, N_CLUSTERS)
            log_cross_cluster_discoveries(
                embeddings, current_labels, concept_names, concept_ids, top_k=10)
            train_ds.update_clusters(current_labels)
            val_ds.update_clusters(current_labels)
            # Recreate loaders to pick up new cluster assignments
            train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                                      num_workers=0, collate_fn=anchor_collate, pin_memory=True)
            model.train()

        # === Training phase ===
        t0 = time.time()
        model.train()
        total_loss, nb = 0, 0

        for inp, node_ids, pos_idx in train_loader:
            inp = inp.to(device)
            node_ids = node_ids.to(device)
            pos_idx = pos_idx.to(device)

            scores = model(inp, node_ids)

            B = scores.shape[0]
            pos_scores = scores.gather(1, pos_idx.unsqueeze(1))
            neg_mask = torch.ones_like(scores, dtype=torch.bool)
            neg_mask.scatter_(1, pos_idx.unsqueeze(1), False)
            neg_scores = scores[neg_mask].reshape(B, -1)
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

            opt.zero_grad(set_to_none=True)
            bpr_loss.backward()
            opt.step()

            total_loss += bpr_loss.item()
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
            torch.save(model.state_dict(), f'{DATA_DIR}/pencil_v5_cluster_best.pt')
            torch.save(current_labels, f'{DATA_DIR}/cluster_labels.pt') if current_labels is not None else None
            marker = " *"

        cluster_info = f"  clusters={N_CLUSTERS}" if epoch % CLUSTER_EVERY == 0 else ""
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}  loss={total_loss/nb:.4f}  "
              f"MRR={mrr:.4f}  Hit@1={hit1:.4f}  Hit@5={hit5:.4f}  "
              f"train={dt_train:.1f}s  eval={dt_eval:.1f}s{cluster_info}{marker}", flush=True)

    print(f"\nBest MRR: {best_mrr:.4f}", flush=True)
    print(f"Total: {time.time()-t_total:.1f}s", flush=True)


if __name__ == '__main__':
    main()
