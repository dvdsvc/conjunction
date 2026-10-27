"""
Temporal LBD: v6 = v5_iso architecture + pairwise ranking on PPR change.

Changes from v5:
  - Reads from fold{fold}_v2_chunks (PPR-based labels)
  - Uses concept_ids_v2.json (L<=3, score>=0.7)
  - Uses graph_v2_{year}.npz for concept features
  - Labels are continuous PPR change (positive = convergence)

Usage:
  python train_temporal_v6.py --fold 0
  python train_temporal_v6.py --train-all
  python train_temporal_v6.py --eval --fold 0
"""
import os, time, argparse, glob, json
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
TEMP_DIR = f'{DATA_DIR}/temporal_v2'

MAX_NODES = 32
EMBED_DIM = 68
SCIBERT_DIM = 768
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
DIST_BINS = 10
DIST_EMBED_DIM = 32

BATCH_SIZE = 16
N_EPOCHS = 30
LR = 3e-4
N_CANDIDATES = 50
PATIENCE = 5
PPR_CHANGE_THRESH = 1e-7  # binarize threshold for BCE

CONCEPT_IDS_FILE = f'{TEMP_DIR}/concept_ids_v2.json'
OLD_CONCEPT_IDS_FILE = f'{DATA_DIR}/concept_ids.json'


def load_scibert_remapped(concept_ids, n_concepts):
    """Load SciBERT embeddings, remap from old concept ordering to new."""
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


# ── Model (same as v5_iso) ──────────────────────────────────────────────

class PencilV6(nn.Module):
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


# ── Pairwise ranking loss ───────────────────────────────────────────────

def pairwise_ranking_loss(scores, targets):
    B, K = scores.shape
    total_loss = torch.tensor(0.0, device=scores.device)
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


# ── Chunk loading ──────────────────────────────────────────────────────

def _load_chunks(prefix, fold_dir):
    files = sorted(glob.glob(f'{fold_dir}/{prefix}_chunk_*.npz'))
    if not files:
        raise FileNotFoundError(f"No chunks: {fold_dir}/{prefix}_chunk_*.npz")
    pairs_list, labels_list = [], []
    for f in files:
        with np.load(f) as d:
            pairs_list.append(d['pairs'])
            labels_list.append(d['labels'])
    return np.concatenate(pairs_list), np.concatenate(labels_list)


# ── Concept feature table ─────────────────────────────────────────────

def build_concept_feature_table(dm_t1, n_concepts, adj, device):
    """Build ego-graph features for each concept using graph_v2 adjacency."""
    MN = MAX_NODES
    degrees = np.array(adj.sum(axis=1)).flatten()
    concept_nids = np.zeros((n_concepts, MN), dtype=np.int64)
    concept_nnz = np.ones(n_concepts, dtype=np.int32)
    for c in range(n_concepts):
        neighbors = adj[c].indices
        concept_nids[c, 0] = c
        n_neigh = min(len(neighbors), MN - 1)
        if n_neigh > 0:
            top_idx = np.argsort(degrees[neighbors])[-n_neigh:]
            concept_nids[c, 1:1 + n_neigh] = neighbors[top_idx]
        concept_nnz[c] = 1 + n_neigh

    concept_feats = np.zeros((n_concepts, MN, EMBED_DIM), dtype=np.float32)
    pos_idx = np.arange(MN)
    concept_feats[:, pos_idx, pos_idx] = 1.0
    concept_feats[:, :, 2 * MN] = 1.0

    # Use BFS distance for node features (PPR is for labels, not for ego-graph features)
    dm_gpu = torch.from_numpy(dm_t1.astype(np.int32)).to(device)
    nids_t = torch.from_numpy(concept_nids).to(device)
    dm_row = dm_gpu[torch.arange(n_concepts, device=device)[:, None], nids_t]
    concept_feats[:, :, 2 * MN + 1] = torch.clamp(dm_row.float(), max=DIST_BINS).cpu().numpy() / DIST_BINS
    concept_feats[:, :, 2 * MN + 2] = 0.0

    for c in range(n_concepts):
        nnz = int(concept_nnz[c])
        for i in range(nnz):
            for j in range(nnz):
                if i != j and dm_t1[concept_nids[c, i], concept_nids[c, j]] == 1:
                    concept_feats[c, i, MN + j] = 1.0

    return concept_feats, concept_nids


# ── Dataset ────────────────────────────────────────────────────────────

class AnchorDataset(torch.utils.data.Dataset):
    def __init__(self, src_to_cands, concept_feats, n_candidates, rng_seed=42):
        self.anchors = [u for u in sorted(src_to_cands.keys())
                        if len(src_to_cands[u]) >= 2]
        self.src_to_cands = src_to_cands
        self.concept_feats = concept_feats
        self.K = n_candidates
        self.rng = np.random.RandomState(rng_seed)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        u = self.anchors[idx]
        cands = self.src_to_cands[u]
        if len(cands) >= self.K:
            idxs = self.rng.choice(len(cands), self.K, replace=False)
        else:
            idxs = self.rng.choice(len(cands), self.K, replace=True)
        vs = [cands[i][0] for i in idxs]
        dcs = np.array([cands[i][1] for i in idxs], dtype=np.float32)
        feats = self.concept_feats[vs]
        nids = np.zeros((len(vs), MAX_NODES), dtype=np.int64)
        for ci, v in enumerate(vs):
            nids[ci, 0] = u
            nids[ci, 1] = v
        return (torch.from_numpy(feats),
                torch.from_numpy(nids),
                torch.from_numpy(dcs))


def anchor_collate(batch):
    inp = torch.stack([b[0] for b in batch])
    nids = torch.stack([b[1] for b in batch])
    dcs = torch.stack([b[2] for b in batch])
    return inp, nids, dcs


def build_anchor_data(pairs, labels):
    src_to_cands = {}
    for (u, v), dc in zip(pairs, labels):
        if u not in src_to_cands:
            src_to_cands[u] = []
        src_to_cands[u].append((v, float(dc)))
    return src_to_cands


# ── Train ──────────────────────────────────────────────────────────────

def train_fold(fold, concept_feats, adj, scibert_table):
    print(f"\n{'='*60}\n  Fold {fold} (v6 PPR + pairwise ranking)\n{'='*60}", flush=True)
    device = torch.device('cuda')

    fold_dir = f'{TEMP_DIR}/fold{fold}_v2_chunks'
    manifest = json.load(open(f'{fold_dir}/manifest.json'))
    print(f"  train: {manifest['window_train']}, val: {manifest['window_val']}", flush=True)

    print(f"  Loading train chunks...", flush=True)
    train_pairs, train_labels = _load_chunks('train', fold_dir)
    src_train = build_anchor_data(train_pairs, train_labels)
    print(f"  Train anchors: {len(src_train):,}", flush=True)

    train_ds = AnchorDataset(src_train, concept_feats, N_CANDIDATES)
    train_loader = torch.utils.data.DataLoader(
        train_ds, BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=anchor_collate, pin_memory=True)

    print(f"  Loading val chunks...", flush=True)
    val_pairs, val_labels = _load_chunks('val', fold_dir)
    src_val = build_anchor_data(val_pairs, val_labels)
    print(f"  Val anchors: {len(src_val):,}", flush=True)

    val_ds = AnchorDataset(src_val, concept_feats, N_CANDIDATES, rng_seed=99)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=anchor_collate, pin_memory=True)

    scibert = scibert_table
    model = PencilV6(scibert).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)

    dummy = torch.zeros(2, 1 + N_CANDIDATES, MAX_NODES, EMBED_DIM, device=device)
    dummy_nid = torch.zeros(2, 1 + N_CANDIDATES, MAX_NODES, dtype=torch.long, device=device)
    model.train()
    _ = model(dummy, dummy_nid)
    torch.cuda.synchronize()

    model_dir = f'{TEMP_DIR}/model_fold{fold}_v6bce'
    os.makedirs(model_dir, exist_ok=True)
    best_disc = -1
    patience_counter = 0

    bce_loss_fn = nn.BCEWithLogitsLoss()
    print(f"\n--- Training (BCE + iso, {N_CANDIDATES} cands, bs={BATCH_SIZE}, patience={PATIENCE}) ---", flush=True)
    for epoch in range(N_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, nb = 0.0, 0

        for inp, node_ids, dcs in train_loader:
            inp = inp.to(device)
            node_ids = node_ids.to(device)
            labels_bin = (dcs > PPR_CHANGE_THRESH).float().to(device)
            scores = model(inp, node_ids)
            loss = bce_loss_fn(scores, labels_bin)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1

        sched.step()
        dt_train = time.time() - t0

        # Validation
        model.eval()
        all_top_means = []
        all_bot_means = []
        val_loss_sum, val_nb = 0.0, 0
        with torch.no_grad():
            for inp, node_ids, dcs in val_loader:
                inp = inp.to(device)
                node_ids = node_ids.to(device)
                labels_bin = (dcs > PPR_CHANGE_THRESH).float().to(device)
                scores = model(inp, node_ids)
                val_loss_sum += bce_loss_fn(scores, labels_bin).item()
                val_nb += 1
                probs = torch.sigmoid(scores).float().cpu().numpy()
                dcs_np = dcs.float().numpy()
                for b in range(probs.shape[0]):
                    ranked = np.argsort(-probs[b])
                    K = len(ranked)
                    top_k = max(1, K // 10)
                    all_top_means.append(dcs_np[b, ranked[:top_k]].mean())
                    all_bot_means.append(dcs_np[b, ranked[top_k:]].mean())

        top_mean = np.mean(all_top_means)
        bot_mean = np.mean(all_bot_means)
        disc = top_mean / max(abs(bot_mean), 1e-8)
        val_loss = val_loss_sum / max(val_nb, 1)
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
              f"val_loss={val_loss:.6f}  disc={disc:.3f}  "
              f"top={top_mean:.6f}  bot={bot_mean:.6f}  "
              f"train={dt_train:.1f}s  eval={dt_eval:.1f}s{marker}", flush=True)

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (patience={PATIENCE})", flush=True)
            break

    print(f"\n  Best disc: {best_disc:.3f}", flush=True)
    return f'{model_dir}/best.pt'


# ── Eval ───────────────────────────────────────────────────────────────

def eval_fold(fold, concept_feats, adj, scibert_table):
    print(f"\n{'='*60}\n  Eval fold {fold} (v6 PPR)\n{'='*60}", flush=True)
    device = torch.device('cuda')

    fold_dir = f'{TEMP_DIR}/fold{fold}_v2_chunks'
    manifest = json.load(open(f'{fold_dir}/manifest.json'))
    print(f"  val: {manifest['window_val']}", flush=True)

    val_pairs, val_labels = _load_chunks('val', fold_dir)
    src_val = build_anchor_data(val_pairs, val_labels)
    print(f"  Val anchors: {len(src_val):,}", flush=True)

    val_ds = AnchorDataset(src_val, concept_feats, N_CANDIDATES, rng_seed=99)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=anchor_collate, pin_memory=True)

    scibert = scibert_table
    model = PencilV6(scibert).to(device)
    model_file = f'{TEMP_DIR}/model_fold{fold}_v6bce/best.pt'
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
    all_nids_list = []

    with torch.no_grad():
        for inp, node_ids, dcs in val_loader:
            inp = inp.to(device)
            node_ids = node_ids.to(device)
            logits = model(inp, node_ids)
            probs = torch.sigmoid(logits).float().cpu().numpy()
            dcs_np = dcs.numpy()
            nids_np = node_ids.cpu().numpy()
            for b in range(probs.shape[0]):
                ranked = np.argsort(-probs[b])
                K = len(ranked)
                top_k = max(1, K // 10)
                all_top_means.append(dcs_np[b, ranked[:top_k]].mean())
                all_bot_means.append(dcs_np[b, ranked[top_k:]].mean())
            all_scores_list.append(probs.flatten())
            all_labels_list.append(dcs_np.flatten())
            all_nids_list.append(nids_np[:, :, 1])

    top_mean = np.mean(all_top_means)
    bot_mean = np.mean(all_bot_means)
    disc = top_mean / max(abs(bot_mean), 1e-8)

    all_scores = np.concatenate(all_scores_list)
    all_labels = np.concatenate(all_labels_list)
    sp_corr, _ = spearmanr(all_scores, all_labels)
    all_nids = np.concatenate(all_nids_list)

    print(f"\n=== Eval fold {fold} (BCE) ===", flush=True)
    print(f"  Disc ratio: {disc:.3f}", flush=True)
    print(f"  Spearman: {sp_corr:.4f}", flush=True)
    print(f"  top_mean: {top_mean:.6f}, bot_mean: {bot_mean:.6f}", flush=True)

    src_ids = np.array(val_ds.anchors)
    np.savez(f'{TEMP_DIR}/eval_v6_fold{fold}.npz',
             scores=all_scores, labels=all_labels,
             src_ids=src_ids, cand_v=all_nids)
    return disc


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--train-all', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--graph-year', type=int, default=2005,
                        help='Year for graph structure (BFS distances for ego-graph features)')
    args = parser.parse_args()

    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    # Load v2 concept IDs
    print("Loading v2 concept IDs...", flush=True)
    with open(CONCEPT_IDS_FILE) as f:
        concept_ids = json.load(f)
    n_concepts = len(concept_ids)
    print(f"  {n_concepts} concepts", flush=True)

    # Load graph for concept features
    graph_year = args.graph_year
    adj = sp.load_npz(f'{TEMP_DIR}/graph_v2_{graph_year}.npz').tocsr()
    print(f"  graph_v2_{graph_year}: {adj.shape[0]} nodes, {adj.nnz:,} edges", flush=True)

    # Compute BFS distances for concept features if needed
    bfs_dist_path = f'{TEMP_DIR}/dist_v2_{graph_year}.npy'
    if not os.path.exists(bfs_dist_path):
        print(f"  Computing BFS distances on graph_v2_{graph_year}...", flush=True)
        from scipy.sparse.csgraph import shortest_path
        t0 = time.time()
        dist = shortest_path(adj, directed=False)
        dist[np.isinf(dist)] = 9999
        np.save(bfs_dist_path, dist.astype(np.int16))
        print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    dm_t1 = np.load(bfs_dist_path).astype(np.int32)

    print("Building concept feature table...", flush=True)
    t0 = time.time()
    concept_feats, concept_nids = build_concept_feature_table(dm_t1, n_concepts, adj, device)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    print("Loading SciBERT (remapped)...", flush=True)
    scibert_table = load_scibert_remapped(concept_ids, n_concepts)

    if args.eval:
        eval_fold(args.fold, concept_feats, adj, scibert_table)
    elif args.train_all:
        for fold in [0]:
            train_fold(fold, concept_feats, adj, scibert_table)
    else:
        train_fold(args.fold, concept_feats, adj, scibert_table)


if __name__ == '__main__':
    main()
