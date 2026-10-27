"""
Temporal LBD: v7 = PPR-based ego-graph + expanded nodes + BCE.

Changes from v6:
  - BFS distance → PPR distance for ego-graph neighbor selection & features
  - MAX_NODES 32 → 64 (more neighbors captured)
  - EMBED_DIM = 2*64 + 1 = 129 (pos one-hot + adjacency + ppr_dist)
  - Removed dist_embed (BFS bins), PPR dist is continuous float in struct_proj
  - cand_pos_embed expanded to 128

Usage:
  python train_temporal_v7.py --fold 0
  python train_temporal_v7.py --train-all
  python train_temporal_v7.py --eval --fold 0
"""
import os, time, argparse, glob, json
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

HAS_SDPA = hasattr(F, 'scaled_dot_product_attention')

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
TEMP_DIR = f'{DATA_DIR}/temporal_v2'

MAX_NODES = 64
EMBED_DIM = 2 * MAX_NODES + 1   # 129: pos(64) + adj(64) + ppr_dist(1)
SCIBERT_DIM = 768
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8

BATCH_SIZE = 16
N_EPOCHS = 30
LR = 3e-4
N_CANDIDATES = 50
PATIENCE = 5
PPR_CHANGE_THRESH = 5e-5

CONCEPT_IDS_FILE = f'{TEMP_DIR}/concept_ids_v2.json'
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


# ── Model ──────────────────────────────────────────────────────────────

class PencilV7(nn.Module):
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
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

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

    def _forward_transformer(self, x, adj_norm, pad_mask_flat, iso_mask):
        B = x.shape[0]
        S = x.shape[1]
        # Combine iso_mask with padding: pad tokens cannot be attended to
        pad_expand = pad_mask_flat.unsqueeze(1)  # [B, 1, S]
        attn_allowed = (~pad_expand).expand(B, S, S)  # [B, S, S]
        iso_allowed = (iso_mask == 0.0).unsqueeze(0)   # [1, S, S]
        bool_mask = (attn_allowed & iso_allowed).unsqueeze(1)  # [B, 1, S, S]

        for layer in self.layers:
            h = layer['ln1'](x)
            # Flash Attention via SDPA
            qkv = layer['qkv'](h)  # [B, S, 3*H]
            q, k, v = qkv.chunk(3, dim=-1)
            # [B, S, H] -> [B, heads, S, head_dim]
            q = q.view(B, S, N_HEADS, HIDDEN // N_HEADS).transpose(1, 2)
            k = k.view(B, S, N_HEADS, HIDDEN // N_HEADS).transpose(1, 2)
            v = v.view(B, S, N_HEADS, HIDDEN // N_HEADS).transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bool_mask, dropout_p=0.1 if self.training else 0.0)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, HIDDEN)
            h = layer['out_proj'](attn_out)
            x = x + h
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
        pad_mask = (inp[:, :, :, :2 * MAX_NODES].abs().sum(dim=3) == 0)
        pad_mask_flat = pad_mask.reshape(B, G * N)
        iso_mask = self._build_iso_mask(G - 1, N, device)
        x = self._forward_transformer(x, adj_norm, pad_mask_flat, iso_mask)
        x_cand = x.reshape(B, G, N, -1)
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (x_cand * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)
        scores = self.head(pooled).squeeze(-1)
        return scores


# ── PPR ego-graph construction ────────────────────────────────────────

def build_concept_feature_table(ppr_mat, n_concepts, device):
    """Build ego-graph features using top-PPR nodes from entire graph (not just 1-hop).

    Feature layout per node [EMBED_DIM=129]:
      [0:64]   position one-hot
      [64:128] adjacency between ego-graph nodes (if directly connected in graph)
      [128]    PPR distance to anchor (log-normalized 0~1)
    """
    MN = MAX_NODES

    # Select top-PPR nodes for each concept (from entire graph, any hop distance)
    ppr_work = ppr_mat.copy()
    np.fill_diagonal(ppr_work, 0)  # exclude self

    concept_nids = np.zeros((n_concepts, MN), dtype=np.int64)
    concept_nids[:, 0] = np.arange(n_concepts)  # self at position 0

    t0 = time.time()
    for c in range(n_concepts):
        n_neigh = min(n_concepts - 1, MN - 1)
        if n_neigh < n_concepts - 1:
            top_idx = np.argpartition(ppr_work[c], -n_neigh)[-n_neigh:]
        else:
            top_idx = np.argsort(ppr_work[c])[-n_neigh:]
        concept_nids[c, 1:1 + n_neigh] = top_idx
    print(f"    Top-PPR selection: {time.time()-t0:.1f}s", flush=True)

    # Build feature matrix
    concept_feats = np.zeros((n_concepts, MN, EMBED_DIM), dtype=np.float32)

    # Position one-hot: [0:MN]
    pos_idx = np.arange(MN)
    concept_feats[:, pos_idx, pos_idx] = 1.0

    # PPR distance feature: channel [2*MN], log-normalized per anchor
    t0 = time.time()
    nids_t = torch.from_numpy(concept_nids).to(device)
    ppr_gpu = torch.from_numpy(ppr_work.astype(np.float32)).to(device)
    ppr_vals = ppr_gpu[torch.arange(n_concepts, device=device)[:, None], nids_t]  # [N, MN]
    ppr_log = torch.log10(ppr_vals.clamp(min=1e-10))
    ppr_min = ppr_log.min(dim=1, keepdim=True).values
    ppr_max = ppr_log.max(dim=1, keepdim=True).values
    ppr_norm = (ppr_log - ppr_min) / (ppr_max - ppr_min).clamp(min=1e-10)
    concept_feats[:, :, 2 * MN] = ppr_norm.cpu().numpy()
    del ppr_gpu, ppr_work
    print(f"    PPR features: {time.time()-t0:.1f}s", flush=True)

    # Adjacency between ego-graph nodes: filled by fill_adjacency_features()
    return concept_feats, concept_nids


def fill_adjacency_features(concept_feats, concept_nids, adj):
    """Fill adjacency channel [MN:2*MN] using vectorized sparse column lookup."""
    MN = MAX_NODES
    n_concepts = len(concept_nids)
    t0 = time.time()
    adj_csc = adj.tocsc()
    for c in range(n_concepts):
        nids_c = concept_nids[c]
        # Get columns for all selected nodes at once: [N, MN] sparse
        cols = adj_csc[:, nids_c].toarray()  # [20949, 64]
        for i in range(MN):
            col_i = cols[:, i]  # neighbors of node i
            for j in range(MN):
                if i != j and col_i[nids_c[j]]:
                    concept_feats[c, i, MN + j] = 1.0
    print(f"    Adjacency fill: {time.time()-t0:.1f}s", flush=True)


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


# ── Train ──────────────────────────────────────────────────────────────

def train_fold(fold, concept_feats_train, concept_feats_val, adj_train, adj_val,
               scibert_table):
    print(f"\n{'='*60}\n  Fold {fold} (v7 PPR-ego + BCE)\n{'='*60}", flush=True)
    device = torch.device('cuda')

    fold_dir = f'{TEMP_DIR}/fold{fold}_v2_chunks'
    manifest = json.load(open(f'{fold_dir}/manifest.json'))
    print(f"  train: {manifest['window_train']}, val: {manifest['window_val']}", flush=True)

    print(f"  Loading train chunks...", flush=True)
    train_pairs, train_labels = _load_chunks('train', fold_dir)
    src_train = build_anchor_data(train_pairs, train_labels)
    print(f"  Train anchors: {len(src_train):,}", flush=True)

    train_ds = AnchorDataset(src_train, concept_feats_train, N_CANDIDATES)
    train_loader = torch.utils.data.DataLoader(
        train_ds, BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=anchor_collate, pin_memory=True)

    print(f"  Loading val chunks...", flush=True)
    val_pairs, val_labels = _load_chunks('val', fold_dir)
    src_val = build_anchor_data(val_pairs, val_labels)
    print(f"  Val anchors: {len(src_val):,}", flush=True)

    val_ds = AnchorDataset(src_val, concept_feats_val, N_CANDIDATES, rng_seed=99)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=anchor_collate, pin_memory=True)

    model = PencilV7(scibert_table).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)

    dummy = torch.zeros(2, 1 + N_CANDIDATES, MAX_NODES, EMBED_DIM, device=device)
    dummy_nid = torch.zeros(2, 1 + N_CANDIDATES, MAX_NODES, dtype=torch.long, device=device)
    model.train()
    _ = model(dummy, dummy_nid)
    torch.cuda.synchronize()

    model_dir = f'{TEMP_DIR}/model_fold{fold}_v7'
    os.makedirs(model_dir, exist_ok=True)
    best_disc = -1
    patience_counter = 0

    POS_WEIGHT = 9.0  # sqrt(81), moderate rebalancing
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=device))
    print(f"\n--- Training (v7 PPR-ego, {N_CANDIDATES} cands, bs={BATCH_SIZE}, patience={PATIENCE}) ---",
          flush=True)

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

def eval_fold(fold, concept_feats_val, adj_val, scibert_table):
    print(f"\n{'='*60}\n  Eval fold {fold} (v7 PPR-ego)\n{'='*60}", flush=True)
    device = torch.device('cuda')

    fold_dir = f'{TEMP_DIR}/fold{fold}_v2_chunks'
    manifest = json.load(open(f'{fold_dir}/manifest.json'))
    print(f"  val: {manifest['window_val']}", flush=True)

    val_pairs, val_labels = _load_chunks('val', fold_dir)
    src_val = build_anchor_data(val_pairs, val_labels)
    print(f"  Val anchors: {len(src_val):,}", flush=True)

    val_ds = AnchorDataset(src_val, concept_feats_val, N_CANDIDATES, rng_seed=99)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=anchor_collate, pin_memory=True)

    model = PencilV7(scibert_table).to(device)
    model_file = f'{TEMP_DIR}/model_fold{fold}_v7/best.pt'
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

    print(f"\n=== Eval fold {fold} (v7 PPR-ego) ===", flush=True)
    print(f"  Disc ratio: {disc:.3f}", flush=True)
    print(f"  Spearman: {sp_corr:.4f}", flush=True)
    print(f"  top_mean: {top_mean:.6f}, bot_mean: {bot_mean:.6f}", flush=True)

    src_ids = np.array(val_ds.anchors)
    all_nids = np.concatenate(all_nids_list)
    np.savez(f'{TEMP_DIR}/eval_v7_fold{fold}.npz',
             scores=all_scores, labels=all_labels,
             src_ids=src_ids, cand_v=all_nids)
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

    # Load v2 concept IDs
    print("Loading v2 concept IDs...", flush=True)
    with open(CONCEPT_IDS_FILE) as f:
        concept_ids = json.load(f)
    n_concepts = len(concept_ids)
    print(f"  {n_concepts} concepts", flush=True)

    # Load SciBERT
    print("Loading SciBERT (remapped)...", flush=True)
    scibert_table = load_scibert_remapped(concept_ids, n_concepts)

    # Load train graph + PPR (2005)
    print(f"Loading graph_v2_{args.train_graph_year}...", flush=True)
    adj_train = sp.load_npz(f'{TEMP_DIR}/graph_v2_{args.train_graph_year}.npz').tocsr()
    print(f"  {adj_train.shape[0]} nodes, {adj_train.nnz:,} edges", flush=True)

    ppr_train_path = f'{TEMP_DIR}/ppr_v2_{args.train_graph_year}.npy'
    print(f"Loading PPR {args.train_graph_year}...", flush=True)
    ppr_train = np.load(ppr_train_path).astype(np.float32)
    print(f"  shape: {ppr_train.shape}", flush=True)

    print("Building train concept feature table (PPR-ego)...", flush=True)
    t0 = time.time()
    concept_feats_train, concept_nids_train = build_concept_feature_table(
        ppr_train, n_concepts, device)
    fill_adjacency_features(concept_feats_train, concept_nids_train, adj_train)
    print(f"  Total: {time.time()-t0:.1f}s", flush=True)

    # Load val graph + PPR (2015)
    print(f"Loading graph_v2_{args.val_graph_year}...", flush=True)
    adj_val = sp.load_npz(f'{TEMP_DIR}/graph_v2_{args.val_graph_year}.npz').tocsr()
    print(f"  {adj_val.shape[0]} nodes, {adj_val.nnz:,} edges", flush=True)

    ppr_val_path = f'{TEMP_DIR}/ppr_v2_{args.val_graph_year}.npy'
    print(f"Loading PPR {args.val_graph_year}...", flush=True)
    ppr_val = np.load(ppr_val_path).astype(np.float32)
    print(f"  shape: {ppr_val.shape}", flush=True)

    print("Building val concept feature table (PPR-ego)...", flush=True)
    t0 = time.time()
    concept_feats_val, concept_nids_val = build_concept_feature_table(
        ppr_val, n_concepts, device)
    fill_adjacency_features(concept_feats_val, concept_nids_val, adj_val)
    print(f"  Total: {time.time()-t0:.1f}s", flush=True)

    if args.eval:
        eval_fold(args.fold, concept_feats_val, adj_val, scibert_table)
    elif args.train_all:
        for fold in [0]:
            train_fold(fold, concept_feats_train, concept_feats_val,
                       adj_train, adj_val, scibert_table)
    else:
        train_fold(args.fold, concept_feats_train, concept_feats_val,
                   adj_train, adj_val, scibert_table)


if __name__ == '__main__':
    main()
