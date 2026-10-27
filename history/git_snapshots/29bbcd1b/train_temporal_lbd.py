"""
Temporal LBD training with PENCIL Graph Transformer.

Combines:
  - PENCIL v5 mix: Graph Transformer with struct+scibert+dist features
  - Phoenix quant: WeightedBCELoss, discrimination_ratio, pairwise_ranking_loss
  - Rolling window: train on 2012→2017, val on 2017→2021; retrain on 2017→2021, test 2021→2024

Usage:
  python train_temporal_lbd.py --fold 0    # train fold 0, val on fold 0 val
  python train_temporal_lbd.py --fold 1    # train fold 1, val on fold 1 val
  python train_temporal_lbd.py --eval      # eval fold 1 model on test data
"""
import os, time, argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from multiprocessing import Process
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
TEMP_DIR = f'{DATA_DIR}/temporal'

# Model config (same as PENCIL v5 mix)
MAX_NODES = 32
EMBED_DIM = 68   # 2*MAX_NODES + 4
SCIBERT_DIM = 768
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
DIST_BINS = 10
DIST_EMBED_DIM = 32

# Training config
BATCH_SIZE = 256
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8
WARMUP_FRAC = 0.1

# Eval config
TOP_K = [1, 3, 5, 10, 50, 100]


# ── Model ───────────────────────────────────────────────────────────────
class PencilTemporal(nn.Module):
    """PENCIL Graph Transformer for temporal link prediction."""

    def __init__(self, scibert_embeddings):
        super().__init__()
        self.register_buffer('scibert_table', scibert_embeddings)
        self.scibert_proj = nn.Linear(SCIBERT_DIM, HIDDEN // 2)
        self.struct_proj = nn.Linear(EMBED_DIM, HIDDEN // 2)
        self.dist_embed = nn.Embedding((DIST_BINS + 2) ** 2, DIST_EMBED_DIM)
        self.combine = nn.Linear(HIDDEN // 2 + HIDDEN // 2 + DIST_EMBED_DIM, HIDDEN)
        self.layers = nn.ModuleList()
        for _ in range(N_LAYERS):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(HIDDEN, N_HEADS, batch_first=True,
                                              dropout=0.1),
                'ln1': nn.LayerNorm(HIDDEN),
                'ffn': nn.Sequential(nn.Linear(HIDDEN, HIDDEN * 4), nn.GELU(),
                                     nn.Linear(HIDDEN * 4, HIDDEN), nn.Dropout(0.1)),
                'ln2': nn.LayerNorm(HIDDEN),
                'mp': nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU()),
            }))
        self.ln_f = nn.LayerNorm(HIDDEN)
        self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
                                  nn.Linear(HIDDEN, 1))

    def forward(self, inp, node_ids):
        """
        inp: [B, MAX_NODES, EMBED_DIM]  (float16/32)
        node_ids: [B, MAX_NODES]  (int64)
        Returns: [B] scores
        """
        B = inp.shape[0]

        # Structural features (no distance features for projection)
        xr = inp.clone()
        xr[:, :, 2 * MAX_NODES + 1] = 0
        xr[:, :, 2 * MAX_NODES + 2] = 0
        se = self.struct_proj(xr)                          # [B, N, H/2]

        # SciBERT features per node
        sc = self.scibert_proj(
            self.scibert_table[node_ids.clamp(0, self.scibert_table.shape[0] - 1)]
        )                                                   # [B, N, H/2]

        # Distance embeddings
        du = (inp[:, :, 2 * MAX_NODES + 1] * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        dv = (inp[:, :, 2 * MAX_NODES + 2] * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        de = self.dist_embed(du * (DIST_BINS + 2) + dv)    # [B, N, DED]

        x = self.combine(torch.cat([se, sc, de], -1))      # [B, N, H]

        # Adjacency for message passing
        adj_m = inp[:, :, MAX_NODES:2 * MAX_NODES]         # [B, N, N]
        an = adj_m / adj_m.sum(-1, keepdim=True).clamp(min=1)

        # Transformer layers
        attn_mask = torch.zeros(MAX_NODES, MAX_NODES, device=x.device)
        for layer in self.layers:
            h = layer['ln1'](x)
            h, _ = layer['attn'](h, h, h, attn_mask=attn_mask)
            x = x + h
            # Message passing
            x = x + layer['mp'](torch.bmm(an, x))
            h = layer['ln2'](x)
            x = x + layer['ffn'](h)

        x = self.ln_f(x)
        pooled = x.mean(dim=1)                             # [B, H]
        return self.head(pooled).squeeze(-1)                # [B]

    @torch.no_grad()
    def score_batch(self, inp, node_ids, inner_bs=512):
        """Score a large batch in chunks (for eval). Returns numpy array."""
        n = inp.shape[0]
        inp = inp.float()  # ensure float32 for model layers
        scores = []
        for st in range(0, n, inner_bs):
            ed = min(st + inner_bs, n)
            s = self.forward(inp[st:ed].to(next(self.parameters()).device),
                             node_ids[st:ed].to(next(self.parameters()).device))
            scores.append(s.float().cpu())
        return torch.cat(scores).numpy()


# ── Dataset ─────────────────────────────────────────────────────────────
class SubgraphDataset(Dataset):
    """Precomputed subgraph dataset."""

    def __init__(self, inp, node_ids, labels):
        self.inp = inp.astype(np.float32)   # float16 → float32 for training
        self.nids = node_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.inp[idx]),
                torch.from_numpy(self.nids[idx]),
                self.labels[idx])


# ── Loss and metrics ────────────────────────────────────────────────────
def pairwise_ranking_loss(scores, targets):
    """Pairwise ranking loss on continuous targets.
    For pairs (i,j) where target[i] > target[j], enforce score[i] > score[j].
    scores: [B], targets: [B] (continuous distance change values)
    """
    n = len(scores)
    if n < 2:
        return torch.tensor(0.0, device=scores.device)

    t_diff = targets.unsqueeze(0) - targets.unsqueeze(1)  # [n, n]
    s_diff = scores.unsqueeze(0) - scores.unsqueeze(1)     # [n, n]
    pair_mask = t_diff > 0
    npairs = pair_mask.sum()
    if npairs == 0:
        return torch.tensor(0.0, device=scores.device)

    return F.relu(1.0 - s_diff[pair_mask]).sum() / npairs.float()


def compute_metrics(scores, targets):
    """Compute disc ratio on continuous distance change values."""
    scores_f = scores.float()
    ranked = torch.argsort(-scores_f)
    n = len(scores_f)
    top_n = max(1, n // 10)
    top_mean = targets[ranked[:top_n]].mean().item()
    bot_mean = targets[ranked[top_n:]].mean().item()
    disc_ratio = top_mean / max(abs(bot_mean), 1e-8)
    return {
        'disc_ratio': disc_ratio,
        'top_mean_change': top_mean,
        'bot_mean_change': bot_mean,
    }


# ── Training loop ───────────────────────────────────────────────────────
def train_fold(fold, resume_from=None):
    """Train PENCIL temporal model on given fold."""
    print(f"{'='*60}")
    print(f"  Training fold {fold}")
    print(f"{'='*60}", flush=True)

    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    # Load precomputed subgraphs
    subgraph_file = f'{TEMP_DIR}/fold{fold}_subgraphs.npz'
    print(f"Loading {subgraph_file}...", flush=True)
    data = np.load(subgraph_file, allow_pickle=True)

    train_ds = SubgraphDataset(data['train_inp'], data['train_nids'], data['train_labels'])
    val_ds = SubgraphDataset(data['val_inp'], data['val_nids'], data['val_labels'])
    print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds)} samples", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Load SciBERT embeddings
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)

    # Build model
    model = PencilTemporal(scibert).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} params", flush=True)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss
    scaler = torch.amp.GradScaler()

    # Tracking
    best_val_disc = -1
    best_val_loss = float('inf')
    patience_counter = 0
    model_dir = f'{TEMP_DIR}/model_fold{fold}'
    os.makedirs(model_dir, exist_ok=True)

    if resume_from and os.path.exists(resume_from):
        model.load_state_dict(torch.load(resume_from, weights_only=True))
        print(f"  Resumed from {resume_from}", flush=True)

    # Training
    print(f"\n--- Training (batch_size={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}) ---",
          flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss = 0
        total_disc = 0
        n_batches = 0

        for inp, nids, labels in train_loader:
            inp = inp.to(device, non_blocking=True)
            nids = nids.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                scores = model(inp, nids)
            scores_f32 = scores.float()
            loss = pairwise_ranking_loss(scores_f32, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            metrics = compute_metrics(scores_f32.detach(), labels)
            total_disc += metrics['disc_ratio']
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_disc = total_disc / n_batches
        dt = time.time() - t0

        # Validation
        model.eval()
        val_scores_all = []
        val_labels_all = []
        val_loss_sum = 0
        val_batches = 0

        with torch.no_grad():
            for inp, nids, labels in val_loader:
                inp = inp.to(device, non_blocking=True)
                nids = nids.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    scores = model(inp, nids)
                scores_f32 = scores.float()
                val_loss_sum += pairwise_ranking_loss(scores_f32, labels).item()
                val_scores_all.append(scores_f32.cpu())
                val_labels_all.append(labels.cpu())
                val_batches += 1

        val_scores = torch.cat(val_scores_all)
        val_labels_cat = torch.cat(val_labels_all)
        val_metrics = compute_metrics(val_scores, val_labels_cat)
        val_loss = val_loss_sum / val_batches

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f} disc={avg_disc:.3f} | "
              f"val_loss={val_loss:.4f} val_disc={val_metrics['disc_ratio']:.3f} "
              f"top={val_metrics['top_mean_change']:.3f} "
              f"bot={val_metrics['bot_mean_change']:.3f} | "
              f"{dt:.1f}s lr={lr_now:.2e}", flush=True)

        # Early stopping on validation discrimination ratio
        if val_metrics['disc_ratio'] > best_val_disc:
            best_val_disc = val_metrics['disc_ratio']
            best_val_loss = val_loss
            patience_counter = 0
            path = f'{model_dir}/best.pt'
            torch.save(model.state_dict(), path)
            print(f"    → saved {path} (disc={best_val_disc:.3f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})", flush=True)
                break

    print(f"\n  Best val disc={best_val_disc:.3f}, loss={best_val_loss:.4f}", flush=True)
    return f'{model_dir}/best.pt'


# ── Eval: R@K ranking ───────────────────────────────────────────────────
def eval_ranking(fold):
    """Evaluate temporal LBD: load precomputed val subgraphs, score, rank, compute metrics."""
    print(f"\n{'='*60}")
    print(f"  Evaluating fold {fold}")
    print(f"{'='*60}", flush=True)

    device = torch.device('cuda')
    torch.cuda.empty_cache()

    # Load precomputed val subgraphs (already built by build_temporal_data.py)
    subgraph_file = f'{TEMP_DIR}/fold{fold}_subgraphs.npz'
    data = np.load(subgraph_file, allow_pickle=True)
    val_pairs = data['val_pairs']
    val_labels = data['val_labels'].astype(np.float32)
    val_inp = data['val_inp']
    val_nids = data['val_nids']
    n_val = len(val_pairs)
    print(f"  Val pairs: {n_val}, dist_change: min={val_labels.min():.0f}, "
          f"max={val_labels.max():.0f}, mean={val_labels.mean():.2f}", flush=True)
    print(f"  Precomputed subgraphs: {val_inp.shape}", flush=True)

    # Load model
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilTemporal(scibert).to(device)
    model_file = f'{TEMP_DIR}/model_fold{fold}/best.pt'
    if not os.path.exists(model_file):
        print(f"  WARNING: {model_file} not found, using random weights", flush=True)
    else:
        model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()

    # Warm up GPU
    _ = model.score_batch(torch.zeros(2, MAX_NODES, EMBED_DIM),
                          torch.zeros(2, MAX_NODES, dtype=torch.long))
    torch.cuda.synchronize()

    # GPU scoring
    print(f"  GPU scoring {n_val:,} pairs...", flush=True)
    t0 = time.time()
    all_scores = model.score_batch(
        torch.from_numpy(val_inp), torch.from_numpy(val_nids), inner_bs=2048)
    torch.cuda.synchronize()
    print(f"  GPU scoring: {time.time()-t0:.1f}s", flush=True)

    # Compute metrics
    scores = all_scores
    labels = val_labels
    ranked = np.argsort(-scores)
    top_frac = max(1, n_val // 10)

    top_mean_change = labels[ranked[:top_frac]].mean()
    bot_mean_change = labels[ranked[top_frac:]].mean()
    disc_ratio = float(top_mean_change / max(abs(bot_mean_change), 1e-8))

    from scipy.stats import spearmanr, pearsonr
    spearman_corr, _ = spearmanr(scores, labels)
    pearson_corr, _ = pearsonr(scores, labels)

    actually_converged = labels >= 2
    n_converged = int(actually_converged.sum())
    if n_converged > 0:
        rank_of_pair = np.zeros(n_val)
        rank_of_pair[ranked] = np.arange(n_val)
        conv_ranks = rank_of_pair[actually_converged] + 1
    else:
        conv_ranks = np.array([])

    print(f"\n=== Temporal LBD eval (fold {fold}) ===", flush=True)
    print(f"  Disc ratio: {disc_ratio:.3f} (top10%_mean={top_mean_change:.3f}, "
          f"bot90%_mean={bot_mean_change:.3f})", flush=True)
    print(f"  Spearman: {spearman_corr:.4f}, Pearson: {pearson_corr:.4f}", flush=True)
    print(f"  Actually converged (dist_change>=2): {n_converged}/{n_val}", flush=True)
    if n_converged > 0:
        for k in [1, 3, 5, 10, 50, 100]:
            if k <= n_val:
                recall = (conv_ranks <= k).mean()
                print(f"  Convergence R@{k}: {recall:.4f}", flush=True)
        print(f"  Median converged rank: {np.median(conv_ranks):.0f} / {n_val}", flush=True)
        print(f"  MRR (converged): {(1.0 / conv_ranks).mean():.4f}", flush=True)
    print(f"  Total eval: {time.time()-t0:.1f}s", flush=True)

    np.save(f'{TEMP_DIR}/eval_fold{fold}_scores.npy',
            np.column_stack([scores, labels]))
    return disc_ratio



def _build_subgraphs_batch(pairs, dm_t1, indptr, indices):
    """Build subgraphs for a batch of pairs using vectorized numpy."""
    MN = MAX_NODES; DB = 10.0
    nb = len(pairs)

    nu = indptr[pairs[:, 0] + 1] - indptr[pairs[:, 0]]
    nv = indptr[pairs[:, 1] + 1] - indptr[pairs[:, 1]]
    u_nbs = np.full((nb, MN - 2), -1, dtype=np.int64)
    v_nbs = np.full((nb, MN - 2), -1, dtype=np.int64)
    u_cnt = np.minimum(nu, MN - 2)
    v_cnt = np.minimum(nv, MN - 2)

    for i in range(nb):
        u_nbs[i, :u_cnt[i]] = indices[indptr[pairs[i, 0]]:indptr[pairs[i, 0]] + u_cnt[i]]
        v_nbs[i, :v_cnt[i]] = indices[indptr[pairs[i, 1]]:indptr[pairs[i, 1]] + v_cnt[i]]

    sel = np.zeros((nb, MN), dtype=np.int64)
    sel[:, 0] = pairs[:, 0]
    sel[:, 1] = pairs[:, 1]
    sel[:, 2:MN - 2 + 2] = np.where(u_nbs >= 0, u_nbs, pairs[:, 0:1])

    du = dm_t1[pairs[:, 0]]
    dv = dm_t1[pairs[:, 1]]

    feat = np.zeros((nb, MN, EMBED_DIM), dtype=np.float32)
    for j in range(MN):
        nodes_j = sel[:, j]
        feat[:, j, j] = 1.0
        feat[:, j, 2 * MN] = 1.0
        feat[:, j, 2 * MN + 1] = np.minimum(du[np.arange(nb), nodes_j], DB) / DB
        feat[:, j, 2 * MN + 2] = np.minimum(dv[np.arange(nb), nodes_j], DB) / DB

    for i in range(nb):
        sel_set = set(sel[i].tolist())
        for j in range(MN):
            a = sel[i, j]
            for k2 in range(indptr[a], indptr[a + 1]):
                b = int(indices[k2])
                if b in sel_set:
                    for k3 in range(MN):
                        if sel[i, k3] == b:
                            feat[i, j, MN + k3] = 1.0
                            break

    return feat.astype(np.float16), sel


def eval_full(fold):
    """Full eval: score ALL eligible pairs, rank, find top convergence candidates."""
    FULL_BATCH = 1000000   # pairs per chunk for eligible pair extraction
    SCORE_BATCH = 100000   # pairs per subgraph build + GPU score batch

    print(f"\n{'='*60}")
    print(f"  Full eval: fold {fold}")
    print(f"{'='*60}", flush=True)

    device = torch.device('cuda')
    torch.cuda.empty_cache()

    # Determine time windows
    if fold == 0:
        t1_year, t2_year = 2017, 2021
    else:
        t1_year, t2_year = 2021, 2024

    # Load distance matrices and graph
    print(f"  Loading dist_{t1_year} + dist_{t2_year}...", flush=True)
    dm_t1 = np.load(f'{TEMP_DIR}/dist_{t1_year}.npy').astype(np.int32)
    dm_t2 = np.load(f'{TEMP_DIR}/dist_{t2_year}.npy').astype(np.int32)
    adj = sp.load_npz(f'{TEMP_DIR}/graph_{t1_year}.npz').tocsr()
    indptr = adj.indptr
    indices = adj.indices
    n_nodes = dm_t1.shape[0]

    # Find all eligible pairs (dist_T1 in [3, 9999), upper triangle)
    print(f"  Finding eligible pairs (dist_{t1_year} >= 3, upper triangle)...", flush=True)
    t0 = time.time()
    eligible_pairs_list = []
    eligible_dc_list = []
    total_upper = n_nodes * (n_nodes - 1) // 2

    for chunk_start in range(0, total_upper, FULL_BATCH):
        chunk_end = min(chunk_start + FULL_BATCH, total_upper)
        rows_ch = np.zeros(chunk_end - chunk_start, dtype=np.int64)
        cols_ch = np.zeros(chunk_end - chunk_start, dtype=np.int64)

        # Map linear index to (row, col) for upper triangle
        idx = np.arange(chunk_start, chunk_end, dtype=np.int64)
        # Solve: row*(2*n - 3 - row)/2 + col - 1 = idx for upper triangle
        # Use binary search approach via numpy
        # row = floor((2*n - 3 - sqrt((2*n-3)^2 - 8*(idx-1))) / 2)
        a_coeff = 2 * n_nodes - 3
        disc = a_coeff.astype(np.float64) ** 2 - 8.0 * (idx.astype(np.float64) - 1)
        disc = np.maximum(disc, 0)
        rows_ch = ((a_coeff - np.sqrt(disc)) / 2).astype(np.int64)
        cols_ch = idx - rows_ch.astype(np.int64) * (2 * n_nodes - 3 - rows_ch) // 2 + rows_ch + 1

        t1_vals = dm_t1[rows_ch, cols_ch]
        eligible = (t1_vals >= 3) & (t1_vals < 9999)
        if eligible.any():
            rows_e = rows_ch[eligible]
            cols_e = cols_ch[eligible]
            t1_e = dm_t1[rows_e, cols_e].astype(np.float32)
            t2_e = dm_t2[rows_e, cols_e].astype(np.float32)
            dc = t1_e - t2_e
            eligible_pairs_list.append(np.column_stack([rows_e, cols_e]))
            eligible_dc_list.append(dc)

        if (chunk_start // FULL_BATCH + 1) % 50 == 0:
            print(f"    processed {chunk_end:,}/{total_upper:,} pairs...", flush=True)

    all_pairs = np.concatenate(eligible_pairs_list)
    all_dc = np.concatenate(eligible_dc_list)
    n_eligible = len(all_pairs)
    dt = time.time() - t0
    print(f"  Eligible pairs: {n_eligible:,} ({100*n_eligible/total_upper:.1f}%), "
          f"found in {dt:.0f}s", flush=True)
    print(f"  dist_change: min={all_dc.min():.0f}, max={all_dc.max():.0f}, "
          f"mean={all_dc.mean():.3f}", flush=True)

    # Load model
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilTemporal(scibert).to(device)
    model_file = f'{TEMP_DIR}/model_fold{fold}/best.pt'
    if not os.path.exists(model_file):
        print(f"  WARNING: {model_file} not found, using random weights", flush=True)
    else:
        model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()

    # Warm up
    _ = model.score_batch(torch.zeros(2, MAX_NODES, EMBED_DIM),
                          torch.zeros(2, MAX_NODES, dtype=torch.long))
    torch.cuda.synchronize()

    # Score all eligible pairs in batches
    print(f"  Scoring {n_eligible:,} pairs (batch={SCORE_BATCH})...", flush=True)
    all_scores = np.zeros(n_eligible, dtype=np.float32)
    n_score_batches = (n_eligible + SCORE_BATCH - 1) // SCORE_BATCH
    t1 = time.time()

    for bi in range(n_score_batches):
        s = bi * SCORE_BATCH
        e = min(s + SCORE_BATCH, n_eligible)
        inp, nids = _build_subgraphs_batch(all_pairs[s:e], dm_t1, indptr, indices)
        scores_batch = model.score_batch(
            torch.from_numpy(inp), torch.from_numpy(nids), inner_bs=2048)
        all_scores[s:e] = scores_batch

        if (bi + 1) % 10 == 0 or bi == n_score_batches - 1:
            elapsed = time.time() - t1
            done = e
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_eligible - done) / rate if rate > 0 else 0
            print(f"    scored {done:,}/{n_eligible:,} ({rate:.0f}/s, ETA {eta:.0f}s)", flush=True)

    dt_score = time.time() - t1
    print(f"  Scoring done: {dt_score:.1f}s ({n_eligible/dt_score:.0f} pairs/s)", flush=True)

    # Compute metrics
    scores = all_scores
    labels = all_dc
    ranked = np.argsort(-scores)
    top_frac = max(1, n_eligible // 10)

    top_mean_change = labels[ranked[:top_frac]].mean()
    bot_mean_change = labels[ranked[top_frac:]].mean()
    disc_ratio = float(top_mean_change / max(abs(bot_mean_change), 1e-8))

    from scipy.stats import spearmanr, pearsonr
    spearman_corr, _ = spearmanr(scores, labels)
    pearson_corr, _ = pearsonr(scores, labels)

    actually_converged = labels >= 2
    n_converged = int(actually_converged.sum())
    if n_converged > 0:
        rank_of_pair = np.zeros(n_eligible)
        rank_of_pair[ranked] = np.arange(n_eligible)
        conv_ranks = rank_of_pair[actually_converged] + 1
    else:
        conv_ranks = np.array([])

    print(f"\n{'='*60}", flush=True)
    print(f"  FULL EVAL (fold {fold}): {n_eligible:,} pairs", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Disc ratio: {disc_ratio:.3f} (top10%_mean={top_mean_change:.3f}, "
          f"bot90%_mean={bot_mean_change:.3f})", flush=True)
    print(f"  Spearman: {spearman_corr:.4f}, Pearson: {pearson_corr:.4f}", flush=True)
    print(f"  Actually converged (dist_change>=2): {n_converged:,}/{n_eligible:,}", flush=True)
    if n_converged > 0:
        for k in [1, 3, 5, 10, 50, 100, 1000]:
            if k <= n_eligible:
                recall = (conv_ranks <= k).mean()
                print(f"  Convergence R@{k}: {recall:.4f}", flush=True)
        print(f"  Median converged rank: {np.median(conv_ranks):.0f} / {n_eligible:,}", flush=True)
        print(f"  MRR (converged): {(1.0 / conv_ranks).mean():.4f}", flush=True)

    # Save top-1000 results
    top_k = 1000
    top_idx = ranked[:top_k]
    print(f"\n  Top-{top_k} pairs by model score:", flush=True)
    print(f"    mean dist_change: {labels[top_idx].mean():.3f}", flush=True)
    print(f"    mean dist_T1: {dm_t1[all_pairs[top_idx, 0], all_pairs[top_idx, 1]].mean():.1f}", flush=True)
    print(f"    convergent in top-{top_k}: {int(labels[top_idx].sum() > 0)}", flush=True)

    # Save results
    np.savez(f'{TEMP_DIR}/eval_full_fold{fold}.npz',
             scores=scores, labels=labels, pairs=all_pairs,
             ranked=ranked[:10000])
    print(f"  Saved {TEMP_DIR}/eval_full_fold{fold}.npz", flush=True)
    print(f"  Total eval: {time.time()-t0:.1f}s", flush=True)
    del dm_t1, dm_t2
    return disc_ratio


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1, help='Fold to train (0 or 1)')
    parser.add_argument('--eval', action='store_true', help='Eval on precomputed val subgraphs')
    parser.add_argument('--train-all', action='store_true', help='Train all folds in parallel')
    parser.add_argument('--eval-all', action='store_true', help='Eval all folds (precomputed)')
    parser.add_argument('--eval-full', action='store_true', help='Full eval: score ALL eligible pairs')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    if args.train_all:
        print("Training both folds in parallel...", flush=True)
        procs = []
        for fold in [0, 1]:
            p = Process(target=train_fold, args=(fold,))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        print("Both folds trained.", flush=True)
    elif args.eval_full:
        eval_full(args.fold)
    elif args.eval_all:
        for fold in [0, 1]:
            eval_ranking(fold)
    elif args.eval:
        eval_ranking(args.fold)
    else:
        best_path = train_fold(args.fold, resume_from=args.resume)
        print(f"\nTraining complete. Best model: {best_path}", flush=True)
        print(f"Run eval with: python {__file__} --fold {args.fold} --eval", flush=True)


if __name__ == '__main__':
    main()
