"""
Temporal LBD training with PENCIL Graph Transformer.
Subgraphs built on-the-fly from chunk files (no precomputed subgraphs).

Usage:
  python train_temporal_lbd.py --fold 0
  python train_temporal_lbd.py --fold 1
  python train_temporal_lbd.py --train-all
  python train_temporal_lbd.py --eval-all
  python train_temporal_lbd.py --eval-full --fold 0
"""
import os, time, argparse, glob, json
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from multiprocessing import Process
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

BATCH_SIZE = 256
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
WARMUP_FRAC = 0.1
SUBGRAPH_BATCH = 50000
DB = 10.0


# ── Model ───────────────────────────────────────────────────────────────
class PencilTemporal(nn.Module):
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
                'attn': nn.MultiheadAttention(HIDDEN, N_HEADS, batch_first=True, dropout=0.1),
                'ln1': nn.LayerNorm(HIDDEN),
                'ffn': nn.Sequential(nn.Linear(HIDDEN, HIDDEN * 4), nn.GELU(),
                                     nn.Linear(HIDDEN * 4, HIDDEN), nn.Dropout(0.1)),
                'ln2': nn.LayerNorm(HIDDEN),
                'mp': nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU()),
            }))
        self.ln_f = nn.LayerNorm(HIDDEN)
        self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

    def forward(self, inp, node_ids):
        B = inp.shape[0]
        xr = inp.clone()
        xr[:, :, 2 * MAX_NODES + 1] = 0
        xr[:, :, 2 * MAX_NODES + 2] = 0
        se = self.struct_proj(xr)
        sc = self.scibert_proj(
            self.scibert_table[node_ids.clamp(0, self.scibert_table.shape[0] - 1)])
        du = (inp[:, :, 2 * MAX_NODES + 1] * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        dv = (inp[:, :, 2 * MAX_NODES + 2] * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        de = self.dist_embed(du * (DIST_BINS + 2) + dv)
        x = self.combine(torch.cat([se, sc, de], -1))
        adj_m = inp[:, :, MAX_NODES:2 * MAX_NODES]
        an = adj_m / adj_m.sum(-1, keepdim=True).clamp(min=1)
        attn_mask = torch.zeros(MAX_NODES, MAX_NODES, device=x.device)
        for layer in self.layers:
            h = layer['ln1'](x)
            h, _ = layer['attn'](h, h, h, attn_mask=attn_mask)
            x = x + h
            x = x + layer['mp'](torch.bmm(an, x))
            h = layer['ln2'](x)
            x = x + layer['ffn'](h)
        x = self.ln_f(x)
        return self.head(x.mean(dim=1)).squeeze(-1)

    @torch.no_grad()
    def score_batch(self, inp, node_ids, inner_bs=512):
        n = inp.shape[0]
        inp = inp.float()
        scores = []
        for st in range(0, n, inner_bs):
            ed = min(st + inner_bs, n)
            s = self.forward(inp[st:ed].to(next(self.parameters()).device),
                             node_ids[st:ed].to(next(self.parameters()).device))
            scores.append(s.float().cpu())
        return torch.cat(scores).numpy()


# ── Chunk loading ───────────────────────────────────────────────────────
def _load_chunks(prefix, fold_dir, max_chunks=None):
    """Load pairs/labels from chunk files into pre-allocated arrays."""
    files = sorted(glob.glob(f'{fold_dir}/{prefix}_chunk_*.npz'))
    if max_chunks:
        files = files[:max_chunks]
    if not files:
        raise FileNotFoundError(f"No chunks found: {fold_dir}/{prefix}_chunk_*.npz")

    sizes = []
    for f in files:
        with np.load(f) as d:
            sizes.append(len(d['pairs']))
    total = sum(sizes)

    pairs = np.empty((total, 2), dtype=np.int64)
    labels = np.empty(total, dtype=np.float32)
    off = 0
    for f, sz in zip(files, sizes):
        with np.load(f) as d:
            pairs[off:off + sz] = d['pairs']
            labels[off:off + sz] = d['labels']
        off += sz
    return pairs, labels


def _load_all_chunks(prefix, fold_dir):
    """Load all chunks, return (pairs, labels) lists (one entry per chunk)."""
    files = sorted(glob.glob(f'{fold_dir}/{prefix}_chunk_*.npz'))
    pairs_list, labels_list = [], []
    for f in files:
        with np.load(f) as d:
            pairs_list.append(d['pairs'])
            labels_list.append(d['labels'])
    return pairs_list, labels_list


# ── On-the-fly subgraph building (GPU-accelerated) ─────────────────────
def _init_gpu_graph(indptr, indices, dm_t1, device):
    """Pre-compute GPU tensors for fast subgraph building."""
    N = len(indptr) - 1
    MN = MAX_NODES
    degs = indptr[1:] - indptr[:-1]
    max_deg = int(degs.max())
    k = min(max_deg, MN - 2)
    # Padded neighbor table [N, k] via CSR slicing
    nb_table = np.zeros((N, k), dtype=np.int64)
    for i in range(N):
        d = min(int(degs[i]), k)
        if d > 0:
            nb_table[i, :d] = indices[indptr[i]:indptr[i] + d]
    return {
        'nb_table': torch.from_numpy(nb_table).to(device),
        'dm_t1': torch.from_numpy(dm_t1).to(device),
    }


def _build_subgraphs_batch(pairs, gpu_ctx):
    """Build subgraphs on GPU using gather ops."""
    MN = MAX_NODES
    nb = len(pairs)
    device = gpu_ctx['nb_table'].device
    nb_table = gpu_ctx['nb_table']  # [N, k]
    dm_gpu = gpu_ctx['dm_t1']       # [N, N] int32

    pairs_t = torch.from_numpy(pairs.astype(np.int64)).to(device)
    u_nodes = pairs_t[:, 0]  # [nb]
    v_nodes = pairs_t[:, 1]
    k = nb_table.shape[1]

    # Gather 1-hop neighbors
    u_nbs = nb_table[u_nodes]  # [nb, k]
    v_nbs = nb_table[v_nodes]
    half = (MN - 2) // 2  # 15

    # Build sel [nb, MN]
    sel = torch.empty((nb, MN), dtype=torch.int64, device=device)
    sel[:, 0] = u_nodes
    sel[:, 1] = v_nodes
    sel[:, 2:2 + half] = u_nbs[:, :half]
    sel[:, 2 + half:MN] = v_nbs[:, :MN - 2 - half]

    # Distance features on GPU
    arange_nb = torch.arange(nb, device=device)
    dm_u = dm_gpu[u_nodes]  # [nb, N]
    dm_v = dm_gpu[v_nodes]
    sel_du = torch.gather(dm_u, 1, sel.clamp(0, dm_gpu.shape[1] - 1))  # [nb, MN]
    sel_dv = torch.gather(dm_v, 1, sel.clamp(0, dm_gpu.shape[1] - 1))

    inp = torch.zeros((nb, MN, EMBED_DIM), dtype=torch.float16, device=device)
    inp[arange_nb[:, None], torch.arange(MN, device=device), torch.arange(MN, device=device)] = 1.0
    inp[:, :, 2 * MN] = 1.0
    inp[:, :, 2 * MN + 1] = torch.clamp(sel_du.float(), max=DB) / DB
    inp[:, :, 2 * MN + 2] = torch.clamp(sel_dv.float(), max=DB) / DB

    # Adjacency via gather + eq
    sel_col = sel[:, :, None].expand(nb, MN, MN)  # [nb, MN, MN]
    nb_sel = torch.gather(nb_table[sel.view(-1)].view(nb, MN, k), 2,
                          sel_col.clamp(0, k - 1))  # [nb, MN, MN]
    adj_sub = (nb_sel == sel_col).float()
    inp[:, :, MN:2 * MN] = adj_sub.half()

    return inp, sel


# ── Loss and metrics ────────────────────────────────────────────────────
def pairwise_ranking_loss(scores, targets):
    """Pairwise ranking loss using element-wise ops (torch.compile friendly)."""
    n = len(scores)
    if n < 2:
        return scores.sum() * 0
    t_diff = targets.unsqueeze(0) - targets.unsqueeze(1)  # [B, B]
    s_diff = scores.unsqueeze(0) - scores.unsqueeze(1)    # [B, B]
    pair_mask = (t_diff > 0).float()
    npairs = pair_mask.sum()
    if npairs == 0:
        return scores.sum() * 0
    # Element-wise: avoid boolean indexing for torch.compile compatibility
    return (F.relu(1.0 - s_diff) * pair_mask).sum() / npairs


def compute_metrics(scores, targets):
    scores_f = scores.float()
    ranked = torch.argsort(-scores_f)
    n = len(scores_f)
    top_n = max(1, n // 10)
    top_mean = targets[ranked[:top_n]].mean().item()
    bot_mean = targets[ranked[top_n:]].mean().item()
    return {'disc_ratio': top_mean / max(abs(bot_mean), 1e-8),
            'top_mean_change': top_mean, 'bot_mean_change': bot_mean}


# ── Training loop ───────────────────────────────────────────────────────
def train_fold(fold, resume_from=None):
    print(f"{'='*60}\n  Training fold {fold}\n{'='*60}", flush=True)
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    fold_dir = f'{TEMP_DIR}/fold{fold}_chunks'
    manifest = json.load(open(f'{fold_dir}/manifest.json'))
    print(f"  Window train: {manifest['window_train']}, val: {manifest['window_val']}", flush=True)

    # Load graph for on-the-fly subgraph building
    train_t1 = int(manifest['window_train'].split('_')[0])
    adj = sp.load_npz(f'{TEMP_DIR}/graph_{train_t1}.npz').tocsr()
    indptr = adj.indptr
    indices = adj.indices
    dm_t1 = np.load(f'{TEMP_DIR}/dist_{train_t1}.npy').astype(np.int32)

    # Move graph data to GPU for fast subgraph building
    print(f"  Building GPU graph tensors...", flush=True)
    gpu_ctx = _init_gpu_graph(indptr, indices, dm_t1, device)

    # Load pairs/labels from chunks
    print(f"  Loading train chunks...", flush=True)
    train_pairs, train_labels = _load_chunks('train', fold_dir)
    # Sample subset for faster training
    SAMPLE_FRAC = 0.33
    if SAMPLE_FRAC < 1.0:
        n_sample = int(len(train_pairs) * SAMPLE_FRAC)
        rng_sample = np.random.RandomState(42)
        idx_sample = rng_sample.choice(len(train_pairs), n_sample, replace=False)
        train_pairs = train_pairs[idx_sample]
        train_labels = train_labels[idx_sample]
    print(f"  Train: {len(train_pairs):,} pairs (sampled {SAMPLE_FRAC*100:.0f}%)", flush=True)

    print(f"  Loading val chunks...", flush=True)
    val_pairs, val_labels = _load_chunks('val', fold_dir)
    # Also sample val for speed
    if SAMPLE_FRAC < 1.0:
        n_sample_v = int(len(val_pairs) * SAMPLE_FRAC)
        idx_v = rng_sample.choice(len(val_pairs), n_sample_v, replace=False)
        val_pairs = val_pairs[idx_v]
        val_labels = val_labels[idx_v]
    print(f"  Val: {len(val_pairs):,} pairs", flush=True)

    # Shuffle indices for training
    rng = np.random.RandomState(42)
    train_idx = np.arange(len(train_pairs))
    rng.shuffle(train_idx)

    # SciBERT + model
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilTemporal(scibert).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} params", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_train_batches = len(train_idx) // BATCH_SIZE
    total_steps = n_train_batches * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler()

    best_val_disc = -1
    best_val_loss = float('inf')
    patience_counter = 0
    model_dir = f'{TEMP_DIR}/model_fold{fold}'
    os.makedirs(model_dir, exist_ok=True)

    if resume_from and os.path.exists(resume_from):
        model.load_state_dict(torch.load(resume_from, weights_only=True))
        print(f"  Resumed from {resume_from}", flush=True)

    print(f"\n--- Training (batch_size={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}) ---", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss = torch.tensor(0.0, device=device); n_batches = 0
        rng.shuffle(train_idx)

        # Build subgraphs in SUBGRAPH_BATCH chunks, train on each
        for chunk_s in range(0, len(train_idx), SUBGRAPH_BATCH):
            chunk_e = min(chunk_s + SUBGRAPH_BATCH, len(train_idx))
            idx_chunk = train_idx[chunk_s:chunk_e]
            pairs_chunk = train_pairs[idx_chunk]
            labels_chunk = train_labels[idx_chunk]

            inp_gpu, nids_gpu = _build_subgraphs_batch(pairs_chunk, gpu_ctx)

            # Train on this chunk (already on GPU)
            for bi in range(0, len(inp_gpu), BATCH_SIZE):
                bj = min(bi + BATCH_SIZE, len(inp_gpu))
                if bj - bi < 2:
                    continue
                inp = inp_gpu[bi:bj].float()
                nids = nids_gpu[bi:bj]
                labels = torch.from_numpy(labels_chunk[bi:bj]).to(device, non_blocking=True)

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

                total_loss += loss.detach()
                n_batches += 1

        avg_loss = (total_loss / max(n_batches, 1)).item()
        avg_disc = 0  # skip per-epoch metrics for speed
        dt = time.time() - t0

        # Validation: build subgraphs per chunk, score all
        model.eval()
        val_scores_all = []; val_labels_all = []
        val_loss_tensor = torch.tensor(0.0, device=device); val_batches = 0
        with torch.no_grad():
            for cs in range(0, len(val_pairs), SUBGRAPH_BATCH):
                ce = min(cs + SUBGRAPH_BATCH, len(val_pairs))
                inp_gpu, nids_gpu = _build_subgraphs_batch(val_pairs[cs:ce], gpu_ctx)
                for bi in range(0, len(inp_gpu), BATCH_SIZE):
                    bj = min(bi + BATCH_SIZE, len(inp_gpu))
                    if bj - bi < 2:
                        continue
                    inp = inp_gpu[bi:bj].float()
                    nids = nids_gpu[bi:bj]
                    labels = torch.from_numpy(val_labels[cs + bi:cs + bj]).to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        scores = model(inp, nids)
                    scores_f32 = scores.float()
                    val_loss_tensor += pairwise_ranking_loss(scores_f32, labels).detach()
                    val_scores_all.append(scores_f32.cpu())
                    val_labels_all.append(labels.cpu())
                    val_batches += 1

        val_scores = torch.cat(val_scores_all)
        val_labels_cat = torch.cat(val_labels_all)
        val_metrics = compute_metrics(val_scores, val_labels_cat)
        val_loss = (val_loss_tensor / max(val_batches, 1)).item()

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f} disc={avg_disc:.3f} | "
              f"val_loss={val_loss:.4f} val_disc={val_metrics['disc_ratio']:.3f} "
              f"top={val_metrics['top_mean_change']:.3f} "
              f"bot={val_metrics['bot_mean_change']:.3f} | "
              f"{dt:.1f}s lr={lr_now:.2e}", flush=True)

        if val_metrics['disc_ratio'] > best_val_disc:
            best_val_disc = val_metrics['disc_ratio']
            best_val_loss = val_loss
            patience_counter = 0
            path = f'{model_dir}/best.pt'
            torch.save(model.state_dict(), path)
            print(f"    -> saved {path} (disc={best_val_disc:.3f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})", flush=True)
                break

    print(f"\n  Best val disc={best_val_disc:.3f}, loss={best_val_loss:.4f}", flush=True)
    return f'{model_dir}/best.pt'


# ── Eval ────────────────────────────────────────────────────────────────
def eval_ranking(fold):
    print(f"\n{'='*60}\n  Evaluating fold {fold}\n{'='*60}", flush=True)
    device = torch.device('cuda')
    torch.cuda.empty_cache()

    fold_dir = f'{TEMP_DIR}/fold{fold}_chunks'
    manifest = json.load(open(f'{fold_dir}/manifest.json'))
    val_t1 = int(manifest['window_val'].split('_')[0])

    adj = sp.load_npz(f'{TEMP_DIR}/graph_{val_t1}.npz').tocsr()
    dm_t1 = np.load(f'{TEMP_DIR}/dist_{val_t1}.npy').astype(np.int32)
    indptr = adj.indptr; indices = adj.indices

    print(f"  Building GPU graph tensors...", flush=True)
    gpu_ctx = _init_gpu_graph(indptr, indices, dm_t1, device)

    val_pairs_list, val_labels_list = _load_all_chunks('val', fold_dir)
    n_val = sum(len(p) for p in val_pairs_list)
    print(f"  Val pairs: {n_val:,}", flush=True)

    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilTemporal(scibert).to(device)
    model_file = f'{TEMP_DIR}/model_fold{fold}/best.pt'
    if not os.path.exists(model_file):
        print(f"  WARNING: {model_file} not found, using random weights", flush=True)
    else:
        model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()

    all_scores = []; all_labels = []
    t0 = time.time()
    for pairs_chunk, labels_chunk in zip(val_pairs_list, val_labels_list):
        inp_gpu, nids_gpu = _build_subgraphs_batch(pairs_chunk, gpu_ctx)
        scores_chunk = model.score_batch(inp_gpu.float(), nids_gpu, inner_bs=2048)
        all_scores.append(scores_chunk)
        all_labels.append(labels_chunk)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    del val_pairs_list, val_labels_list

    from scipy.stats import spearmanr, pearsonr
    ranked = np.argsort(-scores)
    top_frac = max(1, n_val // 10)
    top_mean = labels[ranked[:top_frac]].mean()
    bot_mean = labels[ranked[top_frac:]].mean()
    disc_ratio = float(top_mean / max(abs(bot_mean), 1e-8))
    spearman_corr, _ = spearmanr(scores, labels)
    pearson_corr, _ = pearsonr(scores, labels)

    converged = labels >= 2
    n_conv = int(converged.sum())
    if n_conv > 0:
        rank_of = np.zeros(n_val); rank_of[ranked] = np.arange(n_val)
        conv_ranks = rank_of[converged] + 1

    print(f"\n=== Eval fold {fold} ({n_val:,} pairs) ===", flush=True)
    print(f"  Disc ratio: {disc_ratio:.3f}", flush=True)
    print(f"  Spearman: {spearman_corr:.4f}, Pearson: {pearson_corr:.4f}", flush=True)
    if n_conv > 0:
        for k in [1, 3, 5, 10, 50, 100]:
            if k <= n_val:
                print(f"  R@{k}: {(conv_ranks <= k).mean():.4f}", flush=True)
        print(f"  MRR: {(1.0 / conv_ranks).mean():.4f}", flush=True)
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    np.save(f'{TEMP_DIR}/eval_fold{fold}_scores.npy', np.column_stack([scores, labels]))
    return disc_ratio


def eval_full(fold):
    """Full eval: score ALL eligible pairs."""
    FULL_BATCH = 1_000_000
    print(f"\n{'='*60}\n  Full eval: fold {fold}\n{'='*60}", flush=True)
    device = torch.device('cuda'); torch.cuda.empty_cache()

    fold_dir = f'{TEMP_DIR}/fold{fold}_chunks'
    manifest = json.load(open(f'{fold_dir}/manifest.json'))
    t1_year = int(manifest['window_val'].split('_')[0])
    t2_year = int(manifest['window_val'].split('_')[1])

    dm_t1 = np.load(f'{TEMP_DIR}/dist_{t1_year}.npy').astype(np.int32)
    dm_t2 = np.load(f'{TEMP_DIR}/dist_{t2_year}.npy').astype(np.int32)
    adj = sp.load_npz(f'{TEMP_DIR}/graph_{t1_year}.npz').tocsr()
    indptr = adj.indptr; indices = adj.indices
    n_nodes = dm_t1.shape[0]

    print(f"  Building GPU graph tensors...", flush=True)
    gpu_ctx = _init_gpu_graph(indptr, indices, dm_t1, device)

    print(f"  Finding eligible pairs...", flush=True)
    t0 = time.time()
    el_pairs_list = []; el_dc_list = []
    total_upper = n_nodes * (n_nodes - 1) // 2
    for cs in range(0, total_upper, FULL_BATCH):
        ce = min(cs + FULL_BATCH, total_upper)
        idx = np.arange(cs, ce, dtype=np.int64)
        a_c = 2 * n_nodes - 3
        disc = a_c.astype(np.float64) ** 2 - 8.0 * (idx.astype(np.float64) - 1)
        disc = np.maximum(disc, 0)
        rows = ((a_c - np.sqrt(disc)) / 2).astype(np.int64)
        cols = idx - rows * (2 * n_nodes - 3 - rows) // 2 + rows + 1
        t1v = dm_t1[rows, cols]
        ok = (t1v >= 3) & (t1v < 9999)
        if ok.any():
            re, ce2 = rows[ok], cols[ok]
            dc = dm_t1[re, ce2].astype(np.float32) - dm_t2[re, ce2].astype(np.float32)
            el_pairs_list.append(np.column_stack([re, ce2]))
            el_dc_list.append(dc)

    all_pairs = np.concatenate(el_pairs_list)
    all_dc = np.concatenate(el_dc_list)
    n_eligible = len(all_pairs)
    print(f"  Eligible: {n_eligible:,} ({100*n_eligible/total_upper:.1f}%), {time.time()-t0:.0f}s", flush=True)

    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilTemporal(scibert).to(device)
    model_file = f'{TEMP_DIR}/model_fold{fold}/best.pt'
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()

    all_scores = np.zeros(n_eligible, dtype=np.float32)
    t1 = time.time()
    SCORE_BATCH = 100_000
    for bi in range(0, n_eligible, SCORE_BATCH):
        bj = min(bi + SCORE_BATCH, n_eligible)
        inp_gpu, nids_gpu = _build_subgraphs_batch(all_pairs[bi:bj], gpu_ctx)
        all_scores[bi:bj] = model.score_batch(inp_gpu.float(), nids_gpu, inner_bs=2048)
        if (bi // SCORE_BATCH + 1) % 10 == 0:
            print(f"    {bj:,}/{n_eligible:,} ({bj/(time.time()-t1):.0f}/s)", flush=True)

    from scipy.stats import spearmanr, pearsonr
    ranked = np.argsort(-all_scores)
    top_frac = max(1, n_eligible // 10)
    disc_ratio = float(all_dc[ranked[:top_frac]].mean() / max(abs(all_dc[ranked[top_frac:]].mean()), 1e-8))
    sp_corr, _ = spearmanr(all_scores, all_dc)
    pr_corr, _ = pearsonr(all_scores, all_dc)
    conv_mask = all_dc >= 2

    print(f"\n=== FULL EVAL fold {fold}: {n_eligible:,} pairs ===", flush=True)
    print(f"  Disc ratio: {disc_ratio:.3f}", flush=True)
    print(f"  Spearman: {sp_corr:.4f}, Pearson: {pr_corr:.4f}", flush=True)
    if conv_mask.sum() > 0:
        r_of = np.zeros(n_eligible); r_of[ranked] = np.arange(n_eligible)
        cr = r_of[conv_mask] + 1
        for k in [1, 3, 5, 10, 50, 100, 1000]:
            if k <= n_eligible:
                print(f"  R@{k}: {(cr <= k).mean():.4f}", flush=True)
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)
    np.savez(f'{TEMP_DIR}/eval_full_fold{fold}.npz', scores=all_scores, labels=all_dc,
             pairs=all_pairs, ranked=ranked[:10000])
    return disc_ratio


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train-all', action='store_true')
    parser.add_argument('--eval-all', action='store_true')
    parser.add_argument('--eval-full', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    if args.train_all:
        print("Training both folds (sequential)...", flush=True)
        for fold in [0, 1]:
            print(f"\n{'#'*60}\n  Starting fold {fold}\n{'#'*60}", flush=True)
            train_fold(fold)
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
        print(f"\nBest model: {best_path}", flush=True)


if __name__ == '__main__':
    main()
