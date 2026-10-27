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
from multiprocessing import Pool, Process
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
N_EVAL_WORKERS = 20
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
def _eval_worker_build(args):
    """Worker: build subgraphs for a batch of eval anchors."""
    anchor_batch, all_cands_batch, dist_t1_path, indptr, indices, batch_start, rng_seed = args

    dist_t1 = np.load(dist_t1_path, mmap_mode='r')
    rng = np.random.RandomState(rng_seed)
    MN = MAX_NODES; DB = 10.0

    out_list_inp = []
    out_list_nids = []
    tmp_f = np.zeros((MN, EMBED_DIM), dtype=np.float32)
    tmp_n = np.zeros(MN, dtype=np.int64)

    for local_ai in range(len(anchor_batch)):
        u = int(anchor_batch[local_ai, 0])
        cands = all_cands_batch[local_ai]
        du_row = dist_t1[u]
        u_nbs = indices[indptr[u]:indptr[u + 1]].tolist()

        for ci in range(len(cands)):
            v = int(cands[ci])
            dv_row = dist_t1[v]
            v_nbs = indices[indptr[v]:indptr[v + 1]].tolist()

            # Build subgraph inline
            sel = [u, v]; sel_set = {u, v}
            for nbs in [u_nbs, v_nbs]:
                for nb in nbs:
                    if len(sel) >= MN: break
                    if nb not in sel_set:
                        sel.append(int(nb)); sel_set.add(int(nb))

            ns = len(sel)
            tmp_f[:MN, :] = 0; tmp_n[:MN] = 0
            for i in range(ns):
                tmp_f[i, i] = 1.0; tmp_f[i, 2*MN] = 1.0
                tmp_f[i, 2*MN+1] = min(float(du_row[sel[i]]), DB) / DB
                tmp_f[i, 2*MN+2] = min(float(dv_row[sel[i]]), DB) / DB
            for i in range(ns):
                a = sel[i]
                for j in range(indptr[a], indptr[a+1]):
                    b = int(indices[j])
                    if b in sel_set:
                        for k in range(ns):
                            if sel[k] == b: tmp_f[i, MN+k] = 1.0; break
            for i in range(min(ns, MN)): tmp_n[i] = sel[i]

            out_list_inp.append(tmp_f.copy())
            out_list_nids.append(tmp_n.copy())

    return np.array(out_list_inp, dtype=np.float16), np.array(out_list_nids, dtype=np.int64)


def _eval_build_single(args):
    """Worker: build subgraphs for individual pairs (no anchor grouping)."""
    pairs_batch, _, dist_t1_path, indptr, indices, batch_start, rng_seed = args

    dist_t1 = np.load(dist_t1_path, mmap_mode='r')
    MN = MAX_NODES; DB = 10.0

    out_inp = np.zeros((len(pairs_batch), MN, EMBED_DIM), dtype=np.float16)
    out_nids = np.zeros((len(pairs_batch), MN), dtype=np.int64)
    tmp_f = np.zeros((MN, EMBED_DIM), dtype=np.float32)
    tmp_n = np.zeros(MN, dtype=np.int64)

    for i in range(len(pairs_batch)):
        u = int(pairs_batch[i, 0])
        v = int(pairs_batch[i, 1])
        du_row = dist_t1[u]
        dv_row = dist_t1[v]
        u_nbs = indices[indptr[u]:indptr[u + 1]].tolist()
        v_nbs = indices[indptr[v]:indptr[v + 1]].tolist()

        sel = [u, v]; sel_set = {u, v}
        for nbs in [u_nbs, v_nbs]:
            for nb in nbs:
                if len(sel) >= MN: break
                if nb not in sel_set:
                    sel.append(int(nb)); sel_set.add(int(nb))

        ns = len(sel)
        tmp_f[:MN, :] = 0; tmp_n[:MN] = 0
        for j in range(ns):
            tmp_f[j, j] = 1.0; tmp_f[j, 2*MN] = 1.0
            tmp_f[j, 2*MN+1] = min(float(du_row[sel[j]]), DB) / DB
            tmp_f[j, 2*MN+2] = min(float(dv_row[sel[j]]), DB) / DB
        for j in range(ns):
            a = sel[j]
            for k2 in range(indptr[a], indptr[a+1]):
                b = int(indices[k2])
                if b in sel_set:
                    for k3 in range(ns):
                        if sel[k3] == b: tmp_f[j, MN+k3] = 1.0; break
        for j in range(min(ns, MN)): tmp_n[j] = sel[j]

        out_inp[i] = tmp_f.astype(np.float16)
        out_nids[i] = tmp_n

    return out_inp, out_nids


def eval_ranking(fold):
    """Evaluate temporal LBD: score all val pairs, rank, compute disc ratio + R@K.
    Same approach as quant model: full ranking, no anchor-based sampling.
    """
    N_EVAL_WORKERS = 20

    print(f"\n{'='*60}")
    print(f"  Evaluating fold {fold}")
    print(f"{'='*60}", flush=True)

    device = torch.device('cuda')
    torch.cuda.empty_cache()

    # Load fold data
    subgraph_file = f'{TEMP_DIR}/fold{fold}_subgraphs.npz'
    data = np.load(subgraph_file, allow_pickle=True)
    val_pairs = data['val_pairs']
    val_sample_labels = data['val_sample_labels'].astype(np.float32)
    n_val = len(val_pairs)
    print(f"  Val pairs: {n_val}, dist_change: min={val_sample_labels.min():.0f}, "
          f"max={val_sample_labels.max():.0f}, mean={val_sample_labels.mean():.2f}", flush=True)

    if fold == 0:
        t1_year = 2017
    else:
        t1_year = 2021

    print(f"  Loading graph_{t1_year} and dist_{t1_year}...", flush=True)
    adj = sp.load_npz(f'{TEMP_DIR}/graph_{t1_year}.npz').tocsr()
    dist_t1_path = f'{TEMP_DIR}/dist_{t1_year}.npy'

    indptr = adj.indptr.astype(np.int64)
    indices = adj.indices.astype(np.int64)

    # Load model
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilTemporal(scibert).to(device)
    model_file = f'{TEMP_DIR}/model_fold{fold}/best.pt'
    if not os.path.exists(model_file):
        print(f"  WARNING: {model_file} not found, using random weights", flush=True)
    else:
        model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()

    _ = model.score_batch(torch.zeros(2, MAX_NODES, EMBED_DIM),
                          torch.zeros(2, MAX_NODES, dtype=torch.long))
    torch.cuda.synchronize()

    print(f"  Building subgraphs for all {n_val} val pairs ({N_EVAL_WORKERS} workers)...", flush=True)
    t0 = time.time()

    batch_size = max(1, (n_val + N_EVAL_WORKERS - 1) // N_EVAL_WORKERS)
    batches = []
    for i in range(0, n_val, batch_size):
        end = min(i + batch_size, n_val)
        batches.append((
            val_pairs[i:end],
            np.zeros((0, 2), dtype=np.int64),  # dummy, not used
            dist_t1_path,
            indptr, indices,
            i,
            42 + i,
        ))

    # Reuse worker: each val pair is (u, v), build subgraph for just that pair
    with Pool(N_EVAL_WORKERS) as pool:
        results = pool.map(_eval_build_single, batches)

    out_inp = np.concatenate([r[0] for r in results])
    out_nids = np.concatenate([r[1] for r in results])
    print(f"  Subgraphs built: {len(out_inp):,} in {time.time()-t0:.1f}s", flush=True)

    # GPU scoring
    print("  GPU scoring...", flush=True)
    t1 = time.time()
    all_scores = model.score_batch(
        torch.from_numpy(out_inp), torch.from_numpy(out_nids), inner_bs=2048)
    torch.cuda.synchronize()
    print(f"  GPU scoring: {time.time()-t1:.1f}s", flush=True)

    # Compute metrics: disc ratio on continuous distance change values
    scores = all_scores
    labels = val_sample_labels
    ranked = np.argsort(-scores)  # highest score first
    top_frac = max(1, n_val // 10)

    # Disc ratio: mean distance_change of top 10% / mean distance_change of bottom 90%
    top_mean_change = labels[ranked[:top_frac]].mean()
    bot_mean_change = labels[ranked[top_frac:]].mean()
    disc_ratio = float(top_mean_change / max(abs(bot_mean_change), 1e-8))

    # Rank correlation: does model score order match distance change order?
    from scipy.stats import spearmanr, pearsonr
    spearman_corr, _ = spearmanr(scores, labels)
    pearson_corr, _ = pearsonr(scores, labels)

    # Convergence detection: among pairs that actually converged (dist_change >= 2),
    # what fraction are in model's top-K?
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



# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1, help='Fold to train (0 or 1)')
    parser.add_argument('--eval', action='store_true', help='Run eval only')
    parser.add_argument('--train-all', action='store_true', help='Train all folds in parallel')
    parser.add_argument('--eval-all', action='store_true', help='Eval all folds')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    if args.train_all:
        # Train both folds in parallel (GPU has plenty of capacity)
        print("Training both folds in parallel...", flush=True)
        procs = []
        for fold in [0, 1]:
            p = Process(target=train_fold, args=(fold,))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        print("Both folds trained.", flush=True)
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
