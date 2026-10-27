"""
PENCIL v5 with mixed negative sampling:
  - 50% hard negatives (from precomputed A² top-20)
  - 30% in-batch negatives (other anchors' positives)
  - 20% random from hard neg pool

Architecture: same as v5 iso (isolation mask + position randomization)
"""
import os, time
import numpy as np
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
N_CANDIDATES = 5       # negatives per anchor per step
N_HARD_NEG_POOL = 20   # precomputed hard negatives
DIST_BINS = 10
DIST_EMBED_DIM = 32

# Sampling ratios
HARD_RATIO = 0.5
INBATCH_RATIO = 0.30
RANDOM_RATIO = 0.20


class MixedNegDataset(Dataset):
    """Each item: positive + 20 hard negatives. Sampling done at collate time."""
    def __init__(self, inp, node_ids):
        """
        inp: (N, 1+20, 32, 68) float16
        node_ids: (N, 1+20, 32) int64
        """
        self.inp = inp
        self.node_ids = node_ids
        self.N = len(inp)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.inp[idx], self.node_ids[idx], idx


def mixed_neg_collate(batch, n_candidates=N_CANDIDATES):
    """Build training batch with mixed negative sampling.

    For each anchor: 1 positive + n_candidates negatives, where:
      - ~50% from hard negatives (precomputed A² top-20)
      - ~30% from other anchors' positives (in-batch)
      - ~20% random from hard neg pool
    Positive position is randomized.
    """
    B = len(batch)
    all_inp = torch.zeros(B, 1 + n_candidates, 32, 68)
    all_nids = torch.zeros(B, 1 + n_candidates, 32, dtype=torch.long)
    all_pos_idx = torch.zeros(B, dtype=torch.long)

    all_items = [(b[0], b[1], b[2]) for b in batch]

    n_hard = max(1, int(n_candidates * HARD_RATIO))
    n_inbatch = max(0, int(n_candidates * INBATCH_RATIO))
    n_random = n_candidates - n_hard - n_inbatch

    for i in range(B):
        inp_i, nids_i, anchor_idx = all_items[i]

        # Positive at position 0
        candidates_inp = [inp_i[0]]     # pos
        candidates_nids = [nids_i[0]]

        # 50% hard negatives from pool (indices 1..20)
        hard_pool = list(range(1, 1 + N_HARD_NEG_POOL))
        hard_chosen = np.random.choice(hard_pool, size=n_hard, replace=False)
        for h in hard_chosen:
            candidates_inp.append(inp_i[h])
            candidates_nids.append(nids_i[h])

        # 30% in-batch negatives (other anchors' positives)
        other_indices = [j for j in range(B) if j != i]
        if len(other_indices) >= n_inbatch:
            inbatch_chosen = np.random.choice(other_indices, size=n_inbatch, replace=False)
        else:
            inbatch_chosen = other_indices + list(np.random.choice(other_indices, size=n_inbatch - len(other_indices), replace=True))
        for j in inbatch_chosen:
            other_inp, other_nids, _ = all_items[j]
            candidates_inp.append(other_inp[0])  # other anchor's positive
            candidates_nids.append(other_nids[0])

        # 20% random from hard neg pool
        for _ in range(n_random):
            r = np.random.randint(1, 1 + N_HARD_NEG_POOL)
            candidates_inp.append(inp_i[r])
            candidates_nids.append(nids_i[r])

        # Stack and randomize position
        cand_inp = torch.stack(candidates_inp)   # (1+K, 32, 68)
        cand_nids = torch.stack(candidates_nids)  # (1+K, 32)
        G = 1 + n_candidates
        perm = torch.randperm(G)
        pos_idx = (perm == 0).nonzero(as_tuple=False).item()

        all_inp[i] = cand_inp[perm]
        all_nids[i] = cand_nids[perm]
        all_pos_idx[i] = pos_idx

    return all_inp, all_nids, all_pos_idx


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
                'ffn': nn.Sequential(nn.Linear(HIDDEN, HIDDEN * 4), nn.GELU(),
                    nn.Linear(HIDDEN * 4, HIDDEN), nn.Dropout(0.1)),
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
            start = k * n_nodes; end = (k + 1) * n_nodes
            mask[start:end, :src_end] = 0.0
            mask[start:end, start:end] = 0.0
        return mask

    def forward(self, inp, node_ids):
        B, G, N, _ = inp.shape; device = inp.device
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
        return self.head(pooled).squeeze(-1)


def main():
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(42); np.random.seed(42)

    print("Loading SciBERT embeddings...", flush=True)
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)

    print("Loading precomputed data (20 hard negatives)...", flush=True)
    t0 = time.time()
    train_raw = np.load(f'{DATA_DIR}/train_precomp_v5_hardneg.npz')
    val_raw = np.load(f'{DATA_DIR}/val_precomp_v5_hardneg.npz')

    train_inp = torch.from_numpy(train_raw['inp'].astype(np.float32))
    train_nids = torch.from_numpy(train_raw['node_ids'])
    val_inp = torch.from_numpy(val_raw['inp'].astype(np.float32))
    val_nids = torch.from_numpy(val_raw['node_ids'])
    print(f"  train: {train_inp.shape}, val: {val_inp.shape}, {time.time()-t0:.1f}s", flush=True)

    train_ds = MixedNegDataset(train_inp, train_nids)
    val_ds = MixedNegDataset(val_inp, val_nids)

    model = PencilV5(scibert).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)

    # Warmup
    dummy = torch.zeros(2, 1+N_CANDIDATES, MAX_NODES, EMBED_DIM, device=device)
    dummy_n = torch.zeros(2, 1+N_CANDIDATES, MAX_NODES, dtype=torch.long, device=device)
    _ = model(dummy, dummy_n); torch.cuda.synchronize()

    def val_collate_fn(batch):
        return mixed_neg_collate(batch, n_candidates=N_CANDIDATES)

    print(f"\n=== Training PENCIL v5 mixed-neg (50% hard / 30% in-batch / 20% random) ===", flush=True)
    best_mrr = 0; t_total = time.time()

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0,
                              collate_fn=val_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0,
                            collate_fn=val_collate_fn, pin_memory=True)

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, nb = 0, 0

        for inp, node_ids, pos_idx in train_loader:
            inp = inp.to(device); node_ids = node_ids.to(device); pos_idx = pos_idx.to(device)
            scores = model(inp, node_ids)
            B = scores.shape[0]
            pos_scores = scores.gather(1, pos_idx.unsqueeze(1))
            neg_mask = torch.ones_like(scores, dtype=torch.bool)
            neg_mask.scatter_(1, pos_idx.unsqueeze(1), False)
            neg_scores = scores[neg_mask].reshape(B, -1)
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
            opt.zero_grad(set_to_none=True)
            bpr_loss.backward(); opt.step()
            total_loss += bpr_loss.item(); nb += 1
        sched.step()
        dt_train = time.time() - t0

        model.eval()
        val_ranks = []
        with torch.no_grad():
            for inp, node_ids, pos_idx in val_loader:
                inp = inp.to(device); node_ids = node_ids.to(device); pos_idx = pos_idx.to(device)
                scores = model(inp, node_ids)
                B = scores.shape[0]
                pos_scores = scores.gather(1, pos_idx.unsqueeze(1))
                neg_mask = torch.ones_like(scores, dtype=torch.bool)
                neg_mask.scatter_(1, pos_idx.unsqueeze(1), False)
                neg_scores = scores[neg_mask].reshape(B, -1)
                higher = (neg_scores > pos_scores).sum(dim=1)
                val_ranks.append((higher + 1).cpu())

        ranks = torch.cat(val_ranks).float()
        mrr = (1.0 / ranks).mean().item()
        hit1 = (ranks <= 1).float().mean().item()
        hit5 = (ranks <= 5).float().mean().item()
        dt_eval = time.time() - t0

        marker = ""
        if mrr > best_mrr:
            best_mrr = mrr
            torch.save(model.state_dict(), f'{DATA_DIR}/pencil_v5_mixedneg_best.pt')
            marker = " *"

        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}  loss={total_loss/nb:.4f}  "
              f"MRR={mrr:.4f}  Hit@1={hit1:.4f}  Hit@5={hit5:.4f}  "
              f"train={dt_train:.1f}s  eval={dt_eval:.1f}s{marker}", flush=True)

    print(f"\nBest MRR: {best_mrr:.4f}", flush=True)
    print(f"Total: {time.time()-t_total:.1f}s", flush=True)


if __name__ == '__main__':
    main()
