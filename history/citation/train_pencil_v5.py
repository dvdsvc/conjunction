"""
PENCIL v5: Same graph-structure Transformer as v3, listwise softmax loss.

Architecture (UNCHANGED from v3):
  - 32-node subgraph per (u, v) pair
  - 4-layer Transformer with per-layer graph message passing
  - Message passing: adj_norm @ x → MLP → residual

Changes from v3:
  - Loss: same BPR (pairwise)
  - Training: 1 positive + 5 hard negatives per anchor (same as v3)
  - Data: precomputed node_ids (features reconstructed on GPU, faster than v3's CPU build)
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32
EMBED_DIM = 2 * MAX_NODES + 2 + 2  # 68
SCIBERT_DIM = 768
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
N_EPOCHS = 30
LR = 3e-4
BATCH_SIZE = 128
N_CANDIDATES = 50     # 1 positive + 50 negatives (BPR pairwise)
DIST_BINS = 10
DIST_EMBED_DIM = 32


class AnchorDataset(Dataset):
    """Dataset for listwise training: each item is (1+K) candidates for one anchor."""
    def __init__(self, node_ids):
        """
        Args:
            node_ids: (N, 1+K, 32) int64 — precomputed node IDs
        """
        self.node_ids = node_ids

    def __len__(self):
        return len(self.node_ids)

    def __getitem__(self, idx):
        return self.node_ids[idx]  # (1+K, 32)


def anchor_collate(batch):
    """Collate: stack node_ids, create label (positive always at index 0 in candidates)."""
    node_ids = torch.stack(batch)  # (B, 1+K, 32)
    B = node_ids.shape[0]
    # Labels: positive is always at position 0 in the K candidates (index 1 in node_ids)
    labels = torch.zeros(B, dtype=torch.long)
    return node_ids, labels


def reconstruct_features_gpu(nids, dist_matrix):
    """Reconstruct subgraph features from node_ids (fully vectorized, no Python loops).

    Args:
        nids: (B, 32) int64 — node IDs (on GPU)
        dist_matrix: (N, N) float32 — distance matrix (on GPU)
    Returns:
        (B, 32, 68) float32 — reconstructed features (on GPU)
    """
    B, N = nids.shape
    device = nids.device

    feat = torch.zeros(B, N, EMBED_DIM, device=device)

    # Columns 0..31: one-hot position encoding
    pos_idx = torch.arange(N, device=device)
    feat[:, pos_idx, pos_idx] = 1.0

    # Columns 32..63: adjacency (dist == 1 means direct edge)
    # dist_matrix[nids[:, :, None], nids[:, None, :]] → (B, N, N)
    pair_dist = dist_matrix[nids[:, :, None], nids[:, None, :]]
    feat[:, :, MAX_NODES:MAX_NODES + N] = (pair_dist == 1.0).float()

    # Column 64: mask
    feat[:, :, 2 * MAX_NODES] = (feat[:, :, :MAX_NODES].abs().sum(dim=2) > 0).float()

    # Columns 65, 66: distances from u and v
    u_ids = nids[:, 0]  # (B,)
    v_ids = nids[:, 1]  # (B,)
    feat[:, :, 2 * MAX_NODES + 1] = dist_matrix[u_ids[:, None], nids].clamp(0, DIST_BINS) / DIST_BINS
    feat[:, :, 2 * MAX_NODES + 2] = dist_matrix[v_ids[:, None], nids].clamp(0, DIST_BINS) / DIST_BINS

    return feat


class PencilModel(nn.Module):
    """PENCIL v3/v5 architecture: Transformer with per-layer graph message passing."""
    def __init__(self, scibert_embeddings):
        super().__init__()
        self.register_buffer('scibert_table', scibert_embeddings)
        self.scibert_proj = nn.Linear(SCIBERT_DIM, HIDDEN // 2)
        self.struct_proj = nn.Linear(EMBED_DIM, HIDDEN // 2)
        self.dist_embed = nn.Embedding((DIST_BINS + 2) * (DIST_BINS + 2), DIST_EMBED_DIM)
        self.combine = nn.Linear(HIDDEN // 2 + HIDDEN // 2 + DIST_EMBED_DIM, HIDDEN)
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
            nn.Linear(HIDDEN * 2, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

    def forward(self, inp, node_ids):
        """
        Args:
            inp: (B, 32, 68) — subgraph features
            node_ids: (B, 32) — node IDs for SciBERT lookup
        Returns:
            (B,) — scalar scores
        """
        B, N, _ = inp.shape
        adj = inp[:, :N, MAX_NODES:MAX_NODES + N]
        deg = adj.sum(dim=2, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        dist_u_bin = (inp[:, :, 2 * MAX_NODES + 1] * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        dist_v_bin = (inp[:, :, 2 * MAX_NODES + 2] * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        dist_idx = dist_u_bin * (DIST_BINS + 2) + dist_v_bin
        dist_emb = self.dist_embed(dist_idx)
        x_raw = inp.clone()
        x_raw[:, :, 2 * MAX_NODES + 1] = 0
        x_raw[:, :, 2 * MAX_NODES + 2] = 0
        struct_emb = self.struct_proj(x_raw)
        node_ids_flat = node_ids.reshape(-1).clamp(0, self.scibert_table.shape[0] - 1)
        scibert_emb = self.scibert_proj(
            self.scibert_table[node_ids_flat].reshape(B, N, SCIBERT_DIM))
        x = torch.cat([struct_emb, scibert_emb, dist_emb], dim=-1)
        x = self.combine(x)
        pad_mask = (inp[:, :, :2 * MAX_NODES + 1].abs().sum(dim=2) == 0)
        for layer in self.layers:
            h = layer['ln1'](x)
            h, _ = layer['attn'](h, h, h, key_padding_mask=pad_mask)
            x = x + h
            h_mp = layer['mp'](torch.bmm(adj_norm, x[:, :N]))
            x[:, :N] = x[:, :N] + h_mp
            h = layer['ln2'](x)
            x = x + layer['ffn'](h)
        x = self.ln_f(x)
        return self.head(torch.cat([x[:, 0], x[:, 1]], -1)).squeeze(-1)


def main():
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading SciBERT embeddings...", flush=True)
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    print(f"  shape: {scibert.shape}", flush=True)

    print("Loading precomputed v5 data (node_ids)...", flush=True)
    t0 = time.time()
    train_nids = np.load(f'{DATA_DIR}/train_nids_v5.npy')  # (N, 1+200, 32)
    val_nids = np.load(f'{DATA_DIR}/val_nids_v5.npy')       # (N, 1+100, 32)
    print(f"  train: {train_nids.shape}, val: {val_nids.shape}, loaded in {time.time()-t0:.1f}s", flush=True)

    print("Loading distance matrix...", flush=True)
    dist_matrix = torch.from_numpy(np.load(f'{DATA_DIR}/dist_matrix.npy')).to(device)

    train_ds = AnchorDataset(torch.from_numpy(train_nids))
    val_ds = AnchorDataset(torch.from_numpy(val_nids))

    model = PencilModel(scibert).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)

    # Warmup
    dummy = torch.zeros(4, MAX_NODES, EMBED_DIM, device=device)
    dummy_nid = torch.zeros(4, MAX_NODES, dtype=torch.long, device=device)
    _ = model(dummy, dummy_nid)
    torch.cuda.synchronize()
    print(f"  GPU warmup done", flush=True)

    # JIT warmup for feature reconstruction
    _ = reconstruct_features_gpu(torch.zeros(2, MAX_NODES, dtype=torch.long, device=device), dist_matrix)
    torch.cuda.synchronize()

    print(f"\n=== Training v5 (BPR pairwise, {N_CANDIDATES} negs, {BATCH_SIZE} anchors/batch) ===", flush=True)
    best_mrr = 0
    t_total = time.time()

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=anchor_collate, pin_memory=True)

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, nb = 0, 0

        for node_ids, labels in train_loader:
            B = node_ids.shape[0]
            K = node_ids.shape[1] - 1  # available negatives
            node_ids = node_ids.to(device)

            # Randomly sample N_CANDIDATES negatives
            if K > N_CANDIDATES:
                perm = torch.randperm(K - 1, device=device)[:N_CANDIDATES - 1] + 2
                keep = torch.cat([torch.tensor([0, 1], device=device), perm])
                node_ids = node_ids[:, keep]
                K = N_CANDIDATES

            # Flatten all (1 + K) candidates per anchor
            flat_nids = node_ids.reshape(B * (1 + K), 32)
            inp = reconstruct_features_gpu(flat_nids, dist_matrix)

            # Forward pass
            scores = model(inp, flat_nids).view(B, 1 + K)  # (B, 1+K)

            # BPR loss: pos at index 0, negs at 1..K
            pos_scores = scores[:, 0]           # (B,)
            neg_scores = scores[:, 1:]          # (B, K)
            bpr_loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-8).mean()

            opt.zero_grad(set_to_none=True)
            bpr_loss.backward()
            opt.step()

            total_loss += bpr_loss.item()
            nb += 1

        sched.step()
        dt_train = time.time() - t0

        # Validation: compute MRR on N_CANDIDATES candidates
        model.eval()
        val_ranks = []
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                                num_workers=0, collate_fn=anchor_collate, pin_memory=True)
        with torch.no_grad():
            for node_ids, labels in val_loader:
                B = node_ids.shape[0]
                # Use first 1 + N_CANDIDATES candidates for val
                node_ids = node_ids[:, :1 + N_CANDIDATES].to(device)
                K = N_CANDIDATES

                flat_nids = node_ids.reshape(B * (1 + K), 32)
                inp = reconstruct_features_gpu(flat_nids, dist_matrix)
                scores = model(inp, flat_nids).view(B, 1 + K)

                pos_scores = scores[:, 0]
                neg_scores = scores[:, 1:]
                higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
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
            torch.save(model.state_dict(), f'{DATA_DIR}/pencil_v5_best.pt')
            marker = " *"

        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}  loss={total_loss/nb:.4f}  "
              f"MRR={mrr:.4f}  Hit@1={hit1:.4f}  Hit@5={hit5:.4f}  "
              f"train={dt_train:.1f}s  eval={dt_eval:.1f}s{marker}", flush=True)

    print(f"\nBest MRR: {best_mrr:.4f}", flush=True)
    print(f"Total: {time.time()-t_total:.1f}s", flush=True)


if __name__ == '__main__':
    main()
