"""
PENCIL v5: v3's per-node graph Transformer + Phoenix candidate isolation mask.

Architecture:
  - 32-node subgraph per (u, v) candidate (same as v3)
  - 4-layer Transformer with per-layer graph message passing (same as v3)
  - NEW: All K candidates for same anchor u processed TOGETHER
  - NEW: Candidate isolation mask:
      source nodes (u, pos 0-31): attend to ALL nodes (all candidates)
      candidate k nodes (pos 32k..32k+31): attend to source nodes + own nodes ONLY
  - Each candidate gets pooled embedding → scored by head
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
BATCH_SIZE = 16       # anchors per batch (each with 1+K candidates)
N_CANDIDATES = 5      # 1 positive + 5 negatives
DIST_BINS = 10
DIST_EMBED_DIM = 32


class AnchorDataset(Dataset):
    """Each item is one anchor with (1+K) candidates, precomputed features.
    Positive position is randomized each time to prevent position bias."""
    def __init__(self, inp, node_ids, n_candidates):
        """
        Args:
            inp: (N, 1+K, 32, 68) float32
            node_ids: (N, 1+K, 32) int64
            n_candidates: K (number of negatives, total = 1+K)
        """
        self.inp = inp
        self.node_ids = node_ids
        self.K = n_candidates

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inp = self.inp[idx]    # (1+K, 32, 68)
        nids = self.node_ids[idx]  # (1+K, 32)
        # Randomize positive position: positive is at index 0, move to random pos
        G = 1 + self.K
        perm = torch.randperm(G)
        # Find where original index 0 (positive) ended up
        pos_idx = (perm == 0).nonzero(as_tuple=False).item()
        return inp[perm], nids[perm], pos_idx


def anchor_collate(batch):
    inp = torch.stack([b[0] for b in batch])
    nids = torch.stack([b[1] for b in batch])
    pos_idx = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return inp, nids, pos_idx


class PencilV5(nn.Module):
    """v3 graph Transformer + Phoenix candidate isolation mask.

    Forward:
      inp: (B, 1+K, 32, 68) — B anchors, each with 1+K candidates of 32 nodes
      node_ids: (B, 1+K, 32)

    Process:
      1. Flatten to (B*(1+K)*32, 68), project to hidden
      2. Reshape to (B, (1+K)*32, H) — all candidates' nodes in one sequence
      3. 4 Transformer layers with:
         - Candidate isolation attention mask
         - Per-candidate graph message passing
      4. Pool each candidate's 32 nodes → (B, 1+K, H)
      5. Score head → (B, 1+K)
    """
    def __init__(self, scibert_embeddings):
        super().__init__()
        self.register_buffer('scibert_table', scibert_embeddings)
        self.scibert_proj = nn.Linear(SCIBERT_DIM, HIDDEN // 2)
        self.struct_proj = nn.Linear(EMBED_DIM, HIDDEN // 2)
        self.dist_embed = nn.Embedding((DIST_BINS + 2) * (DIST_BINS + 2), DIST_EMBED_DIM)
        self.combine = nn.Linear(HIDDEN // 2 + HIDDEN // 2 + DIST_EMBED_DIM, HIDDEN)

        # Candidate position embedding (distinguishes source vs neg1 vs neg2 ...)
        self.cand_pos_embed = nn.Embedding(64, HIDDEN)  # up to 64 candidates

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

        # Score head: takes pooled candidate embedding → scalar
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

    def _build_iso_mask(self, n_cands, n_nodes, device):
        """Build candidate isolation mask.

        Returns (S, S) float mask where S = (1+n_cands) * n_nodes:
          -10000 = blocked, 0 = allowed

        Source (candidate 0): attends to ALL nodes
        Candidate k (k>=1): attends to source nodes + own nodes only
        """
        S = (1 + n_cands) * n_nodes
        # Start with all blocked
        mask = torch.full((S, S), -10000.0, device=device)

        src_end = n_nodes  # source occupies [0, src_end)

        # Source attends to everything
        mask[:src_end, :] = 0.0

        # Each candidate attends to source + self
        for k in range(1 + n_cands):
            start = k * n_nodes
            end = (k + 1) * n_nodes
            mask[start:end, :src_end] = 0.0       # attend to source
            mask[start:end, start:end] = 0.0       # attend to self

        return mask  # (S, S)

    def forward(self, inp, node_ids):
        """
        Args:
            inp: (B, 1+K, 32, 68)
            node_ids: (B, 1+K, 32)
        Returns:
            (B, 1+K) — scores per candidate
        """
        B, G, N, _ = inp.shape  # G = 1+K
        device = inp.device

        # --- Build node features (same as v3) ---
        flat_inp = inp.reshape(B * G * N, -1)
        flat_nids = node_ids.reshape(B * G * N)

        # Struct projection (zero out distance cols)
        x_raw = flat_inp.clone()
        x_raw[:, 2 * MAX_NODES + 1] = 0
        x_raw[:, 2 * MAX_NODES + 2] = 0
        struct_emb = self.struct_proj(x_raw)

        # SciBERT
        nids_flat = flat_nids.clamp(0, self.scibert_table.shape[0] - 1)
        scibert_emb = self.scibert_proj(self.scibert_table[nids_flat])

        # Distance embedding
        dist_u_bin = (flat_inp[:, 2 * MAX_NODES + 1] * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        dist_v_bin = (flat_inp[:, 2 * MAX_NODES + 2] * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        dist_idx = dist_u_bin * (DIST_BINS + 2) + dist_v_bin
        dist_emb = self.dist_embed(dist_idx)

        x = self.combine(torch.cat([struct_emb, scibert_emb, dist_emb], dim=-1))

        # Add candidate position embedding
        # cand_idx: for each node, which candidate it belongs to (0=source, 1=neg1, ...)
        cand_idx = torch.arange(G, device=device).unsqueeze(1).expand(-1, N).reshape(-1)
        cand_idx = cand_idx.unsqueeze(0).expand(B, -1).reshape(-1)  # (B*G*N,)
        x = x + self.cand_pos_embed(cand_idx)

        # Reshape to (B, G*N, H) — all candidates' nodes in one sequence
        x = x.reshape(B, G * N, -1)

        # --- Adjacency for graph message passing ---
        adj = inp[:, :, :, MAX_NODES:MAX_NODES + N]  # (B, G, N, N)
        deg = adj.sum(dim=3, keepdim=True).clamp(min=1)
        adj_norm = adj / deg  # (B, G, N, N)

        # --- Padding mask ---
        pad_mask = (inp[:, :, :, :2 * MAX_NODES + 1].abs().sum(dim=3) == 0)  # (B, G, N)
        pad_mask_flat = pad_mask.reshape(B, G * N)  # (B, G*N)

        # --- Candidate isolation mask ---
        iso_mask = self._build_iso_mask(G - 1, N, device)  # (G*N, G*N)

        # --- Transformer layers ---
        for layer in self.layers:
            h = layer['ln1'](x)
            h, _ = layer['attn'](h, h, h,
                                  attn_mask=iso_mask,
                                  key_padding_mask=pad_mask_flat)
            x = x + h

            # Graph message passing: per-candidate adj_norm @ nodes
            # x: (B, G*N, H) → (B, G, N, H)
            x_cand = x.reshape(B, G, N, -1)
            h_mp = layer['mp'](torch.einsum('bgnh,bgnm->bgmh', x_cand, adj_norm))
            x_cand = x_cand + h_mp
            x = x.reshape(B, G * N, -1)
            x[:, :] = x_cand.reshape(B, G * N, -1)

            h = layer['ln2'](x)
            x = x + layer['ffn'](h)

        x = self.ln_f(x)

        # --- Pool each candidate's 32 nodes → score ---
        # Masked mean-pool
        x_cand = x.reshape(B, G, N, -1)  # (B, G, N, H)
        valid = (~pad_mask).unsqueeze(-1).float()  # (B, G, N, 1)
        pooled = (x_cand * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)  # (B, G, H)

        scores = self.head(pooled).squeeze(-1)  # (B, G)
        return scores


def main():
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading SciBERT embeddings...", flush=True)
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    print(f"  shape: {scibert.shape}", flush=True)

    # Load v3 precomputed data (features + node_ids)
    print("Loading v3 precomputed data...", flush=True)
    t0 = time.time()
    train_raw = np.load(f'{DATA_DIR}/train_precomp_v3.npz')
    val_raw = np.load(f'{DATA_DIR}/val_precomp_v3.npz')

    # Reshape: (N*G, 32, 68) → (N, G, 32, 68)
    G = 1 + N_CANDIDATES  # 6
    train_inp = torch.from_numpy(train_raw['inp'].astype(np.float32)).reshape(-1, G, MAX_NODES, EMBED_DIM)
    train_nids = torch.from_numpy(train_raw['node_ids']).reshape(-1, G, MAX_NODES)
    val_inp = torch.from_numpy(val_raw['inp'].astype(np.float32)).reshape(-1, G, MAX_NODES, EMBED_DIM)
    val_nids = torch.from_numpy(val_raw['node_ids']).reshape(-1, G, MAX_NODES)
    print(f"  train: {train_inp.shape}, val: {val_inp.shape}, loaded in {time.time()-t0:.1f}s", flush=True)

    train_ds = AnchorDataset(train_inp, train_nids, N_CANDIDATES)
    val_ds = AnchorDataset(val_inp, val_nids, N_CANDIDATES)

    model = PencilV5(scibert).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)

    # Warmup
    dummy_inp = torch.zeros(2, G, MAX_NODES, EMBED_DIM, device=device)
    dummy_nid = torch.zeros(2, G, MAX_NODES, dtype=torch.long, device=device)
    _ = model(dummy_inp, dummy_nid)
    torch.cuda.synchronize()
    print(f"  GPU warmup done", flush=True)

    print(f"\n=== Training PENCIL v5 (v3 data + isolation mask, {N_CANDIDATES} negs) ===", flush=True)
    best_mrr = 0
    t_total = time.time()

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=anchor_collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            num_workers=0, collate_fn=anchor_collate, pin_memory=True)

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, nb = 0, 0

        for inp, node_ids, pos_idx in train_loader:
            inp = inp.to(device)
            node_ids = node_ids.to(device)
            pos_idx = pos_idx.to(device)

            scores = model(inp, node_ids)  # (B, G)

            # BPR loss: pos at pos_idx, negs at all other positions
            B = scores.shape[0]
            pos_scores = scores.gather(1, pos_idx.unsqueeze(1))  # (B, 1)
            # Build mask for negatives (all positions except pos_idx)
            neg_mask = torch.ones_like(scores, dtype=torch.bool)
            neg_mask.scatter_(1, pos_idx.unsqueeze(1), False)
            neg_scores = scores[neg_mask].reshape(B, -1)  # (B, K)
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

            opt.zero_grad(set_to_none=True)
            bpr_loss.backward()
            opt.step()

            total_loss += bpr_loss.item()
            nb += 1

        sched.step()
        dt_train = time.time() - t0

        # Validation
        model.eval()
        val_ranks = []
        with torch.no_grad():
            for inp, node_ids, pos_idx in val_loader:
                inp = inp.to(device)
                node_ids = node_ids.to(device)
                pos_idx = pos_idx.to(device)
                scores = model(inp, node_ids)
                B = scores.shape[0]
                pos_scores = scores.gather(1, pos_idx.unsqueeze(1))  # (B, 1)
                neg_mask = torch.ones_like(scores, dtype=torch.bool)
                neg_mask.scatter_(1, pos_idx.unsqueeze(1), False)
                neg_scores = scores[neg_mask].reshape(B, -1)  # (B, K)
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
            torch.save(model.state_dict(), f'{DATA_DIR}/pencil_v5_iso_best.pt')
            marker = " *"

        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}  loss={total_loss/nb:.4f}  "
              f"MRR={mrr:.4f}  Hit@1={hit1:.4f}  Hit@5={hit5:.4f}  "
              f"train={dt_train:.1f}s  eval={dt_eval:.1f}s{marker}", flush=True)

    print(f"\nBest MRR: {best_mrr:.4f}", flush=True)
    print(f"Total: {time.time()-t_total:.1f}s", flush=True)


if __name__ == '__main__':
    main()
