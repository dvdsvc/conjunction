"""
Train PENCIL v5 model using v3's precomputed features (direct features, not node_ids).
Uses BPR loss with 5 hard negatives (same as v3) but v5's training loop.
Goal: verify that v5's training code produces comparable results to v3.
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
BATCH_SIZE = 128
N_CANDIDATES = 5     # Same as v3: 1 positive + 5 negatives
DIST_BINS = 10
DIST_EMBED_DIM = 32


class V3Dataset(Dataset):
    """Dataset from v3 precomputed data: (inp, node_ids, label)."""
    def __init__(self, inp, node_ids, labels):
        """
        Args:
            inp: (N*G, 32, 68) float16 — precomputed features
            node_ids: (N*G, 32) int64 — node IDs
            labels: (N*G,) float32 — 1.0 for positive, 0.0 for negative
        """
        self.inp = inp
        self.node_ids = node_ids
        self.labels = labels
        self.G = 6  # 1 positive + 5 negatives per anchor

    def __len__(self):
        return len(self.labels) // self.G

    def __getitem__(self, idx):
        start = idx * self.G
        return (self.inp[start:start + self.G],
                self.node_ids[start:start + self.G])


def anchor_collate(batch):
    inp = torch.stack([torch.from_numpy(b[0].astype(np.float32)) for b in batch])
    nids = torch.stack([torch.from_numpy(b[1]) for b in batch])
    return inp, nids


class PencilModel(nn.Module):
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

    print("Loading v3 precomputed data...", flush=True)
    t0 = time.time()
    train_data = np.load(f'{DATA_DIR}/train_precomp_v3.npz')
    val_data = np.load(f'{DATA_DIR}/val_precomp_v3.npz')
    print(f"  train: inp={train_data['inp'].shape}, nids={train_data['node_ids'].shape}", flush=True)
    print(f"  val: inp={val_data['inp'].shape}, nids={val_data['node_ids'].shape}", flush=True)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    train_ds = V3Dataset(train_data['inp'], train_data['node_ids'], train_data['label'])
    val_ds = V3Dataset(val_data['inp'], val_data['node_ids'], val_data['label'])

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

    print(f"\n=== Training (v3 data, BPR, {N_CANDIDATES} negs) ===", flush=True)
    best_mrr = 0
    t_total = time.time()

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=anchor_collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=0, collate_fn=anchor_collate, pin_memory=True)

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, nb = 0, 0

        for inp, node_ids in train_loader:
            B = inp.shape[0]
            G = inp.shape[1]  # 1+K
            inp = inp.to(device)
            node_ids = node_ids.to(device)

            # Score all candidates
            scores = model(inp.view(B * G, MAX_NODES, EMBED_DIM),
                           node_ids.view(B * G, MAX_NODES)).view(B, G)

            # BPR loss: pos at index 0, negs at 1..K
            pos_scores = scores[:, 0]
            neg_scores = scores[:, 1:]
            bpr_loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-8).mean()

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
            for inp, node_ids in val_loader:
                B = inp.shape[0]
                G = inp.shape[1]
                inp = inp.to(device)
                node_ids = node_ids.to(device)
                scores = model(inp.view(B * G, MAX_NODES, EMBED_DIM),
                               node_ids.view(B * G, MAX_NODES)).view(B, G)
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
            torch.save(model.state_dict(), f'{DATA_DIR}/pencil_v5_v3data_best.pt')
            marker = " *"

        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}  loss={total_loss/nb:.4f}  "
              f"MRR={mrr:.4f}  Hit@1={hit1:.4f}  Hit@5={hit5:.4f}  "
              f"train={dt_train:.1f}s  eval={dt_eval:.1f}s{marker}", flush=True)

    print(f"\nBest MRR: {best_mrr:.4f}", flush=True)
    print(f"Total: {time.time()-t_total:.1f}s", flush=True)


if __name__ == '__main__':
    main()
