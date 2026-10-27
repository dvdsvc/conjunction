"""
Fast PENCIL training v2: MAX_NODES=32, DRNL + SciBERT embeddings.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32
EMBED_DIM = 2 * MAX_NODES + 2 + 2  # 70: structural features
SCIBERT_DIM = 768
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
N_EPOCHS = 30
LR = 3e-4
N_HARD_NEG = 5
BATCH_SIZE = 256
DIST_BINS = 10
DIST_EMBED_DIM = 32


class AnchorDataset(Dataset):
    def __init__(self, inp, task, label, node_ids):
        G = 1 + N_HARD_NEG
        n_anchors = len(label) // G
        self.inp = inp[:n_anchors * G].view(n_anchors, G, *inp.shape[1:])
        self.task = task[:n_anchors * G].view(n_anchors, G, *task.shape[1:])
        self.label = label[:n_anchors * G].view(n_anchors, G)
        self.node_ids = node_ids[:n_anchors * G].view(n_anchors, G, *node_ids.shape[1:])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.inp[idx], self.task[idx], self.label[idx], self.node_ids[idx]


def anchor_collate(batch):
    inp = torch.cat([b[0] for b in batch], dim=0)
    task = torch.cat([b[1] for b in batch], dim=0)
    label = torch.cat([b[2] for b in batch], dim=0)
    node_ids = torch.cat([b[3] for b in batch], dim=0)
    return inp, task, label, node_ids


class PencilModel(nn.Module):
    def __init__(self, scibert_embeddings):
        super().__init__()
        # Frozen SciBERT lookup
        self.register_buffer('scibert_table', scibert_embeddings)  # (N_concepts, 768)
        self.scibert_proj = nn.Linear(SCIBERT_DIM, HIDDEN // 2)  # project to half hidden

        # Structural projection
        self.struct_proj = nn.Linear(EMBED_DIM, HIDDEN // 2)

        # Distance embedding
        self.dist_embed = nn.Embedding((DIST_BINS + 2) * (DIST_BINS + 2), DIST_EMBED_DIM)

        # Combined projection: struct(128) + scibert(128) + dist(32) = 288 → HIDDEN
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

        # Adjacency
        adj = inp[:, :N, MAX_NODES:MAX_NODES + N]
        deg = adj.sum(dim=2, keepdim=True).clamp(min=1)
        adj_norm = adj / deg

        # DRNL distance embedding
        dist_u_raw = inp[:, :, 2 * MAX_NODES + 1]
        dist_v_raw = inp[:, :, 2 * MAX_NODES + 2]
        dist_u_bin = (dist_u_raw * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        dist_v_bin = (dist_v_raw * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        dist_idx = dist_u_bin * (DIST_BINS + 2) + dist_v_bin
        dist_emb = self.dist_embed(dist_idx)  # (B, N, 32)

        # Structural features (zero out dist columns, handled by dist_embed)
        x_raw = inp.clone()
        x_raw[:, :, 2 * MAX_NODES + 1] = 0
        x_raw[:, :, 2 * MAX_NODES + 2] = 0
        struct_emb = self.struct_proj(x_raw)  # (B, N, 128)

        # SciBERT semantic embeddings
        node_ids_flat = node_ids.reshape(-1)  # (B*N,)
        # Clamp to valid range
        node_ids_flat = node_ids_flat.clamp(0, self.scibert_table.shape[0] - 1)
        scibert_emb = self.scibert_table[node_ids_flat]  # (B*N, 768)
        scibert_emb = scibert_emb.reshape(B, N, SCIBERT_DIM)
        scibert_emb = self.scibert_proj(scibert_emb)  # (B, N, 128)

        # Combine all features
        x = torch.cat([struct_emb, scibert_emb, dist_emb], dim=-1)  # (B, N, 288)
        x = self.combine(x)  # (B, N, HIDDEN)

        # Padding mask
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


@torch.no_grad()
def evaluate_mrr(model, dataset, device, batch_size=64):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=anchor_collate, pin_memory=True)

    all_ranks = []
    for inp, task, label, node_ids in loader:
        inp = inp.to(device, non_blocking=True)
        node_ids = node_ids.to(device, non_blocking=True)
        scores = model(inp, node_ids)
        G = 1 + N_HARD_NEG
        n_anchors = scores.shape[0] // G
        scores = scores.view(n_anchors, G)
        pos_scores = scores[:, 0]
        neg_scores = scores[:, 1:]
        higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
        rank = higher + 1
        all_ranks.append(rank.cpu())

    ranks = torch.cat(all_ranks).float()
    mrr = (1.0 / ranks).mean().item()
    hit1 = (ranks <= 1).float().mean().item()
    hit3 = (ranks <= 3).float().mean().item()
    hit5 = (ranks <= 5).float().mean().item()
    return mrr, hit1, hit3, hit5


def main():
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # Load SciBERT embeddings
    print("Loading SciBERT embeddings...", flush=True)
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    print(f"  shape: {scibert.shape}", flush=True)

    # Load precomputed data
    print("Loading precomputed tensors v2...", flush=True)
    t0 = time.time()
    train_data = np.load(f'{DATA_DIR}/train_precomp_v2.npz')
    val_data = np.load(f'{DATA_DIR}/val_precomp_v2.npz')
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    train_inp = torch.from_numpy(train_data['inp'].astype(np.float32)).pin_memory()
    train_task = torch.from_numpy(train_data['task'].astype(np.float32)).pin_memory()
    train_label = torch.from_numpy(train_data['label'].astype(np.float32)).pin_memory()
    train_node_ids = torch.from_numpy(train_data['node_ids'].astype(np.int64)).pin_memory()
    val_inp = torch.from_numpy(val_data['inp'].astype(np.float32)).pin_memory()
    val_task = torch.from_numpy(val_data['task'].astype(np.float32)).pin_memory()
    val_label = torch.from_numpy(val_data['label'].astype(np.float32)).pin_memory()
    val_node_ids = torch.from_numpy(val_data['node_ids'].astype(np.int64)).pin_memory()

    G = 1 + N_HARD_NEG
    print(f"  train: {len(train_label)} samples ({len(train_label)//G} anchors)", flush=True)
    print(f"  val:   {len(val_label)} samples ({len(val_label)//G} anchors)", flush=True)

    train_ds = AnchorDataset(train_inp, train_task, train_label, train_node_ids)
    val_ds = AnchorDataset(val_inp, val_task, val_label, val_node_ids)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=anchor_collate, pin_memory=True)

    model = PencilModel(scibert).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    bpr_loss_fn = nn.LogSigmoid()

    # Warmup
    dummy = torch.zeros(4, MAX_NODES, EMBED_DIM, device=device)
    dummy_nid = torch.zeros(4, MAX_NODES, dtype=torch.long, device=device)
    _ = model(dummy, dummy_nid)
    torch.cuda.synchronize()
    print(f"  GPU warmup done, mem: {torch.cuda.memory_allocated()/1e9:.3f}GB", flush=True)

    print(f"\n=== Training v2 (SciBERT+DRNL, {BATCH_SIZE} anchors/batch, {N_EPOCHS} epochs) ===", flush=True)
    best_mrr = 0
    t_total = time.time()

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, nb = 0, 0

        for inp, task, label, node_ids in train_loader:
            inp = inp.to(device, non_blocking=True)
            node_ids = node_ids.to(device, non_blocking=True)
            scores = model(inp, node_ids)

            n_anchors = scores.shape[0] // G
            scores = scores.view(n_anchors, G)
            pos_scores = scores[:, 0]
            neg_scores = scores[:, 1:]

            pos_expanded = pos_scores.repeat_interleave(N_HARD_NEG)
            bpr_loss = -bpr_loss_fn(pos_expanded - neg_scores.reshape(-1)).mean()

            opt.zero_grad(set_to_none=True)
            bpr_loss.backward()
            opt.step()

            total_loss += bpr_loss.item()
            nb += 1

        sched.step()
        dt_train = time.time() - t0

        t1 = time.time()
        mrr, hit1, hit3, hit5 = evaluate_mrr(model, val_ds, device)
        dt_eval = time.time() - t1

        marker = ""
        if mrr > best_mrr:
            best_mrr = mrr
            torch.save(model.state_dict(), f'{DATA_DIR}/pencil_v2_best.pt')
            marker = " *"

        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}  bpr_loss={total_loss/nb:.4f}  "
              f"MRR={mrr:.4f}  Hit@1={hit1:.4f}  Hit@3={hit3:.4f}  Hit@5={hit5:.4f}  "
              f"train={dt_train:.1f}s  eval={dt_eval:.1f}s{marker}", flush=True)

    print(f"\nBest MRR: {best_mrr:.4f}", flush=True)
    print(f"Total: {time.time()-t_total:.1f}s", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
