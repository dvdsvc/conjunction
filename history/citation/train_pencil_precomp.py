"""
Fast PENCIL training on precomputed subgraph tensors.
Key fix: shuffle by anchor group (not individual samples) to preserve BPR pairing.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 128
EMBED_DIM = 2 * MAX_NODES + 2  # 258
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
N_EPOCHS = 30
LR = 3e-4
N_HARD_NEG = 5
BATCH_SIZE = 256  # anchors per batch (each anchor = 6 subgraphs)


class AnchorDataset(Dataset):
    """Each item = one anchor with its 1 positive + N_HARD_NEG negatives."""
    def __init__(self, inp, task, label):
        group = 1 + N_HARD_NEG
        n_anchors = len(label) // group
        # Reshape to (n_anchors, group, ...)
        self.inp = inp[:n_anchors * group].view(n_anchors, group, *inp.shape[1:])
        self.task = task[:n_anchors * group].view(n_anchors, group, *task.shape[1:])
        self.label = label[:n_anchors * group].view(n_anchors, group)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.inp[idx], self.task[idx], self.label[idx]


def anchor_collate(batch):
    """batch = list of (inp_group, task_group, label_group) each (6, 128, 258)"""
    inp = torch.cat([b[0] for b in batch], dim=0)   # (B*6, 128, 258)
    task = torch.cat([b[1] for b in batch], dim=0)   # (B*6, 2, 258)
    label = torch.cat([b[2] for b in batch], dim=0)  # (B*6,)
    return inp, task, label


class PencilModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(EMBED_DIM, HIDDEN)
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

    def forward(self, inp):
        B, N, _ = inp.shape
        adj = inp[:, :N, MAX_NODES:MAX_NODES + N]
        deg = adj.sum(dim=2, keepdim=True).clamp(min=1)
        adj_norm = adj / deg

        pad_mask = (inp.abs().sum(dim=2) == 0)

        x = self.proj(inp)
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
    """Compute MRR/Hit@k preserving anchor grouping."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=anchor_collate, pin_memory=True)

    all_ranks = []
    for inp, task, label in loader:
        inp = inp.to(device, non_blocking=True)
        scores = model(inp)
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

    print("Loading precomputed tensors...", flush=True)
    t0 = time.time()
    train_data = np.load(f'{DATA_DIR}/train_precomp.npz')
    val_data = np.load(f'{DATA_DIR}/val_precomp.npz')
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    train_inp = torch.from_numpy(train_data['inp'].astype(np.float32)).pin_memory()
    train_task = torch.from_numpy(train_data['task'].astype(np.float32)).pin_memory()
    train_label = torch.from_numpy(train_data['label'].astype(np.float32)).pin_memory()
    val_inp = torch.from_numpy(val_data['inp'].astype(np.float32)).pin_memory()
    val_task = torch.from_numpy(val_data['task'].astype(np.float32)).pin_memory()
    val_label = torch.from_numpy(val_data['label'].astype(np.float32)).pin_memory()

    G = 1 + N_HARD_NEG
    print(f"  train: {len(train_label)} samples ({len(train_label)//G} anchors)", flush=True)
    print(f"  val:   {len(val_label)} samples ({len(val_label)//G} anchors)", flush=True)

    train_ds = AnchorDataset(train_inp, train_task, train_label)
    val_ds = AnchorDataset(val_inp, val_task, val_label)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=anchor_collate, pin_memory=True)

    model = PencilModel().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    bpr_loss_fn = nn.LogSigmoid()

    # Warmup
    dummy = torch.zeros(4, MAX_NODES, EMBED_DIM, device=device)
    _ = model(dummy)
    torch.cuda.synchronize()
    print(f"  GPU warmup done, mem: {torch.cuda.memory_allocated()/1e9:.3f}GB", flush=True)

    print(f"\n=== Training (BPR anchor-shuffle, {BATCH_SIZE} anchors/batch, {N_EPOCHS} epochs) ===", flush=True)
    best_mrr = 0
    t_total = time.time()

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, nb = 0, 0

        for inp, task, label in train_loader:
            inp = inp.to(device, non_blocking=True)
            scores = model(inp)

            # inp comes as (B*6, 128, 258) with groups of 6 intact
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
            torch.save(model.state_dict(), f'{DATA_DIR}/pencil_bpr_precomp_best.pt')
            marker = " *"

        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}  bpr_loss={total_loss/nb:.4f}  "
              f"MRR={mrr:.4f}  Hit@1={hit1:.4f}  Hit@3={hit3:.4f}  Hit@5={hit5:.4f}  "
              f"train={dt_train:.1f}s  eval={dt_eval:.1f}s{marker}", flush=True)

    print(f"\nBest MRR: {best_mrr:.4f}", flush=True)
    print(f"Total: {time.time()-t_total:.1f}s", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
