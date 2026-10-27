"""
Self-contained PENCIL training: JSONL → numpy → train → eval.
Single script, no external dependencies beyond torch/numpy.
"""

import json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 128
EMBED_DIM = 2 * MAX_NODES + 2  # 258
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
BATCH_TRAIN = 1024
BATCH_VAL = 2048
N_EPOCHS = 10
LR = 1e-3
N_WORKERS_LOAD = 40

# ── Step 1: JSONL → numpy (parallel) ──

def _convert_line(line):
    """Convert one JSONL sample to numpy arrays."""
    d = json.loads(line)
    n = d['num_nodes']
    node_ids = d['node_ids']
    edges = d['edges']

    # Input: (MAX_NODES, EMBED_DIM) adjacency_row encoding
    inp = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float16)

    # One-hot node IDs + is_node flag
    for i in range(min(n, MAX_NODES)):
        inp[i, i] = 1.0
        inp[i, 2 * MAX_NODES] = 1.0  # is_node

    # Adjacency rows
    for a, b in edges:
        if a < MAX_NODES and b < MAX_NODES:
            inp[a, MAX_NODES + b] = 1.0
            inp[b, MAX_NODES + a] = 1.0

    # Task tokens: src=0, dst=1
    task = np.zeros((2, EMBED_DIM), dtype=np.float16)
    # src
    task[0, :MAX_NODES] = inp[0, :MAX_NODES]
    task[0, MAX_NODES:2*MAX_NODES] = inp[0, MAX_NODES:2*MAX_NODES]
    task[0, 2*MAX_NODES + 1] = 1.0  # is_task
    # dst
    task[1, :MAX_NODES] = inp[1, :MAX_NODES]
    task[1, MAX_NODES:2*MAX_NODES] = inp[1, MAX_NODES:2*MAX_NODES]
    task[1, 2*MAX_NODES + 1] = 1.0

    return inp, task, float(d['label']), n


def load_and_convert(jsonl_path, npz_path, n_workers=N_WORKERS_LOAD):
    """Load JSONL, convert to numpy, cache to npz."""
    if os.path.exists(npz_path):
        print(f"  loading cached {npz_path}", flush=True)
        data = np.load(npz_path)
        return (torch.from_numpy(data['inp'].astype(np.float32)),
                torch.from_numpy(data['task'].astype(np.float32)),
                torch.from_numpy(data['label']),
                torch.from_numpy(data['nn']))

    print(f"  converting {jsonl_path} with {n_workers} workers...", flush=True)
    t0 = time.time()

    with open(jsonl_path) as f:
        lines = f.readlines()

    print(f"    read {len(lines)} lines in {time.time()-t0:.1f}s", flush=True)

    with Pool(n_workers) as pool:
        results = pool.map(_convert_line, lines, chunksize=500)

    inp = np.stack([r[0] for r in results])
    task = np.stack([r[1] for r in results])
    label = np.array([r[2] for r in results], dtype=np.float32)
    nn = np.array([r[3] for r in results], dtype=np.int64)

    # Save cache
    np.savez_compressed(npz_path, inp=inp, task=task, label=label, nn=nn)
    print(f"  saved {npz_path} in {time.time()-t0:.1f}s, {len(label)} samples", flush=True)

    return (torch.from_numpy(inp.astype(np.float32)),
            torch.from_numpy(task.astype(np.float32)),
            torch.from_numpy(label),
            torch.from_numpy(nn))


# ── Step 2: Dataset ──

class TensorDataset(Dataset):
    def __init__(self, inp, task, label, nn):
        self.inp = inp
        self.task = task
        self.label = label
        self.nn = nn

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.inp[idx], self.task[idx], self.label[idx], self.nn[idx]


# ── Step 3: Model ──

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

    def forward(self, inp, task, nn_):
        B, N, _ = inp.shape
        MN = MAX_NODES

        # Adjacency from encoding
        adj = inp[:, :N, MN:MN+N]
        deg = adj.sum(dim=2, keepdim=True).clamp(min=1)
        adj_norm = adj / deg

        x = self.proj(inp)

        for layer in self.layers:
            h = layer['ln1'](x)
            h, _ = layer['attn'](h, h, h)
            x = x + h
            h_mp = layer['mp'](torch.bmm(adj_norm, x[:, :N]))
            x[:, :N] = x[:, :N] + h_mp
            h = layer['ln2'](x)
            x = x + layer['ffn'](h)

        x = self.ln_f(x)
        src, dst = x[:, 0], x[:, 1]
        return self.head(torch.cat([src, dst], -1)).squeeze(-1)


# ── Step 4: Train & Eval ──

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, correct, total, nb = 0, 0, 0, 0
    for inp, task, label, nn_ in loader:
        inp, task, label, nn_ = [t.to(device) for t in (inp, task, label, nn_)]
        logits = model(inp, task, nn_)
        total_loss += loss_fn(logits, label).item()
        correct += ((logits > 0).float() == label).sum().item()
        total += label.shape[0]
        nb += 1
    return total_loss / max(nb, 1), correct / max(total, 1)


def main():
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    # Load & convert
    print("Step 1: Loading data...", flush=True)
    inp, task, label, nn_ = load_and_convert(
        f'{DATA_DIR}/train.jsonl', f'{DATA_DIR}/train_cache.npz')

    print(f"  total: {len(label)}, pos: {(label==1).sum().item()}, neg: {(label==0).sum().item()}", flush=True)

    # Split 80/20
    n = len(label)
    idx = torch.randperm(n)
    tr, va = idx[:int(n*0.8)], idx[int(n*0.8):]

    train_ds = TensorDataset(inp[tr], task[tr], label[tr], nn_[tr])
    val_ds = TensorDataset(inp[va], task[va], label[va], nn_[va])
    print(f"  train: {len(train_ds)}, val: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, BATCH_TRAIN, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_VAL, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    print("Step 2: Building model...", flush=True)
    model = PencilModel().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    loss_fn = nn.BCEWithLogitsLoss()

    # Warmup: one dummy batch
    dummy = torch.zeros(4, MAX_NODES, EMBED_DIM, device=device)
    dummy_task = torch.zeros(4, 2, EMBED_DIM, device=device)
    dummy_nn = torch.full((4,), MAX_NODES, dtype=torch.long, device=device)
    _ = model(dummy, dummy_task, dummy_nn)
    torch.cuda.synchronize()
    print(f"  GPU warmup done, mem: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # Train
    print("\nStep 3: Training...", flush=True)
    best_val_acc = 0
    t_total = time.time()

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, nb = 0, 0

        for inp_b, task_b, label_b, nn_b in train_loader:
            inp_b = inp_b.to(device, non_blocking=True)
            task_b = task_b.to(device, non_blocking=True)
            label_b = label_b.to(device, non_blocking=True)
            nn_b = nn_b.to(device, non_blocking=True)

            logits = model(inp_b, task_b, nn_b)
            loss = loss_fn(logits, label_b)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            nb += 1

        sched.step()
        val_loss, val_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{DATA_DIR}/pencil_best.pt')
            marker = " *"

        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}  loss={total_loss/nb:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}  {dt:.1f}s{marker}", flush=True)

    print(f"\nBest val_acc: {best_val_acc:.4f}", flush=True)
    print(f"Total time: {time.time()-t_total:.1f}s", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
