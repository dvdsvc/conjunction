"""
Quick PENCIL training - fast version.
Converts JSONL to numpy, single-process DataLoader.
"""

import json, os, time, sys
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 128
EMBED_DIM = 2 * MAX_NODES + 2  # adjacency_row encoding

# ─────────────────── Convert JSONL → numpy ───────────────────

def jsonl_to_numpy(jsonl_path, npz_path):
    """Convert JSONL dataset to numpy arrays for fast loading."""
    if os.path.exists(npz_path):
        print(f"  {npz_path} exists, skipping", flush=True)
        return

    print(f"  converting {jsonl_path}...", flush=True)
    samples_input = []
    samples_task = []
    samples_label = []

    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            n = d['num_nodes']
            node_ids = d['node_ids']
            edges = d['edges']

            # Build input embedding: (n, EMBED_DIM)
            inp = np.zeros((n, EMBED_DIM), dtype=np.float16)
            for i in range(min(n, MAX_NODES)):
                inp[i, i] = 1.0  # one-hot node ID
                inp[i, 2*MAX_NODES] = 1.0  # is_node

            # Adjacency rows
            for a, b in edges:
                if a < n and b < n and b < MAX_NODES:
                    inp[a, MAX_NODES + b] = 1.0
                if a < n and b < n and a < MAX_NODES:
                    inp[b, MAX_NODES + a] = 1.0

            # Pad to MAX_NODES
            padded = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float16)
            padded[:n] = inp

            # Task embedding: src + dst
            task = np.zeros((2, EMBED_DIM), dtype=np.float16)
            task[0, :MAX_NODES] = padded[0, :MAX_NODES]
            task[0, MAX_NODES:2*MAX_NODES] = padded[0, MAX_NODES:2*MAX_NODES]
            task[0, 2*MAX_NODES+1] = 1.0  # is_task
            task[1, :MAX_NODES] = padded[1, :MAX_NODES]
            task[1, MAX_NODES:2*MAX_NODES] = padded[1, MAX_NODES:2*MAX_NODES]
            task[1, 2*MAX_NODES+1] = 1.0

            samples_input.append(padded)
            samples_task.append(task)
            samples_label.append(d['label'])

    inp_arr = np.stack(samples_input)  # (N, MAX_NODES, EMBED_DIM)
    task_arr = np.stack(samples_task)  # (N, 2, EMBED_DIM)
    label_arr = np.array(samples_label, dtype=np.float32)

    np.savez_compressed(npz_path,
                        input=inp_arr.astype(np.float16),
                        task=task_arr.astype(np.float16),
                        label=label_arr,
                        n_nodes=np.array([MAX_NODES]*len(label_arr)))
    print(f"  saved {npz_path}: {len(label_arr)} samples", flush=True)

# ─────────────────── Dataset ───────────────────

class NumpyDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.input = torch.from_numpy(data['input'].astype(np.float32))
        self.task = torch.from_numpy(data['task'].astype(np.float32))
        self.label = torch.from_numpy(data['label'])
        self.n_nodes = torch.from_numpy(data['n_nodes'])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'input_embeds': self.input[idx],
            'task_embeds': self.task[idx],
            'label': self.label[idx],
            'num_nodes': self.n_nodes[idx],
        }

# ─────────────────── Model ───────────────────

class PencilModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_layers=4, n_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = MAX_NODES

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=0.1),
                'ln1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(0.1),
                ),
                'ln2': nn.LayerNorm(hidden_dim),
                'mp_proj': nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU()),
            }))
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_embeds, task_embeds, num_nodes):
        B, N, D = input_embeds.shape
        MN = self.max_nodes

        # Extract adjacency from input encoding
        # input_embeds[:, :, MN:2*MN] = adjacency rows
        adj = input_embeds[:, :N, MN:MN+N]  # (B, N, N)
        degree = adj.sum(dim=2, keepdim=True).clamp(min=1)
        adj_norm = adj / degree

        # Project input
        x = self.input_proj(input_embeds)  # (B, N, hidden)

        for layer in self.layers:
            # Self-attention
            h = layer['ln1'](x)
            h, _ = layer['attn'](h, h, h)
            x = x + h

            # Message passing (only on node tokens, not padding)
            h_mp = torch.bmm(adj_norm, x[:, :N])
            h_mp = layer['mp_proj'](h_mp)
            x[:, :N] = x[:, :N] + h_mp

            # FFN
            h = layer['ln2'](x)
            h = layer['ffn'](h)
            x = x + h

        x = self.ln_f(x)

        # Extract src (node 0) and dst (node 1) hidden states
        src_h = x[:, 0]  # (B, hidden)
        dst_h = x[:, 1]  # (B, hidden)
        combined = torch.cat([src_h, dst_h], dim=-1)  # (B, 2*hidden)

        return self.head(combined).squeeze(-1)

# ─────────────────── Eval ───────────────────

@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    n_batch = 0

    for batch in loader:
        inp = batch['input_embeds'].to(device)
        task = batch['task_embeds'].to(device)
        nn_ = batch['num_nodes'].to(device)
        labels = batch['label'].to(device)

        logits = model(inp, task, nn_)
        loss = loss_fn(logits, labels)

        preds = (logits > 0).float()
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
        total_loss += loss.item()
        n_batch += 1

    return total_loss / max(n_batch, 1), correct / max(total, 1)

# ─────────────────── Main ───────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Step 1: Generate val/test if needed
    if not os.path.exists(f'{DATA_DIR}/val.jsonl'):
        print("Generating val/test...", flush=True)
        gen_val_test()

    # Step 2: Convert JSONL → numpy
    print("Converting to numpy...", flush=True)
    for split in ['train', 'val', 'test']:
        jsonl_to_numpy(f'{DATA_DIR}/{split}.jsonl', f'{DATA_DIR}/{split}.npz')

    # Step 3: Load datasets
    print("Loading numpy datasets...", flush=True)
    train_ds = NumpyDataset(f'{DATA_DIR}/train.npz')
    val_ds = NumpyDataset(f'{DATA_DIR}/val.npz')
    print(f"  train: {len(train_ds)}, val: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Step 4: Model
    model = PencilModel(EMBED_DIM, hidden_dim=256, n_layers=4, n_heads=8).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = nn.BCEWithLogitsLoss()

    # Step 5: Train
    print("\nTraining...", flush=True)
    best_val_acc = 0
    for epoch in range(10):
        model.train()
        t0 = time.time()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            inp = batch['input_embeds'].to(device)
            task = batch['task_embeds'].to(device)
            nn_ = batch['num_nodes'].to(device)
            labels = batch['label'].to(device)

            logits = model(inp, task, nn_)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)
        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{DATA_DIR}/pencil_model.pt')
            improved = " *"

        print(f"Epoch {epoch+1:2d}/10: loss={avg_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} lr={lr:.6f} {dt:.1f}s{improved}", flush=True)

    print(f"\nBest val_acc: {best_val_acc:.4f}", flush=True)
    print("Done.", flush=True)

def gen_val_test():
    """Generate val/test: positives from train + random negatives."""
    print("  generating val/test from existing data...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz')
    n_concepts = adj.shape[0]
    rows = adj.tocsr().indptr
    cols = adj.tocsr().indices
    degree = np.array(adj.sum(axis=1)).flatten()

    # Load positive pairs
    pos_pairs = []
    with open(f'{DATA_DIR}/train.jsonl') as f:
        for line in f:
            d = json.loads(line)
            if d['label'] == 1:
                pos_pairs.append((d['node_ids'][d['src_idx']], d['node_ids'][d['dst_idx']]))

    rng = np.random.RandomState(42)
    n = len(pos_pairs)
    val_pos = pos_pairs[n//10:2*n//10]
    test_pos = pos_pairs[2*n//10:3*n//10]

    def build_split(pos_samples, name, n_neg=5):
        samples = []
        for u, v in pos_samples:
            # Positive: build subgraph
            samples.append(build_sample(u, v, 1, rows, cols, degree, n_concepts))
            # Random negatives
            neg_count = 0
            attempts = 0
            while neg_count < n_neg and attempts < 200:
                neg = rng.randint(0, n_concepts)
                if neg != u and neg != v:
                    samples.append(build_sample(u, neg, 0, rows, cols, degree, n_concepts))
                    neg_count += 1
                attempts += 1
        with open(f'{DATA_DIR}/{name}.jsonl', 'w') as f:
            for s in samples:
                f.write(json.dumps(s) + '\n')
        print(f"  {name}: {len(samples)} samples", flush=True)

    def build_sample(u, v, label, rows, cols, degree, n_concepts):
        nu = set(cols[rows[u]:rows[u+1]])
        nv = set(cols[rows[v]:rows[v+1]])
        shared = sorted(nu & nv, key=lambda x: -degree[x])
        only_u = sorted(nu - nv - {v}, key=lambda x: -degree[x])
        only_v = sorted(nv - nu - {u}, key=lambda x: -degree[x])
        nodes = [u, v]
        rem = MAX_NODES - 2
        for lst in [shared, only_u, only_v]:
            for nn in lst:
                if rem <= 0: break
                nodes.append(nn)
                rem -= 1
            if rem <= 0: break
        nset = set(nodes)
        lmap = {c: i for i, c in enumerate(nodes)}
        edges_local = []
        for a in nodes:
            for b in cols[rows[a]:rows[a+1]]:
                if b in nset and a < b:
                    edges_local.append([lmap[a], lmap[b]])
        return {
            'num_nodes': len(nodes),
            'node_ids': nodes,
            'edges': edges_local,
            'src_idx': 0, 'dst_idx': 1, 'label': label,
        }

    build_split(val_pos, 'val')
    build_split(test_pos, 'test')

if __name__ == '__main__':
    main()
