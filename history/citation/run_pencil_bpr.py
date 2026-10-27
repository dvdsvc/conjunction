"""
PENCIL with BPR loss + hard negative sampling.
For each positive (u, v+), sample 5 hard negatives (2-hop disconnected).
Evaluate: MRR and Hit@10 (ranking metrics).
"""

import json, os, time, random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 128
EMBED_DIM = 2 * MAX_NODES + 2
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
N_EPOCHS = 10
LR = 1e-3
N_HARD_NEG = 5

# ── Globals (set in main, used by dataset workers via fork) ──
_adj_rows = None
_adj_cols = None
_degree = None

# ── Dataset: BPR triplets with on-the-fly subgraph construction ──

class BPRDataset(Dataset):
    """
    Each sample: (anchor u, positive v+)
    At __getitem__, builds subgraph for (u, v+) and N_HARD_NEG hard negatives.
    Returns: positive subgraph embedding + negative subgraph embeddings.
    """
    def __init__(self, pos_pairs, adj_csr, degree, n_hard_neg=N_HARD_NEG, max_nodes=MAX_NODES):
        self.pos_pairs = pos_pairs  # list of (u_idx, v_idx)
        self.adj = adj_csr
        self.degree = degree
        self.n_hard_neg = n_hard_neg
        self.max_nodes = max_nodes
        self.rows = adj_csr.indptr
        self.cols = adj_csr.indices

    def __len__(self):
        return len(self.pos_pairs)

    def _build_subgraph(self, u, v):
        """Build subgraph for (u, v) pair."""
        nu = set(self.cols[self.rows[u]:self.rows[u+1]])
        nv = set(self.cols[self.rows[v]:self.rows[v+1]])
        shared = sorted(nu & nv, key=lambda x: -self.degree[x])
        only_u = sorted(nu - nv - {v}, key=lambda x: -self.degree[x])
        only_v = sorted(nv - nu - {u}, key=lambda x: -self.degree[x])

        nodes = [u, v]
        rem = self.max_nodes - 2
        for lst in [shared, only_u, only_v]:
            for n in lst:
                if rem <= 0: break
                nodes.append(n)
                rem -= 1
            if rem <= 0: break

        nset = set(nodes)
        lmap = {c: i for i, c in enumerate(nodes)}

        # Build encoding: (n, EMBED_DIM)
        n = len(nodes)
        x = np.zeros((n, EMBED_DIM), dtype=np.float32)
        for i in range(n):
            if i < MAX_NODES:
                x[i, i] = 1.0
            x[i, 2*MAX_NODES] = 1.0  # is_node

        for a in nodes:
            for b in self.cols[self.rows[a]:self.rows[a+1]]:
                if b in nset:
                    ia, ib = lmap[a], lmap[b]
                    if ib < MAX_NODES:
                        x[ia, MAX_NODES + ib] = 1.0

        # Pad to max_nodes
        padded = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
        padded[:n] = x

        # Task tokens
        task = np.zeros((2, EMBED_DIM), dtype=np.float32)
        task[0] = padded[0]
        task[0, 2*MAX_NODES + 1] = 1.0
        task[1] = padded[1]
        task[1, 2*MAX_NODES + 1] = 1.0

        return padded, task, n

    def _sample_hard_neg(self, u, pos_v):
        """Sample a hard negative: 2-hop disconnected from u."""
        nu = self.cols[self.rows[u]:self.rows[u+1]]
        # Sample 20 neighbors randomly
        if len(nu) > 20:
            nu = nu[np.random.choice(len(nu), 20, replace=False)]
        for n1 in nu:
            n2s = self.cols[self.rows[n1]:self.rows[n1+1]]
            if len(n2s) > 0:
                neg = np.random.choice(n2s)
                if neg != u and neg != pos_v:
                    # Check not already connected
                    u_neighbors = set(self.cols[self.rows[u]:self.rows[u+1]])
                    if neg not in u_neighbors:
                        return neg
        # Fallback: random
        return np.random.randint(0, self.adj.shape[0])

    def __getitem__(self, idx):
        u, v = self.pos_pairs[idx]

        # Positive subgraph
        pos_inp, pos_task, pos_n = self._build_subgraph(u, v)

        # Hard negative subgraphs
        neg_inps = []
        neg_tasks = []
        neg_ns = []
        for _ in range(self.n_hard_neg):
            neg_v = self._sample_hard_neg(u, v)
            ni, nt, nn = self._build_subgraph(u, neg_v)
            neg_inps.append(ni)
            neg_tasks.append(nt)
            neg_ns.append(nn)

        return {
            'pos_inp': torch.from_numpy(pos_inp),
            'pos_task': torch.from_numpy(pos_task),
            'pos_n': pos_n,
            'neg_inp': torch.stack([torch.from_numpy(x) for x in neg_inps]),
            'neg_task': torch.stack([torch.from_numpy(x) for x in neg_tasks]),
            'neg_n': torch.tensor(neg_ns),
        }


def bpr_collate(batch):
    B = len(batch)
    pos_inp = torch.stack([b['pos_inp'] for b in batch])
    pos_task = torch.stack([b['pos_task'] for b in batch])
    pos_n = torch.tensor([b['pos_n'] for b in batch])
    neg_inp = torch.cat([b['neg_inp'] for b in batch])  # (B*5, MAX_NODES, EMBED_DIM)
    neg_task = torch.cat([b['neg_task'] for b in batch])
    neg_n = torch.cat([b['neg_n'] for b in batch])
    return pos_inp, pos_task, pos_n, neg_inp, neg_task, neg_n


# ── Model ──

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
                    nn.Linear(HIDDEN, HIDDEN*4), nn.GELU(),
                    nn.Linear(HIDDEN*4, HIDDEN), nn.Dropout(0.1)),
                'ln2': nn.LayerNorm(HIDDEN),
                'mp': nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU()),
            }))
        self.ln_f = nn.LayerNorm(HIDDEN)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN*2, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

    def _encode(self, inp, nn_):
        """Encode a batch of subgraphs → score per sample."""
        B, N, _ = inp.shape
        adj = inp[:, :N, MAX_NODES:MAX_NODES+N]
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
        return self.head(torch.cat([x[:, 0], x[:, 1]], -1)).squeeze(-1)

    def forward(self, pos_inp, pos_task, pos_n, neg_inp, neg_task, neg_n):
        pos_scores = self._encode(pos_inp, pos_n)       # (B,)
        neg_scores = self._encode(neg_inp, neg_n)        # (B*5,)
        return pos_scores, neg_scores


# ── Eval: MRR & Hit@k ──

@torch.no_grad()
def evaluate_mrr(model, val_pairs, adj_csr, degree, device, n_candidates=100):
    """
    For each positive (u, v+), rank v+ among n_candidates random concepts.
    Compute MRR and Hit@10.
    """
    model.eval()
    rng = np.random.RandomState(42)
    n_concepts = adj_csr.shape[0]
    rows = adj_csr.indptr
    cols = adj_csr.indices

    ranks = []
    sample_size = min(1000, len(val_pairs))
    sampled = rng.choice(len(val_pairs), sample_size, replace=False)

    dataset = BPRDataset(
        [val_pairs[i] for i in sampled], adj_csr, degree,
        n_hard_neg=n_candidates-1, max_nodes=MAX_NODES)
    loader = DataLoader(dataset, batch_size=64, collate_fn=bpr_collate, num_workers=0)

    for pos_inp, pos_task, pos_n, neg_inp, neg_task, neg_n in loader:
        pos_inp = pos_inp.to(device)
        pos_task = pos_task.to(device)
        pos_n = pos_n.to(device)
        neg_inp = neg_inp.to(device)
        neg_task = neg_task.to(device)
        neg_n = neg_n.to(device)

        pos_scores = model._encode(pos_inp, pos_n)  # (B,)
        neg_scores = model._encode(neg_inp, neg_n)   # (B*99,)
        neg_scores = neg_scores.view(pos_scores.shape[0], -1)  # (B, 99)

        # Rank: how many negatives score higher than positive?
        higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)  # (B,)
        rank = higher + 1  # 1-indexed rank
        ranks.append(rank.cpu())

    ranks = torch.cat(ranks).float()
    mrr = (1.0 / ranks).mean().item()
    hit10 = (ranks <= 10).float().mean().item()
    hit1 = (ranks <= 1).float().mean().item()
    return mrr, hit10, hit1


# ── Main ──

def main():
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load graph
    print("Loading graph...", flush=True)
    adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz')
    adj_csr = adj.tocsr()
    degree = np.array(adj.sum(axis=1)).flatten().astype(np.float32)
    n_concepts = adj.shape[0]
    print(f"  concepts: {n_concepts}, edges: {adj.nnz}", flush=True)

    # Load positive pairs from train.jsonl (just the positive samples)
    print("Loading positive pairs...", flush=True)
    pos_pairs = []
    with open(f'{DATA_DIR}/train.jsonl') as f:
        for line in f:
            d = json.loads(line)
            if d['label'] == 1:
                pos_pairs.append((d['node_ids'][d['src_idx']], d['node_ids'][d['dst_idx']]))
    print(f"  positive pairs: {len(pos_pairs)}", flush=True)

    # Split
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(pos_pairs))
    n_train = int(len(pos_pairs) * 0.8)
    train_pairs = [pos_pairs[i] for i in perm[:n_train]]
    val_pairs = [pos_pairs[i] for i in perm[n_train:]]
    print(f"  train: {len(train_pairs)}, val: {len(val_pairs)}", flush=True)

    # Dataset & DataLoader
    train_ds = BPRDataset(train_pairs, adj_csr, degree)
    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True,
        num_workers=16, collate_fn=bpr_collate,
        pin_memory=True, persistent_workers=True)
    print(f"  train batches: {len(train_loader)}", flush=True)

    # Model
    model = PencilModel().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    bpr_loss_fn = nn.LogSigmoid()

    # Warmup
    dummy = torch.zeros(2, MAX_NODES, EMBED_DIM, device=device)
    dummy_task = torch.zeros(2, 2, EMBED_DIM, device=device)
    dummy_n = torch.tensor([MAX_NODES, MAX_NODES], device=device)
    _ = model._encode(dummy, dummy_n)
    torch.cuda.synchronize()
    print(f"  GPU warmup done", flush=True)

    # Train
    print("\n=== Training (BPR + Hard Negatives) ===", flush=True)
    best_mrr = 0
    t_total = time.time()

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, nb = 0, 0

        for pos_inp, pos_task, pos_n, neg_inp, neg_task, neg_n in train_loader:
            pos_inp = pos_inp.to(device, non_blocking=True)
            pos_task = pos_task.to(device, non_blocking=True)
            pos_n = pos_n.to(device, non_blocking=True)
            neg_inp = neg_inp.to(device, non_blocking=True)
            neg_task = neg_task.to(device, non_blocking=True)
            neg_n = neg_n.to(device, non_blocking=True)

            pos_scores, neg_scores = model(pos_inp, pos_task, pos_n,
                                           neg_inp, neg_task, neg_n)

            # BPR loss: -log σ(pos - neg) for each hard negative
            # pos_scores: (B,), neg_scores: (B*5,)
            B = pos_scores.shape[0]
            pos_expanded = pos_scores.repeat_interleave(N_HARD_NEG)  # (B*5,)
            bpr_loss = -bpr_loss_fn(pos_expanded - neg_scores).mean()

            opt.zero_grad(set_to_none=True)
            bpr_loss.backward()
            opt.step()

            total_loss += bpr_loss.item()
            nb += 1

        sched.step()
        dt_train = time.time() - t0

        # Eval
        t1 = time.time()
        mrr, hit10, hit1 = evaluate_mrr(model, val_pairs, adj_csr, degree, device)
        dt_eval = time.time() - t1

        marker = ""
        if mrr > best_mrr:
            best_mrr = mrr
            torch.save(model.state_dict(), f'{DATA_DIR}/pencil_bpr_best.pt')
            marker = " *"

        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}  bpr_loss={total_loss/nb:.4f}  "
              f"MRR={mrr:.4f}  Hit@10={hit10:.4f}  Hit@1={hit1:.4f}  "
              f"train={dt_train:.1f}s  eval={dt_eval:.1f}s{marker}", flush=True)

    print(f"\nBest MRR: {best_mrr:.4f}", flush=True)
    print(f"Total: {time.time()-t_total:.1f}s", flush=True)
    print("Done.", flush=True)

if __name__ == '__main__':
    main()
