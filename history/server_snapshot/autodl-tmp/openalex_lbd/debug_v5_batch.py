import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '/root/autodl-tmp/openalex_lbd')
from train_pencil_v5 import *

device = torch.device('cuda')
torch.manual_seed(42)

scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
train_nids = np.load(f'{DATA_DIR}/train_nids_v5.npy')
dist_matrix = torch.from_numpy(np.load(f'{DATA_DIR}/dist_matrix.npy')).to(device)

train_ds = AnchorDataset(torch.from_numpy(train_nids))
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=anchor_collate)

model = PencilModel(scibert).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
ce_loss_fn = nn.CrossEntropyLoss()

# Get one batch
node_ids, labels = next(iter(train_loader))
B = node_ids.shape[0]
K = node_ids.shape[1] - 1
node_ids = node_ids.to(device)
labels = labels.to(device)

# Random subsample of 99 negatives
perm = torch.randperm(K - 1, device=device)[:N_CANDIDATES - 1] + 2
keep = torch.cat([torch.tensor([0, 1], device=device), perm])
node_ids_sub = node_ids[:, keep]
K_sub = N_CANDIDATES

cand_nids = node_ids_sub[:, 1:]
flat_nids = cand_nids.reshape(B * K_sub, 32)
inp = reconstruct_features_gpu(flat_nids, dist_matrix)

# Check features
print(f"inp stats: mean={inp.mean():.4f}, std={inp.std():.4f}, min={inp.min():.4f}, max={inp.max():.4f}")
print(f"adj nonzero: {(inp[:, :, 32:64] > 0).sum().item()}")
print(f"mask nonzero: {(inp[:, :, 64] > 0).sum().item()}")

# Forward
model.train()
scores = model(inp, flat_nids).view(B, K_sub)
loss = ce_loss_fn(scores, labels)

print(f"\nInitial loss: {loss.item():.4f} (expected ~{np.log(K_sub):.4f})")
print(f"Score stats: mean={scores.mean():.4f}, std={scores.std():.4f}")
print(f"Pos score mean: {scores[:, 0].mean():.4f}")
print(f"Neg score mean: {scores[:, 1:].mean():.4f}")

# Backward
opt.zero_grad()
loss.backward()
grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
print(f"\nGradient norms (top 5):")
for n, g in sorted(grad_norms.items(), key=lambda x: -x[1])[:5]:
    print(f"  {n}: {g:.6f}")
print(f"Total grad norm: {sum(g**2 for g in grad_norms.values())**0.5:.4f}")

# 5 training steps
print("\n--- Training 5 steps ---")
for step in range(5):
    opt.zero_grad()
    scores = model(inp, flat_nids).view(B, K_sub)
    loss = ce_loss_fn(scores, labels)
    loss.backward()
    opt.step()
    pos_rank = (scores[:, 1:] > scores[:, 0:1]).sum(dim=1).float().mean().item()
    print(f"  step {step}: loss={loss.item():.4f}, avg_pos_rank={pos_rank:.1f}/99")
