import numpy as np
import torch

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'

# Check training data
train_nids = np.load(f'{DATA_DIR}/train_nids_v5.npy')
print(f"train_nids shape: {train_nids.shape}")
print(f"  dtype: {train_nids.dtype}")
print(f"  anchor 0 positive (nids[0,0]): {train_nids[0, 0]}")
print(f"  anchor 0 neg1 (nids[0,1]): {train_nids[0, 1]}")
print(f"  anchor 0 neg2 (nids[0,2]): {train_nids[0, 2]}")
print(f"  nids[0,0,0] = {train_nids[0, 0, 0]}, nids[0,0,1] = {train_nids[0, 0, 1]}")
print(f"  nids[0,1,0] = {train_nids[0, 1, 0]}, nids[0,1,1] = {train_nids[0, 1, 1]}")

# Check if u,v are distinct from negatives
u, v = train_nids[0, 0, 0], train_nids[0, 0, 1]
print(f"\n  u={u}, v={v}")
print(f"  neg1 first 2: {train_nids[0, 1, 0]}, {train_nids[0, 1, 1]}")
print(f"  Are neg features distinct from pos? pos_nids[0] vs neg_nids[0]: {not np.array_equal(train_nids[0, 0], train_nids[0, 1])}")

# Check distance matrix
dist = np.load(f'{DATA_DIR}/dist_matrix.npy')
print(f"\ndist_matrix shape: {dist.shape}, dtype: {dist.dtype}")
print(f"  dist[u,v] = {dist[u, v]}")
print(f"  dist[u, train_nids[0,1,0]] = {dist[u, train_nids[0,1,0]]}")
print(f"  dist[u, u] = {dist[u, u]}")
print(f"  min={dist.min()}, max={dist.max()}")
print(f"  n_dist_eq_1 = {(dist == 1).sum()}")

# Quick feature reconstruction test
import sys
sys.path.insert(0, '/root/autodl-tmp/openalex_lbd')
from train_pencil_v5 import reconstruct_features_gpu, PencilModel, MAX_NODES, EMBED_DIM, DIST_BINS

device = torch.device('cuda')
dist_gpu = torch.from_numpy(dist).to(device)
scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)

# Test reconstruction for 1 positive + 1 negative
nids_batch = torch.from_numpy(train_nids[0, :2]).to(device)  # (2, 32)
feat = reconstruct_features_gpu(nids_batch, dist_gpu)
print(f"\nReconstructed features shape: {feat.shape}")
print(f"  pos feat[0,:3,32:36] (adj of first 3 nodes):\n  {feat[0, :3, 32:36]}")
print(f"  pos feat[0,:,64] (mask): {feat[0, :, 64]}")
print(f"  pos feat[0,:,65] (dist_u): {feat[0, :, 65]}")
print(f"  pos feat[0,:,66] (dist_v): {feat[0, :, 66]}")
print(f"  neg feat[1,:3,32:36] (adj):\n  {feat[1, :3, 32:36]}")
print(f"  neg feat[1,:,65] (dist_u): {feat[1, :, 65]}")

# Test model forward pass
model = PencilModel(scibert).to(device)
scores = model(feat, nids_batch)
print(f"\nModel scores: {scores}")

# Test backward
import torch.nn as nn
loss = nn.CrossEntropyLoss()(scores.unsqueeze(0), torch.tensor([0], device=device))
loss.backward()
grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
print(f"Loss: {loss.item():.4f}")
print(f"Grad norm: {grad_norm:.6f}")

# Overfit test: 10 steps on same batch
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
labels = torch.tensor([0], device=device)
for step in range(10):
    opt.zero_grad()
    s = model(feat, nids_batch)
    l = nn.CrossEntropyLoss()(s.unsqueeze(0), labels)
    l.backward()
    opt.step()
    print(f"  step {step}: loss={l.item():.4f}, scores={s.detach().cpu().numpy()}")
