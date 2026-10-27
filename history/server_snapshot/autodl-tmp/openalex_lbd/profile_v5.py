import time, numpy as np, torch
import sys
sys.path.insert(0, '/root/autodl-tmp/openalex_lbd')
from train_pencil_v5 import *

device = torch.device('cuda')
scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
train_nids = np.load(f'{DATA_DIR}/train_nids_v5.npy')
dist_matrix = torch.from_numpy(np.load(f'{DATA_DIR}/dist_matrix.npy')).to(device)
train_ds = AnchorDataset(torch.from_numpy(train_nids))

from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=anchor_collate)
model = PencilModel(scibert).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)

# Profile 3 batches
for bi, (node_ids, labels) in enumerate(train_loader):
    if bi >= 3: break
    B = node_ids.shape[0]
    K = node_ids.shape[1] - 1
    t0 = time.time()
    node_ids = node_ids.to(device)
    labels = labels.to(device)
    t1 = time.time()

    perm = torch.randperm(K - 1, device=device)[:N_CANDIDATES - 1] + 2
    keep = torch.cat([torch.tensor([0, 1], device=device), perm])
    node_ids = node_ids[:, keep]
    K = N_CANDIDATES
    t2 = time.time()

    cand_nids = node_ids[:, 1:]
    flat_nids = cand_nids.reshape(B * K, 32)
    t3 = time.time()

    inp = reconstruct_features_gpu(flat_nids, dist_matrix)
    torch.cuda.synchronize()
    t4 = time.time()

    scores = model(inp, flat_nids).view(B, K)
    torch.cuda.synchronize()
    t5 = time.time()

    loss = torch.nn.CrossEntropyLoss()(scores, labels)
    t6 = time.time()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    torch.cuda.synchronize()
    t7 = time.time()

    print(f"Batch {bi}: to_dev={t1-t0:.3f}s subsample={t2-t1:.3f}s reshape={t3-t2:.3f}s "
          f"recon={t4-t3:.3f}s forward={t5-t4:.3f}s loss={t6-t5:.3f}s "
          f"backward+step={t7-t6:.3f}s TOTAL={t7-t0:.3f}s")
