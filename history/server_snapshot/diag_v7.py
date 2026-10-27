import os, json, glob
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
TEMP_DIR = f'{DATA_DIR}/temporal_v2'

MAX_NODES = 64
EMBED_DIM = 129
SCIBERT_DIM = 768
HIDDEN = 256
N_HEADS = 8
PPR_CHANGE_THRESH = 1e-7

# Import model class from train_temporal_v7
import importlib.util
spec = importlib.util.spec_from_file_location('v7', '/root/train_temporal_v7.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PencilV7 = mod.PencilV7

device = torch.device('cuda')

# Load minimal data
print('Loading concept IDs...')
with open(f'{TEMP_DIR}/concept_ids_v2.json') as f:
    concept_ids = json.load(f)
n_concepts = len(concept_ids)

print('Loading SciBERT...')
scibert_table = mod.load_scibert_remapped(concept_ids, n_concepts)

print('Loading PPR + graph (train)...')
ppr_train = np.load(f'{TEMP_DIR}/ppr_v2_2005.npy').astype(np.float32)
adj_train = sp.load_npz(f'{TEMP_DIR}/graph_v2_2005.npz').tocsr()

print('Building feature table...')
concept_feats, concept_nids = mod.build_concept_feature_table(ppr_train, n_concepts, device)
mod.fill_adjacency_features(concept_feats, concept_nids, adj_train)

print('Loading train chunks...')
train_pairs, train_labels = mod._load_chunks('train', f'{TEMP_DIR}/fold0_v2_chunks')
src_train = mod.build_anchor_data(train_pairs, train_labels)
train_ds = mod.AnchorDataset(src_train, concept_feats, 50, rng_seed=42)

# Build model
model = PencilV7(scibert_table).to(device)
dummy = torch.zeros(2, 51, MAX_NODES, EMBED_DIM, device=device)
dummy_nid = torch.zeros(2, 51, MAX_NODES, dtype=torch.long, device=device)
model.train()
_ = model(dummy, dummy_nid)

# === Diagnostic 1: Raw model outputs (before sigmoid) ===
print('\n=== DIAG 1: Raw model outputs ===')
model.eval()
with torch.no_grad():
    batch = [train_ds[i] for i in range(4)]
    inp = torch.stack([b[0] for b in batch]).to(device)
    nids = torch.stack([b[1] for b in batch]).to(device)
    dcs = torch.stack([b[2] for b in batch])
    raw_logits = model(inp, nids)
    probs = torch.sigmoid(raw_logits)
    
    print(f'  Raw logits: min={raw_logits.min():.4f}, max={raw_logits.max():.4f}, mean={raw_logits.mean():.4f}, std={raw_logits.std():.4f}')
    print(f'  Sigmoid probs: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}')
    print(f'  Labels (dcs): min={dcs.min():.6f}, max={dcs.max():.6f}')
    print(f'  Labels binary: {(dcs > PPR_CHANGE_THRESH).float().mean():.2%} positive')

# === Diagnostic 2: Gradient flow ===
print('\n=== DIAG 2: Gradient flow ===')
model.train()
batch = [train_ds[i] for i in range(4)]
inp = torch.stack([b[0] for b in batch]).to(device)
nids = torch.stack([b[1] for b in batch]).to(device)
dcs = torch.stack([b[2] for b in batch])
labels_bin = (dcs > PPR_CHANGE_THRESH).float().to(device)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Before step
params_before = {n: p.clone() for n, p in model.named_parameters()}

scores = model(inp, nids)
loss = nn.BCEWithLogitsLoss()(scores, labels_bin)
opt.zero_grad()
loss.backward()

print(f'  Loss: {loss.item():.6f}')
print(f'  Scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}')

# Check gradients
for name, p in model.named_parameters():
    if p.grad is not None:
        grad_norm = p.grad.norm().item()
        grad_max = p.grad.abs().max().item()
        if 'head' in name or 'combine' in name or 'qkv' in name:
        # if grad_norm > 0:
            print(f'  {name}: grad_norm={grad_norm:.6f}, grad_max={grad_max:.6f}')
    else:
        print(f'  {name}: NO GRADIENT')

opt.step()

# Check if params changed
params_changed = 0
total_params = 0
for name, p in model.named_parameters():
    total_params += 1
    diff = (p - params_before[name]).abs().max().item()
    if diff > 1e-10:
        params_changed += 1
        if 'head' in name or 'combine' in name:
            print(f'  {name} changed: max_diff={diff:.8f}')

print(f'\n  Params changed: {params_changed}/{total_params}')
print('DIAG DONE')
