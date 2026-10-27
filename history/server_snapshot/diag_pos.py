import json, sys, os
import numpy as np

os.chdir('/root/autodl-tmp/openalex_data/pencil_data/temporal_v2')
TEMP_DIR = '/root/autodl-tmp/openalex_data/pencil_data/temporal_v2'

base = f'{TEMP_DIR}/precomputed_v9/fold0'
deltas = np.load(f'{base}/val_deltas.npy')
u_loc  = np.load(f'{base}/val_u_local.npy')

with open(f'{base}/val_pairs_by_src.json') as f:
    pairs_by_src = json.load(f)

print(f'Total pairs: {len(deltas):,}', flush=True)
print(f'Unique anchors: {len(pairs_by_src):,}', flush=True)

pos_per_anchor = []
for src_str, pairs in pairs_by_src.items():
    idxs = [p[0] for p in pairs]
    anchor_deltas = deltas[idxs]
    n_pos = int((anchor_deltas >= 2).sum())
    pos_per_anchor.append(n_pos)

pos_per_anchor = np.array(pos_per_anchor)
print(f'\nPositive count per anchor (in all candidates):', flush=True)
for v in range(min(11, pos_per_anchor.max()+1)):
    n = int((pos_per_anchor == v).sum())
    pct = n / len(pos_per_anchor) * 100
    print(f'  {v} positives: {n:>6,} anchors ({pct:.1f}%)', flush=True)
print(f'  >10 positives: {int((pos_per_anchor > 10).sum()):>6,}', flush=True)
print(f'\nMean: {pos_per_anchor.mean():.2f}, Median: {np.median(pos_per_anchor):.0f}', flush=True)
print(f'Max: {pos_per_anchor.max()}, Anchors with 0 pos: {int((pos_per_anchor==0).sum()):,}', flush=True)

# Now simulate what __getitem__ does: pick up to 50 candidates
# The dataset uses N_CANDIDATES=50, sampling from pairs_by_src
# Let's see: for each anchor, how many positives in a typical 50-candidate batch
print(f'\n--- Simulating 50-candidate sampling ---', flush=True)
rng = np.random.RandomState(42)
K = 50
pos_in_k = []
for src_str, pairs in pairs_by_src.items():
    idxs = [p[0] for p in pairs]
    anchor_deltas = deltas[idxs]
    n_total = len(idxs)
    if n_total <= K:
        n_pos = int((anchor_deltas >= 2).sum())
    else:
        sampled = rng.choice(n_total, K, replace=False)
        n_pos = int((anchor_deltas[sampled] >= 2).sum())
    pos_in_k.append(n_pos)

pos_in_k = np.array(pos_in_k)
print(f'Positives in 50-candidate sample:', flush=True)
for v in range(min(11, pos_in_k.max()+1)):
    n = int((pos_in_k == v).sum())
    pct = n / len(pos_in_k) * 100
    print(f'  {v} positives: {n:>6,} anchors ({pct:.1f}%)', flush=True)
print(f'Mean: {pos_in_k.mean():.2f}, Median: {np.median(pos_in_k):.0f}', flush=True)
