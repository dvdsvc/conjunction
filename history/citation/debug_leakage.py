"""Check for data leakage between train and val precomputed data."""
import numpy as np
import re

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'

def _extract_val_pairs(data_dir):
    pos_pairs = []
    with open(f'{data_dir}/train.jsonl') as f:
        for line in f:
            if '"label": 1' not in line and '"label":1' not in line: continue
            src_m = re.search(r'"src_idx"\s*:\s*(\d+)', line)
            dst_m = re.search(r'"dst_idx"\s*:\s*(\d+)', line)
            if not src_m or not dst_m: continue
            src_idx = int(src_m.group(1))
            dst_idx = int(dst_m.group(1))
            nid_m = re.search(r'"node_ids"\s*:\s*\[([^\]]+)\]', line)
            if not nid_m: continue
            nids = [int(x) for x in nid_m.group(1).split(',')]
            pos_pairs.append((nids[src_idx], nids[dst_idx]))
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(pos_pairs))
    n_train = int(len(pos_pairs) * 0.8)
    return ([pos_pairs[i] for i in perm[:n_train]],
            [pos_pairs[i] for i in perm[n_train:]])

train_pairs, val_pairs = _extract_val_pairs(DATA_DIR)

# Load precomputed data
train_data = np.load(f'{DATA_DIR}/train_precomp_v3.npz')
val_data = np.load(f'{DATA_DIR}/val_precomp_v3.npz')

train_nids = train_data['node_ids']  # (N_train*6, 32)
val_nids = val_data['node_ids']      # (N_val*6, 32)
train_labels = train_data['label']
val_labels = val_data['label']

G = 6  # 1+5

print(f"Train: {train_nids.shape[0]//G} anchors, {train_nids.shape[0]} samples")
print(f"Val:   {val_nids.shape[0]//G} anchors, {val_nids.shape[0]} samples")
print(f"Train positive count: {train_labels.sum()}")
print(f"Val positive count:   {val_labels.sum()}")

# Check 1: do train and val positive pairs overlap?
train_pos_pairs = set()
for i in range(0, len(train_nids), G):
    u, v = int(train_nids[i, 0]), int(train_nids[i, 1])
    train_pos_pairs.add((u, v))

val_pos_pairs = set()
for i in range(0, len(val_nids), G):
    u, v = int(val_nids[i, 0]), int(val_nids[i, 1])
    val_pos_pairs.add((u, v))

overlap = train_pos_pairs & val_pos_pairs
print(f"\n=== Positive pair overlap ===")
print(f"  Train unique pairs: {len(train_pos_pairs)}")
print(f"  Val unique pairs:   {len(val_pos_pairs)}")
print(f"  Overlap:            {len(overlap)}")
if len(overlap) > 0:
    print(f"  WARNING: DATA LEAKAGE! {len(overlap)} pairs in both train and val")
    for p in list(overlap)[:5]:
        print(f"    {p}")
else:
    print(f"  OK: no overlap")

# Check 2: do train and val use the same NEGATIVE candidates?
# Compare first few negative subgraphs
print(f"\n=== Negative candidate comparison ===")
# Train anchor 0 neg 1
train_neg0_nids = set(train_nids[1])  # anchor 0, neg 1
# Val anchor 0 neg 1
val_neg0_nids = set(val_nids[1])
print(f"  Train[0] neg1 node_ids: {sorted(train_neg0_nids)[:10]}")
print(f"  Val[0] neg1 node_ids:   {sorted(val_neg0_nids)[:10]}")
print(f"  Same: {train_neg0_nids == val_neg0_nids}")

# Check 3: are the val positive subgraphs trivially distinguishable?
# Look at adjacency features (col 32:64) for positive vs negative
print(f"\n=== Feature distinguishability (val anchor 0) ===")
val_inp = val_data['inp']  # (N_val*6, 32, 68) float16
for idx, name in [(0, 'positive'), (1, 'neg1'), (2, 'neg2')]:
    feat = val_inp[idx].astype(np.float32)
    # Count edges in adjacency
    adj_sum = feat[:, 32:64].sum()
    # Distance features
    dist_u = feat[:, 65]
    dist_v = feat[:, 66]
    print(f"  {name}: adj_sum={adj_sum:.0f}, dist_u_mean={dist_u[dist_u>0].mean():.3f}, dist_v_mean={dist_v[dist_v>0].mean():.3f}")

# Check 4: for positive pair, what are dist_u[0] and dist_v[0]?
# pos node 0 = u, pos node 1 = v
print(f"\n=== Distance at anchor nodes (val) ===")
for ai in range(5):
    pos_feat = val_inp[ai * G].astype(np.float32)
    print(f"  anchor {ai}: dist_u[0]={pos_feat[0,65]:.3f}, dist_v[0]={pos_feat[0,66]:.3f}, "
          f"dist_u[1]={pos_feat[1,65]:.3f}, dist_v[1]={pos_feat[1,66]:.3f}")
    # For negative
    neg_feat = val_inp[ai * G + 1].astype(np.float32)
    print(f"  neg1:      dist_u[0]={neg_feat[0,65]:.3f}, dist_v[0]={neg_feat[0,66]:.3f}, "
          f"dist_u[1]={neg_feat[1,65]:.3f}, dist_v[1]={neg_feat[1,66]:.3f}")

# Check 5: Is the positive trivially identifiable from features alone?
# (e.g., positive always has dist_u[0]=0, dist_v[1]=0, but negatives don't)
print(f"\n=== Can we distinguish pos from neg by simple feature check? ===")
for ai in range(min(10, len(val_pairs))):
    pos_feat = val_inp[ai * G].astype(np.float32)
    neg_feat = val_inp[ai * G + 1].astype(np.float32)
    # Positive: node 0 = u (dist_u=0), node 1 = v (dist_v=0)
    # Negative: node 0 = u (dist_u=0), node 1 = neg_cand (dist_v != 0 usually)
    pos_has_u_at_0 = pos_feat[0, 65] == 0  # dist_u(node0) = 0 → node0 IS u
    pos_has_v_at_1 = pos_feat[1, 66] == 0  # dist_v(node1) = 0 → node1 IS v
    neg_has_u_at_0 = neg_feat[0, 65] == 0
    neg_has_v_at_1 = neg_feat[1, 66] == 0  # should NOT be 0 (neg != v)
    print(f"  anchor {ai}: pos_u@0={pos_has_u_at_0}, pos_v@1={pos_has_v_at_1}, "
          f"neg_u@0={neg_has_u_at_0}, neg_v@1={neg_has_v_at_1}")
