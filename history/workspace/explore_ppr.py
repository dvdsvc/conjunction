"""Explore PPR distribution and find better normalization."""
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA_DIR = '/PATH/TO/ppr_data'

for year in [2012, 2024]:
    print(f"\n=== Year {year} ===")
    d = np.load(f'{DATA_DIR}/ppr_dist_{year}.npy').astype(np.float32)
    n = d.shape[0]

    # Upper triangle only (exclude diagonal)
    mask = np.triu_indices(n, k=1)
    upper = d[mask]

    # Raw distance (already -log(PPR) + clip + min-max from script)
    # But that was clipped wrong. Reload raw PPR instead.
    print(f"  Stored dist: min={upper.min():.4f} max={upper.max():.4f}")

print("\n\n=== Recomputing from raw PPR with better normalization ===")

import scipy.sparse as sp

ALPHA = 0.15
N_ITER = 30

def compute_ppr_raw(adj_csr, alpha=ALPHA, n_iter=N_ITER):
    n = adj_csr.shape[0]
    deg = np.array(adj_csr.sum(axis=1)).flatten()
    deg_inv = np.zeros(n, dtype=np.float32)
    deg_inv[deg > 0] = 1.0 / deg[deg > 0]
    M = (sp.diags(deg_inv) @ adj_csr).T.tocsr()

    ppr = np.zeros((n, n), dtype=np.float32)
    batch_size = 64
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        B = batch_end - batch_start
        r = np.zeros((n, B), dtype=np.float32)
        for i in range(B):
            r[batch_start + i, i] = 1.0
        for _ in range(n_iter):
            r = (1 - alpha) * (M @ r) + alpha * r
        ppr[:, batch_start:batch_end] = r
    return ppr


for year in [2012, 2024]:
    print(f"\n=== Year {year} ===")
    adj = sp.load_npz(f'{DATA_DIR}/graph_{year}.npz').tocsr()
    ppr = compute_ppr_raw(adj)

    mask = np.triu_indices(ppr.shape[0], k=1)
    upper_ppr = ppr[mask]

    # Method 1: -log(PPR), then rank-normalize (percentile rank)
    dist_log = -np.log(upper_ppr + 1e-25)
    # Rank normalization: each value → its percentile rank in [0,1]
    ranks = np.argsort(np.argsort(dist_log)).astype(np.float32)
    dist_rank = ranks / len(ranks)

    print(f"  Method 1: rank-normalized -log(PPR)")
    print(f"    min={dist_rank.min():.4f} max={dist_rank.max():.4f}")
    for pct in [1, 10, 25, 50, 75, 90, 99]:
        print(f"    p{pct}: {np.percentile(dist_rank, pct):.4f}")

    # Method 2: log(1 + 1/(PPR+eps)), rank-normalize
    dist_inv = np.log(1 + 1.0 / (upper_ppr + 1e-25))
    ranks2 = np.argsort(np.argsort(dist_inv)).astype(np.float32)
    dist_rank2 = ranks2 / len(ranks2)

    print(f"  Method 2: rank-normalized log(1+1/PPR)")
    print(f"    min={dist_rank2.min():.4f} max={dist_rank2.max():.4f}")
    for pct in [1, 10, 25, 50, 75, 90, 99]:
        print(f"    p{pct}: {np.percentile(dist_rank2, pct):.4f}")

    # Method 3: -log(PPR), clip at [0.1%, 99.9%], min-max
    p001, p999 = np.percentile(dist_log, [0.1, 99.9])
    dist_clip = np.clip(dist_log, p001, p999)
    dist_clip = (dist_clip - p001) / (p999 - p001)

    print(f"  Method 3: -log(PPR), clip [0.1%, 99.9%], min-max")
    print(f"    min={dist_clip.min():.4f} max={dist_clip.max():.4f}")
    for pct in [1, 10, 25, 50, 75, 90, 99]:
        print(f"    p{pct}: {np.percentile(dist_clip, pct):.4f}")

    # PPR change signal: what does raw PPR look like?
    print(f"\n  Raw PPR distribution:")
    nz = upper_ppr[upper_ppr > 0]
    for pct in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
        print(f"    p{pct}: {np.percentile(nz, pct):.10f}")

    # Save raw PPR for change computation
    np.save(f'{DATA_DIR}/ppr_raw_{year}.npy', ppr.astype(np.float16))
    print(f"  Saved raw PPR: {ppr.nbytes/1e6:.0f}MB")

# Now compute PPR change
print("\n\n=== PPR Change (t2 - t1) ===")
ppr_2012 = np.load(f'{DATA_DIR}/ppr_raw_2012.npy').astype(np.float32)
ppr_2024 = np.load(f'{DATA_DIR}/ppr_raw_2024.npy').astype(np.float32)
ppr_change = ppr_2024 - ppr_2012

upper_change = ppr_change[mask]
print(f"  PPR change stats:")
print(f"    min={upper_change.min():.8f} max={upper_change.max():.8f}")
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    p{pct}: {np.percentile(upper_change, pct):.8f}")

# Positive change = convergence
pos = upper_change[upper_change > 0]
neg = upper_change[upper_change < 0]
zero = (upper_change == 0).sum()
print(f"\n  Positive (convergent): {len(pos):,} ({len(pos)/len(upper_change)*100:.1f}%)")
print(f"  Negative (divergent): {len(neg):,} ({len(neg)/len(upper_change)*100:.1f}%)")
print(f"  Zero: {zero:,} ({zero/len(upper_change)*100:.1f}%)")

# What about dist_change from -log?
dist_change = -np.log(ppr_2024 + 1e-25) + np.log(ppr_2012 + 1e-25)
# = log(ppr_2012 / ppr_2024)
upper_dc = dist_change[mask]
print(f"\n  -log dist change (= log ratio):")
print(f"    min={upper_dc.min():.4f} max={upper_dc.max():.4f}")
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    p{pct}: {np.percentile(upper_dc, pct):.4f}")

# How many pairs have PPR change > 0 AND were previously "distant"?
# This is the LBD signal: previously distant pairs becoming closer
far_mask = -np.log(ppr_2012 + 1e-25) > np.percentile(-np.log(ppr_2012[mask] + 1e-25), 75)
convergent_far = (ppr_change[mask] > 0) & far_mask
print(f"\n  Far pairs (p75+) with positive PPR change: {convergent_far.sum():,}")
