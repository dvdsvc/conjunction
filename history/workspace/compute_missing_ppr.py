"""Compute missing PPR matrices for 2012, 2021, 2024.
Uses the exact same method as build_temporal_v2.py (alpha=0.15, 30 iters, column-stochastic).
"""
import os, time
import numpy as np
import scipy.sparse as sp

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data/temporal_l3'
PPR_ALPHA = 0.15
PPR_ITERS = 30

def compute_ppr(adj_csr, alpha=PPR_ALPHA, n_iter=PPR_ITERS):
    """Full PPR matrix via power iteration. Returns dense (n, n) float32."""
    n = adj_csr.shape[0]
    deg = np.array(adj_csr.sum(axis=1)).flatten()
    deg_inv = np.zeros(n, dtype=np.float32)
    deg_inv[deg > 0] = 1.0 / deg[deg > 0]
    M = (sp.diags(deg_inv) @ adj_csr).T.tocsr()

    ppr = np.zeros((n, n), dtype=np.float32)
    batch_size = 64
    for bs in range(0, n, batch_size):
        be = min(bs + batch_size, n)
        B = be - bs
        r = np.zeros((n, B), dtype=np.float32)
        for i in range(B):
            r[bs + i, i] = 1.0
        for _ in range(n_iter):
            r = (1 - alpha) * (M @ r) + alpha * r
        ppr[:, bs:be] = r
        if (bs // batch_size) % 20 == 0:
            print(f"    batch {bs//batch_size+1}/{(n+batch_size-1)//batch_size}", flush=True)
    return ppr

years = [2012, 2021, 2024]
for year in years:
    out_path = f'{DATA_DIR}/ppr_v2_{year}.npy'
    if os.path.exists(out_path):
        print(f"ppr_v2_{year}.npy already exists, skipping.", flush=True)
        continue

    graph_path = f'{DATA_DIR}/graph_v2_{year}.npz'
    print(f"\nLoading graph_v2_{year}...", flush=True)
    adj = sp.load_npz(graph_path).tocsr()
    print(f"  {adj.shape[0]} nodes, {adj.nnz:,} edges", flush=True)

    print(f"Computing PPR for {year}...", flush=True)
    t0 = time.time()
    ppr = compute_ppr(adj)
    print(f"  Done in {time.time()-t0:.0f}s", flush=True)
    print(f"  Col sums (first 3): {ppr.sum(axis=0)[:3]}", flush=True)

    np.save(out_path, ppr.astype(np.float32))
    print(f"  Saved {out_path} ({ppr.nbytes / 1e6:.0f} MB)", flush=True)
    del ppr, adj

print("\nAll done.", flush=True)
