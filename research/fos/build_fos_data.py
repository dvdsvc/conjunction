"""
Build FOS data for v15 training.

Converts FOS Art_Benchmark data into the format expected by train_temporal_v15:
  - Cumulative graph snapshots (.npz)
  - PPR matrices (.npy)
  - SciBERT embeddings (.pt)
  - Concept ID ordering (.json)

Usage:
  python build_fos_data.py
"""
import os, time, json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

FOS_DIR = '/root/future-of-science/FOS_Benchmark/Art_Business'
OUT_DIR = '/root/autodl-tmp/openalex_data/fos_data'

YEARS = [2005, 2010, 2015, 2020, 2024]
PPR_ALPHA = 0.15
PPR_ITERS = 10


def build_graph_snapshots():
    """Build cumulative graph snapshots from all_edges.csv."""
    edges = pd.read_csv(
        f'{FOS_DIR}/edges/all_edges.csv', header=None,
        names=['src', 'dst', 'ts'])

    nodes = pd.read_pickle(f'{FOS_DIR}/node_embeddings/full_features.pkl')
    node_list = nodes['node_id'].tolist()
    id2idx = {nid: i for i, nid in enumerate(node_list)}
    n = len(node_list)

    print(f'Nodes: {n}, Total edges: {len(edges):,}')

    for year in YEARS:
        out_path = f'{OUT_DIR}/graph_fos_{year}.npz'
        if os.path.exists(out_path):
            adj = sp.load_npz(out_path)
            print(f'  {year}: exists ({int(adj.nnz/2):,} undirected edges)')
            continue

        mask = edges['ts'] <= year
        sub = edges[mask]
        rows = [id2idx[s] for s in sub['src']]
        cols = [id2idx[d] for d in sub['dst']]
        adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        adj = (adj + adj.T).tocsr()
        adj[adj > 0] = 1

        sp.save_npz(out_path, adj)
        deg = np.array(adj.sum(axis=1)).flatten()
        print(f'  {year}: {int(adj.nnz/2):,} undirected edges, '
              f'mean_deg={deg.mean():.1f}, median={np.median(deg):.0f}')


def build_scibert():
    """Extract SciBERT embeddings from FOS full_features.pkl."""
    out_path = f'{OUT_DIR}/concept_scibert.pt'
    ids_path = f'{OUT_DIR}/concept_ids.json'
    if os.path.exists(out_path):
        print('  concept_scibert.pt exists')
        return

    nodes = pd.read_pickle(f'{FOS_DIR}/node_embeddings/full_features.pkl')
    node_list = nodes['node_id'].tolist()

    embeddings = np.stack(nodes['embedding'].values)
    print(f'  Embeddings shape: {embeddings.shape}')

    torch.save(torch.from_numpy(embeddings).float(), out_path)
    with open(ids_path, 'w') as f:
        json.dump(node_list, f)

    print(f'  Saved concept_scibert.pt ({os.path.getsize(out_path)/1e6:.1f}MB)')
    print(f'  Saved concept_ids.json ({len(node_list)} concepts)')


def compute_ppr(adj_csr, alpha=PPR_ALPHA, n_iter=PPR_ITERS):
    """Full PPR matrix via power iteration."""
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
    return ppr


def build_ppr():
    """Compute PPR for each snapshot."""
    for year in YEARS:
        out_path = f'{OUT_DIR}/ppr_fos_{year}.npy'
        if os.path.exists(out_path):
            print(f'  PPR {year}: exists')
            continue

        adj = sp.load_npz(f'{OUT_DIR}/graph_fos_{year}.npz').tocsr()
        print(f'  PPR {year}: computing ({adj.shape[0]} nodes, {adj.nnz:,} edges)...')
        t0 = time.time()
        ppr = compute_ppr(adj)
        np.save(out_path, ppr)
        elapsed = time.time() - t0
        print(f'  PPR {year}: done in {elapsed:.0f}s '
              f'({os.path.getsize(out_path)/1e6:.0f}MB)')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('=== Building graph snapshots ===')
    build_graph_snapshots()

    print('\n=== Building SciBERT embeddings ===')
    build_scibert()

    print('\n=== Computing PPR ===')
    build_ppr()

    print('\nDone! Data in', OUT_DIR)


if __name__ == '__main__':
    main()
