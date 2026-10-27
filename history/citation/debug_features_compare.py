"""Directly compare v3 precomputed features vs v5 reconstructed features."""
import numpy as np
import re

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32
EMBED_DIM = 68
DIST_BINS = 10

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

if __name__ == '__main__':
    import scipy.sparse as sp

    # Load v3 precomputed data
    print("Loading v3 precomputed data...")
    v3_data = np.load(f'{DATA_DIR}/train_precomp_v3.npz')
    v3_inp = v3_data['inp']  # float16
    v3_nids = v3_data['node_ids']
    v3_labels = v3_data['label']
    print(f"  v3 inp shape: {v3_inp.shape}, nids shape: {v3_nids.shape}")
    print(f"  v3 labels sum: {v3_labels.sum():.0f} (positive count)")

    # Load v5 precomputed data
    print("Loading v5 precomputed data...")
    v5_train_nids = np.load(f'{DATA_DIR}/train_nids_v5.npy')
    print(f"  v5 train_nids shape: {v5_train_nids.shape}")

    dist_matrix = np.load(f'{DATA_DIR}/dist_matrix.npy')
    train_pairs, val_pairs = _extract_val_pairs(DATA_DIR)

    # Compare first 3 anchors (positive subgraph = index 0 in each group)
    G_v3 = 6  # 1+5 for v3
    G_v5 = 201  # 1+200 for v5

    for ai in range(3):
        u, v = train_pairs[ai]
        print(f"\n=== train anchor {ai}: u={u}, v={v} ===")

        # v3 positive: index ai*G_v3
        v3_feat = v3_inp[ai * G_v3].astype(np.float32)
        v3_nid = v3_nids[ai * G_v3]
        print(f"  v3 nids first 5: {v3_nid[:5]}")
        print(f"  v3 nids[0]={v3_nid[0]}, nids[1]={v3_nid[1]}")
        print(f"  v3 col 65 (dist_u) row 0: {v3_feat[0, 65]:.4f}, row 1: {v3_feat[1, 65]:.4f}")
        print(f"  v3 col 66 (dist_v) row 0: {v3_feat[0, 66]:.4f}, row 1: {v3_feat[1, 66]:.4f}")

        # v5 positive: index ai (first in the 1+200 group)
        v5_nid = v5_train_nids[ai, 0]
        print(f"  v5 nids first 5: {v5_nid[:5]}")
        print(f"  v5 nids[0]={v5_nid[0]}, nids[1]={v5_nid[1]}")

        # Reconstruct v5 features using distance matrix (same as GPU training)
        from numba import njit
        @njit(cache=True)
        def _reconstruct(nids, u, v, dist_matrix, dist_u, dist_v, out):
            n = 0
            for i in range(MAX_NODES):
                if nids[i] != 0 or i == 0: n = i + 1
            for i in range(MAX_NODES):
                for j in range(EMBED_DIM): out[i, j] = 0.0
            for i in range(n):
                out[i, i] = 1.0
                out[i, 2 * MAX_NODES] = 1.0
                out[i, 2 * MAX_NODES + 1] = min(dist_u[nids[i]], float(DIST_BINS)) / float(DIST_BINS)
                out[i, 2 * MAX_NODES + 2] = min(dist_v[nids[i]], float(DIST_BINS)) / float(DIST_BINS)
                for j in range(n):
                    if dist_matrix[nids[i], nids[j]] == 1:
                        out[i, MAX_NODES + j] = 1.0

        # Use v5's u,v (from nids[0], nids[1])
        v5_u, v5_v = int(v5_nid[0]), int(v5_nid[1])
        tmp_out = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
        _reconstruct(v5_nid, v5_u, v5_v, dist_matrix,
                     dist_matrix[v5_u], dist_matrix[v5_v], tmp_out)
        v5_feat = tmp_out.copy()

        print(f"  v5 reconstructed col 65 row 0: {v5_feat[0, 65]:.4f}, row 1: {v5_feat[1, 65]:.4f}")
        print(f"  v5 reconstructed col 66 row 0: {v5_feat[0, 66]:.4f}, row 1: {v5_feat[1, 66]:.4f}")

        # Compare node_ids
        same_nids = np.array_equal(v3_nid, v5_nid)
        print(f"  node_ids same: {same_nids}")

        # Compare features
        if same_nids:
            diff = np.abs(v3_feat - v5_feat)
            max_diff = diff.max()
            print(f"  feature max diff: {max_diff:.6f}")
            if max_diff > 1e-3:
                # Find differences
                rows, cols = np.where(diff > 1e-3)
                for r, c in zip(rows[:20], cols[:20]):
                    col_name = {65: 'dist_u', 66: 'dist_v'}.get(c, f'col{c}')
                    print(f"    row={r}, {col_name}: v3={v3_feat[r,c]:.4f}, v5={v5_feat[r,c]:.4f}")
        else:
            # Find where they differ
            diff_mask = v3_nid != v5_nid
            print(f"  nids differ at {diff_mask.sum()} positions")
            for pos in np.where(diff_mask)[0][:5]:
                print(f"    pos {pos}: v3={v3_nid[pos]}, v5={v5_nid[pos]}")

    # Check distance matrix properties
    print(f"\n=== Distance matrix properties ===")
    print(f"  shape: {dist_matrix.shape}")
    print(f"  dtype: {dist_matrix.dtype}")
    print(f"  min: {dist_matrix.min()}, max: {dist_matrix.max()}")
    print(f"  unique values: {np.unique(dist_matrix)[:20]}")
    # Check if symmetric
    u, v = train_pairs[0]
    print(f"  dist[{u},{v}] = {dist_matrix[u,v]}, dist[{v},{u}] = {dist_matrix[v,u]}")
    # Check diagonal
    print(f"  dist[{u},{u}] = {dist_matrix[u,u]}")
