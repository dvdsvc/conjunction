"""Debug: compare v5 training features vs eval features for same (u,v) pair."""
import time
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
    dist_matrix = np.load(f'{DATA_DIR}/dist_matrix.npy')
    train_pairs, val_pairs = _extract_val_pairs(DATA_DIR)

    # Load v5 training data
    train_nids = np.load(f'{DATA_DIR}/train_nids_v5.npy')
    val_nids = np.load(f'{DATA_DIR}/val_nids_v5.npy')

    # Load eval data
    eval_nids = np.load(f'{DATA_DIR}/eval_subgraph_nids.npy', mmap_mode='r')
    eval_cands_len = np.load(f'{DATA_DIR}/eval_cands_len.npy')

    # Check: find first val pair (u, v) in both val_nids_v5 and eval_subgraph_nids
    for ai in range(3):
        u, v = val_pairs[ai]
        print(f"\n=== anchor {ai}: u={u}, v={v} ===")

        # Find positive in val_nids_v5 (first candidate should be positive)
        val_nid = val_nids[ai, 0]  # positive subgraph node_ids
        print(f"  val_nids_v5[0] = {val_nid[:8]}...")
        print(f"  val_nids_v5[0] first two: u={val_nid[0]}, v={val_nid[1]}")
        # Check: is val_nids_v5[0,0]==u and val_nids_v5[0,1]==v?
        print(f"  matches (u,v): {val_nid[0]==u and val_nid[1]==v}")

        # Find positive in eval data
        NC = int(eval_cands_len[ai])
        eval_nid_ai = eval_nids[ai]
        pos_idx = -1
        for j in range(NC):
            if eval_nid_ai[j, 0] == u and eval_nid_ai[j, 1] == v:
                pos_idx = j
                break
        print(f"  eval pos_idx = {pos_idx}/{NC}")

        if pos_idx >= 0:
            eval_pos_nid = eval_nid_ai[pos_idx]
            print(f"  eval_nids[pos] = {eval_pos_nid[:8]}...")
            # Compare node_ids
            same = np.array_equal(val_nid, eval_pos_nid)
            print(f"  node_ids identical: {same}")
            if not same:
                diff_mask = val_nid != eval_pos_nid
                print(f"  diff positions: {np.where(diff_mask)[0]}")
                print(f"  val_nids_v5 values: {val_nid[diff_mask]}")
                print(f"  eval_nids values: {eval_pos_nid[diff_mask]}")

    # Also check: what does GPU reconstruction produce vs Numba?
    print("\n\n=== Feature reconstruction comparison ===")
    import torch
    from numba import njit

    @njit(cache=True)
    def _reconstruct_features(nids, u, v, dist_matrix, dist_u, dist_v, out):
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

    def reconstruct_features_gpu(nids, dist_matrix_gpu):
        """Same as train_pencil_v5.py"""
        B, N = nids.shape
        device = nids.device
        EMBED_DIM = 68
        feat = torch.zeros(B, N, EMBED_DIM, device=device)
        pos_idx = torch.arange(N, device=device)
        feat[:, pos_idx, pos_idx] = 1.0
        pair_dist = dist_matrix_gpu[nids[:, :, None], nids[:, None, :]]
        feat[:, :, MAX_NODES:MAX_NODES + N] = (pair_dist == 1.0).float()
        feat[:, :, 2 * MAX_NODES] = (feat[:, :, :MAX_NODES].abs().sum(dim=2) > 0).float()
        u_ids = nids[:, 0]
        v_ids = nids[:, 1]
        feat[:, :, 2 * MAX_NODES + 1] = dist_matrix_gpu[u_ids[:, None], nids].clamp(0, DIST_BINS) / DIST_BINS
        feat[:, :, 2 * MAX_NODES + 2] = dist_matrix_gpu[v_ids[:, None], nids].clamp(0, DIST_BINS) / DIST_BINS
        return feat

    dist_matrix_gpu = torch.from_numpy(dist_matrix).cuda()

    # Use first val pair
    u, v = val_pairs[0]
    nids_np = eval_nids[0, 0:1].copy()  # positive subgraph
    print(f"  u={u}, v={v}")

    # Numba reconstruction
    numba_out = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
    _reconstruct_features(nids_np[0], u, v, dist_matrix, dist_matrix[u], dist_matrix[v], numba_out)

    # GPU reconstruction
    nids_gpu = torch.from_numpy(nids_np).cuda()
    gpu_out = reconstruct_features_gpu(nids_gpu, dist_matrix_gpu)[0].cpu().numpy()

    print(f"  numba_out shape: {numba_out.shape}")
    print(f"  gpu_out shape: {gpu_out.shape}")
    print(f"  numba_out sum: {numba_out.sum():.4f}")
    print(f"  gpu_out sum: {gpu_out.sum():.4f}")
    print(f"  equal: {np.allclose(numba_out, gpu_out, atol=1e-6)}")

    if not np.allclose(numba_out, gpu_out, atol=1e-6):
        diff = np.abs(numba_out - gpu_out)
        print(f"  max diff: {diff.max():.6f}")
        for col in range(EMBED_DIM):
            col_diff = diff[:, col].max()
            if col_diff > 1e-6:
                print(f"  col {col}: max_diff={col_diff:.6f}")
                # Show first few different rows
                row_diffs = diff[:, col]
                for row in range(MAX_NODES):
                    if row_diffs[row] > 1e-6:
                        print(f"    row {row}: numba={numba_out[row, col]:.6f}, gpu={gpu_out[row, col]:.6f}")

    # Check training data: val_nids_v5[0] positive
    print("\n=== val_nids_v5[0] positive feature check ===")
    nids_train = val_nids[0, 0:1].copy()
    train_numba = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
    _reconstruct_features(nids_train[0], u, v, dist_matrix, dist_matrix[u], dist_matrix[v], train_numba)

    train_gpu = reconstruct_features_gpu(torch.from_numpy(nids_train).cuda(), dist_matrix_gpu)[0].cpu().numpy()
    print(f"  train numba vs GPU equal: {np.allclose(train_numba, train_gpu, atol=1e-6)}")
    print(f"  eval vs train numba equal: {np.allclose(numba_out, train_numba, atol=1e-6)}")
    if not np.allclose(numba_out, train_numba, atol=1e-6):
        diff = np.abs(numba_out - train_numba)
        print(f"  max diff: {diff.max():.6f}")
        diff_mask = diff > 1e-6
        rows, cols = np.where(diff_mask)
        for r, c in zip(rows[:10], cols[:10]):
            print(f"    row={r}, col={c}: eval={numba_out[r,c]:.6f}, train={train_numba[r,c]:.6f}")
