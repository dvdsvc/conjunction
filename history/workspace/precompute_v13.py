"""
Precompute v13: Same sampling as v12, but labels = dist_t2 (endpoint distance).
v12's weighted BFS delta (first-hop-free) used for pair selection.
dist_t2 = weighted BFS distance in t2 graph (first-hop-free), unreachable = max+1.
"""
import os, time, json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
TEMP_DIR = f'{DATA_DIR}/temporal_l3'
OUT_DIR = f'{TEMP_DIR}/precomputed_v13'
MAX_MERGED = 256


def build_weighted_adj(adj_csr):
    deg = np.array(adj_csr.sum(axis=1)).flatten().astype(np.float64)
    adj_sym = (adj_csr + adj_csr.T).tocsr()
    adj_sym[adj_sym > 0] = 1
    rows, cols = adj_sym.nonzero()
    avg_deg = (deg[rows] + deg[cols]) / 2.0
    weights = np.log(avg_deg + 1).astype(np.float32)
    return sp.csr_matrix((weights, (rows, cols)), shape=adj_sym.shape)


def compute_weighted_bfs_first_hop_free(adj, out_path, label="dist"):
    """Weighted BFS with first-hop-free. Returns float32[n,n], unreachable=-1."""
    if os.path.exists(out_path):
        print(f"    {label} exists: {out_path}", flush=True)
        return np.load(out_path, mmap_mode='r')

    n = adj.shape[0]
    t0 = time.time()
    print(f"    Building weighted adj ({n} nodes)...", flush=True)
    w = build_weighted_adj(adj)
    print(f"    Computing shortest paths...", flush=True)
    dist = shortest_path(w, directed=False, method='D')
    dist -= 1  # first-hop free
    dist[dist == np.inf - 1] = -1
    dist = dist.astype(np.float32)
    np.save(out_path, dist)
    print(f"    {label} done: {time.time()-t0:.0f}s", flush=True)
    return dist


def fast_clustering_coefficients(adj_csr):
    adj_sym = (adj_csr + adj_csr.T)
    adj_sym[adj_sym > 0] = 1
    adj_sym = adj_sym.tocsr()
    n = adj_sym.shape[0]
    cc = np.zeros(n, dtype=np.float32)
    for u in range(n):
        neighbors = adj_sym[u].indices
        k = len(neighbors)
        if k < 2:
            continue
        neighbor_set = set(neighbors.tolist())
        triangles = 0
        for a in range(k):
            na = neighbors[a]
            for nb in adj_sym[na].indices:
                if nb in neighbor_set and nb != na:
                    triangles += 1
        cc[u] = triangles / (k * (k - 1))
    return cc


def compute_all_pair_features(u_arr, v_arr, ppr, adj_csr):
    n = adj_csr.shape[0]
    deg = np.array(adj_csr.sum(axis=1)).flatten().astype(np.float64)
    cc = fast_clustering_coefficients(adj_csr)
    A2 = (adj_csr @ adj_csr).tocsr()
    A2.setdiag(0)
    log_inv = np.zeros(n)
    mask = deg > 1
    log_inv[mask] = 1.0 / np.log(np.maximum(deg[mask], 2))
    AA_mat = (adj_csr @ sp.diags(log_inv) @ adj_csr.T).tocsr()
    AA_mat.setdiag(0)
    ppr_row_sum = np.array(ppr.sum(axis=1)).flatten()
    cn_all = np.array(A2[u_arr, v_arr]).flatten().astype(np.float64)
    aa_all = np.array(AA_mat[u_arr, v_arr]).flatten()
    deg_u = deg[u_arr]
    deg_v = deg[v_arr]
    union = deg_u + deg_v - cn_all
    jaccard = cn_all / np.maximum(union, 1)
    ppr_uv = ppr[u_arr, v_arr]
    ppr_vu = ppr[v_arr, u_arr]
    wcn = ppr_row_sum[u_arr] + ppr_row_sum[v_arr]
    cc_u = cc[u_arr]
    cc_v = cc[v_arr]
    avg_ppr_u = ppr_row_sum[u_arr] / np.maximum(deg_u, 1)
    avg_ppr_v = ppr_row_sum[v_arr] / np.maximum(deg_v, 1)
    feats = np.column_stack([
        ppr_uv, ppr_vu, cn_all, wcn, jaccard, aa_all,
        deg_u, deg_v, cc_u, cc_v,
        avg_ppr_u, avg_ppr_v,
        np.log1p(deg_u), np.log1p(deg_v),
        jaccard,
    ]).astype(np.float32)
    return feats


def precompute_two_hop_neighborhoods(adj_csr, max_nodes=MAX_MERGED):
    n = adj_csr.shape[0]
    deg = np.array(adj_csr.sum(axis=1)).flatten()
    A2 = (adj_csr @ adj_csr).tocsr()
    A2.setdiag(0)
    A2[A2 > 0] = 1
    neighborhoods = []
    for u in range(n):
        nbs_2hop = A2[u].indices.tolist()
        nbs_1hop = adj_csr[u].indices.tolist()
        all_nbs = set(nbs_1hop) | set(nbs_2hop)
        all_nbs.discard(u)
        if len(all_nbs) > max_nodes - 1:
            nb_arr = np.array(list(all_nbs), dtype=np.int32)
            degs = deg[nb_arr]
            top_idx = np.argsort(degs)[-(max_nodes - 1):]
            nb_arr = nb_arr[top_idx]
        else:
            nb_arr = np.array(sorted(all_nbs), dtype=np.int32)
        result = np.empty(len(nb_arr) + 1, dtype=np.int32)
        result[0] = u
        result[1:] = nb_arr
        neighborhoods.append(result)
    return neighborhoods


def build_merged_subgraph_fast(u, v, two_hop_nbs, deg, max_nodes=MAX_MERGED):
    nbs_u = two_hop_nbs[u]
    nbs_v = two_hop_nbs[v]
    merged = np.union1d(nbs_u, nbs_v)
    if len(merged) > max_nodes:
        degs = deg[merged]
        top_idx = np.argsort(degs)[-max_nodes:]
        merged = np.sort(merged[top_idx])
    node_to_local = {int(n): i for i, n in enumerate(merged)}
    return merged, node_to_local.get(int(u), 0), node_to_local.get(int(v), 0)


def process_split(split_name, out_fold, adj_t1, delta, dist_t2, ppr, deg, cc, rng):
    """Same sampling as v12 (delta-based), but labels = dist_t2."""
    print(f"\n  Processing {split_name}...", flush=True)

    # Filter pairs from upper triangle using delta (same as v12)
    print("    Filtering pairs (delta-based)...", flush=True)
    t0 = time.time()
    rows, cols = np.triu_indices(delta.shape[0], k=1)
    all_deltas_flat = delta[rows, cols]

    pos_mask = all_deltas_flat >= 2
    neg_mask = all_deltas_flat == 0

    n_pos = int(pos_mask.sum())
    n_neg_target = min(int(neg_mask.sum()), n_pos * 5)

    if n_pos == 0:
        print(f"    WARNING: no positive examples for {split_name}", flush=True)
        return False

    pos_rows, pos_cols = rows[pos_mask], cols[pos_mask]
    neg_all_idx = np.where(neg_mask)[0]
    rng.shuffle(neg_all_idx)
    neg_sel = neg_all_idx[:n_neg_target]
    neg_rows, neg_cols = rows[neg_sel], cols[neg_sel]

    all_u = np.concatenate([pos_rows, neg_rows])
    all_v = np.concatenate([pos_cols, neg_cols])

    # Labels = dist_t2 instead of delta
    pos_labels = dist_t2[pos_rows, pos_cols].copy()
    neg_labels = dist_t2[neg_rows, neg_cols].copy()
    # Set unreachable to large value
    reachable = dist_t2 > 0
    max_dist = float(dist_t2[reachable].max()) if reachable.any() else 10.0
    INF_DIST = max_dist + 1.0
    pos_labels[pos_labels < 0] = INF_DIST
    neg_labels[neg_labels < 0] = INF_DIST
    all_labels = np.concatenate([pos_labels, neg_labels]).astype(np.float32)

    perm = rng.permutation(len(all_u))
    all_u = all_u[perm]
    all_v = all_v[perm]
    all_labels = all_labels[perm]

    del delta, all_deltas_flat, rows, cols

    print(f"    pos={n_pos:,}, neg={n_neg_target:,}, total={len(all_u):,} "
          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"    INF_DIST={INF_DIST:.2f}", flush=True)

    n_total = len(all_u)

    # Group by source
    print("    Grouping pairs by source...", flush=True)
    t0 = time.time()
    src_order = np.argsort(all_u, kind='stable')
    sorted_src = all_u[src_order]
    change_pts = np.where(np.diff(sorted_src))[0] + 1
    groups = np.split(src_order, change_pts)
    keys = sorted_src[change_pts - 1]
    keys = np.concatenate([[sorted_src[0]], keys])
    pairs_by_src = {int(k): g.tolist() for k, g in zip(keys, groups)}
    print(f"      {len(pairs_by_src):,} source nodes ({time.time()-t0:.0f}s)", flush=True)

    # Pair features
    print("    Computing pair features (vectorized)...", flush=True)
    t0 = time.time()
    pair_feats = compute_all_pair_features(all_u, all_v, ppr, adj_t1)
    print(f"      Done: {time.time()-t0:.0f}s", flush=True)

    # 2-hop neighborhoods
    print("    Precomputing 2-hop neighborhoods...", flush=True)
    t0 = time.time()
    two_hop_nbs = precompute_two_hop_neighborhoods(adj_t1, MAX_MERGED)
    print(f"      Done: {time.time()-t0:.0f}s", flush=True)

    # Merged subgraph indices
    print(f"    Building merged subgraph indices ({n_total:,})...", flush=True)
    deg_arr = np.array(adj_t1.sum(axis=1)).flatten().astype(np.float64)
    merged_nodes = np.zeros((n_total, MAX_MERGED), dtype=np.int32)
    u_local_arr = np.zeros(n_total, dtype=np.int32)
    v_local_arr = np.zeros(n_total, dtype=np.int32)

    t0 = time.time()
    for i in range(n_total):
        pu, pv = int(all_u[i]), int(all_v[i])
        nodes, u_loc, v_loc = build_merged_subgraph_fast(
            pu, pv, two_hop_nbs, deg_arr, MAX_MERGED)
        nn = len(nodes)
        merged_nodes[i, :nn] = nodes
        u_local_arr[i] = u_loc
        v_local_arr[i] = v_loc
        if (i + 1) % 500000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_total - i - 1)
            print(f"      {i+1:,}/{n_total:,} ({elapsed/60:.1f}min, "
                  f"ETA {eta/60:.1f}min)", flush=True)

    print(f"      Done: {(time.time()-t0)/60:.1f}min", flush=True)

    # Save (labels stored as 'deltas' for compatibility)
    np.save(f'{out_fold}/{split_name}_pair_feats.npy', pair_feats)
    np.save(f'{out_fold}/{split_name}_merged_nodes.npy', merged_nodes)
    np.save(f'{out_fold}/{split_name}_deltas.npy', all_labels)
    np.save(f'{out_fold}/{split_name}_u_local.npy', u_local_arr)
    np.save(f'{out_fold}/{split_name}_v_local.npy', v_local_arr)

    print("    Saving pairs_by_src JSON...", flush=True)
    pairs_by_src_np = {}
    for k, idxs in pairs_by_src.items():
        pairs_by_src_np[str(k)] = [[int(i), int(all_v[i])] for i in idxs]
    with open(f'{out_fold}/{split_name}_pairs_by_src.json', 'w') as f:
        json.dump(pairs_by_src_np, f)

    total_gb = (pair_feats.nbytes + merged_nodes.nbytes +
                all_labels.nbytes + u_local_arr.nbytes +
                v_local_arr.nbytes) / 1e9
    print(f"      Saved: {total_gb:.2f}GB", flush=True)
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=None)
    args = parser.parse_args()

    with open(f'{DATA_DIR}/concept_ids_l3.json') as f:
        concept_ids = json.load(f)
    print(f"Concepts: {len(concept_ids)}", flush=True)

    folds = [
        (0, 2005, 2015, 2015, 2024),
        (1, 2017, 2024, 2017, 2021),
    ]
    if args.fold is not None:
        folds = [f for f in folds if f[0] == args.fold]

    for fold, train_t1, train_t2, val_t1, val_t2 in folds:
        print(f"\n{'='*60}", flush=True)
        print(f"  Fold {fold}: train {train_t1}→{train_t2}, val {val_t1}→{val_t2}",
              flush=True)

        out_fold = f'{OUT_DIR}/fold{fold}'
        if os.path.exists(f'{out_fold}/meta.json'):
            print(f"  Already exists, skipping.", flush=True)
            continue
        os.makedirs(out_fold, exist_ok=True)

        rng = np.random.RandomState(42 + fold)

        # Train split
        print("  Loading train graphs...", flush=True)
        adj_train_t1 = sp.load_npz(f'{TEMP_DIR}/graph_v2_{train_t1}.npz').tocsr()
        adj_train_t2 = sp.load_npz(f'{TEMP_DIR}/graph_v2_{train_t2}.npz').tocsr()
        ppr_train = np.load(f'{TEMP_DIR}/ppr_v2_{train_t1}.npy', mmap_mode='r')

        print("  Computing delta (weighted BFS, first-hop-free)...", flush=True)
        delta_train = compute_weighted_bfs_first_hop_free(
            adj_train_t1, f'{out_fold}/train_bfs_delta.npy', "delta_t1")
        delta_t2_train = compute_weighted_bfs_first_hop_free(
            adj_train_t2, f'{out_fold}/train_dist_t2.npy', "dist_t2")

        # delta = dist_t1 - dist_t2 (collapse measure)
        print("  Computing delta = dist_t1 - dist_t2...", flush=True)
        mask = delta_train > 0
        delta_combined = np.zeros_like(delta_train)
        delta_combined[mask] = delta_train[mask] - delta_t2_train[mask]

        deg_train = np.array(adj_train_t1.sum(axis=1)).flatten().astype(np.int32)
        print("  Computing clustering coefficients...", flush=True)
        t0 = time.time()
        cc_train = fast_clustering_coefficients(adj_train_t1)
        print(f"    Done: {time.time()-t0:.0f}s", flush=True)

        train_ok = process_split('train', out_fold, adj_train_t1,
                                 delta_combined, delta_t2_train,
                                 ppr_train, deg_train, cc_train, rng)
        del adj_train_t1, adj_train_t2, ppr_train, deg_train, cc_train
        del delta_train, delta_t2_train, delta_combined

        if not train_ok:
            continue

        # Val split
        print("  Loading val graphs...", flush=True)
        adj_val_t1 = sp.load_npz(f'{TEMP_DIR}/graph_v2_{val_t1}.npz').tocsr()
        adj_val_t2 = sp.load_npz(f'{TEMP_DIR}/graph_v2_{val_t2}.npz').tocsr()
        ppr_val = np.load(f'{TEMP_DIR}/ppr_v2_{val_t1}.npy', mmap_mode='r')

        print("  Computing val delta...", flush=True)
        delta_val_t1 = compute_weighted_bfs_first_hop_free(
            adj_val_t1, f'{out_fold}/val_bfs_delta.npy', "delta_t1")
        delta_val_t2 = compute_weighted_bfs_first_hop_free(
            adj_val_t2, f'{out_fold}/val_dist_t2.npy', "dist_t2")

        mask = delta_val_t1 > 0
        delta_val_combined = np.zeros_like(delta_val_t1)
        delta_val_combined[mask] = delta_val_t1[mask] - delta_val_t2[mask]

        deg_val = np.array(adj_val_t1.sum(axis=1)).flatten().astype(np.int32)
        print("  Computing val clustering coefficients...", flush=True)
        t0 = time.time()
        cc_val = fast_clustering_coefficients(adj_val_t1)
        print(f"    Done: {time.time()-t0:.0f}s", flush=True)

        val_ok = process_split('val', out_fold, adj_val_t1,
                               delta_val_combined, delta_val_t2,
                               ppr_val, deg_val, cc_val, rng)
        del adj_val_t1, adj_val_t2, ppr_val, deg_val, cc_val
        del delta_val_t1, delta_val_t2, delta_val_combined

        if val_ok:
            meta = {
                'fold': fold,
                'train_window': f'{train_t1}→{train_t2}',
                'val_window': f'{val_t1}→{val_t2}',
                'MAX_MERGED': MAX_MERGED,
                'pair_feat_dim': 15,
                'label': 'dist_t2 (endpoint distance, first-hop-free)',
                'sampling': 'delta >= 2 (pos), delta == 0 (neg), 5x ratio',
            }
            with open(f'{out_fold}/meta.json', 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"\n  Fold {fold} done.", flush=True)


if __name__ == '__main__':
    main()
