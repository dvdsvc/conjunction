"""
Precompute v14: Gravity model with 2-hop candidate pool.

Changes from v13 (architecture only, all other details preserved):
  1. Candidate pool: t1 graph 2-hop neighbors (replaces delta≥2 / delta==0 sampling)
  2. Remove t2 unreachable pairs (no max+1 sentinel — just skip them)
  3. Add 6 gravity features to existing 15 → 21-dim pair features

Gravity features (appended as cols 15–20):
  15. pure_cn          — common neighbors with degree < median (de-hubbed)
  16. weighted_gravity  — Σ 1/log(deg(w)) for pure common neighbors
  17. neighbor_cohesion — triangle density among common neighbors
  18. second_order_gravity — |N2(u) ∩ N2(v)| / |N2(u) ∪ N2(v)|
  19. subgraph_connectivity — edge count / node count in N1(u) ∪ N1(v)
  20. resistance_distance — log(1 + parallel conductance of direct + 2-hop paths)

Label: dist_t2 (weighted BFS first-hop-free), unreachable pairs removed.

Usage:
  python precompute_v14.py --fold 0
  python precompute_v14.py          # both folds
"""
import os, time, json, argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
TEMP_DIR = f'{DATA_DIR}/temporal_l3'
OUT_DIR = f'{TEMP_DIR}/precomputed_v14'
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


# ── Original 15-dim features (identical to v13) ──────────────────────

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


# ── 6-dim gravity features (vectorized, batched) ──────────────────────

def compute_gravity_features(u_arr, v_arr, adj_csr, deg):
    """Compute 6 gravity features. Fully vectorized, batched to avoid OOM."""
    n = adj_csr.shape[0]
    adj_sym = (adj_csr + adj_csr.T).tocsr()
    adj_sym[adj_sym > 0] = 1

    A2 = (adj_sym @ adj_sym).tocsr()
    A2.setdiag(0)
    A3 = (A2 @ adj_sym).tocsr()
    deg_median = np.median(deg[deg > 0])
    is_pure = (deg < deg_median).astype(np.float64)
    A2_pure = (adj_sym @ sp.diags(is_pure) @ adj_sym.T).tocsr()
    A2_pure.setdiag(0)
    log_inv = np.zeros(n, dtype=np.float64)
    mask = deg > 1
    log_inv[mask] = 1.0 / np.log(np.maximum(deg[mask], 2))
    log_inv_pure = log_inv * is_pure
    AA_pure = (adj_sym @ sp.diags(log_inv_pure) @ adj_sym.T).tocsr()
    AA_pure.setdiag(0)
    A2_bin = A2.copy()
    A2_bin[A2_bin > 0] = 1
    wA2 = (adj_csr @ adj_csr).tocsr()
    wA2.setdiag(0)

    # Features 1-3: vectorized via matrix lookup (fast, no batching needed)
    pure_cn = np.array(A2_pure[u_arr, v_arr]).flatten().astype(np.float32)
    weighted_gravity = np.array(AA_pure[u_arr, v_arr]).flatten().astype(np.float32)
    cn_all = np.array(A2[u_arr, v_arr]).flatten().astype(np.float64)
    triangles = np.array(A3[u_arr, v_arr]).flatten().astype(np.float64)
    neighbor_cohesion = (triangles / np.maximum(cn_all * (cn_all - 1), 1)).astype(np.float32)
    neighbor_cohesion[cn_all < 2] = 0.0

    # Features 4-6: batched to avoid large [N, n] sparse matrices
    n_pairs = len(u_arr)
    second_order = np.zeros(n_pairs, dtype=np.float32)
    subgraph_conn = np.zeros(n_pairs, dtype=np.float32)

    # Feature 6: resistance_distance (vectorized, no batching needed)
    w_direct = np.array(adj_csr[u_arr, v_arr]).flatten().astype(np.float64)
    w_direct_rev = np.array(adj_csr[v_arr, u_arr]).flatten().astype(np.float64)
    w_direct = np.maximum(w_direct, w_direct_rev)
    parallel_weight = np.array(wA2[u_arr, v_arr]).flatten().astype(np.float64)
    resistance_distance = np.log1p(w_direct + parallel_weight / 2.0).astype(np.float32)

    # Batch loop for features 4 and 5
    BATCH = 50000
    for start in range(0, n_pairs, BATCH):
        end = min(start + BATCH, n_pairs)
        u_b = u_arr[start:end]
        v_b = v_arr[start:end]
        cn_b = cn_all[start:end]
        deg_u = deg[u_b]
        deg_v = deg[v_b]

        # 4. second_order_gravity
        row_u = A2_bin[u_b]
        row_v = A2_bin[v_b]
        n2_inter = np.array(row_u.multiply(row_v).sum(axis=1)).flatten()
        n2_size = np.array(A2_bin.sum(axis=1)).flatten()
        n2_union = n2_size[u_b] + n2_size[v_b] - n2_inter
        second_order[start:end] = (n2_inter / np.maximum(n2_union, 1)).astype(np.float32)

        # 5. subgraph_connectivity
        union_sizes = deg_u + deg_v - cn_b
        union_indicator = row_u + row_v
        union_indicator[union_indicator > 0] = 1
        # No need to remove u,v: A2 diagonal is 0, so they're not in the union
        deg_col = sp.csr_matrix(deg.reshape(-1, 1))
        deg_sum = (union_indicator @ deg_col).toarray().flatten()
        is_nb = np.array(adj_sym[u_b, v_b]).flatten()
        cross_u = deg_u - 1.0 - is_nb
        cross_v = deg_v - 1.0 - is_nb
        n_edges = (deg_sum - cross_u - cross_v) / 2.0
        subgraph_conn[start:end] = (n_edges / np.maximum(union_sizes, 1)).astype(np.float32)

    return np.column_stack([
        pure_cn, weighted_gravity, neighbor_cohesion,
        second_order, subgraph_conn, resistance_distance,
    ])


# ── Subgraph building (identical to v13) ──────────────────────────────

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


# ── Process split ─────────────────────────────────────────────────────

def process_split(split_name, out_fold, adj_t1, dist_t2, ppr, deg, cc, rng):
    """v14: candidate pool from 2-hop neighbors, filter unreachable, 21-dim features."""
    print(f"\n  Processing {split_name}...", flush=True)

    # Step 1: Build 2-hop candidate pool (replaces delta-based sampling)
    print("    Building 2-hop candidate pool...", flush=True)
    t0 = time.time()
    n = adj_t1.shape[0]
    A2_t1 = (adj_t1 @ adj_t1).tocsr()
    A2_t1.setdiag(0)
    A2_t1[A2_t1 > 0] = 1

    all_u_list = []
    all_v_list = []
    all_labels_list = []
    pairs_by_src = {}
    pair_idx = 0

    for u in range(n):
        nbs_1hop = set(adj_t1[u].indices.tolist())
        nbs_2hop = set(A2_t1[u].indices.tolist())
        all_nbs = (nbs_1hop | nbs_2hop) - {u}

        cands = []
        for v in all_nbs:
            d = dist_t2[u, v]
            if d >= 0:  # reachable in t2
                cands.append((pair_idx, int(v), float(d)))
                pair_idx += 1

        if cands:
            start_idx = len(all_u_list)
            for _, v, d in cands:
                all_u_list.append(u)
                all_v_list.append(v)
                all_labels_list.append(d)
            pairs_by_src[u] = [[start_idx + j, cands[j][1]] for j in range(len(cands))]

    all_u = np.array(all_u_list, dtype=np.int64)
    all_v = np.array(all_v_list, dtype=np.int64)
    all_labels = np.array(all_labels_list, dtype=np.float32)
    del A2_t1

    n_total = len(all_u)
    n_anchors = len(pairs_by_src)
    print(f"      {n_anchors:,} anchors, {n_total:,} pairs ({time.time()-t0:.0f}s)", flush=True)

    if n_total == 0:
        print(f"    WARNING: no valid pairs for {split_name}", flush=True)
        return False

    # Label distribution
    print(f"    dist_t2 stats: min={all_labels.min():.3f}, max={all_labels.max():.3f}, "
          f"mean={all_labels.mean():.3f}", flush=True)
    for thresh in [0.5, 1.0, 2.0, 5.0, 10.0]:
        pct = 100.0 * (all_labels <= thresh).sum() / n_total
        print(f"      dist_t2 <= {thresh}: {pct:.1f}%", flush=True)

    # Step 2: Original 15-dim pair features (same as v13)
    print("    Computing pair features (15-dim, same as v13)...", flush=True)
    t0 = time.time()
    pair_feats_15 = compute_all_pair_features(all_u, all_v, ppr, adj_t1)
    print(f"      Done: {time.time()-t0:.0f}s", flush=True)

    # Step 3: 6-dim gravity features (new)
    print("    Computing gravity features (6-dim)...", flush=True)
    t0 = time.time()
    pair_feats_6 = compute_gravity_features(all_u, all_v, adj_t1,
                                             deg.astype(np.float64))
    print(f"      Done: {time.time()-t0:.0f}s", flush=True)

    # Step 4: Concatenate → [N, 21]
    pair_feats = np.column_stack([pair_feats_15, pair_feats_6]).astype(np.float32)
    del pair_feats_15, pair_feats_6
    print(f"    Pair features: {pair_feats.shape}", flush=True)

    # Step 5: 2-hop neighborhoods (same as v13)
    print("    Precomputing 2-hop neighborhoods...", flush=True)
    t0 = time.time()
    two_hop_nbs = precompute_two_hop_neighborhoods(adj_t1, MAX_MERGED)
    print(f"      Done: {time.time()-t0:.0f}s", flush=True)

    # Step 6: Merged subgraph indices (same as v13)
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

    # Step 7: Save (same format as v13, labels stored as 'deltas' for compatibility)
    np.save(f'{out_fold}/{split_name}_pair_feats.npy', pair_feats)
    np.save(f'{out_fold}/{split_name}_merged_nodes.npy', merged_nodes)
    np.save(f'{out_fold}/{split_name}_deltas.npy', all_labels)
    np.save(f'{out_fold}/{split_name}_u_local.npy', u_local_arr)
    np.save(f'{out_fold}/{split_name}_v_local.npy', v_local_arr)

    print("    Saving pairs_by_src JSON...", flush=True)
    pairs_by_src_np = {}
    for k, idxs in pairs_by_src.items():
        pairs_by_src_np[str(k)] = idxs  # already [[pair_idx, v_global], ...]
    with open(f'{out_fold}/{split_name}_pairs_by_src.json', 'w') as f:
        json.dump(pairs_by_src_np, f)

    total_gb = (pair_feats.nbytes + merged_nodes.nbytes +
                all_labels.nbytes + u_local_arr.nbytes +
                v_local_arr.nbytes) / 1e9
    print(f"      Saved: {total_gb:.2f}GB", flush=True)
    return True


# ── Main (rolling windows) ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=None)
    args = parser.parse_args()

    with open(f'{DATA_DIR}/concept_ids_l3.json') as f:
        concept_ids = json.load(f)
    print(f"Concepts: {len(concept_ids)}", flush=True)

    # Rolling windows: each fold = one consecutive time step
    # Fold 0-3: training windows, Fold 4: held-out test window
    rolling_folds = [
        (0, 2005, 2012),
        (1, 2012, 2015),
        (2, 2015, 2017),
        (3, 2017, 2021),
        (4, 2021, 2024),
    ]
    if args.fold is not None:
        rolling_folds = [f for f in rolling_folds if f[0] == args.fold]

    for fold, t1, t2 in rolling_folds:
        print(f"\n{'='*60}", flush=True)
        print(f"  Fold {fold}: {t1}→{t2}", flush=True)

        out_fold = f'{OUT_DIR}/fold{fold}'
        if os.path.exists(f'{out_fold}/meta.json'):
            print(f"  Already exists, skipping.", flush=True)
            continue
        os.makedirs(out_fold, exist_ok=True)

        rng = np.random.RandomState(42 + fold)

        print(f"  Loading graphs {t1} and {t2}...", flush=True)
        adj_t1 = sp.load_npz(f'{TEMP_DIR}/graph_v2_{t1}.npz').tocsr()
        adj_t2 = sp.load_npz(f'{TEMP_DIR}/graph_v2_{t2}.npz').tocsr()
        ppr = np.load(f'{TEMP_DIR}/ppr_v2_{t1}.npy', mmap_mode='r')

        print(f"  Computing dist_t2 (weighted BFS on {t2} graph)...", flush=True)
        dist_t2 = compute_weighted_bfs_first_hop_free(
            adj_t2, f'{out_fold}/dist_t2.npy', "dist_t2")

        deg = np.array(adj_t1.sum(axis=1)).flatten().astype(np.int32)
        print("  Computing clustering coefficients...", flush=True)
        t0 = time.time()
        cc = fast_clustering_coefficients(adj_t1)
        print(f"    Done: {time.time()-t0:.0f}s", flush=True)

        ok = process_split('train', out_fold, adj_t1, dist_t2, ppr,
                           deg, cc, rng)
        del adj_t1, adj_t2, ppr, deg, cc, dist_t2

        if ok:
            meta = {
                'fold': fold,
                'window': f'{t1}→{t2}',
                'MAX_MERGED': MAX_MERGED,
                'pair_feat_dim': 21,
                'label': 'dist_t2 (endpoint distance, first-hop-free, unreachable removed)',
                'sampling': 't1 2-hop neighbors, filter t2 unreachable',
                'gravity_features': [
                    'pure_cn', 'weighted_gravity', 'neighbor_cohesion',
                    'second_order_gravity', 'subgraph_connectivity', 'resistance_distance'
                ],
            }
            with open(f'{out_fold}/meta.json', 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"\n  Fold {fold} ({t1}→{t2}) done.", flush=True)


if __name__ == '__main__':
    main()
