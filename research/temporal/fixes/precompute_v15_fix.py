"""
Precompute v15 fix: preserve the v15 task definition while repairing pair/group
index consistency and endpoint retention.

Changes from v14:
  1. Candidate pool filtered by t1 features (no label leakage):
     - Adamic-Adar > 0 at t1
  2. Label: relevance grades instead of continuous dist_t2
     - rel=4: dist_t2 ≤ 1.0 (near direct neighbor)
     - rel=3: dist_t2 ∈ (1.0, 2.0]
     - rel=2: dist_t2 ∈ (2.0, 5.0]
     - rel=1: dist_t2 ∈ (5.0, 10.0]
     - rel=0: dist_t2 > 10 or unreachable
  3. Features: 21-dim (same as v14, gravity features computed before filtering)

Usage:
  python precompute_v15_fix.py --fold 0
  python precompute_v15_fix.py
"""
import os, time, json, argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
TEMP_DIR = f'{DATA_DIR}/temporal_l3'
OUT_DIR = f'{TEMP_DIR}/precomputed_v15_fix'
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
    if os.path.exists(out_path):
        print(f"    {label} exists: {out_path}", flush=True)
        return np.load(out_path, mmap_mode='r')
    n = adj.shape[0]
    t0 = time.time()
    print(f"    Building weighted adj ({n} nodes)...", flush=True)
    w = build_weighted_adj(adj)
    print(f"    Computing shortest paths...", flush=True)
    dist = shortest_path(w, directed=False, method='D')
    dist -= 1
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


# ── Original 15-dim features ──────────────────────────────────────────

def compute_all_pair_features(u_arr, v_arr, ppr, adj_csr, cc):
    n = adj_csr.shape[0]
    deg = np.array(adj_csr.sum(axis=1)).flatten().astype(np.float64)
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

    pure_cn = np.array(A2_pure[u_arr, v_arr]).flatten().astype(np.float32)
    weighted_gravity = np.array(AA_pure[u_arr, v_arr]).flatten().astype(np.float32)
    cn_all = np.array(A2[u_arr, v_arr]).flatten().astype(np.float64)
    triangles = np.array(A3[u_arr, v_arr]).flatten().astype(np.float64)
    neighbor_cohesion = (triangles / np.maximum(cn_all * (cn_all - 1), 1)).astype(np.float32)
    neighbor_cohesion[cn_all < 2] = 0.0

    w_direct = np.array(adj_csr[u_arr, v_arr]).flatten().astype(np.float64)
    w_direct_rev = np.array(adj_csr[v_arr, u_arr]).flatten().astype(np.float64)
    w_direct = np.maximum(w_direct, w_direct_rev)
    parallel_weight = np.array(wA2[u_arr, v_arr]).flatten().astype(np.float64)
    resistance_distance = np.log1p(w_direct + parallel_weight / 2.0).astype(np.float32)

    n_pairs = len(u_arr)
    second_order = np.zeros(n_pairs, dtype=np.float32)
    subgraph_conn = np.zeros(n_pairs, dtype=np.float32)

    BATCH = 50000
    for start in range(0, n_pairs, BATCH):
        end = min(start + BATCH, n_pairs)
        u_b = u_arr[start:end]
        v_b = v_arr[start:end]
        cn_b = cn_all[start:end]
        deg_u = deg[u_b]
        deg_v = deg[v_b]

        row_u = A2_bin[u_b]
        row_v = A2_bin[v_b]
        n2_inter = np.array(row_u.multiply(row_v).sum(axis=1)).flatten()
        n2_size = np.array(A2_bin.sum(axis=1)).flatten()
        n2_union = n2_size[u_b] + n2_size[v_b] - n2_inter
        second_order[start:end] = (n2_inter / np.maximum(n2_union, 1)).astype(np.float32)

        union_sizes = deg_u + deg_v - cn_b
        union_indicator = row_u + row_v
        union_indicator[union_indicator > 0] = 1
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


# ── Subgraph building ─────────────────────────────────────────────────

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
        # The two endpoints must survive degree-based truncation. Falling back
        # to local index 0 silently attaches an endpoint to the wrong concept.
        required = np.unique(np.array([u, v], dtype=merged.dtype))
        optional = merged[~np.isin(merged, required)]
        keep_optional = max_nodes - len(required)
        if keep_optional > 0:
            top_idx = np.argsort(deg[optional])[-keep_optional:]
            merged = np.sort(np.concatenate([required, optional[top_idx]]))
        else:
            merged = np.sort(required[:max_nodes])
    node_to_local = {int(n): i for i, n in enumerate(merged)}
    if int(u) not in node_to_local or int(v) not in node_to_local:
        raise RuntimeError(f"Subgraph truncation dropped endpoint: ({u}, {v})")
    return merged, node_to_local[int(u)], node_to_local[int(v)]


# ── Relevance grade assignment ────────────────────────────────────────

def dist_to_relevance(dist_t2):
    """Convert continuous dist_t2 to discrete relevance grades.
    rel=4: dist_t2 ≤ 1.0 (near direct neighbor)
    rel=3: dist_t2 ∈ (1.0, 2.0]
    rel=2: dist_t2 ∈ (2.0, 5.0]
    rel=1: dist_t2 ∈ (5.0, 10.0]
    rel=0: dist_t2 > 10 or unreachable
    """
    rel = np.zeros(len(dist_t2), dtype=np.int32)
    rel[dist_t2 <= 1.0] = 4
    rel[(dist_t2 > 1.0) & (dist_t2 <= 2.0)] = 3
    rel[(dist_t2 > 2.0) & (dist_t2 <= 5.0)] = 2
    rel[(dist_t2 > 5.0) & (dist_t2 <= 10.0)] = 1
    # dist_t2 > 10 stays at 0
    return rel


# ── Process split ─────────────────────────────────────────────────────

def process_split(split_name, out_fold, adj_t1, dist_t2, ppr, deg, cc, rng):
    """v15: 2-hop candidates → t1 feature filtering → relevance grades."""
    print(f"\n  Processing {split_name}...", flush=True)

    # Step 1: Build 2-hop candidate pool (same as v14)
    print("    Building 2-hop candidate pool...", flush=True)
    t0 = time.time()
    n = adj_t1.shape[0]
    A2_t1 = (adj_t1 @ adj_t1).tocsr()
    A2_t1.setdiag(0)
    A2_t1[A2_t1 > 0] = 1

    all_u_list = []
    all_v_list = []
    all_d_list = []
    pairs_by_src_pre = {}  # u -> [(v, dist_t2), ...]

    for u in range(n):
        nbs_1hop = set(adj_t1[u].indices.tolist())
        nbs_2hop = set(A2_t1[u].indices.tolist())
        all_nbs = (nbs_1hop | nbs_2hop) - {u}

        cands = []
        for v in all_nbs:
            d = dist_t2[u, v]
            if d >= 0:  # reachable in t2
                cands.append((int(v), float(d)))
                all_u_list.append(u)
                all_v_list.append(int(v))
                all_d_list.append(float(d))
        if cands:
            pairs_by_src_pre[u] = cands

    del A2_t1

    u_raw = np.array(all_u_list, dtype=np.int64)
    v_raw = np.array(all_v_list, dtype=np.int64)
    d_raw = np.array(all_d_list, dtype=np.float32)
    n_raw = len(u_raw)
    print(f"      Raw 2-hop candidates: {n_raw:,} ({time.time()-t0:.0f}s)", flush=True)

    # Step 2: Compute 21-dim features (before filtering, to get t1 features)
    print("    Computing pair features (21-dim)...", flush=True)
    t0 = time.time()
    feats_15 = compute_all_pair_features(u_raw, v_raw, ppr, adj_t1, cc)
    print(f"      15-dim done: {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    feats_6 = compute_gravity_features(u_raw, v_raw, adj_t1, deg.astype(np.float64))
    print(f"      6-dim done: {time.time()-t0:.0f}s", flush=True)
    pair_feats_raw = np.column_stack([feats_15, feats_6]).astype(np.float32)
    del feats_15, feats_6

    # Step 3: t1 feature filtering (NO label leakage)
    # Relaxed: only require aa_all > 0 (has positive Adamic-Adar)
    # pure_cn ≥ 1 is too strict (only 0.1% pass) because most common neighbors
    # have degree > median in sparse graphs
    aa_all = pair_feats_raw[:, 5]  # Adamic-Adar from original features

    filter_mask = aa_all > 0
    n_filtered = int(filter_mask.sum())
    print(f"    t1 feature filter: {n_raw:,} → {n_filtered:,} "
          f"({100*n_filtered/max(n_raw,1):.1f}%)", flush=True)

    u_filt = u_raw[filter_mask]
    v_filt = v_raw[filter_mask]
    d_filt = d_raw[filter_mask]
    pair_feats = pair_feats_raw[filter_mask]
    del u_raw, v_raw, d_raw, pair_feats_raw

    # Step 4: Assign relevance grades
    rel = dist_to_relevance(d_filt)
    print(f"    Relevance grade distribution:", flush=True)
    for g in range(5):
        count = int((rel == g).sum())
        print(f"      rel={g}: {count:,} ({100*count/max(n_filtered,1):.1f}%)", flush=True)

    # Step 5: Reorder pairs by source, then build contiguous pair indices.
    # The original v15 derived keys from change_pts - 1, which duplicated the
    # first key and shifted every following candidate group to the wrong anchor.
    if len(u_filt) == 0:
        print(f"    WARNING: no valid pairs for {split_name}", flush=True)
        return False

    src_order = np.argsort(u_filt, kind='stable')
    pair_feats = pair_feats[src_order]
    rel = rel[src_order]
    u_filt = u_filt[src_order]
    v_filt = v_filt[src_order]
    d_filt = d_filt[src_order]

    pairs_by_src = {}
    group_starts = np.concatenate(([0], np.where(np.diff(u_filt) != 0)[0] + 1))
    group_ends = np.concatenate((group_starts[1:], [len(u_filt)]))
    for start, end in zip(group_starts, group_ends):
        anchor = int(u_filt[start])
        if not np.all(u_filt[start:end] == anchor):
            raise RuntimeError(f"Non-contiguous source group for anchor {anchor}")
        pairs_by_src[anchor] = [
            [pair_idx, int(v_filt[pair_idx])]
            for pair_idx in range(int(start), int(end))
        ]

    n_total = len(u_filt)
    n_anchors = len(pairs_by_src)
    print(f"    After filter: {n_anchors:,} anchors, {n_total:,} pairs", flush=True)

    mapped_pairs = sum(len(items) for items in pairs_by_src.values())
    if mapped_pairs != n_total:
        raise RuntimeError(
            f"Pair mapping mismatch: mapped={mapped_pairs}, arrays={n_total}")

    # Step 6: 2-hop neighborhoods
    print("    Precomputing 2-hop neighborhoods...", flush=True)
    t0 = time.time()
    two_hop_nbs = precompute_two_hop_neighborhoods(adj_t1, MAX_MERGED)
    print(f"      Done: {time.time()-t0:.0f}s", flush=True)

    # Step 7: Merged subgraph indices
    print(f"    Building merged subgraph indices ({n_total:,})...", flush=True)
    deg_arr = np.array(adj_t1.sum(axis=1)).flatten().astype(np.float64)
    # -1 is padding; concept ID 0 is a valid node and must not be masked out.
    merged_nodes = np.full((n_total, MAX_MERGED), -1, dtype=np.int32)
    u_local_arr = np.zeros(n_total, dtype=np.int32)
    v_local_arr = np.zeros(n_total, dtype=np.int32)

    t0 = time.time()
    for i in range(n_total):
        pu, pv = int(u_filt[i]), int(v_filt[i])
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

    # Step 8: Save (labels = relevance grades, stored as 'deltas' for compatibility)
    np.save(f'{out_fold}/{split_name}_pair_feats.npy', pair_feats)
    np.save(f'{out_fold}/{split_name}_merged_nodes.npy', merged_nodes)
    np.save(f'{out_fold}/{split_name}_deltas.npy', rel.astype(np.float32))
    np.save(f'{out_fold}/{split_name}_dist_t2.npy', d_filt)  # also save raw dist for eval
    np.save(f'{out_fold}/{split_name}_u_local.npy', u_local_arr)
    np.save(f'{out_fold}/{split_name}_v_local.npy', v_local_arr)

    print("    Saving pairs_by_src JSON...", flush=True)
    pairs_by_src_np = {}
    for k, idxs in pairs_by_src.items():
        pairs_by_src_np[str(k)] = idxs
    with open(f'{out_fold}/{split_name}_pairs_by_src.json', 'w') as f:
        json.dump(pairs_by_src_np, f)

    total_gb = (pair_feats.nbytes + merged_nodes.nbytes +
                rel.nbytes + d_filt.nbytes +
                u_local_arr.nbytes + v_local_arr.nbytes) / 1e9
    print(f"      Saved: {total_gb:.2f}GB", flush=True)
    return True


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=None)
    args = parser.parse_args()

    with open(f'{DATA_DIR}/concept_ids_l3.json') as f:
        concept_ids = json.load(f)
    print(f"Concepts: {len(concept_ids)}", flush=True)

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
                'label': 'relevance grades (4/3/2/1/0) based on dist_t2',
                'sampling': 't1 1-or-2-hop, filter: aa_all>0',
                'fixes': [
                    'source-group keys use group starts',
                    'pair indices rebuilt after source sorting',
                    'merged subgraphs always retain both endpoints',
                    'merged-node padding uses -1 so concept 0 remains valid',
                ],
                'relevance': {
                    '4': 'dist_t2 ≤ 1.0',
                    '3': 'dist_t2 ∈ (1.0, 2.0]',
                    '2': 'dist_t2 ∈ (2.0, 5.0]',
                    '1': 'dist_t2 ∈ (5.0, 10.0]',
                    '0': 'dist_t2 > 10',
                },
            }
            with open(f'{out_fold}/meta.json', 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"\n  Fold {fold} ({t1}→{t2}) done.", flush=True)


if __name__ == '__main__':
    main()
