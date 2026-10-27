"""
Precompute FOS v15: topology-only features + relevance grades + LambdaRank.

Adapted from precompute_v15.py for the FOS Art_Business temporal graph.
New features (topology-only, no semantics):
  - k-core values (u, v)
  - Fiedler vector positions (u, v) — global spectral position
  - Inter-community edge density
  - Neighbor degree distribution (mean, var) for u and v
  - Common neighbor k-core mean
  - Triangle closure rate (u, v)
  - 2-hop and 3-hop path counts

Total pair features: 36-dim (21 original + 6 gravity + 9 new topology)

Usage:
  python precompute_fos_v15.py --fold 0
  python precompute_fos_v15.py
"""
import os, time, json, argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh
import cupy as cp

DATA_DIR = '/root/autodl-tmp/openalex_data/fos_data'
OUT_DIR = f'{DATA_DIR}/precomputed_fos_v15'
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
    """Clustering coefficient via A^3 diagonal (vectorized, no Python loops)."""
    adj_sym = (adj_csr + adj_csr.T).tocsr()
    adj_sym[adj_sym > 0] = 1
    A2 = (adj_sym @ adj_sym).tocsr()
    A3 = (A2 @ adj_sym).tocsr()
    # triangles[u] = A3[u,u] / 2 for undirected graphs
    diag = np.array(A3.diagonal()).flatten() / 2.0
    deg = np.array(adj_sym.sum(axis=1)).flatten().astype(np.float64)
    max_tri = deg * (deg - 1) / 2.0
    cc = np.zeros(adj_csr.shape[0], dtype=np.float32)
    mask = max_tri > 0
    cc[mask] = (diag[mask] / max_tri[mask]).astype(np.float32)
    return cc


# ── Topology features (k-core, Fiedler) ─────────────────────────────

def compute_kcore(adj_csr):
    """Compute core number for each node using networkx."""
    import networkx as nx
    adj_sym = (adj_csr + adj_csr.T).tocsr()
    adj_sym[adj_sym > 0] = 1
    G = nx.from_scipy_sparse_array(adj_sym)
    core_dict = nx.core_number(G)
    kcore = np.zeros(adj_csr.shape[0], dtype=np.float32)
    for node, val in core_dict.items():
        kcore[node] = val
    return kcore


def compute_fiedler(adj_csr):
    """Compute Fiedler vector (2nd smallest eigenvector of graph Laplacian)."""
    adj_sym = (adj_csr + adj_csr.T).tocsr()
    adj_sym[adj_sym > 0] = 1
    deg = np.array(adj_sym.sum(axis=1)).flatten()
    L = sp.diags(deg) - adj_sym  # Laplacian
    # Find 2nd smallest eigenvalue/eigenvector
    try:
        vals, vecs = eigsh(L.astype(np.float64), k=2, which='SM', tol=1e-6)
        # Sort by eigenvalue, take the 2nd smallest
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]].astype(np.float32)
    except Exception as e:
        print(f"    WARNING: Fiedler computation failed ({e}), using zeros", flush=True)
        fiedler = np.zeros(adj_csr.shape[0], dtype=np.float32)
    return fiedler


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


# ── 6-dim gravity features ──────────────────────────────────────────

def compute_gravity_features(u_arr, v_arr, adj_csr, deg):
    """All n×n matrices precomputed once, then indexed with [u_arr, v_arr]."""
    n = adj_csr.shape[0]
    adj_sym = (adj_csr + adj_csr.T).tocsr()
    adj_sym[adj_sym > 0] = 1
    adj_dense = adj_sym.toarray().astype(np.float32)
    deg_f = deg.astype(np.float32)
    deg_median = np.median(deg_f[deg_f > 0])
    is_pure = (deg_f < deg_median).astype(np.float32)

    # n×n matrices (all 3238×3238 ≈ 40MB each)
    A2_dense = adj_dense @ adj_dense
    np.fill_diagonal(A2_dense, 0)
    A2_bin = (A2_dense > 0).astype(np.float32)
    A3_dense = A2_bin @ adj_dense

    adj_pure = adj_dense * is_pure[np.newaxis, :]
    A2_pure = adj_dense @ adj_pure
    np.fill_diagonal(A2_pure, 0)

    log_inv = np.zeros(n, dtype=np.float32)
    mask = deg_f > 1
    log_inv[mask] = 1.0 / np.log(np.maximum(deg_f[mask], 2))
    log_inv_pure = log_inv * is_pure
    AA_pure = adj_dense * log_inv_pure[np.newaxis, :]
    AA_pure = adj_dense @ AA_pure
    np.fill_diagonal(AA_pure, 0)

    wA2 = adj_dense @ adj_dense
    np.fill_diagonal(wA2, 0)

    # Precomputed pair-level n×n matrices
    A2_overlap = A2_bin @ A2_bin.T  # |N2(u) ∩ N2(v)|
    overlap_deg_all = adj_dense @ (adj_dense * deg_f[np.newaxis, :]).T  # sum(deg[w]) for shared neighbors
    adj_deg_sum = adj_dense @ deg_f  # (n,)

    # Extract per-pair values (1D, no large intermediates)
    pure_cn = A2_pure[u_arr, v_arr].astype(np.float32)
    weighted_gravity = AA_pure[u_arr, v_arr].astype(np.float32)
    cn_all = A2_dense[u_arr, v_arr]
    triangles = A3_dense[u_arr, v_arr]

    neighbor_cohesion = np.where(cn_all >= 2,
                                 triangles / np.maximum(cn_all * (cn_all - 1), 1),
                                 0.0).astype(np.float32)

    w_direct = np.maximum(adj_dense[u_arr, v_arr], adj_dense[v_arr, u_arr])
    parallel_weight = wA2[u_arr, v_arr]
    resistance_distance = np.log1p(w_direct.astype(np.float64) + parallel_weight.astype(np.float64) / 2.0).astype(np.float32)

    # Second-order: |N2(u) ∩ N2(v)| / |N2(u) ∪ N2(v)|
    A2_row_sum = A2_bin.sum(axis=1)
    n2_inter = A2_overlap[u_arr, v_arr]
    n2_union = A2_row_sum[u_arr] + A2_row_sum[v_arr] - n2_inter
    second_order = (n2_inter / np.maximum(n2_union, 1)).astype(np.float32)

    # Subgraph connectivity
    union_sizes = deg_f[u_arr] + deg_f[v_arr] - cn_all
    deg_sum = adj_deg_sum[u_arr] + adj_deg_sum[v_arr] - overlap_deg_all[u_arr, v_arr]
    is_nb = adj_dense[u_arr, v_arr]
    n_edges = (deg_sum - (deg_f[u_arr] - 1.0 - is_nb) - (deg_f[v_arr] - 1.0 - is_nb)) / 2.0
    subgraph_conn = (n_edges / np.maximum(union_sizes, 1)).astype(np.float32)

    return np.column_stack([
        pure_cn, weighted_gravity, neighbor_cohesion,
        second_order, subgraph_conn, resistance_distance,
    ]).astype(np.float32)


# ── New topology features (9-dim) ───────────────────────────────────

def compute_topology_features(u_arr, v_arr, adj_csr, kcore, fiedler, deg):
    """Compute topology-only features for pairs (fully vectorized via dense matrices)."""
    adj_sym = (adj_csr + adj_csr.T).tocsr()
    adj_sym[adj_sym > 0] = 1
    adj_dense = adj_sym.toarray().astype(np.float32)  # 3238x3238 = 40MB
    n = adj_csr.shape[0]
    deg_f = deg.astype(np.float32)

    # k-core and Fiedler for u, v (vectorized)
    kcore_u = kcore[u_arr]
    kcore_v = kcore[v_arr]
    fiedler_u = fiedler[u_arr]
    fiedler_v = fiedler[v_arr]

    # 2-hop and 3-hop path counts (precomputed n×n matrices)
    A2_dense = adj_dense @ adj_dense
    np.fill_diagonal(A2_dense, 0)
    A2_bin = (A2_dense > 0).astype(np.float32)
    paths_2hop = A2_bin[u_arr, v_arr]
    A3_dense = A2_bin @ adj_dense
    paths_3hop = A3_dense[u_arr, v_arr].astype(np.float32)

    # Neighbor degree stats (vectorized)
    sum_nb_deg = adj_dense @ deg_f
    mean_nb_deg = np.zeros(n, dtype=np.float32)
    mask_deg = deg_f > 0
    mean_nb_deg[mask_deg] = sum_nb_deg[mask_deg] / deg_f[mask_deg]
    deg_sq = deg_f ** 2
    sum_nb_deg_sq = adj_dense @ deg_sq
    mean_nb_deg_sq = np.zeros(n, dtype=np.float32)
    mean_nb_deg_sq[mask_deg] = sum_nb_deg_sq[mask_deg] / deg_f[mask_deg]
    var_nb_deg = np.maximum(mean_nb_deg_sq - mean_nb_deg ** 2, 0)

    neighbor_deg_mean_u = mean_nb_deg[u_arr]
    neighbor_deg_var_u = var_nb_deg[u_arr]
    neighbor_deg_mean_v = mean_nb_deg[v_arr]
    neighbor_deg_var_v = var_nb_deg[v_arr]

    # Triangle closure rate (clustering coefficient from A3 diagonal)
    diag_A3 = np.array(A3_dense.diagonal()) / 2.0
    max_tri = deg_f * (deg_f - 1) / 2.0
    tri_closure_all = np.zeros(n, dtype=np.float32)
    tri_mask = max_tri > 0
    tri_closure_all[tri_mask] = (diag_A3[tri_mask] / max_tri[tri_mask]).astype(np.float32)
    triangle_closure_u = tri_closure_all[u_arr]
    triangle_closure_v = tri_closure_all[v_arr]

    # CN kcore mean and community edge density (GPU-accelerated via cuPy)
    kcore_f = kcore.astype(np.float32)
    n_pairs = len(u_arr)
    cn_kcore_mean = np.zeros(n_pairs, dtype=np.float32)
    community_edge_density = np.zeros(n_pairs, dtype=np.float32)

    # Move adj_dense and kcore to GPU once
    adj_gpu = cp.asarray(adj_dense)
    kcore_gpu = cp.asarray(kcore_f)

    BATCH = 50000
    for start in range(0, n_pairs, BATCH):
        end = min(start + BATCH, n_pairs)
        u_b = u_arr[start:end]
        v_b = v_arr[start:end]
        batch_size = end - start

        # GPU arrays
        nu = adj_gpu[u_b].copy()
        nv = adj_gpu[v_b].copy()
        idx = cp.arange(batch_size)
        nu[idx, v_b] = 0
        nv[idx, u_b] = 0

        # CN kcore mean
        cn_mask = nu * nv
        cn_count = cn_mask.sum(axis=1)
        cn_kcore_sum = cn_mask @ kcore_gpu
        result = cp.where(cn_count > 0, cn_kcore_sum / cn_count, 0.0).astype(cp.float32)
        cn_kcore_mean[start:end] = cp.asnumpy(result)

        # Community edge density: nu @ adj @ nv diagonal
        cross = cp.einsum('bi,ij,bj->b', nu, adj_gpu, nv)
        nu_size = nu.sum(axis=1)
        nv_size = nv.sum(axis=1)
        result = (cross / cp.maximum(nu_size * nv_size, 1)).astype(cp.float32)
        community_edge_density[start:end] = cp.asnumpy(result)

    return np.column_stack([
        kcore_u, kcore_v, fiedler_u, fiedler_v,
        community_edge_density,
        neighbor_deg_mean_u, neighbor_deg_var_u,
        neighbor_deg_mean_v, neighbor_deg_var_v,
        cn_kcore_mean,
        triangle_closure_u, triangle_closure_v,
        paths_2hop, paths_3hop,
    ]).astype(np.float32)


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
        degs = deg[merged]
        top_idx = np.argsort(degs)[-max_nodes:]
        merged = np.sort(merged[top_idx])
    node_to_local = {int(n): i for i, n in enumerate(merged)}
    return merged, node_to_local.get(int(u), 0), node_to_local.get(int(v), 0)


# ── Relevance grade assignment ────────────────────────────────────────

def dist_to_relevance(dist_t2):
    rel = np.zeros(len(dist_t2), dtype=np.int32)
    rel[dist_t2 <= 1.0] = 4
    rel[(dist_t2 > 1.0) & (dist_t2 <= 2.0)] = 3
    rel[(dist_t2 > 2.0) & (dist_t2 <= 5.0)] = 2
    rel[(dist_t2 > 5.0) & (dist_t2 <= 10.0)] = 1
    return rel


# ── Process split ─────────────────────────────────────────────────────

def process_split(split_name, out_fold, adj_t1, dist_t2, ppr, deg, cc,
                  kcore, fiedler, rng):
    """v15: 2-hop candidates → feature filtering → relevance grades."""
    print(f"\n  Processing {split_name}...", flush=True)

    # Step 1: Build 2-hop candidate pool
    print("    Building 2-hop candidate pool...", flush=True)
    t0 = time.time()
    n = adj_t1.shape[0]
    A2_t1 = (adj_t1 @ adj_t1).tocsr()
    A2_t1.setdiag(0)
    A2_t1[A2_t1 > 0] = 1

    all_u_list = []
    all_v_list = []
    all_d_list = []
    pairs_by_src_pre = {}

    for u in range(n):
        nbs_1hop = set(adj_t1[u].indices.tolist())
        nbs_2hop = set(A2_t1[u].indices.tolist())
        all_nbs = (nbs_1hop | nbs_2hop) - {u}

        cands = []
        for v in all_nbs:
            d = dist_t2[u, v]
            if d >= 0:
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

    # Step 2: Compute features
    print("    Computing pair features...", flush=True)
    t0 = time.time()
    feats_15 = compute_all_pair_features(u_raw, v_raw, ppr, adj_t1, cc)
    print(f"      15-dim done: {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    feats_6 = compute_gravity_features(u_raw, v_raw, adj_t1, deg.astype(np.float64))
    print(f"      6-dim done: {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    feats_topo = compute_topology_features(u_raw, v_raw, adj_t1, kcore, fiedler,
                                           deg.astype(np.float64))
    print(f"      14-dim topology done: {time.time()-t0:.0f}s", flush=True)
    pair_feats_raw = np.column_stack([feats_15, feats_6, feats_topo]).astype(np.float32)
    del feats_15, feats_6, feats_topo

    # Step 3: Feature filtering (aa_all > 0)
    aa_all = pair_feats_raw[:, 5]
    filter_mask = aa_all > 0
    n_filtered = int(filter_mask.sum())
    print(f"    t1 feature filter: {n_raw:,} -> {n_filtered:,} "
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

    # Step 5: Rebuild pairs_by_src
    src_order = np.argsort(u_filt, kind='stable')
    sorted_u = u_filt[src_order]
    change_pts = np.where(np.diff(sorted_u))[0] + 1
    groups = np.split(src_order, change_pts)
    keys = sorted_u[change_pts - 1] if len(change_pts) > 0 else np.array([sorted_u[0]])
    if len(change_pts) > 0:
        keys = np.concatenate([[sorted_u[0]], keys])

    pairs_by_src = {}
    for k, g in zip(keys, groups):
        pairs_by_src[int(k)] = [[int(i), int(v_filt[i])] for i in g.tolist()]

    n_total = len(u_filt)
    n_anchors = len(pairs_by_src)
    print(f"    After filter: {n_anchors:,} anchors, {n_total:,} pairs", flush=True)

    if n_total == 0:
        print(f"    WARNING: no valid pairs for {split_name}", flush=True)
        return False

    pair_feats = pair_feats[src_order]
    rel = rel[src_order]
    u_filt = u_filt[src_order]
    v_filt = v_filt[src_order]
    d_filt = d_filt[src_order]

    # Step 6: 2-hop neighborhoods
    print("    Precomputing 2-hop neighborhoods...", flush=True)
    t0 = time.time()
    two_hop_nbs = precompute_two_hop_neighborhoods(adj_t1, MAX_MERGED)
    print(f"      Done: {time.time()-t0:.0f}s", flush=True)

    # Step 7: Merged subgraph indices
    print(f"    Building merged subgraph indices ({n_total:,})...", flush=True)
    deg_arr = np.array(adj_t1.sum(axis=1)).flatten().astype(np.float64)
    merged_nodes = np.zeros((n_total, MAX_MERGED), dtype=np.int32)
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

    # Step 8: Save
    np.save(f'{out_fold}/{split_name}_pair_feats.npy', pair_feats)
    np.save(f'{out_fold}/{split_name}_merged_nodes.npy', merged_nodes)
    np.save(f'{out_fold}/{split_name}_deltas.npy', rel.astype(np.float32))
    np.save(f'{out_fold}/{split_name}_dist_t2.npy', d_filt)
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

    with open(f'{DATA_DIR}/concept_ids.json') as f:
        concept_ids = json.load(f)
    print(f"Concepts: {len(concept_ids)}", flush=True)

    rolling_folds = [
        (0, 2005, 2010),
        (1, 2010, 2015),
        (2, 2015, 2020),
        (3, 2020, 2024),
    ]
    if args.fold is not None:
        rolling_folds = [f for f in rolling_folds if f[0] == args.fold]

    for fold, t1, t2 in rolling_folds:
        print(f"\n{'='*60}", flush=True)
        print(f"  Fold {fold}: {t1}->{t2}", flush=True)

        out_fold = f'{OUT_DIR}/fold{fold}'
        if os.path.exists(f'{out_fold}/meta.json'):
            print(f"  Already exists, skipping.", flush=True)
            continue
        os.makedirs(out_fold, exist_ok=True)

        rng = np.random.RandomState(42 + fold)

        print(f"  Loading graphs {t1} and {t2}...", flush=True)
        adj_t1 = sp.load_npz(f'{DATA_DIR}/graph_fos_{t1}.npz').tocsr()
        adj_t2 = sp.load_npz(f'{DATA_DIR}/graph_fos_{t2}.npz').tocsr()
        ppr = np.load(f'{DATA_DIR}/ppr_fos_{t1}.npy', mmap_mode='r')

        print(f"  Computing dist_t2 (weighted BFS on {t2} graph)...", flush=True)
        dist_t2 = compute_weighted_bfs_first_hop_free(
            adj_t2, f'{out_fold}/dist_t2.npy', "dist_t2")

        deg = np.array(adj_t1.sum(axis=1)).flatten().astype(np.int32)
        print("  Computing clustering coefficients...", flush=True)
        t0 = time.time()
        cc = fast_clustering_coefficients(adj_t1)
        print(f"    Done: {time.time()-t0:.0f}s", flush=True)

        print("  Computing k-core...", flush=True)
        t0 = time.time()
        kcore = compute_kcore(adj_t1)
        print(f"    Done: {time.time()-t0:.0f}s", flush=True)

        print("  Computing Fiedler vector...", flush=True)
        t0 = time.time()
        fiedler = compute_fiedler(adj_t1)
        print(f"    Done: {time.time()-t0:.0f}s", flush=True)

        ok = process_split('train', out_fold, adj_t1, dist_t2, ppr,
                           deg, cc, kcore, fiedler, rng)
        del adj_t1, adj_t2, ppr, deg, cc, dist_t2, kcore, fiedler

        if ok:
            meta = {
                'fold': fold,
                'window': f'{t1}->{t2}',
                'MAX_MERGED': MAX_MERGED,
                'pair_feat_dim': 35,
                'label': 'relevance grades (4/3/2/1/0) based on dist_t2',
                'features': '21 original + 6 gravity + 8 topology (kcore, fiedler, '
                            'community_edge_density, neighbor_deg_stats, cn_kcore_mean, '
                            'triangle_closure, paths_2hop, paths_3hop)',
                'relevance': {
                    '4': 'dist_t2 <= 1.0',
                    '3': 'dist_t2 in (1.0, 2.0]',
                    '2': 'dist_t2 in (2.0, 5.0]',
                    '1': 'dist_t2 in (5.0, 10.0]',
                    '0': 'dist_t2 > 10',
                },
            }
            with open(f'{out_fold}/meta.json', 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"\n  Fold {fold} ({t1}->{t2}) done.", flush=True)


if __name__ == '__main__':
    main()
