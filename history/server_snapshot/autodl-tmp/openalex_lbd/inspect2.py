import numpy as np, json, re, scipy.sparse as sp
DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'

concept_ids = json.load(open(f'{DATA_DIR}/concept_ids.json'))
concept_names = json.load(open(f'{DATA_DIR}/concept_names.json'))
def get_name(idx):
    if idx < 0 or idx >= len(concept_ids): return f"?{idx}"
    return concept_names.get(concept_ids[idx], concept_ids[idx])

def _extract_val_pairs(data_dir):
    pos_pairs = []
    with open(f'{data_dir}/train.jsonl') as f:
        for line in f:
            if '"label": 1' not in line and '"label":1' not in line: continue
            src_m = re.search(r'"src_idx"\s*:\s*(\d+)', line)
            dst_m = re.search(r'"dst_idx"\s*:\s*(\d+)', line)
            if not src_m or not dst_m: continue
            src_idx = int(src_m.group(1)); dst_idx = int(dst_m.group(1))
            nid_m = re.search(r'"node_ids"\s*:\s*\[([^\]]+)\]', line)
            if not nid_m: continue
            nids = [int(x) for x in nid_m.group(1).split(',')]
            pos_pairs.append((nids[src_idx], nids[dst_idx]))
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(pos_pairs))
    n_train = int(len(pos_pairs) * 0.8)
    return [pos_pairs[i] for i in perm[n_train:]]

val_pairs = _extract_val_pairs(DATA_DIR)
val_data = np.load(f'{DATA_DIR}/val_precomp_v3.npz')
val_inp = val_data['inp'].astype(np.float32).reshape(-1, 6, 32, 68)
val_nids = val_data['node_ids'].reshape(-1, 6, 32)

# Correct node count: check cols 0:65 (position + adjacency + mask)
def count_nn_real(feat):
    return int(np.greater(np.abs(feat[:, :65]).sum(axis=1), 0).sum())

print("=== Real Subgraph Statistics ===")
pos_nns = [count_nn_real(val_inp[i, 0]) for i in range(len(val_pairs))]
neg_nns = [count_nn_real(val_inp[i, 1]) for i in range(len(val_pairs))]
print(f"Pos subgraph size: mean={np.mean(pos_nns):.1f}, median={np.median(pos_nns):.0f}, range=[{min(pos_nns)}, {max(pos_nns)}]")
print(f"Neg subgraph size: mean={np.mean(neg_nns):.1f}, median={np.median(neg_nns):.0f}, range=[{min(neg_nns)}, {max(neg_nns)}]")

# Detailed examples
print(f"\n=== 10 Concrete Examples ===")
for ai in range(10):
    u, v = val_pairs[ai]
    u_name, v_name = get_name(u), get_name(v)
    pos_feat = val_inp[ai, 0]
    pos_nids_arr = val_nids[ai, 0]
    nn = count_nn_real(pos_feat)
    
    # Find u and v positions
    u_idx = v_idx = None
    for j in range(nn):
        if pos_feat[j, 65] == 0: u_idx = j
        if pos_feat[j, 66] == 0: v_idx = j
    
    # Get node names
    all_names = [get_name(int(pos_nids_arr[j])) for j in range(nn)]
    
    # u and v neighbors
    u_nbs = set(); v_nbs = set()
    if u_idx is not None:
        u_nbs = {j for j in range(nn) if j != u_idx and pos_feat[u_idx, 32+j] > 0}
    if v_idx is not None:
        v_nbs = {j for j in range(nn) if j != v_idx and pos_feat[v_idx, 32+j] > 0}
    cross = sum(1 for a in u_nbs for b in v_nbs if pos_feat[a, 32+b] > 0)
    
    u_nb_names = [get_name(int(pos_nids_arr[j])) for j in sorted(u_nbs)]
    v_nb_names = [get_name(int(pos_nids_arr[j])) for j in sorted(v_nbs)]
    
    # Shared neighbors between u and v in subgraph
    u_nbs_set = {j for j in range(nn) if j != u_idx and pos_feat[u_idx, 32+j] > 0} if u_idx is not None else set()
    v_nbs_set = {j for j in range(nn) if j != v_idx and pos_feat[v_idx, 32+j] > 0} if v_idx is not None else set()
    shared_in_subgraph = u_nbs_set & v_nbs_set
    shared_names = [get_name(int(pos_nids_arr[j])) for j in sorted(shared_in_subgraph)]
    
    # Distance values
    du_u = pos_feat[u_idx, 65] if u_idx is not None else -1
    dv_v = pos_feat[v_idx, 66] if v_idx is not None else -1
    # Distance from u to v node
    du_v = pos_feat[v_idx, 65] if v_idx is not None else -1
    dv_u = pos_feat[u_idx, 66] if u_idx is not None else -1
    
    print(f"\n[{ai}] {u_name} -> {v_name}")
    print(f"    subgraph: {nn} nodes")
    print(f"    u@node[{u_idx}], v@node[{v_idx}]")
    print(f"    dist(u,u)={du_u:.2f}, dist(v,u)={dv_u:.2f}, dist(u,v)={du_v:.2f}, dist(v,v)={dv_v:.2f}")
    print(f"    u neighbors ({len(u_nbs)}): {', '.join(u_nb_names[:8])}")
    print(f"    v neighbors ({len(v_nbs)}): {', '.join(v_nb_names[:8])}")
    print(f"    shared u&v neighbors in subgraph: {', '.join(shared_names[:6])}")
    print(f"    cross-links (u-nb <-> v-nb): {cross}")
    # Show top-5 concept domain names
    top_names = [all_names[j] for j in range(min(5, nn))]
    print(f"    top subgraph concepts: {', '.join(top_names)}")

# Ablation with correct node counting
print(f"\n=== Global Structural Patterns ===")
adj = sp.load_npz(f'{DATA_DIR}/adj_pre2021.npz').tocsr()

cross_pos, cross_neg = [], []
u_deg_pos, u_deg_neg = [], []
v_deg_pos, v_deg_neg = [], []
shared_nb_full = []
nn_pos, nn_neg = [], []

for i in range(len(val_pairs)):
    u, v = val_pairs[i]
    shared_nb_full.append(len(set(adj[u].indices) & set(adj[v].indices)))
    
    for feat, cl, ud, vd, nn_l in [
        (val_inp[i, 0], cross_pos, u_deg_pos, v_deg_pos, nn_pos),
        (val_inp[i, 1], cross_neg, u_deg_neg, v_deg_neg, nn_neg)]:
        nn = count_nn_real(feat)
        nn_l.append(nn)
        ui = vi = None
        for j in range(nn):
            if feat[j, 65] == 0: ui = j
            if feat[j, 66] == 0: vi = j
        if ui is not None and vi is not None:
            un = {j for j in range(nn) if j != ui and feat[ui, 32+j] > 0}
            vn = {j for j in range(nn) if j != vi and feat[vi, 32+j] > 0}
            ud.append(len(un)); vd.append(len(vn))
            cl.append(sum(1 for a in un for b in vn if feat[a, 32+b] > 0))
        else:
            ud.append(0); vd.append(0); cl.append(0)

print(f"  Subgraph nodes: pos={np.mean(nn_pos):.1f}, neg={np.mean(nn_neg):.1f}")
print(f"  u degree: pos={np.mean(u_deg_pos):.1f}, neg={np.mean(u_deg_neg):.1f}")
print(f"  v degree: pos={np.mean(v_deg_pos):.1f}, neg={np.mean(v_deg_neg):.1f}")
print(f"  Cross-links: pos={np.mean(cross_pos):.1f}, neg={np.mean(cross_neg):.1f}")
print(f"  Shared nb (full graph): mean={np.mean(shared_nb_full):.1f}")

# Score correlation
print(f"\n=== What drives the score? ===")
# Look at the actual feature differences
print(f"Feature means (pos vs neg):")
for col, name in [(65, 'dist_u'), (66, 'dist_v')]:
    pos_vals = [val_inp[i, 0, j, col] for i in range(len(val_pairs)) for j in range(count_nn_real(val_inp[i, 0]))]
    neg_vals = [val_inp[i, 1, j, col] for i in range(len(val_pairs)) for j in range(count_nn_real(val_inp[i, 1]))]
    print(f"  {name}: pos_mean={np.mean(pos_vals):.4f}, neg_mean={np.mean(neg_vals):.4f}")
    # How many nodes have dist_u=0 (is u) vs dist_v=0 (is v)
    pos_u0 = sum(1 for v in pos_vals if v == 0)
    print(f"    pos nodes with {name}=0: {pos_u0}/{len(pos_vals)}")

# The REAL question: what's different between pos and neg subgraphs?
print(f"\n=== Key insight: node overlap between pos and neg subgraphs ===")
for ai in range(10):
    pos_nid_set = set(int(val_nids[ai, 0, j]) for j in range(count_nn_real(val_inp[ai, 0])))
    neg_nid_set = set(int(val_nids[ai, 1, j]) for j in range(count_nn_real(val_inp[ai, 1])))
    overlap = pos_nid_set & neg_nid_set
    u, v = val_pairs[ai]
    print(f"  [{ai}] {get_name(u)}->{get_name(v)}: pos={len(pos_nid_set)} nodes, neg={len(neg_nid_set)} nodes, overlap={len(overlap)}")

print(f"\nDone.")
