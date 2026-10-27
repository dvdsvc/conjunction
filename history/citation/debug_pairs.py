"""Debug: find what (u,v) pairs are ACTUALLY in val_nids_v5 vs eval_subgraph_nids."""
import numpy as np
import re

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'

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

train_pairs, val_pairs = _extract_val_pairs(DATA_DIR)
print(f"len(train_pairs)={len(train_pairs)}, len(val_pairs)={len(val_pairs)}")

val_nids = np.load(f'{DATA_DIR}/val_nids_v5.npy')
print(f"val_nids_v5 shape: {val_nids.shape}")

eval_nids = np.load(f'{DATA_DIR}/eval_subgraph_nids.npy', mmap_mode='r')
eval_cands_len = np.load(f'{DATA_DIR}/eval_cands_len.npy')
print(f"eval_subgraph_nids shape: {eval_nids.shape}")

# For first 5 val pairs, check if the positive subgraph matches
print("\n=== Comparing val_pairs vs val_nids_v5[0] (first 5) ===")
for ai in range(5):
    u, v = val_pairs[ai]
    train_pos = val_nids[ai, 0]  # positive in training data
    print(f"  val_pairs[{ai}]: u={u}, v={v}")
    print(f"  val_nids[{ai},0]: first 2 = ({train_pos[0]}, {train_pos[1]})")
    print(f"  match: {train_pos[0]==u and train_pos[1]==v}")

# For first 5 eval anchors, check the positive pair
print("\n=== eval_subgraph_nids first 5 anchors ===")
for ai in range(5):
    NC = int(eval_cands_len[ai])
    nids_ai = eval_nids[ai]
    # Find the positive (where node_ids[0]==u and node_ids[1]==v)
    # Actually, we need to find which candidate has u,v matching val_pairs
    u, v = val_pairs[ai]
    pos_idx = -1
    for j in range(NC):
        if nids_ai[j, 0] == u and nids_ai[j, 1] == v:
            pos_idx = j
            break
    print(f"  eval_pairs[{ai}] from val_pairs: u={u}, v={v}")
    print(f"  found in eval_nids[{ai}]: pos_idx={pos_idx}")
    if pos_idx >= 0:
        print(f"  eval_nids[{ai},{pos_idx}] first 2: ({nids_ai[pos_idx,0]}, {nids_ai[pos_idx,1]})")
    # Also check: what are the u,v for the first candidate?
    print(f"  eval_nids[{ai},0] first 2: ({nids_ai[0,0]}, {nids_ai[0,1]})")
    print(f"  Does eval_nids[{ai},0] match val_pairs[{ai}]: {nids_ai[0,0]==u and nids_ai[0,1]==v}")

# Key question: are val_pairs the same as what eval uses?
print("\n=== Checking if val_pairs == eval positive pairs ===")
mismatches = 0
for ai in range(min(20, len(val_pairs))):
    u, v = val_pairs[ai]
    NC = int(eval_cands_len[ai])
    nids_ai = eval_nids[ai]
    # The eval script uses val_pairs[ai] to find pos, but what (u,v) does eval_nids[ai] correspond to?
    # precompute_eval_nids.py should tell us
    if nids_ai[0, 0] != u or nids_ai[0, 1] != v:
        mismatches += 1
        print(f"  MISMATCH ai={ai}: val_pairs=({u},{v}), eval_nids[0]=({nids_ai[0,0]},{nids_ai[0,1]})")
if mismatches == 0:
    print(f"  All first 20 match!")
else:
    print(f"  {mismatches} mismatches in first 20")
