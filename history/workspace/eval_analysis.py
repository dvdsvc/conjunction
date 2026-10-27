"""Show concrete concept pairs with names."""
import numpy as np, json

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
TEMP_DIR = f'{DATA_DIR}/temporal_v2'
K_EVAL = 50

d = np.load(f'{TEMP_DIR}/eval_v5_fold0.npz')
scores = d['scores']
labels = d['labels']
src_ids = d['src_ids']
cand_v = d['cand_v']
n = len(src_ids)

scores_2d = scores[:n*K_EVAL].reshape(n, K_EVAL)
labels_2d = labels[:n*K_EVAL].reshape(n, K_EVAL)

names_dict = json.load(open(f'{DATA_DIR}/concept_names.json'))
cids = json.load(open(f'{DATA_DIR}/concept_ids.json'))
idx_to_name = {i: names_dict.get(cid, cid) for i, cid in enumerate(cids)}

print(f"Total anchors: {n}\n")

# Show best examples: top-5 has 4+ convergent
shown = 0
for i in range(n):
    ranked = np.argsort(-scores_2d[i])
    top5_dc = [labels_2d[i, ranked[k]] for k in range(5)]
    if sum(dc > 0 for dc in top5_dc) < 4:
        continue

    u = int(src_ids[i])
    u_name = idx_to_name.get(u, f"idx_{u}")
    print(f"=== {u_name} ===")
    for k in range(min(10, K_EVAL)):
        idx = ranked[k]
        v = int(cand_v[i, idx])
        v_name = idx_to_name.get(v, f"idx_{v}")
        dc = labels_2d[i, idx]
        sc = scores_2d[i, idx]
        marker = " [CONVERGENT]" if dc > 0 else ""
        print(f"  #{k+1:2d} {v_name}  score={sc:.3f} dc={dc:.1f}{marker}")
    print()

    shown += 1
    if shown >= 15:
        break
