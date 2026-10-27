import json, numpy as np
from collections import Counter

# Check graph connectivity by level
adj = np.load("/root/autodl-tmp/openalex_data/pencil_data/temporal_v2/graph_2012.npz")  # will try CSR
import scipy.sparse as sp
adj = sp.load_npz("/root/autodl-tmp/openalex_data/pencil_data/temporal_v2/graph_2012.npz").tocsr()

cids_list = json.load(open("/root/autodl-tmp/openalex_data/pencil_data/concept_ids.json"))

concept_levels = {}
with open("/root/autodl-tmp/openalex_data/concepts.jsonl") as f:
    for line in f:
        w = json.loads(line)
        for c in w["concepts"]:
            concept_levels[c["id"]] = c["level"]

# Check degree distribution by level
degrees = np.array(adj.sum(axis=1)).flatten()
l2_degrees = []
l3_degrees = []
l1_degrees = []
for i, cid in enumerate(cids_list):
    lvl = concept_levels.get(cid, -1)
    deg = degrees[i]
    if lvl == 2: l2_degrees.append(deg)
    elif lvl == 3: l3_degrees.append(deg)
    elif lvl == 1: l1_degrees.append(deg)

print("Degree by level:")
print(f"  L1 (n={len(l1_degrees)}): mean={np.mean(l1_degrees):.1f}, median={np.median(l1_degrees):.0f}, max={max(l1_degrees) if l1_degrees else 0}")
print(f"  L2 (n={len(l2_degrees)}): mean={np.mean(l2_degrees):.1f}, median={np.median(l2_degrees):.0f}, max={max(l2_degrees) if l2_degrees else 0}")
print(f"  L3 (n={len(l3_degrees)}): mean={np.mean(l3_degrees):.1f}, median={np.median(l3_degrees):.0f}, max={max(l3_degrees) if l3_degrees else 0}")

# Check: how many L3 concepts are isolated (degree=0)?
iso_l3 = sum(1 for d in l3_degrees if d == 0)
iso_l2 = sum(1 for d in l2_degrees if d == 0)
print(f"\nIsolated (degree=0): L2={iso_l2}, L3={iso_l3}")
print(f"  L3 isolation rate: {iso_l3/len(l3_degrees)*100:.1f}%")
print(f"  L2 isolation rate: {iso_l2/len(l2_degrees)*100:.1f}%")

# Check the dist matrix - what are typical distances for L3 vs L2?
dm = np.load("/root/autodl-tmp/openalex_data/pencil_data/temporal_v2/dist_2012.npy")

# Sample some L2-L2, L3-L3 distances
l2_idx = [i for i,cid in enumerate(cids_list) if concept_levels.get(cid,-1)==2]
l3_idx = [i for i,cid in enumerate(cids_list) if concept_levels.get(cid,-1)==3]

np.random.seed(42)
if len(l2_idx) > 1000:
    l2_sample = np.random.choice(l2_idx, 1000, replace=False)
else:
    l2_sample = np.array(l2_idx)
if len(l3_idx) > 1000:
    l3_sample = np.random.choice(l3_idx, 1000, replace=False)
else:
    l3_sample = np.array(l3_idx)

l2_dists = dm[np.ix_(l2_sample, l2_sample)].flatten()
l3_dists = dm[np.ix_(l3_sample, l3_sample)].flatten()

from collections import Counter
print(f"\nL2-L2 distance distribution (sampled 1000x1000):")
d2_cnt = Counter(l2_dists.tolist())
for d in sorted(d2_cnt)[:8]:
    print(f"  dist={d}: {d2_cnt[d]:,} ({d2_cnt[d]/len(l2_dists)*100:.1f}%)")

print(f"\nL3-L3 distance distribution (sampled 1000x1000):")
d3_cnt = Counter(l3_dists.tolist())
for d in sorted(d3_cnt)[:8]:
    print(f"  dist={d}: {d3_cnt[d]:,} ({d3_cnt[d]/len(l3_dists)*100:.1f}%)")
