"""Build L<=3 score>=0.7 graph (2012), save npz."""
import json, numpy as np
import scipy.sparse as sp
from itertools import combinations

prefix = "https://openalex.org/"
YEAR = 2012

concept_to_idx = {}
idx = 0
edges = set()

for path in ["/root/autodl-tmp/openalex_data/geology_papers.jsonl",
             "/root/autodl-tmp/openalex_data/climate_papers.jsonl"]:
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            if "year" not in p or "concepts" not in p: continue
            if int(p["year"]) > YEAR: continue
            valid = []
            for c in p["concepts"]:
                cid = c["id"]
                if cid.startswith(prefix): cid = cid[len(prefix):]
                lvl = c.get("level", 1)
                score = c.get("score", 0)
                if lvl <= 3 and score >= 0.7:
                    if cid not in concept_to_idx:
                        concept_to_idx[cid] = idx
                        idx += 1
                    valid.append(concept_to_idx[cid])
            for a, b in combinations(valid, 2):
                edges.add((min(a,b), max(a,b)))

n = len(concept_to_idx)
rows = [e[0] for e in edges]
cols = [e[1] for e in edges]
adj = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
adj = adj + adj.T

sp.save_npz("/tmp/graph_l3_s07_2012.npz", adj)

# Save concept mapping
import json as j2
j2.dump(concept_to_idx, open("/tmp/concept_map_l3_s07.json", "w"))

# Also save concept names for later
names = j2.load(open("/root/autodl-tmp/openalex_data/pencil_data/concept_names.json"))
cids_list = j2.load(open("/root/autodl-tmp/openalex_data/pencil_data/concept_ids.json"))
# Build name map for new concepts
name_map = {}
for cid in concept_to_idx:
    if cid in names:
        name_map[cid] = names[cid]
j2.dump(name_map, open("/tmp/concept_names_l3_s07.json", "w"))

from scipy.sparse.csgraph import connected_components
n_comp, labels = connected_components(adj, directed=False)
comp_sizes = np.bincount(labels)
deg = np.array(adj.sum(axis=1)).flatten()

print(f"nodes={n} edges={adj.nnz} mean_deg={deg.mean():.1f} max_deg={deg.max()}")
print(f"components={n_comp} largest={comp_sizes.max()} isolated={int((deg==0).sum())}")
print(f"Saved to /tmp/graph_l3_s07_2012.npz")
