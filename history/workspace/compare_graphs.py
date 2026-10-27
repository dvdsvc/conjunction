import json, numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from itertools import combinations

prefix = "https://openalex.org/"

def build_graph(level_max, score_min, year_cutoff):
    concept_to_idx = {}
    idx = 0
    edges = set()
    for path in ["/root/autodl-tmp/openalex_data/geology_papers.jsonl", "/root/autodl-tmp/openalex_data/climate_papers.jsonl"]:
        with open(path) as f:
            for line in f:
                p = json.loads(line)
                if "year" not in p or "concepts" not in p: continue
                if int(p["year"]) > year_cutoff: continue
                valid = []
                for c in p["concepts"]:
                    cid = c["id"]
                    if cid.startswith(prefix): cid = cid[len(prefix):]
                    lvl = c.get("level", 1)
                    score = c.get("score", 0)
                    if lvl <= level_max and score >= score_min:
                        if cid not in concept_to_idx:
                            concept_to_idx[cid] = idx
                            idx += 1
                        valid.append(concept_to_idx[cid])
                for a, b in combinations(valid, 2):
                    edges.add((min(a,b), max(a,b)))
    n = len(concept_to_idx)
    if not edges:
        return sp.csr_matrix((n,n)), concept_to_idx
    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    adj = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    return adj + adj.T, concept_to_idx

configs = [
    (3, 0.3, "L<=3, score>=0.3 (current)"),
    (2, 0.5, "L<=2, score>=0.5"),
    (2, 0.6, "L<=2, score>=0.6"),
    (3, 0.5, "L<=3, score>=0.5"),
    (3, 0.7, "L<=3, score>=0.7"),
]

import sys
for level_max, score_min, label in configs:
    print(f"[{label}] building...", flush=True)
    adj, cmap = build_graph(level_max, score_min, 2012)
    n = adj.shape[0]
    deg = np.array(adj.sum(axis=1)).flatten()
    n_comp, labels = connected_components(adj, directed=False)
    comp_sizes = np.bincount(labels)
    largest_idx = np.argmax(comp_sizes)
    largest_nodes = np.where(labels == largest_idx)[0]

    sub = adj[np.ix_(largest_nodes, largest_nodes)]
    if sub.shape[0] > 1 and sub.shape[0] < 20000:
        dist = shortest_path(sub, directed=False)
        finite = dist[(dist > 0) & (dist < 9999)]
        diam = finite.max() if len(finite) > 0 else -1
        avg_d = finite.mean() if len(finite) > 0 else -1
    else:
        diam, avg_d = -1, -1
        if sub.shape[0] >= 20000:
            print(f"  (skipping diameter for {sub.shape[0]} nodes)", flush=True)

    print(f"{label}")
    print(f"  nodes={n:,} edges={adj.nnz:,} mean_deg={deg.mean():.1f} max_deg={deg.max():.0f}")
    print(f"  components={n_comp} largest={comp_sizes.max():,} isolated={int((deg==0).sum())}")
    print(f"  diameter={diam:.0f} avg_dist={avg_d:.2f}")
    print(flush=True)
