import json, numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from itertools import combinations
import sys

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

config = (int(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
level_max, score_min, label = config
print(f"[{label}] building...", flush=True)
adj, cmap = build_graph(level_max, score_min, 2012)
n = adj.shape[0]
deg = np.array(adj.sum(axis=1)).flatten()
n_comp, labels = connected_components(adj, directed=False)
comp_sizes = np.bincount(labels)
largest_idx = np.argmax(comp_sizes)
largest_nodes = np.where(labels == largest_idx)[0]

print(f"{label}")
print(f"  nodes={n:,} edges={adj.nnz:,} mean_deg={deg.mean():.1f} max_deg={deg.max():.0f}")
print(f"  components={n_comp} largest={comp_sizes.max():,} isolated={int((deg==0).sum())}", flush=True)

if len(largest_nodes) < 15000:
    sub = adj[np.ix_(largest_nodes, largest_nodes)]
    print(f"  computing diameter for {len(largest_nodes)} nodes...", flush=True)
    dist = shortest_path(sub, directed=False)
    finite = dist[(dist > 0) & (dist < 9999)]
    diam = finite.max() if len(finite) > 0 else -1
    avg_d = finite.mean() if len(finite) > 0 else -1
    print(f"  diameter={diam:.0f} avg_dist={avg_d:.2f}", flush=True)
else:
    print(f"  (skipping diameter for {len(largest_nodes)} nodes)", flush=True)
