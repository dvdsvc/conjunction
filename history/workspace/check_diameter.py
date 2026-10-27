"""Build L<=3, score>=0.7 graph and compute exact diameter."""
import json, numpy as np, time
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from itertools import combinations

prefix = "https://openalex.org/"
YEAR = 2012

# Build graph
print("Building L<=3, score>=0.7 graph...", flush=True)
t0 = time.time()
concept_to_idx = {}
idx = 0
edges = set()
papers = 0

for path in ["/PATH/TO/geology_papers.jsonl",
             "/PATH/TO/climate_papers.jsonl"]:
    # try server path via scp... actually we need the data
    pass

# Data is on server, need to build there. Let's use a different approach:
# build on server and save the graph npz, then download.
print("Need paper data on server. Writing server script instead.")
