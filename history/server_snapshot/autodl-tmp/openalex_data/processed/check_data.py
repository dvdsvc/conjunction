import torch, os
DATA = "/root/autodl-tmp/openalex_data/processed"

fd = torch.load(f"{DATA}/features.pt", map_location="cpu", weights_only=False)
print("=== features.pt keys ===")
print(list(fd.keys()))

concepts = fd["concepts"]
print(f"\nconcepts: list of {len(concepts)} lists")
print(f"  sample: {concepts[0][:5]}")

pids = fd["paper_ids"]
print(f"\npaper_ids: {len(pids)} items")
print(f"  sample: {pids[0]}")

titles = fd.get("titles")
if titles is not None:
    print(f"\ntitles: {len(titles)} items")
    for t in titles[:3]:
        print(f"  {t[:80]}")
else:
    print("\nNO titles in features.pt")

p = f"{DATA}/gnn_embeddings.pt"
if os.path.exists(p):
    gnn = torch.load(p, map_location="cpu", weights_only=False)
    print(f"\ngnn_embeddings: shape={gnn.shape}, dtype={gnn.dtype}")
else:
    print("\nNO gnn_embeddings.pt")

ref_lists = torch.load(f"{DATA}/ref_lists.pt", map_location="cpu", weights_only=False)
total_edges = sum(len(r) for r in ref_lists)
empty = sum(1 for r in ref_lists if len(r)==0)
nonempty = len(ref_lists) - empty
avg_len = sum(len(r) for r in ref_lists if len(r)>0) / max(nonempty, 1)
print(f"\nref_lists: {len(ref_lists)} papers")
print(f"  empty: {empty}, non-empty: {nonempty}, avg len: {avg_len:.1f}")
print(f"  total edges: {total_edges:,}")

bridges = torch.load(f"{DATA}/lbd_bridges.pt", map_location="cpu", weights_only=False)
print(f"\nlbd_bridges: {len(bridges):,} pairs")
if isinstance(bridges[0], dict):
    print(f"  sample keys: {list(bridges[0].keys())}")
