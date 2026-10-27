import torch, os, sys
DATA = "/root/autodl-tmp/openalex_data/processed"

print("Loading features.pt (weights_only=False)...", flush=True)
fd = torch.load(f"{DATA}/features.pt", map_location="cpu", weights_only=False)
print(f"keys: {list(fd.keys())}", flush=True)

for k, v in fd.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: Tensor shape={v.shape} dtype={v.dtype}")
    elif isinstance(v, list):
        print(f"  {k}: list len={len(v)}")
        if len(v) > 0:
            print(f"    sample[0] type={type(v[0])} val={str(v[0])[:100]}")
    else:
        print(f"  {k}: {type(v).__name__} = {str(v)[:100]}")
    sys.stdout.flush()

# free memory
del fd

print("\nLoading gnn_embeddings.pt...", flush=True)
gnn = torch.load(f"{DATA}/gnn_embeddings.pt", map_location="cpu", weights_only=False)
if isinstance(gnn, torch.Tensor):
    print(f"  shape={gnn.shape} dtype={gnn.dtype}")
else:
    print(f"  type={type(gnn)}")
del gnn

print("\nref_lists stats...", flush=True)
ref_lists = torch.load(f"{DATA}/ref_lists.pt", map_location="cpu", weights_only=False)
n = len(ref_lists)
total = sum(len(r) for r in ref_lists)
empty = sum(1 for r in ref_lists if len(r)==0)
print(f"  n_papers={n}, total_edges={total:,}, empty={empty}, non-empty={n-empty}")
print(f"  sample[0][:5]={ref_lists[0][:5]}")
del ref_lists

print("\nlbd_bridges stats...", flush=True)
bridges = torch.load(f"{DATA}/lbd_bridges.pt", map_location="cpu", weights_only=False)
print(f"  n_bridges={len(bridges):,}")
if isinstance(bridges[0], dict):
    print(f"  keys={list(bridges[0].keys())}")
    print(f"  sample={bridges[0]}")
elif isinstance(bridges[0], (list, tuple)):
    print(f"  sample={bridges[0]}")
print("Done.", flush=True)
