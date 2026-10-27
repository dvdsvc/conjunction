import torch, sys
print("start", flush=True)
sys.stdout.flush()

fd = torch.load("/root/autodl-tmp/openalex_data/processed/features.pt",
                map_location="cpu", weights_only=False)
print(f"loaded, keys={list(fd.keys())}", flush=True)

for k, v in fd.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: Tensor {v.shape} {v.dtype}", flush=True)
    elif isinstance(v, list):
        sample = str(v[0])[:80] if v else "empty"
        print(f"  {k}: list[{len(v)}] sample={sample}", flush=True)
    else:
        print(f"  {k}: {type(v).__name__}={str(v)[:60]}", flush=True)
print("done", flush=True)
