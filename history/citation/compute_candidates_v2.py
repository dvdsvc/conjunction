#!/usr/bin/env python3
"""Compute candidate vectors as SINGLE TOKEN (no reference list).

Each paper is represented by its own features only:
  title_emb + gnn_emb + scalar + concept → Transformer → single token → L2 norm

No reference list is included — avoids data leakage.
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F

from phoenix_model import PhoenixCitationTransformer


def infer_model_config(state_dict, scalar_dim):
    d_model = state_dict["feature_proj.0.weight"].shape[0]
    total_in = state_dict["feature_proj.0.weight"].shape[1]
    ce = state_dict["concept_proj.weight"].shape[0]
    has_title = "title_proj.0.weight" in state_dict
    if has_title:
        tp = state_dict["title_proj.0.weight"].shape[0]
        ge = total_in - scalar_dim - ce - tp
    else:
        tp = 0
        ge = total_in - scalar_dim - ce
    n_heads = 12 if d_model == 384 else d_model // 64
    n_layers = len([k for k in state_dict if k.endswith(".attn_norm.weight")])
    return dict(d_model=d_model, n_heads=n_heads, n_kv_heads=n_heads,
                n_layers=n_layers, concept_emb_dim=ce, gnn_emb_dim=ge,
                title_emb_dim=384 if has_title else 0, title_proj_dim=tp)


def load_phoenix(ckpt_path, config, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = infer_model_config(ckpt["model"], config["scalar_feature_dim"])
    model = PhoenixCitationTransformer(
        scalar_dim=config["scalar_feature_dim"],
        concept_vocab_size=config["concept_vocab_size"],
        **arch, dropout=0.0, output_multiplier=0.125,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, arch


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), weights_only=False)
    gnn_emb = torch.load(os.path.join(args.data_dir, "gnn_embeddings.pt"), weights_only=False)
    with open(os.path.join(args.data_dir, "config.json")) as f:
        config = json.load(f)

    title_path = os.path.join(args.data_dir, "title_embeddings.pt")
    has_title = os.path.exists(title_path)
    if has_title:
        title_embeddings = torch.load(title_path, weights_only=False)
        print(f"Title embeddings: {title_embeddings.shape}", flush=True)
    else:
        title_embeddings = None

    features = fd["features"]
    concepts = fd["concepts"]
    concept_dim = config["concept_vocab_size"]
    n_total = features.size(0)
    scalar_dim = features.shape[1]
    gnn_dim = gnn_emb.shape[1]
    title_dim = title_embeddings.shape[1] if has_title else 0

    ckpt_path = os.path.join(args.data_dir, "checkpoints/phoenix_best.pt")
    phoenix, arch = load_phoenix(ckpt_path, config, args.device)
    d_model = arch["d_model"]
    print(f"Phoenix loaded: d_model={d_model}", flush=True)

    # Compute candidate vectors as SINGLE TOKEN (no refs)
    print(f"Computing {n_total:,} candidate vectors (single token, batch={args.batch_size})...", flush=True)
    all_vecs = torch.zeros(n_total, d_model)
    t0 = time.time()

    for start in range(0, n_total, args.batch_size):
        end = min(start + args.batch_size, n_total)
        idxs = list(range(start, end))
        B = len(idxs)

        # Single token per paper: just its own features
        b_scalars = features[idxs].to(args.device)               # (B, scalar_dim)
        b_gnns = gnn_emb[idxs].to(args.device)                   # (B, gnn_dim)
        b_titles = title_embeddings[idxs].to(args.device) if has_title else None  # (B, title_dim)

        # Build concept multi-hot
        b_cmh = torch.zeros(B, concept_dim, device=args.device)
        for i, idx in enumerate(idxs):
            c = concepts[idx]
            if c:
                valid = [x for x in c if x < concept_dim]
                if valid:
                    b_cmh[i].scatter_(0, torch.tensor(valid, device=args.device), 1.0)

        # Project through Phoenix (single token → trivial attention)
        c_proj = phoenix.concept_proj(b_cmh)                # (B, concept_emb_dim)
        if has_title:
            t_proj = phoenix.title_proj(b_titles)           # (B, title_proj_dim)
            x = torch.cat([b_scalars, c_proj, t_proj, b_gnns], dim=-1)
        else:
            x = torch.cat([b_scalars, c_proj, b_gnns], dim=-1)

        x = phoenix.feature_proj(x).unsqueeze(1)            # (B, 1, d_model)

        # Transformer blocks (single token, mask=1)
        mask = torch.ones(B, 1, 1, device=args.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for block in phoenix.blocks:
                x = block(x, mask)
            x = phoenix.final_norm(x)

        vec = x.squeeze(1)                                   # (B, d_model)
        vec = F.normalize(vec.float(), p=2, dim=-1)
        all_vecs[start:end] = vec.cpu()

        if (start // args.batch_size) % 100 == 0 and start > 0:
            elapsed = time.time() - t0
            rate = start / elapsed
            eta = (n_total - start) / rate / 60
            print(f"  [{start:,}/{n_total:,}] {rate:.0f}/s ETA {eta:.1f}m", flush=True)

    elapsed = time.time() - t0
    print(f"Done in {elapsed/60:.1f}m ({n_total/max(elapsed,1):.0f} vec/s)", flush=True)

    # Save
    out_path = os.path.join(args.data_dir, "phoenix_candidate_vectors.pt")
    torch.save(all_vecs, out_path)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Saved: {out_path}  shape={all_vecs.shape}  {size_mb:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
