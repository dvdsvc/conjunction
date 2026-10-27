#!/usr/bin/env python3
"""Optimized batch inference through Phoenix Transformer for all 3.55M papers.

Memory-efficient: NO pre-padded concept tensors. Builds padded batches on-the-fly.
Uses model's built-in scatter on GPU for concept multi-hot.
"""
import argparse, json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from phoenix_model import PhoenixCitationTransformer


def infer_model_config(state_dict, scalar_dim):
    d_model = state_dict["feature_proj.0.weight"].shape[0]
    total_in = state_dict["feature_proj.0.weight"].shape[1]
    ce = state_dict["concept_proj.weight"].shape[0]
    ge = total_in - scalar_dim - ce
    n_heads = 12 if d_model == 384 else d_model // 64
    n_layers = len([k for k in state_dict if k.endswith(".attn_norm.weight")])
    return dict(d_model=d_model, n_heads=n_heads, n_kv_heads=n_heads,
                n_layers=n_layers, concept_emb_dim=ce, gnn_emb_dim=ge)


def build_attn_mask(max_refs, device):
    """Full bidirectional attention mask (matches batch_user_vectors)."""
    S = 1 + max_refs
    mask = torch.ones(1, S, S, device=device)
    return mask


@torch.no_grad()
def batch_inference(phoenix, features, gnn_emb, ref_lists, concepts,
                    max_refs, max_n_concepts, device, batch_size=1024, use_fp16=True):
    """Run Phoenix Transformer on all papers in batches."""
    N = features.size(0)
    d_model = phoenix.d_model
    scalar_dim = features.size(1)
    gnn_dim = gnn_emb.size(1)
    concept_vocab_size = phoenix.concept_proj.in_features
    seq_len = 1 + max_refs

    all_vecs = torch.zeros(N, d_model)
    attn_mask = build_attn_mask(max_refs, device)
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_fp16 \
              else __import__("contextlib").nullcontext()

    t0 = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        # Gather ref indices for this batch
        ref_lists_batch = ref_lists[start:end]
        max_actual_refs = min(max(len(r) for r in ref_lists_batch), max_refs)

        # Build ref index tensor (B, max_actual_refs), padded with -1
        ref_idx = torch.full((B, max_actual_refs), -1, dtype=torch.long)
        for i, r in enumerate(ref_lists_batch):
            rl = min(len(r), max_actual_refs)
            if rl > 0:
                ref_idx[i, :rl] = torch.tensor(r[:rl], dtype=torch.long)

        # Vectorized feature gathering for refs
        valid = ref_idx >= 0
        flat_idx = ref_idx[valid]  # all valid ref indices
        ref_features_all = features[flat_idx]
        ref_gnn_all = gnn_emb[flat_idx]

        # Scatter back into padded tensors
        b_scalar = torch.zeros(B, seq_len, scalar_dim)
        b_gnn = torch.zeros(B, seq_len, gnn_dim)
        b_scalar[:, 0] = features[start:end]
        b_gnn[:, 0] = gnn_emb[start:end]

        pos = 0
        for i in range(B):
            rl = valid[i].sum().item()
            if rl > 0:
                b_scalar[i, 1:1+rl] = ref_features_all[pos:pos+rl]
                b_gnn[i, 1:1+rl] = ref_gnn_all[pos:pos+rl]
                pos += rl

        # Concept gathering (per-sample, unavoidable due to variable length)
        b_concept = torch.zeros(B, seq_len, max_n_concepts, dtype=torch.long)
        for i in range(B):
            # Self concepts
            c = concepts[start + i]
            cl = min(len(c), max_n_concepts)
            if cl > 0:
                b_concept[i, 0, :cl] = torch.tensor(c[:cl], dtype=torch.long)
            # Ref concepts
            for j, ref in enumerate(ref_lists_batch[i][:max_refs]):
                c = concepts[ref]
                cl = min(len(c), max_n_concepts)
                if cl > 0:
                    b_concept[i, j + 1, :cl] = torch.tensor(c[:cl], dtype=torch.long)

        # Move to GPU
        b_scalar = b_scalar.to(device, non_blocking=True)
        b_gnn = b_gnn.to(device, non_blocking=True)
        b_concept = b_concept.to(device, non_blocking=True)
        b_mask = attn_mask.expand(B, -1, -1)

        with amp_ctx:
            concept_mh = torch.zeros(B, seq_len, concept_vocab_size,
                                     device=device, dtype=torch.float16 if use_fp16 else torch.float32)
            concept_mh.scatter_(2, b_concept.clamp(min=0), 1.0)
            concept_emb = phoenix.concept_proj(concept_mh)

            x = torch.cat([b_scalar, concept_emb, b_gnn], dim=-1)
            x = phoenix.feature_proj(x)
            for block in phoenix.blocks:
                x = block(x, b_mask)
            x = phoenix.final_norm(x)
            user_repr = F.normalize(x.mean(dim=1), p=2, dim=-1)

        all_vecs[start:end] = user_repr.cpu()

        if (start // batch_size) % 20 == 0 and start > 0:
            elapsed = time.time() - t0
            rate = start / elapsed
            eta = (N - start) / rate
            print(f"  [{start:,}/{N:,}] {elapsed:.0f}s elapsed, {eta:.0f}s remaining", flush=True)

    total_time = time.time() - t0
    print(f"Done: {N:,} vectors in {total_time:.0f}s ({N/total_time:.0f} vec/s)")
    return all_vecs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--max-refs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), map_location="cpu", weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), map_location="cpu", weights_only=False)
    gnn_emb = torch.load(os.path.join(args.data_dir, "gnn_embeddings.pt"), map_location="cpu", weights_only=False)
    with open(os.path.join(args.data_dir, "config.json")) as f: config = json.load(f)

    print("Loading Phoenix Transformer...", flush=True)
    ckpt = torch.load(os.path.join(args.data_dir, "checkpoints/phoenix_best.pt"),
                       map_location=args.device, weights_only=False)
    arch = infer_model_config(ckpt["model"], config["scalar_feature_dim"])
    phoenix = PhoenixCitationTransformer(
        scalar_dim=config["scalar_feature_dim"],
        concept_vocab_size=config["concept_vocab_size"],
        **arch, dropout=0.0, output_multiplier=0.125,
    ).to(args.device)
    phoenix.load_state_dict(ckpt["model"])
    phoenix.eval()
    print(f"Phoenix: d={arch['d_model']}, layers={arch['n_layers']}")

    features = fd["features"]
    concepts = fd["concepts"]
    N = features.size(0)
    max_n_concepts = max(len(c) for c in concepts)
    max_n_concepts = max(max_n_concepts, 1)
    print(f"Papers: {N:,}, max_concepts: {max_n_concepts}")

    print(f"\nRunning Phoenix inference (batch={args.batch_size}, fp16={not args.no_fp16})...", flush=True)
    all_vecs = batch_inference(
        phoenix, features, gnn_emb, ref_lists, concepts,
        args.max_refs, max_n_concepts, args.device,
        batch_size=args.batch_size, use_fp16=not args.no_fp16)

    out_path = os.path.join(args.data_dir, "phoenix_candidate_vectors.pt")
    torch.save(all_vecs, out_path)
    print(f"\nSaved: {out_path}  shape={all_vecs.shape}")


if __name__ == "__main__":
    main()
