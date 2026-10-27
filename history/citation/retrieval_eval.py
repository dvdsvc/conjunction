#!/usr/bin/env python3
"""Two-tower retrieval evaluation for LBD bridge discovery.

User Tower: Phoenix Transformer (user+history → mean pool → L2 norm) → 384-dim
Candidate Tower: GNN emb → feature_proj (via Phoenix weights) → L2 norm → 384-dim
ANN: Faiss IndexFlatIP (cosine similarity on L2-normalized vectors)
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
import faiss

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


def load_model(ckpt_path, config, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = infer_model_config(ckpt["model"], config["scalar_feature_dim"])
    model = PhoenixCitationTransformer(
        scalar_dim=config["scalar_feature_dim"],
        concept_vocab_size=config["concept_vocab_size"],
        **arch, dropout=0.0, output_multiplier=0.125,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, arch


def pad_concepts(list_of_lists):
    max_k = max((len(c) for c in list_of_lists), default=1)
    max_k = max(max_k, 1)
    out = torch.zeros(len(list_of_lists), max_k, dtype=torch.long)
    for i, cidx in enumerate(list_of_lists):
        for j, c in enumerate(cidx):
            out[i, j] = c
    return out


@torch.no_grad()
def get_user_vector(model, user_scalar, user_concepts, user_gnn,
                    hist_scalars, hist_concepts, hist_gnns, device):
    """User Tower: Phoenix Transformer → mean pool → L2 normalize (384-dim)."""
    scalar = torch.cat([user_scalar, hist_scalars], 0).unsqueeze(0)
    gnn    = torch.cat([user_gnn,    hist_gnns],    0).unsqueeze(0)
    c_idx  = pad_concepts([user_concepts] + hist_concepts).unsqueeze(0).to(device)

    S = scalar.size(1)
    mask = torch.ones(1, S, S, device=device)

    # Concept multi-hot
    concept_mh = torch.zeros(1, S, model.concept_proj.in_features, device=device)
    concept_mh.scatter_(2, c_idx.clamp(min=0), 1.0)

    x = torch.cat([scalar.to(device), model.concept_proj(concept_mh), gnn.to(device)], dim=-1)
    x = model.feature_proj(x)
    for block in model.blocks:
        x = block(x, mask)
    x = model.final_norm(x)

    user_repr = x.mean(dim=1)  # (1, d_model)
    return F.normalize(user_repr, p=2, dim=-1).squeeze(0)


@torch.no_grad()
def project_gnn_to_dmodel(model, gnn_emb_batch, device):
    """Candidate Tower: GNN emb → feature_proj → L2 normalize (384-dim).

    Uses Phoenix feature_proj weights with zero scalar + zero concept,
    so GNN embedding is projected through the same learned projection.
    """
    B = gnn_emb_batch.size(0)
    total_in = model.feature_proj[0].in_features
    concept_dim = model.concept_proj.out_features
    gnn_dim = gnn_emb_batch.size(1)
    scalar_dim = total_in - concept_dim - gnn_dim
    scalar_zeros = torch.zeros(B, scalar_dim, device=device)
    concept_zeros = torch.zeros(B, concept_dim, device=device)

    x = torch.cat([scalar_zeros, concept_zeros, gnn_emb_batch.to(device)], dim=-1)
    x = model.feature_proj(x)
    return F.normalize(x, p=2, dim=-1)


def build_faiss_index(vectors_np):
    """Build Faiss IP index (cosine sim) from L2-normalized vectors."""
    d = vectors_np.shape[1]
    faiss.normalize_L2(vectors_np)
    index = faiss.IndexFlatIP(d)
    index.add(vectors_np)
    return index, vectors_np


def evaluate_retrieval(data, device, model=None, arch=None,
                       n_bridges=5000, max_history=50, k_values=None):
    if k_values is None:
        k_values = [10, 50, 100, 500, 1000, 5000]

    bridges = torch.load(os.path.join(data["data_dir"], "lbd_bridges.pt"), weights_only=False)
    features = data["features"]
    gnn_emb  = data["gnn_embeddings"]
    ref_lists = data["ref_lists"]

    year_threshold = (2021 - 1970) / 56.0
    test_bridges = [b for b in bridges if features[b["source_idx"]][0].item() > year_threshold]
    print(f"Test bridges: {len(test_bridges):,}")
    if len(test_bridges) > n_bridges:
        test_bridges = random.sample(test_bridges, n_bridges)

    # Build Faiss index on candidate embeddings
    print("Projecting candidates to d_model space...", flush=True)
    if model is not None:
        # Project all GNN embeddings to d_model via Phoenix feature_proj
        batch_size = 10000
        all_proj = []
        for i in range(0, gnn_emb.size(0), batch_size):
            batch = gnn_emb[i:i+batch_size]
            proj = project_gnn_to_dmodel(model, batch, device)
            all_proj.append(proj.cpu().numpy())
            if i % 100000 == 0:
                print(f"  projected {i:,}/{gnn_emb.size(0):,}", flush=True)
        cand_np = np.concatenate(all_proj, axis=0).astype(np.float32)
    else:
        # Baseline: raw GNN embeddings (64-dim)
        cand_np = gnn_emb.numpy().astype(np.float32)

    print(f"Building Faiss index ({cand_np.shape[0]:,} vectors, dim={cand_np.shape[1]})...", flush=True)
    index, cand_np = build_faiss_index(cand_np.copy())
    print(f"Index ready: {index.ntotal:,} vectors")

    max_k = max(k_values)
    hits = {k: 0 for k in k_values}
    n_eval = 0

    t0 = time.time()
    for pair in test_bridges:
        src_idx, tgt_idx = pair["source_idx"], pair["target_idx"]
        src_refs = ref_lists[src_idx]
        if len(src_refs) < 2:
            continue
        if len(src_refs) > max_history:
            src_refs = random.sample(src_refs, max_history)

        if model is not None:
            user_vec = get_user_vector(
                model,
                features[src_idx].unsqueeze(0), data["concepts"][src_idx], gnn_emb[src_idx].unsqueeze(0),
                features[src_refs], [data["concepts"][i] for i in src_refs], gnn_emb[src_refs],
                device,
            )
            user_vec_np = user_vec.cpu().numpy().astype(np.float32).reshape(1, -1)
        else:
            # Baseline: mean-pool GNN embeddings
            all_gnn = torch.cat([gnn_emb[src_idx].unsqueeze(0), gnn_emb[src_refs]], dim=0)
            user_vec = F.normalize(all_gnn.mean(dim=0, keepdim=True), p=2, dim=-1)
            user_vec_np = user_vec.cpu().numpy().astype(np.float32).reshape(1, -1)

        faiss.normalize_L2(user_vec_np)
        _, indices = index.search(user_vec_np, max_k)

        for k in k_values:
            if tgt_idx in set(indices[0][:k].tolist()):
                hits[k] += 1

        n_eval += 1
        if n_eval % 500 == 0:
            elapsed = time.time() - t0
            r100 = hits.get(100, 0) / n_eval
            r1000 = hits.get(1000, 0) / n_eval
            print(f"  [{n_eval}] R@100={r100:.4f}  R@1000={r1000:.4f}  ({elapsed:.0f}s)", flush=True)

    if n_eval == 0:
        print("No valid pairs!"); return

    mode = "Phoenix Transformer user tower" if model is not None else "GNN mean-pool baseline"
    print(f"\nRetrieval Evaluation — {mode} (TEST SET, {n_eval} pairs)")
    print("=" * 60)
    for k in k_values:
        print(f"  Recall@{k:5d}: {hits[k]/n_eval:.4f}  ({hits[k]}/{n_eval})")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n-bridges", type=int, default=5000)
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--mode", choices=["baseline", "phoenix", "both"], default="both")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), weights_only=False)
    gnn_path = os.path.join(args.data_dir, "gnn_embeddings.pt")
    gnn_emb = torch.load(gnn_path, weights_only=False) if os.path.exists(gnn_path) else torch.zeros(fd["features"].size(0), 64)
    with open(os.path.join(args.data_dir, "config.json")) as f: config = json.load(f)

    data = dict(data_dir=args.data_dir, features=fd["features"], concepts=fd["concepts"],
                gnn_embeddings=gnn_emb, paper_ids=fd["paper_ids"],
                paper_id_to_idx=fd["paper_id_to_idx"], ref_lists=ref_lists, config=config)

    model, arch = None, None
    if args.mode in ("phoenix", "both"):
        ckpt_path = args.checkpoint or os.path.join(args.data_dir, "checkpoints/phoenix_best.pt")
        model, arch = load_model(ckpt_path, config, args.device)
        print(f"Phoenix loaded: d={arch['d_model']} L={arch['n_layers']}")

    if args.mode in ("baseline", "both"):
        print("\n--- GNN mean-pool baseline ---")
        evaluate_retrieval(data, args.device, model=None, arch=None,
                           n_bridges=args.n_bridges, max_history=args.max_history)

    if args.mode in ("phoenix", "both"):
        print("\n--- Phoenix Transformer user tower ---")
        evaluate_retrieval(data, args.device, model=model, arch=arch,
                           n_bridges=args.n_bridges, max_history=args.max_history)


if __name__ == "__main__":
    main()
