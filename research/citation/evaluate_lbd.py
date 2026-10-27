#!/usr/bin/env python3
"""Evaluate Phoenix on LBD bridge prediction — TEST-SET ONLY (>2021)."""
import argparse, json, os, random
import torch
from phoenix_model import PhoenixCitationTransformer


def pad_concepts(list_of_lists):
    max_k = max((len(c) for c in list_of_lists), default=1)
    max_k = max(max_k, 1)
    out = torch.zeros(len(list_of_lists), max_k, dtype=torch.long)
    for i, cidx in enumerate(list_of_lists):
        for j, c in enumerate(cidx):
            out[i, j] = c
    return out


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
    print(f"Arch: d={arch['d_model']} h={arch['n_heads']} L={arch['n_layers']}")
    model = PhoenixCitationTransformer(
        scalar_dim=config["scalar_feature_dim"],
        concept_vocab_size=config["concept_vocab_size"],
        **arch, dropout=0.0, output_multiplier=0.125,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def score_candidates(model, user_s, user_c, user_g, hist_s, hist_c, hist_g,
                     cand_s, cand_c, cand_g, device):
    scalar = torch.cat([user_s, hist_s, cand_s], 0).unsqueeze(0)
    gnn    = torch.cat([user_g, hist_g, cand_g], 0).unsqueeze(0)
    c_idx  = pad_concepts([user_c] + hist_c + cand_c).unsqueeze(0)

    n_uh = 1 + hist_s.size(0)
    n_c  = cand_s.size(0)
    S = n_uh + n_c

    mask = torch.zeros(1, S, S, device=device)
    for i in range(n_uh):
        mask[0, i, :i+1] = 1.0
    for i in range(n_uh, S):
        mask[0, i, :n_uh] = 1.0
        mask[0, i, i] = 1.0

    logits = model(scalar.to(device), c_idx.to(device), gnn.to(device), mask)
    return logits[0, n_uh:]


def evaluate_lbd(model, data, device, n_neg=99, max_history=50, n_bridges=5000):
    bridges = torch.load(os.path.join(data["data_dir"], "lbd_bridges.pt"), weights_only=False)
    if not bridges:
        print("No bridge pairs!"); return

    # Filter to TEST set only: source paper year > 2021
    # year_norm = (year - 1970) / 56, so year > 2021 => year_norm > 51/56 ≈ 0.9107
    features = data["features"]
    year_threshold = (2021 - 1970) / 56.0  # 0.9107

    test_bridges = []
    for b in bridges:
        src_year_norm = features[b["source_idx"]][0].item()
        if src_year_norm > year_threshold:
            test_bridges.append(b)

    print(f"Total bridges: {len(bridges):,}, test bridges (>2021): {len(test_bridges):,}")

    if len(test_bridges) == 0:
        print("No test-set bridge pairs!"); return

    if len(test_bridges) > n_bridges:
        test_bridges = random.sample(test_bridges, n_bridges)
        print(f"Sampled {n_bridges} test bridges")

    concepts = data["concepts"]
    gnn_emb  = data["gnn_embeddings"]
    n_total  = features.size(0)
    ref_lists = data["ref_lists"]

    recalls_at = [1, 5, 10, 20]
    hits = {k: 0 for k in recalls_at}
    mrr_sum = 0
    n_eval = 0

    for pair in test_bridges:
        src_idx, tgt_idx = pair["source_idx"], pair["target_idx"]
        src_refs = ref_lists[src_idx]
        if len(src_refs) < 2:
            continue
        if len(src_refs) > max_history:
            src_refs = random.sample(src_refs, max_history)

        neg_indices = []
        exclude = {src_idx, tgt_idx} | set(src_refs)
        attempts = 0
        while len(neg_indices) < n_neg and attempts < n_neg * 3:
            idx = random.randint(0, n_total - 1)
            if idx not in exclude:
                neg_indices.append(idx)
            attempts += 1
        if len(neg_indices) < n_neg:
            continue

        cand_indices = [tgt_idx] + neg_indices
        random.shuffle(cand_indices)
        target_pos = cand_indices.index(tgt_idx)

        scores = score_candidates(
            model,
            features[src_idx].unsqueeze(0), concepts[src_idx], gnn_emb[src_idx].unsqueeze(0),
            features[src_refs],  [concepts[i] for i in src_refs], gnn_emb[src_refs],
            features[cand_indices], [concepts[i] for i in cand_indices], gnn_emb[cand_indices],
            device,
        )

        target_score = scores[target_pos].item()
        rank = (scores > target_score).sum().item() + 1

        for k in recalls_at:
            if rank <= k: hits[k] += 1
        mrr_sum += 1.0 / rank
        n_eval += 1

        if n_eval % 500 == 0:
            print(f"  [{n_eval}] R@10={hits[10]/n_eval:.4f}  R@20={hits[20]/n_eval:.4f}  MRR={mrr_sum/n_eval:.4f}", flush=True)

    if n_eval == 0:
        print("No valid bridge pairs!"); return

    print(f"\nLBD Bridge Evaluation — TEST SET ONLY (source year > 2021)")
    print(f"  ({n_eval} pairs, {n_neg} negatives)")
    print("=" * 55)
    for k in recalls_at:
        print(f"  Recall@{k:2d}: {hits[k]/n_eval:.4f}  ({hits[k]}/{n_eval})")
    print(f"  MRR:       {mrr_sum/n_eval:.4f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n-neg", type=int, default=99)
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--n-bridges", type=int, default=5000)
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

    ckpt_path = args.checkpoint or os.path.join(args.data_dir, "checkpoints/phoenix_best.pt")
    model = load_model(ckpt_path, config, args.device)
    evaluate_lbd(model, data, args.device, n_neg=args.n_neg,
                 max_history=args.max_history, n_bridges=args.n_bridges)

if __name__ == "__main__":
    main()
