"""Evaluate Multi-Interest model on real LBD bridge retrieval.

1. Pre-compute candidate vectors (single title embedding → model → L2)
2. Build Faiss index
3. For each test source: get K interest vectors → search K times → merge
4. Report recall@K
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn.functional as F
import faiss

from multi_interest_model import MultiInterestModel


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--n-eval", type=int, default=5000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("Loading data...", flush=True)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"),
                           map_location="cpu", weights_only=False)
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges_real.pt"),
                         map_location="cpu", weights_only=False)
    title_embeddings = torch.load(os.path.join(args.data_dir, "title_embeddings.pt"),
                                  map_location="cpu", weights_only=False)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"),
                    map_location="cpu", weights_only=False)
    features = fd["features"]
    n_total = title_embeddings.size(0)
    title_dim = title_embeddings.size(1)

    # Load model
    ckpt_path = os.path.join(args.data_dir, "checkpoints/multi_interest_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MultiInterestModel(
        title_dim=title_dim, d_model=ckpt["d_model"], k_max=ckpt["k_max"]
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    k_max = ckpt["k_max"]
    d_model = ckpt["d_model"]
    print(f"Model: d_model={d_model}, k_max={k_max}, epoch={ckpt['epoch']}", flush=True)

    # ── Step 1: Compute candidate vectors ──
    cand_path = os.path.join(args.data_dir, "multi_interest_candidates.pt")
    if os.path.exists(cand_path):
        print("Loading cached candidate vectors...", flush=True)
        all_cands = torch.load(cand_path, map_location="cpu", weights_only=False)
    else:
        print(f"Computing {n_total:,} candidate vectors...", flush=True)
        batch_size = 4096
        all_cands = torch.zeros(n_total, d_model)
        t0 = time.time()
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch = title_embeddings[start:end].to(args.device)
            vecs = model.cand_forward(batch).cpu()
            all_cands[start:end] = vecs
            if start % (batch_size * 100) == 0 and start > 0:
                elapsed = time.time() - t0
                rate = start / elapsed
                eta = (n_total - start) / rate / 60
                print(f"  [{start:,}/{n_total:,}] {rate:.0f}/s ETA={eta:.1f}m", flush=True)
        torch.save(all_cands, cand_path)
        print(f"Saved {cand_path}", flush=True)

    # ── Step 2: Build Faiss index ──
    print("Building Faiss index...", flush=True)
    corpus = all_cands.numpy().astype(np.float32)
    index = faiss.IndexFlatIP(d_model)
    index.add(corpus)
    print(f"Index: {index.ntotal:,} vectors", flush=True)

    # ── Step 3: Prepare test bridges ──
    src_all = bridges["source_idx"].numpy()
    tgt_all = bridges["target_idx"].numpy()
    year_thr = (2021 - 1970) / 56.0
    mask = features[src_all, 0].numpy() > year_thr
    src_test = src_all[mask]
    tgt_test = tgt_all[mask]
    print(f"Test bridges: {len(src_test):,}", flush=True)

    import random
    rng = random.Random(42)
    idxs = rng.sample(range(len(src_test)), min(args.n_eval, len(src_test)))
    src_eval = src_test[idxs]
    tgt_eval = tgt_test[idxs]
    print(f"Evaluating: {len(idxs)}", flush=True)

    # ── Step 4: Multi-interest retrieval ──
    print("Retrieving with multi-interest vectors...", flush=True)
    t0 = time.time()
    max_refs = 50
    max_k = 5000
    all_ranks = []

    batch_size = 64
    for batch_start in range(0, len(src_eval), batch_size):
        batch_end = min(batch_start + batch_size, len(src_eval))
        B = batch_end - batch_start

        # Build user title sequences
        max_len = 0
        ref_seqs = []
        for i in range(B):
            refs = ref_lists[src_eval[batch_start + i]][:max_refs]
            ref_seqs.append(refs)
            max_len = max(max_len, len(refs))

        D = title_dim
        user_titles = torch.zeros(B, max_len, D)
        seq_len = torch.zeros(B, dtype=torch.long)
        for i, refs in enumerate(ref_seqs):
            if refs:
                user_titles[i, :len(refs)] = title_embeddings[refs]
            seq_len[i] = len(refs)

        user_titles = user_titles.to(args.device)
        seq_len = seq_len.to(args.device)

        # K interest vectors: (B, K, d_model)
        interests = model.user_forward(user_titles, seq_len)

        # Search each interest separately → merge
        for i in range(B):
            target = tgt_eval[batch_start + i]
            best_rank = max_k

            for k in range(k_max):
                q = interests[i, k].cpu().numpy().astype(np.float32).reshape(1, -1)
                scores, indices = index.search(q, max_k)
                matches = np.where(indices[0] == target)[0]
                if len(matches) > 0:
                    best_rank = min(best_rank, int(matches[0]))

            all_ranks.append(best_rank)

        if (batch_start // batch_size) % 10 == 0 and batch_start > 0:
            elapsed = time.time() - t0
            rate = batch_start / elapsed
            eta = (len(src_eval) - batch_start) / rate / 60
            print(f"  [{batch_start}/{len(src_eval)}] {rate:.0f}/s ETA={eta:.1f}m", flush=True)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s", flush=True)

    # ── Step 5: Report ──
    all_ranks = np.array(all_ranks)
    print(f"\n─── Multi-Interest Retrieval (k_max={k_max}) ───")
    print(f"  Papers: {len(all_ranks)}")
    print(f"{'K':>6}  {'Recall':>8}  {'Hits':>8}")
    print("─" * 30)
    for k in [1, 5, 10, 20, 50, 100, 500, 1000, 5000]:
        recall = (all_ranks < k).sum() / len(all_ranks)
        hits = int((all_ranks < k).sum())
        print(f"{k:>6}  {recall*100:>7.1f}%  {hits:>4}/{len(all_ranks)}")

    found = all_ranks < max_k
    mrr = (1.0 / (all_ranks[found] + 1)).sum() / len(all_ranks)
    print(f"\nMRR@{max_k}: {mrr:.4f}")
    print(f"Found in top-{max_k}: {found.sum()}/{len(all_ranks)} ({found.sum()/len(all_ranks)*100:.1f}%)")


if __name__ == "__main__":
    main()
