#!/usr/bin/env python3
"""Build REAL LBD bridge pairs: A→B→C, where A does NOT cite C.

Real LBD bridges = 2-hop indirect connections that are missing direct edges.
Also filters for cross-domain (low concept overlap between A and C).
"""
import argparse, json, os, time
from collections import Counter
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--max-refs", type=int, default=50, help="Cap refs per paper for speed")
    p.add_argument("--concept-overlap-thr", type=float, default=0.1,
                   help="Max Jaccard overlap between A and C concepts to be 'cross-domain'")
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), map_location="cpu", weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), map_location="cpu", weights_only=False)

    features = fd["features"]
    concepts = fd["concepts"]
    n_total = len(ref_lists)
    print(f"Papers: {n_total:,}", flush=True)

    # Convert ref_lists to sets for fast lookup
    print("Converting ref_lists to sets...", flush=True)
    ref_sets = [set(r[:args.max_refs]) for r in ref_lists]

    # For each paper A:
    #   for each B in refs(A):
    #     for each C in refs(B):
    #       if C not in refs(A) and C != A:
    #         compute concept overlap between A and C
    #         if low overlap → bridge pair
    print("Finding 2-hop candidates...", flush=True)
    t0 = time.time()

    bridges = []
    stats = {"total_2hop": 0, "already_cited": 0, "self_loop": 0, "same_domain": 0, "kept": 0}

    for a in range(n_total):
        refs_a = ref_sets[a]
        if not refs_a:
            continue
        # Collect all 2-hop candidates
        candidates = set()
        for b in refs_a:
            if b < n_total:
                candidates.update(ref_lists[b][:args.max_refs])
        # Remove self and already cited
        candidates -= refs_a
        candidates.discard(a)
        stats["total_2hop"] += len(candidates) + len(refs_a)  # total before filter
        stats["already_cited"] += len(refs_a)

        if not candidates:
            continue

        # Concept overlap filter
        concepts_a = concepts[a]
        if not concepts_a:
            continue
        set_a = set(concepts_a)
        for c in candidates:
            concepts_c = concepts[c]
            if not concepts_c:
                continue
            set_c = set(concepts_c)
            inter = len(set_a & set_c)
            union = len(set_a | set_c)
            jaccard = inter / union if union > 0 else 1.0
            if jaccard <= args.concept_overlap_thr:
                bridges.append({"source_idx": a, "target_idx": c, "jaccard": jaccard})
                stats["kept"] += 1
            else:
                stats["same_domain"] += 1

        if (a + 1) % 500000 == 0:
            elapsed = time.time() - t0
            rate = (a + 1) / elapsed
            eta = (n_total - a - 1) / rate / 60
            print(f"  [{a+1:,}/{n_total:,}] bridges={stats['kept']:,} "
                  f"rate={rate:.0f}/s ETA={eta:.1f}m", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s", flush=True)
    print(f"\nStats:", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v:,}", flush=True)

    print(f"\nTotal real bridges: {len(bridges):,}", flush=True)

    # Year distribution
    year_norm_thr = (2021 - 1970) / 56.0
    pre_2021 = sum(1 for b in bridges if features[b["source_idx"]][0].item() <= year_norm_thr)
    post_2021 = len(bridges) - pre_2021
    print(f"  Pre-2021 (train): {pre_2021:,}", flush=True)
    print(f"  Post-2021 (test): {post_2021:,}", flush=True)

    # Sample stats
    jaccards = [b["jaccard"] for b in bridges[:100000]]
    print(f"  Jaccard mean={np.mean(jaccards):.4f} median={np.median(jaccards):.4f}", flush=True)

    out_path = os.path.join(args.data_dir, "lbd_bridges_real.pt")
    torch.save(bridges, out_path)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
