#!/usr/bin/env python3
"""Build REAL LBD bridge pairs: A→B→C, where A does NOT cite C.

Memory-efficient: stores as 3 parallel numpy arrays, not dict list.
"""
import argparse, os, time
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--max-refs", type=int, default=50)
    p.add_argument("--concept-overlap-thr", type=float, default=0.1)
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), map_location="cpu", weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), map_location="cpu", weights_only=False)

    features = fd["features"]
    concepts = fd["concepts"]
    n_total = len(ref_lists)
    print(f"Papers: {n_total:,}", flush=True)

    # Pre-allocate output arrays (grow in chunks)
    CHUNK = 5_000_000
    src_buf = np.zeros(CHUNK, dtype=np.int32)
    tgt_buf = np.zeros(CHUNK, dtype=np.int32)
    jac_buf = np.zeros(CHUNK, dtype=np.float32)
    buf_idx = 0

    # Final arrays (list of flushed chunks)
    flushed_src = []
    flushed_tgt = []
    flushed_jac = []

    def flush_buffer():
        nonlocal buf_idx
        if buf_idx > 0:
            flushed_src.append(src_buf[:buf_idx].copy())
            flushed_tgt.append(tgt_buf[:buf_idx].copy())
            flushed_jac.append(jac_buf[:buf_idx].copy())
            buf_idx = 0

    def add_bridge(a, c, j):
        nonlocal buf_idx
        src_buf[buf_idx] = a
        tgt_buf[buf_idx] = c
        jac_buf[buf_idx] = j
        buf_idx += 1
        if buf_idx >= CHUNK:
            flush_buffer()

    print("Converting ref_lists to sets...", flush=True)
    ref_sets = [set(r[:args.max_refs]) for r in ref_lists]

    print("Finding 2-hop candidates...", flush=True)
    t0 = time.time()
    total_2hop = 0
    same_domain = 0

    for a in range(n_total):
        refs_a = ref_sets[a]
        if not refs_a:
            continue
        candidates = set()
        for b in refs_a:
            if b < n_total:
                candidates.update(ref_lists[b][:args.max_refs])
        candidates -= refs_a
        candidates.discard(a)
        total_2hop += len(candidates)

        if not candidates:
            continue

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
                add_bridge(a, c, jaccard)
            else:
                same_domain += 1

        if (a + 1) % 500000 == 0:
            elapsed = time.time() - t0
            rate = (a + 1) / elapsed
            eta = (n_total - a - 1) / rate / 60
            n_bridges = sum(len(c) for c in flushed_src) + buf_idx
            print(f"  [{a+1:,}/{n_total:,}] bridges={n_bridges:,} "
                  f"rate={rate:.0f}/s ETA={eta:.1f}m", flush=True)

    flush_buffer()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s", flush=True)

    # Merge chunks
    all_src = np.concatenate(flushed_src) if flushed_src else np.array([], dtype=np.int32)
    all_tgt = np.concatenate(flushed_tgt) if flushed_tgt else np.array([], dtype=np.int32)
    all_jac = np.concatenate(flushed_jac) if flushed_jac else np.array([], np.float32)
    n_bridges = len(all_src)

    print(f"total_2hop: {total_2hop:,}", flush=True)
    print(f"same_domain: {same_domain:,}", flush=True)
    print(f"bridges: {n_bridges:,}", flush=True)

    # Year distribution
    year_norm_thr = (2021 - 1970) / 56.0
    pre_2021 = (features[all_src, 0].numpy() <= year_norm_thr).sum()
    post_2021 = n_bridges - pre_2021
    print(f"Pre-2021 (train): {pre_2021:,}", flush=True)
    print(f"Post-2021 (test): {post_2021:,}", flush=True)
    print(f"Jaccard mean={all_jac.mean():.4f} median={np.median(all_jac):.4f}", flush=True)

    # Save as torch tensors (much smaller than dict list)
    out_path = os.path.join(args.data_dir, "lbd_bridges_real.pt")
    torch.save({
        "source_idx": torch.from_numpy(all_src),
        "target_idx": torch.from_numpy(all_tgt),
        "jaccard": torch.from_numpy(all_jac),
    }, out_path)
    import os as _os
    size_mb = _os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSaved: {out_path}  {size_mb:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
