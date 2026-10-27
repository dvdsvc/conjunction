#!/usr/bin/env python3
"""Build REAL LBD bridge pairs: A→B→C, where A does NOT cite C.

Multiprocessing version for 208-core machines.
"""
import argparse, os, time, multiprocessing as mp
import numpy as np
import torch

# Shared data (set in worker init)
SHARED = {}

def worker_init(ref_lists_data, concepts_data, max_refs, concept_overlap_thr, n_total):
    SHARED["ref_lists"] = ref_lists_data
    SHARED["concepts"] = concepts_data
    SHARED["max_refs"] = max_refs
    SHARED["thr"] = concept_overlap_thr
    SHARED["n_total"] = n_total

def process_range(args):
    start, end = args
    ref_lists = SHARED["ref_lists"]
    concepts = SHARED["concepts"]
    max_refs = SHARED["max_refs"]
    thr = SHARED["thr"]
    n_total = SHARED["n_total"]

    src_out = []
    tgt_out = []
    jac_out = []

    for a in range(start, end):
        refs_a_raw = ref_lists[a]
        if not refs_a_raw:
            continue
        refs_a = set(refs_a_raw[:max_refs])
        candidates = set()
        for b in refs_a:
            if b < n_total:
                bl = ref_lists[b]
                if bl:
                    candidates.update(bl[:max_refs])
        candidates -= refs_a
        candidates.discard(a)
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
            if jaccard <= thr:
                src_out.append(a)
                tgt_out.append(c)
                jac_out.append(jaccard)

    return (np.array(src_out, dtype=np.int32),
            np.array(tgt_out, dtype=np.int32),
            np.array(jac_out, dtype=np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--max-refs", type=int, default=50)
    p.add_argument("--concept-overlap-thr", type=float, default=0.1)
    p.add_argument("--n-workers", type=int, default=80)
    p.add_argument("--chunk-size", type=int, default=5000)
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), map_location="cpu", weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), map_location="cpu", weights_only=False)
    features = fd["features"]
    concepts = fd["concepts"]
    n_total = len(ref_lists)
    print(f"Papers: {n_total:,}  Workers: {args.n_workers}", flush=True)

    # Build ranges
    ranges = []
    for i in range(0, n_total, args.chunk_size):
        ranges.append((i, min(i + args.chunk_size, n_total)))
    print(f"Chunks: {len(ranges)}", flush=True)

    t0 = time.time()
    with mp.Pool(args.n_workers, initializer=worker_init,
                 initargs=(ref_lists, concepts, args.max_refs, args.concept_overlap_thr, n_total)) as pool:
        results = []
        done = 0
        for src, tgt, jac in pool.imap_unordered(process_range, ranges, chunksize=4):
            if len(src) > 0:
                results.append((src, tgt, jac))
            done += 1
            if done % 200 == 0:
                elapsed = time.time() - t0
                n_bridges = sum(len(r[0]) for r in results)
                rate = done * args.chunk_size / elapsed
                eta = (len(ranges) - done) / 200 * (elapsed / max(done, 1)) / 60
                print(f"  [{done}/{len(ranges)}] bridges={n_bridges:,} "
                      f"rate={rate:.0f}/s ETA={eta:.1f}m", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s", flush=True)

    # Merge
    all_src = np.concatenate([r[0] for r in results]) if results else np.array([], dtype=np.int32)
    all_tgt = np.concatenate([r[1] for r in results]) if results else np.array([], dtype=np.int32)
    all_jac = np.concatenate([r[2] for r in results]) if results else np.array([], np.float32)
    n_bridges = len(all_src)
    print(f"Bridges: {n_bridges:,}", flush=True)

    # Year stats
    year_norm_thr = (2021 - 1970) / 56.0
    pre_2021 = (features[all_src, 0].numpy() <= year_norm_thr).sum()
    post_2021 = n_bridges - pre_2021
    print(f"Pre-2021: {pre_2021:,}", flush=True)
    print(f"Post-2021: {post_2021:,}", flush=True)
    print(f"Jaccard mean={all_jac.mean():.4f} median={np.median(all_jac):.4f}", flush=True)

    # Save
    out_path = os.path.join(args.data_dir, "lbd_bridges_real.pt")
    torch.save({
        "source_idx": torch.from_numpy(all_src),
        "target_idx": torch.from_numpy(all_tgt),
        "jaccard": torch.from_numpy(all_jac),
    }, out_path)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSaved: {out_path}  {size_mb:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
