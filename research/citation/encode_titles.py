#!/usr/bin/env python3
"""Encode paper titles into embeddings via sentence-transformers.

Input:  titles.jsonl + features.pt (for paper_id ordering)
Output: title_embeddings.pt  (n_papers, 384)

Usage:
  python encode_titles.py --data-dir /root/autodl-tmp/openalex_data/processed
"""
import argparse, json, os, time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--titles-path", default=None,
                   help="titles.jsonl path (default: <data-dir>/../titles.jsonl)")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    titles_path = args.titles_path or os.path.join(args.data_dir, "../titles.jsonl")

    # 1. Load paper_id ordering from features.pt
    print("Loading paper_ids from features.pt...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"),
                    map_location="cpu", weights_only=False)
    paper_ids = fd["paper_ids"]   # list[str] — index order
    n = len(paper_ids)
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    print(f"  {n:,} papers in index")

    # 2. Load titles
    print(f"Loading titles from {titles_path}...", flush=True)
    titles = [""] * n
    loaded = 0
    with open(titles_path) as f:
        for line in f:
            d = json.loads(line)
            idx = id_to_idx.get(d["id"])
            if idx is not None:
                titles[idx] = d.get("title") or ""
                loaded += 1
    print(f"  {loaded:,} titles matched, {n - loaded:,} empty")

    # 3. Encode
    print(f"Loading model '{args.model}'...", flush=True)
    model = SentenceTransformer(args.model, device=args.device)
    emb_dim = model.get_sentence_embedding_dimension()
    print(f"  embedding dim = {emb_dim}")

    print(f"Encoding {n:,} titles (batch_size={args.batch_size})...", flush=True)
    t0 = time.time()

    # sentence-transformers handles batching internally with show_progress_bar
    embeddings = model.encode(
        titles,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f}m  ({n/max(elapsed,1):.0f} titles/s)")

    # 4. Save
    emb_tensor = torch.from_numpy(embeddings).float()
    out_path = os.path.join(args.data_dir, "title_embeddings.pt")
    torch.save(emb_tensor, out_path)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Saved: {out_path}  shape={emb_tensor.shape}  {size_mb:.0f}MB")


if __name__ == "__main__":
    main()
