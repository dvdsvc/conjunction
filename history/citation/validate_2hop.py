#!/usr/bin/env python3
"""Validate 2-hop title embedding approach with IDF filtering.

For 1000 test papers:
1. Build 2-hop citation neighborhoods
2. Compute IDF from global citation counts
3. Filter to top-30 most distinctive papers per 2-hop set
4. Encode titles with sentence-transformers
5. Compare discrimination: positive pairs vs random pairs
"""
import argparse, json, os, random, time, math
from collections import Counter
import torch
import torch.nn.functional as F
import numpy as np


def load_titles_from_jsonl(path, paper_id_to_idx, max_papers=None):
    """Load titles from OpenAlex JSONL files."""
    import orjson
    titles = {}
    count = 0
    for line in open(path, "rb"):
        d = orjson.loads(line)
        pid = d["id"]
        if pid in paper_id_to_idx:
            idx = paper_id_to_idx[pid]
            title = d.get("title", "")
            if title:
                titles[idx] = title
                count += 1
                if max_papers and count >= max_papers:
                    break
    return titles


def build_2hop_neighborhood(idx, ref_lists, max_1hop=50):
    """Build 1-hop and 2-hop citation sets for a paper."""
    refs_1hop = ref_lists[idx][:max_1hop]
    refs_2hop = set()
    for r in refs_1hop:
        for r2 in ref_lists[r]:
            if r2 != idx and r2 not in refs_1hop:
                refs_2hop.add(r2)
    return refs_1hop, list(refs_2hop)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--raw-dir", default="/root/autodl-tmp/openalex_data")
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--top-k-2hop", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), map_location="cpu", weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), map_location="cpu", weights_only=False)
    gnn_emb = torch.load(os.path.join(args.data_dir, "gnn_embeddings.pt"), map_location="cpu", weights_only=False)

    features = fd["features"]
    paper_id_to_idx = fd["paper_id_to_idx"]
    N = features.size(0)
    print(f"Papers: {N:,}")

    # Step 1: Compute global citation counts for IDF
    print("Computing global citation counts...", flush=True)
    t0 = time.time()
    citation_count = Counter()
    for i in range(N):
        for r in ref_lists[i]:
            citation_count[r] += 1
    print(f"Citation counts done in {time.time()-t0:.0f}s, {len(citation_count):,} cited papers")

    # IDF = log(N / citation_count)
    idf = {}
    for pid, cnt in citation_count.items():
        idf[pid] = math.log(N / max(cnt, 1))

    # Step 2: Load titles (from JSONL files)
    print("Loading titles from JSONL...", flush=True)
    t0 = time.time()
    titles = {}
    for name in ["climate_papers.jsonl", "geology_papers.jsonl"]:
        path = os.path.join(args.raw_dir, name)
        if os.path.exists(path):
            loaded = load_titles_from_jsonl(path, paper_id_to_idx)
            titles.update(loaded)
            print(f"  {name}: {len(loaded):,} titles")
    print(f"Total titles loaded: {len(titles):,} in {time.time()-t0:.0f}s")

    # Step 3: Select test papers (with enough refs and titles)
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges.pt"), map_location="cpu", weights_only=False)
    year_norm_threshold = (2021 - 1970) / 56.0
    test_bridges = [b for b in bridges if features[b["source_idx"]][0].item() > year_norm_threshold]
    random.seed(42)
    random.shuffle(test_bridges)

    test_papers = []
    for b in test_bridges:
        src = b["source_idx"]
        tgt = b["target_idx"]
        if src in titles and tgt in titles and len(ref_lists[src]) >= 3 and len(ref_lists[tgt]) >= 3:
            test_papers.append((src, tgt))
            if len(test_papers) >= args.n_test:
                break
    print(f"Test pairs: {len(test_papers)}")

    # Step 4: Build 2-hop neighborhoods + IDF filtering
    print("Building 2-hop neighborhoods with IDF filtering...", flush=True)
    t0 = time.time()

    def get_filtered_2hop(idx, top_k=30):
        refs_1hop, refs_2hop = build_2hop_neighborhood(idx, ref_lists)
        # Filter: only keep papers we have titles for
        refs_1hop_titled = [r for r in refs_1hop if r in titles]
        refs_2hop_titled = [r for r in refs_2hop if r in titles]
        # IDF filter on 2-hop: keep top-k most distinctive
        refs_2hop_titled.sort(key=lambda r: idf.get(r, 0), reverse=True)
        refs_2hop_filtered = refs_2hop_titled[:top_k]
        return refs_1hop_titled, refs_2hop_filtered

    # Collect all unique paper indices needed
    all_needed = set()
    neighborhoods = {}
    for src, tgt in test_papers:
        for paper_idx in [src, tgt]:
            if paper_idx not in neighborhoods:
                h1, h2 = get_filtered_2hop(paper_idx, args.top_k_2hop)
                neighborhoods[paper_idx] = (h1, h2)
                all_needed.add(paper_idx)
                all_needed.update(h1)
                all_needed.update(h2)

    print(f"Unique papers to encode: {len(all_needed):,} in {time.time()-t0:.0f}s")

    # Step 5: Encode titles with sentence-transformers
    print("Loading sentence-transformers...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device=args.device)
    print(f"Model loaded. Encoding {len(all_needed)} titles...", flush=True)

    # Encode in batches
    all_needed_list = sorted(all_needed)
    batch_size = 1024
    all_embeddings = {}

    t0 = time.time()
    for start in range(0, len(all_needed_list), batch_size):
        end = min(start + batch_size, len(all_needed_list))
        batch_indices = all_needed_list[start:end]
        batch_titles = [titles.get(idx, "") for idx in batch_indices]
        # Replace empty with placeholder
        batch_titles = [t if t else "unknown paper" for t in batch_titles]

        embs = model.encode(batch_titles, batch_size=512, show_progress_bar=False,
                           convert_to_tensor=True, normalize_embeddings=True)
        for i, idx in enumerate(batch_indices):
            all_embeddings[idx] = embs[i].cpu()

        if start % 5000 == 0 and start > 0:
            print(f"  [{start}/{len(all_needed_list)}]", flush=True)

    print(f"Encoding done in {time.time()-t0:.0f}s")

    # Step 6: Build paper representations
    def build_paper_repr(idx, mode="2hop_idf"):
        """Build paper representation using different strategies."""
        h1, h2 = neighborhoods[idx]

        self_emb = all_embeddings.get(idx, torch.zeros(384))

        if mode == "self_only":
            return self_emb

        elif mode == "1hop":
            if not h1:
                return self_emb
            h1_embs = torch.stack([all_embeddings[r] for r in h1 if r in all_embeddings])
            h1_mean = h1_embs.mean(dim=0)
            h1_mean = F.normalize(h1_mean, p=2, dim=-1)
            return torch.cat([self_emb, h1_mean])

        elif mode == "2hop_avg":
            # All 2-hop without IDF filter
            _, h2_all = build_2hop_neighborhood(idx, ref_lists)
            h2_all = [r for r in h2_all if r in all_embeddings]
            if not h2_all:
                return self_emb
            h2_embs = torch.stack([all_embeddings[r] for r in h2_all])
            h2_mean = F.normalize(h2_embs.mean(dim=0), p=2, dim=-1)
            h1_embs = torch.stack([all_embeddings[r] for r in h1 if r in all_embeddings])
            h1_mean = F.normalize(h1_embs.mean(dim=0), p=2, dim=-1)
            return torch.cat([self_emb, h1_mean, h2_mean])

        elif mode == "2hop_idf":
            # IDF-filtered 2-hop
            if not h1 and not h2:
                return self_emb
            parts = [self_emb]
            if h1:
                h1_embs = torch.stack([all_embeddings[r] for r in h1 if r in all_embeddings])
                if len(h1_embs) > 0:
                    parts.append(F.normalize(h1_embs.mean(dim=0), p=2, dim=-1))
            if h2:
                h2_embs = torch.stack([all_embeddings[r] for r in h2 if r in all_embeddings])
                if len(h2_embs) > 0:
                    parts.append(F.normalize(h2_embs.mean(dim=0), p=2, dim=-1))
            return torch.cat(parts)

    # Step 7: Compare discrimination across modes
    modes = ["self_only", "1hop", "2hop_avg", "2hop_idf"]

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Mode: {mode}")

        # Build representations for all test papers
        src_reprs = []
        tgt_reprs = []
        for src, tgt in test_papers:
            src_reprs.append(build_paper_repr(src, mode))
            tgt_reprs.append(build_paper_repr(tgt, mode))

        src_reprs = torch.stack(src_reprs)
        tgt_reprs = torch.stack(tgt_reprs)

        # Positive pairs (src → tgt)
        pos_sims = F.cosine_similarity(src_reprs, tgt_reprs, dim=1).numpy()

        # Random pairs
        rng = random.Random(123)
        rand_sims = []
        for _ in range(500):
            i = rng.randint(0, len(test_papers)-1)
            j = rng.randint(0, len(test_papers)-1)
            sim = F.cosine_similarity(src_reprs[i:i+1], tgt_reprs[j:j+1]).item()
            rand_sims.append(sim)
        rand_sims = np.array(rand_sims)

        # Also compare with gnn_emb baseline
        gnn_src = torch.stack([gnn_emb[src] for src, _ in test_papers])
        gnn_tgt = torch.stack([gnn_emb[tgt] for _, tgt in test_papers])
        gnn_pos = F.cosine_similarity(gnn_src, gnn_tgt, dim=1).numpy()
        gnn_rand = []
        for _ in range(500):
            i = rng.randint(0, len(test_papers)-1)
            j = rng.randint(0, len(test_papers)-1)
            sim = F.cosine_similarity(gnn_src[i:i+1], gnn_tgt[j:j+1]).item()
            gnn_rand.append(sim)
        gnn_rand = np.array(gnn_rand)

        print(f"  Embedding dim: {src_reprs.shape[1]}")
        print(f"  Positive pairs:  mean={pos_sims.mean():.4f}  std={pos_sims.std():.4f}")
        print(f"  Random pairs:    mean={rand_sims.mean():.4f}  std={rand_sims.std():.4f}")
        print(f"  Discrimination:  {pos_sims.mean() - rand_sims.mean():+.4f}")
        print(f"  --- gnn_emb baseline ---")
        print(f"  Positive pairs:  mean={gnn_pos.mean():.4f}")
        print(f"  Random pairs:    mean={gnn_rand.mean():.4f}")
        print(f"  Discrimination:  {gnn_pos.mean() - gnn_rand.mean():+.4f}")


if __name__ == "__main__":
    main()
