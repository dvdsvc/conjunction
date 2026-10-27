#!/usr/bin/env python3
"""Validate 2-hop concept-based paper representation.

No sentence-transformers needed. Uses concept IDs as features.

For each paper:
  1-hop: papers it cites
  2-hop: papers cited by 1-hop papers
  Representation: IDF-weighted concept vector from 2-hop neighborhood

Compares discrimination: positive pairs vs random pairs.
Also downloads sentence-transformers model in parallel.
"""
import argparse, json, os, random, time, math
from collections import Counter
import torch
import torch.nn.functional as F
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--top-k-2hop", type=int, default=30)
    p.add_argument("--download-model", action="store_true")
    args = p.parse_args()

    # Optionally download sentence-transformers model
    if args.download_model:
        print("Downloading sentence-transformers model...", flush=True)
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Download failed: {e}")

    print("Loading data...", flush=True)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"), map_location="cpu", weights_only=False)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"), map_location="cpu", weights_only=False)

    concepts = fd["concepts"]  # list of lists of concept IDs
    features = fd["features"]
    N = features.size(0)
    print(f"Papers: {N:,}")

    # Concept vocabulary
    concept_vocab_size = max(max(c) if c else 0 for c in concepts) + 1
    print(f"Concept vocab: {concept_vocab_size}")

    # Step 1: Global concept IDF
    print("Computing concept IDF...", flush=True)
    concept_doc_count = Counter()
    for c in concepts:
        for cid in c:
            concept_doc_count[cid] += 1
    concept_idf = {}
    for cid, cnt in concept_doc_count.items():
        concept_idf[cid] = math.log(N / max(cnt, 1))

    # Step 2: Global paper citation count (for paper-level IDF)
    print("Computing citation counts...", flush=True)
    citation_count = Counter()
    for i in range(N):
        for r in ref_lists[i]:
            citation_count[r] += 1

    # Step 3: Select test papers
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges.pt"), map_location="cpu", weights_only=False)
    year_norm_threshold = (2021 - 1970) / 56.0
    test_bridges = [b for b in bridges if features[b["source_idx"]][0].item() > year_norm_threshold]
    random.seed(42)
    random.shuffle(test_bridges)

    test_pairs = []
    for b in test_bridges:
        src, tgt = b["source_idx"], b["target_idx"]
        if len(ref_lists[src]) >= 3 and len(ref_lists[tgt]) >= 3:
            test_pairs.append((src, tgt))
            if len(test_pairs) >= args.n_test:
                break
    print(f"Test pairs: {len(test_pairs)}")

    # Step 4: Build concept representations with different strategies
    def get_concepts_1hop(idx):
        """All unique concept IDs from 1-hop citations."""
        all_concepts = set()
        for r in ref_lists[idx][:50]:
            all_concepts.update(concepts[r])
        return all_concepts

    def get_concepts_2hop(idx, top_k_papers=30, idf_filter=True):
        """Concepts from 2-hop neighborhood, optionally filtered."""
        refs_1hop = ref_lists[idx][:50]

        # Collect 2-hop papers
        refs_2hop = []
        for r in refs_1hop:
            for r2 in ref_lists[r]:
                if r2 != idx and r2 not in refs_1hop:
                    refs_2hop.append(r2)

        # IDF filter on 2-hop papers: keep most distinctive
        if idf_filter:
            refs_2hop.sort(key=lambda r: citation_count.get(r, 0))
            refs_2hop = refs_2hop[:top_k_papers]

        # Collect all concept IDs
        all_concepts = set()
        for r in refs_1hop:
            all_concepts.update(concepts[r])
        for r in refs_2hop:
            all_concepts.update(concepts[r])

        return all_concepts

    def concept_to_sparse_vec(concept_set, vocab_size):
        """Convert concept set to sparse IDF-weighted vector."""
        vec = torch.zeros(vocab_size)
        for cid in concept_set:
            if cid < vocab_size:
                vec[cid] = concept_idf.get(cid, 0)
        # L2 normalize
        norm = vec.norm()
        if norm > 0:
            vec = vec / norm
        return vec

    # Test different modes
    modes = {
        "self_concepts": lambda idx: set(concepts[idx]),
        "1hop_concepts": get_concepts_1hop,
        "2hop_no_filter": lambda idx: get_concepts_2hop(idx, top_k_papers=1000, idf_filter=False),
        "2hop_idf_top30": lambda idx: get_concepts_2hop(idx, top_k_papers=30, idf_filter=True),
        "2hop_idf_top100": lambda idx: get_concepts_2hop(idx, top_k_papers=100, idf_filter=True),
    }

    for mode_name, mode_fn in modes.items():
        print(f"\n{'='*60}")
        print(f"Mode: {mode_name}")

        t0 = time.time()

        # Build representations
        src_vecs = []
        tgt_vecs = []
        for src, tgt in test_pairs:
            src_concepts = mode_fn(src)
            tgt_concepts = mode_fn(tgt)
            src_vecs.append(concept_to_sparse_vec(src_concepts, concept_vocab_size))
            tgt_vecs.append(concept_to_sparse_vec(tgt_concepts, concept_vocab_size))

        src_mat = torch.stack(src_vecs)  # (N, vocab)
        tgt_mat = torch.stack(tgt_vecs)

        # Positive pairs: cosine sim
        pos_sims = F.cosine_similarity(src_mat, tgt_mat, dim=1).numpy()

        # Random pairs
        rng = random.Random(123)
        n_test = min(1000, len(test_pairs))
        rand_sims = []
        for _ in range(n_test):
            i = rng.randint(0, len(test_pairs)-1)
            j = rng.randint(0, len(test_pairs)-1)
            sim = F.cosine_similarity(src_mat[i:i+1], tgt_mat[j:j+1]).item()
            rand_sims.append(sim)
        rand_sims = np.array(rand_sims)

        # Stats
        src_nnz = np.array([v.count_nonzero() for v in src_vecs])
        tgt_nnz = np.array([v.count_nonzero() for v in tgt_vecs])

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Avg concepts per paper: src={src_nnz.mean():.0f}, tgt={tgt_nnz.mean():.0f}")
        print(f"  Positive pairs:  mean={pos_sims.mean():.4f}  std={pos_sims.std():.4f}")
        print(f"  Random pairs:    mean={rand_sims.mean():.4f}  std={rand_sims.std():.4f}")
        print(f"  Discrimination:  {pos_sims.mean() - rand_sims.mean():+.4f}")

    # Baseline: gnn_emb
    print(f"\n{'='*60}")
    print("Baseline: gnn_emb (64d)")
    gnn_emb = torch.load(os.path.join(args.data_dir, "gnn_embeddings.pt"), map_location="cpu", weights_only=False)
    gnn_src = torch.stack([gnn_emb[s] for s, _ in test_pairs])
    gnn_tgt = torch.stack([gnn_emb[t] for _, t in test_pairs])
    gnn_pos = F.cosine_similarity(gnn_src, gnn_tgt, dim=1).numpy()
    rng = random.Random(123)
    gnn_rand = []
    for _ in range(n_test):
        i = rng.randint(0, len(test_pairs)-1)
        j = rng.randint(0, len(test_pairs)-1)
        sim = F.cosine_similarity(gnn_src[i:i+1], gnn_tgt[j:j+1]).item()
        gnn_rand.append(sim)
    gnn_rand = np.array(gnn_rand)
    print(f"  Positive pairs:  mean={gnn_pos.mean():.4f}")
    print(f"  Random pairs:    mean={gnn_rand.mean():.4f}")
    print(f"  Discrimination:  {gnn_pos.mean() - gnn_rand.mean():+.4f}")


if __name__ == "__main__":
    main()
