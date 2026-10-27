"""Concept co-occurrence graph statistics - parallel version (208 cores)."""

import json
from collections import defaultdict, Counter
from itertools import combinations
from multiprocessing import Pool
import os, time

MIN_SCORE = 0.3
MAX_LEVEL = 3
N_WORKERS = 160

def parse_line(line):
    d = json.loads(line)
    cids = []
    for c in d.get('concepts', []):
        cid = c['id']
        if '/' in cid:
            cid = cid.rsplit('/', 1)[-1]
        if 1 <= c['level'] <= MAX_LEVEL and c['score'] >= MIN_SCORE:
            cids.append(cid)
    return (d['id'], d['year'], cids) if cids else None

def load_papers_parallel(path):
    with open(path) as f:
        lines = f.readlines()
    with Pool(N_WORKERS) as pool:
        results = pool.map(parse_line, lines, chunksize=5000)
    return [r for r in results if r is not None]

def process_chunk_edges(args):
    """Build edge set + degree for a chunk of papers. Returns (edges_set, degree_counter)."""
    papers_chunk, year_cutoff, mode = args
    edges = set()
    degree = Counter()
    for pid, year, cids in papers_chunk:
        if mode == 'pre' and year > year_cutoff:
            continue
        if mode == 'post' and year <= year_cutoff:
            continue
        uniq = sorted(set(cids))
        for c in uniq:
            degree[c] += 1
        for a, b in combinations(uniq, 2):
            edges.add((a, b))
    return edges, degree

def parallel_build_edges(papers, year_cutoff, mode, n_workers=N_WORKERS):
    chunk_size = (len(papers) + n_workers - 1) // n_workers
    chunks = []
    for i in range(0, len(papers), chunk_size):
        chunks.append((papers[i:i+chunk_size], year_cutoff, mode))

    all_edges = set()
    all_degree = Counter()

    t0 = time.time()
    with Pool(n_workers) as pool:
        for i, (edges, deg) in enumerate(pool.imap_unordered(process_chunk_edges, chunks, chunksize=1)):
            all_edges.update(edges)
            all_degree += deg
            if (i+1) % 20 == 0:
                print(f"  chunks done: {i+1}/{len(chunks)}, edges so far: {len(all_edges)}, time: {time.time()-t0:.1f}s", flush=True)

    return all_edges, all_degree

def estimate_2hop(args):
    """Count 2-hop disconnected pairs for a sample of concepts."""
    concept, neighbors_pre, connected = args
    n1 = neighbors_pre[concept]
    candidates = set()
    for n in n1:
        candidates.update(neighbors_pre[n])
    candidates.discard(concept)
    disconnected = candidates - n1 - connected
    return len(disconnected)

if __name__ == '__main__':
    t_start = time.time()

    print(f"Loading papers with {N_WORKERS} workers...", flush=True)
    climate = load_papers_parallel('/root/autodl-tmp/openalex_data/climate_papers.jsonl')
    geology = load_papers_parallel('/root/autodl-tmp/openalex_data/geology_papers.jsonl')
    all_papers = climate + geology
    print(f"  climate: {len(climate)}, geology: {len(geology)}, total: {len(all_papers)}", flush=True)

    year_dist = Counter(y for _, y, _ in all_papers)
    pre2021 = sum(cnt for y, cnt in year_dist.items() if y <= 2020)
    post2021 = sum(cnt for y, cnt in year_dist.items() if y >= 2021)
    print(f"  year range: {min(year_dist)}-{max(year_dist)}", flush=True)
    print(f"  pre-2021: {pre2021}, post-2021: {post2021}", flush=True)

    all_concepts = set()
    for _, _, cids in all_papers:
        all_concepts.update(cids)
    print(f"  unique concepts (level 1-3, score>={MIN_SCORE}): {len(all_concepts)}", flush=True)
    print(f"  load time: {time.time()-t_start:.1f}s", flush=True)

    # Pre-2021 graph
    print("\nBuilding pre-2021 concept graph (parallel)...", flush=True)
    t0 = time.time()
    pre_edges, concept_degree = parallel_build_edges(all_papers, 2020, 'pre')
    print(f"  pre-2021 edges: {len(pre_edges)}, concepts: {len(concept_degree)}", flush=True)
    print(f"  time: {time.time()-t0:.1f}s", flush=True)

    print("\nTop 20 concepts by degree:", flush=True)
    for cid, deg in concept_degree.most_common(20):
        print(f"  {cid}: {deg} papers", flush=True)

    # Post-2021 graph
    print("\nBuilding post-2021 concept pairs (parallel)...", flush=True)
    t0 = time.time()
    post_edges, _ = parallel_build_edges(all_papers, 2020, 'post')
    print(f"  post-2021 edges: {len(post_edges)}", flush=True)
    print(f"  time: {time.time()-t0:.1f}s", flush=True)

    # Positives
    positives = post_edges - pre_edges
    print(f"\n*** Positive pairs (new post-2021 bridges): {len(positives)} ***", flush=True)
    print(f"*** Pre-2021 edges: {len(pre_edges)} ***", flush=True)

    # Hard negative estimation
    print("\nEstimating hard negatives...", flush=True)
    neighbors = defaultdict(set)
    for a, b in pre_edges:
        neighbors[a].add(b)
        neighbors[b].add(a)

    import random
    random.seed(42)
    sample = random.sample(list(concept_degree.keys()), min(300, len(concept_degree)))

    tasks = [(c, neighbors, {c}) for c in sample]
    with Pool(min(N_WORKERS, len(tasks))) as pool:
        results = pool.map(estimate_2hop, tasks, chunksize=10)

    avg = sum(results) / len(results)
    n_concepts = len(concept_degree)
    print(f"  avg 2-hop disconnected per concept (n={len(sample)}): {avg:.0f}", flush=True)
    print(f"  estimated total hard negatives: ~{avg * n_concepts / 1e6:.1f}M", flush=True)

    print(f"\nTotal time: {time.time()-t_start:.1f}s", flush=True)
    print("Done.", flush=True)
