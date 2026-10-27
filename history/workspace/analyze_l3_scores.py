"""Analyze L3 concept score distribution to pick good threshold."""
import json
from collections import Counter, defaultdict

PAPERS = [
    '/root/autodl-tmp/openalex_data/geology_papers.jsonl',
    '/root/autodl-tmp/openalex_data/climate_papers.jsonl',
]

# Collect score distributions per level
level_scores = defaultdict(list)
concept_mentions = defaultdict(int)  # how many papers mention each concept
concept_levels = {}  # cid -> level
concept_scores_sum = defaultdict(float)

prefix = 'https://openalex.org/'
total_papers = 0
for path in PAPERS:
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            if 'year' not in p or 'concepts' not in p:
                continue
            total_papers += 1
            for c in p['concepts']:
                cid = c['id']
                if cid.startswith(prefix):
                    cid = cid[len(prefix):]
                lvl = c.get('level', 1)
                score = c.get('score', 0)
                level_scores[lvl].append(score)
                concept_mentions[cid] += 1
                concept_levels[cid] = lvl
                concept_scores_sum[cid] += score

print(f"Total papers: {total_papers:,}")
print(f"Total unique concepts: {len(concept_mentions):,}")

# Score distribution per level
for lvl in sorted(level_scores):
    scores = level_scores[lvl]
    import numpy as np
    s = np.array(scores)
    print(f"\nLevel {lvl} (n={len(scores):,} mentions, {len([c for c,l in concept_levels.items() if l==lvl]):,} unique):")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{pct}: {np.percentile(s, pct):.3f}")
    print(f"  >=0.5: {(s>=0.5).sum():,} ({(s>=0.5).mean()*100:.1f}%)")
    print(f"  >=0.6: {(s>=0.6).sum():,} ({(s>=0.6).mean()*100:.1f}%)")
    print(f"  >=0.7: {(s>=0.7).sum():,} ({(s>=0.7).mean()*100:.1f}%)")
    print(f"  >=0.8: {(s>=0.8).sum():,} ({(s>=0.8).mean()*100:.1f}%)")

# How many unique concepts per level at different score thresholds?
print("\n\nUnique concepts per level at score thresholds:")
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    level_counts = defaultdict(int)
    for cid, lvl in concept_levels.items():
        # Check if concept has ANY mention with score >= threshold
        # We need to re-scan... use average score instead
        avg_score = concept_scores_sum[cid] / concept_mentions[cid]
        if avg_score >= threshold:
            level_counts[lvl] += 1
    total = sum(level_counts.values())
    print(f"\n  avg_score >= {threshold}: total={total:,}")
    for lvl in sorted(level_counts):
        print(f"    L{lvl}: {level_counts[lvl]:,}")
