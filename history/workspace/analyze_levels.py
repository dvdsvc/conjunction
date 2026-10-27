import json, numpy as np, glob
from collections import Counter

names = json.load(open("../concept_names.json"))
cids_list = json.load(open("../concept_ids.json"))

concept_levels = {}
with open("/root/autodl-tmp/openalex_data/concepts.jsonl") as f:
    for line in f:
        w = json.loads(line)
        for c in w["concepts"]:
            concept_levels[c["id"]] = c["level"]

val_files = sorted(glob.glob("fold0_chunks/val_chunk_*.npz"))

dc2_level_pairs = Counter()
dc2_concept_names = Counter()
dc2_total = 0
for f in val_files:
    d = np.load(f)
    pairs = d["pairs"]
    labels = d["labels"]
    mask = labels >= 2
    dc2_total += mask.sum()
    for (u,v), dc in zip(pairs[mask], labels[mask]):
        u_name = names.get(cids_list[u], str(u))
        v_name = names.get(cids_list[v], str(v))
        u_cid = cids_list[u]
        v_cid = cids_list[v]
        u_lvl = concept_levels.get(u_cid, -1)
        v_lvl = concept_levels.get(v_cid, -1)
        combo = (min(u_lvl,v_lvl), max(u_lvl,v_lvl))
        dc2_level_pairs[combo] += 1
        dc2_concept_names[u_name] += 1
        dc2_concept_names[v_name] += 1

print(f"dc>=2 total: {dc2_total:,}")
print(f"\ndc>=2 by level combo:")
for combo, cnt in sorted(dc2_level_pairs.items(), key=lambda x:-x[1]):
    print(f"  L{combo[0]}-L{combo[1]}: {cnt:,} ({cnt/dc2_total*100:.1f}%)")

print(f"\nTop 20 hub concepts in dc>=2:")
for name, cnt in dc2_concept_names.most_common(20):
    print(f"  {cnt:4d}x  {name}")
