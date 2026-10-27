import json, numpy as np
from collections import Counter

# Our graph concepts
cids_list = json.load(open("concept_ids.json"))
print(f"Our concept_ids.json: {len(cids_list)} concepts")

# Get level for each concept from concepts.jsonl
concept_levels = {}
with open("/root/autodl-tmp/openalex_data/concepts.jsonl") as f:
    for line in f:
        w = json.loads(line)
        for c in w["concepts"]:
            concept_levels[c["id"]] = c["level"]

our_levels = Counter()
for cid in cids_list:
    lvl = concept_levels.get(cid, -1)
    our_levels[lvl] += 1

print(f"\nOur concepts by level:")
for l in sorted(our_levels):
    print(f"  Level {l}: {our_levels[l]}")

# Show some L2 examples
l2_names = json.load(open("concept_names.json"))
l2_examples = [(l2_names[cid], cid) for cid in cids_list if concept_levels.get(cid, -1) == 2][:10]
print(f"\nL2 examples: {[x[0] for x in l2_examples]}")

l3_examples = [(l2_names[cid], cid) for cid in cids_list if concept_levels.get(cid, -1) == 3][:10]
print(f"L3 examples: {[x[0] for x in l3_examples]}")
