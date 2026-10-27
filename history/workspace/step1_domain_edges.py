#!/usr/bin/env python3
"""Step 1: Add domain labels and filter cross-domain edges.

Reads raw JSONL files + processed data.
Outputs:
  - domain.pt: (n_papers,) int tensor, 0=climate, 1=geology, 2=both
  - cross_domain_edges.pt: (2, n_cross_edges) edge_index for GNN retraining
"""
import os, orjson, torch
from tqdm import tqdm

CLIMATE_CONCEPTS = {"C49204034", "C91586092"}
GEOLOGY_CONCEPTS = {"C1965285", "C17409809", "C199289684"}

DATA_DIR = "/root/autodl-tmp/openalex_data/processed"
RAW_DIR = "/root/autodl-tmp/openalex_data"


def main():
    # Load paper ID mapping
    fd = torch.load(os.path.join(DATA_DIR, "features.pt"), map_location="cpu", weights_only=False)
    pid_to_idx = fd["paper_id_to_idx"]
    paper_ids = fd["paper_ids"]

    # Classify each paper by reading raw JSONL
    print("Classifying papers by domain...")
    domain = torch.full((len(paper_ids),), -1, dtype=torch.long)  # -1 = unknown

    for name in ["climate_papers.jsonl", "geology_papers.jsonl"]:
        path = os.path.join(RAW_DIR, name)
        for line in tqdm(open(path, "rb"), desc=name):
            d = orjson.loads(line)
            pid = d["id"]
            if pid not in pid_to_idx:
                continue
            idx = pid_to_idx[pid]
            cids = set()
            for c in d.get("concepts", []):
                cid = c["id"].split("/")[-1]
                if c.get("level", 0) <= 2:
                    cids.add(cid)
            is_c = bool(cids & CLIMATE_CONCEPTS)
            is_g = bool(cids & GEOLOGY_CONCEPTS)
            if is_c and is_g:
                domain[idx] = 2  # both
            elif is_c:
                domain[idx] = 0  # climate
            elif is_g:
                domain[idx] = 1  # geology

    n_climate = (domain == 0).sum().item()
    n_geology = (domain == 1).sum().item()
    n_both = (domain == 2).sum().item()
    n_unknown = (domain == -1).sum().item()
    print(f"Climate: {n_climate:,}, Geology: {n_geology:,}, Both: {n_both:,}, Unknown: {n_unknown:,}")

    torch.save(domain, os.path.join(DATA_DIR, "domain.pt"))
    print(f"Saved domain.pt")

    # Filter edges: keep only cross-domain
    adj = torch.load(os.path.join(DATA_DIR, "adjacency.pt"), map_location="cpu", weights_only=False)
    ei = adj["edge_index"]  # (2, 34M)
    src_domain = domain[ei[0]]
    dst_domain = domain[ei[1]]

    # Cross-domain: src and dst are different domains (not both, not unknown)
    cross_mask = (
        (src_domain >= 0) & (dst_domain >= 0) &
        (src_domain != 2) & (dst_domain != 2) &
        (src_domain != dst_domain)
    )
    cross_edges = ei[:, cross_mask]
    print(f"Total edges: {ei.size(1):,}")
    print(f"Cross-domain edges: {cross_edges.size(1):,}")

    torch.save(cross_edges, os.path.join(DATA_DIR, "cross_domain_edges.pt"))
    print(f"Saved cross_domain_edges.pt")

    # Stats on bridge_score
    bs = fd["features"][:, 3]
    cross_src_bs = bs[cross_edges[0]]
    print(f"Cross-domain source bridge_score: mean={cross_src_bs.mean():.4f}")


if __name__ == "__main__":
    main()
