"""Optimized CitationDataset for CPU-bound data loading.

Key optimizations vs phoenix_model.py:
  - All features precomputed as numpy arrays in __init__
  - Pre-filtered valid papers (≥3 refs)
  - Pre-built neighbor arrays (flat) for faster negative sampling
  - __getitem__ returns numpy arrays (not torch tensors) for zero-copy collation
  - collate_fn converts numpy→torch once per batch (not per sample)
"""

import numpy as np
import torch


class CitationDataset(torch.utils.data.Dataset):
    """Optimized dataset — all heavy computation in __init__, cheap randomization in __getitem__."""

    def __init__(
        self,
        paper_indices,
        ref_lists,
        features,
        concepts,
        concept_vocab_size,
        gnn_embeddings,
        adj,
        mask_ratio=0.2,
        n_negatives=5,
        max_refs=50,
        max_candidates=10,
    ):
        self.mask_ratio = mask_ratio
        self.n_negatives = n_negatives
        self.max_refs = max_refs
        self.max_candidates = max_candidates

        # Pre-filter to papers with enough references
        self.valid_indices = np.array([
            i for i in paper_indices if len(ref_lists[i]) >= 3
        ], dtype=np.int64)
        print(f"Dataset: {len(self.valid_indices):,} papers with ≥3 refs "
              f"(out of {len(paper_indices):,})")

        # Pre-build ref lists (only for valid papers)
        self.refs = [ref_lists[i] for i in self.valid_indices]

        # Convert features to numpy (vectorized gather in __getitem__)
        self.features_np = features.numpy().astype(np.float32)  # (n, 5)
        self.gnn_np = gnn_embeddings.numpy().astype(np.float32)  # (n, gnn_dim)

        # Pre-build concept indices (list-of-lists-of-lists)
        self.concept_indices = concepts  # list of lists

        # Pre-build flat neighbor arrays for fast negative sampling
        self._build_neighbor_arrays(adj)

    def _build_neighbor_arrays(self, adj):
        """Flatten adjacency dict into arrays for O(1) random neighbor lookup."""
        n = len(self.valid_indices)
        self.neighbor_starts = np.zeros(n + 1, dtype=np.int64)
        all_neighbors = []
        for i, idx in enumerate(self.valid_indices):
            neighbors = adj.get(idx, [])
            all_neighbors.append(neighbors)
            self.neighbor_starts[i + 1] = self.neighbor_starts[i] + len(neighbors)

        if self.neighbor_starts[-1] > 0:
            self.neighbor_flat = np.concatenate(
                [np.array(nbs, dtype=np.int64) for nbs in all_neighbors]
            ) if all_neighbors else np.array([], dtype=np.int64)
        else:
            self.neighbor_flat = np.array([], dtype=np.int64)

    def __len__(self):
        return len(self.valid_indices)

    def _sample_hard_negatives(self, valid_idx, exclude_set, n):
        """Fast negative sampling using pre-built flat neighbor arrays."""
        start = self.neighbor_starts[valid_idx]
        end = self.neighbor_starts[valid_idx + 1]

        if start == end:
            # No neighbors: random fallback
            return self.valid_indices[
                np.random.randint(0, len(self.valid_indices), size=n)
            ].tolist()

        neighbors = self.neighbor_flat[start:end]

        # 2-hop: sample some 1-hop neighbors and expand
        if len(neighbors) > 20:
            sample_1hop = neighbors[np.random.choice(len(neighbors), 20, replace=False)]
        else:
            sample_1hop = neighbors

        candidates = set()
        for nb in sample_1hop:
            nb_idx = np.searchsorted(self.valid_indices, nb)
            if nb_idx < len(self.valid_indices) and self.valid_indices[nb_idx] == nb:
                s = self.neighbor_starts[nb_idx]
                e = self.neighbor_starts[nb_idx + 1]
                if e > s:
                    hop2 = self.neighbor_flat[s:e]
                    if len(hop2) > 10:
                        hop2 = hop2[np.random.choice(len(hop2), 10, replace=False)]
                    for x in hop2:
                        if x not in exclude_set:
                            candidates.add(int(x))

        # Also add direct neighbors as candidates
        for nb in neighbors:
            if int(nb) not in exclude_set:
                candidates.add(int(nb))

        candidates = list(candidates)
        if len(candidates) >= n:
            return [candidates[i] for i in np.random.choice(len(candidates), n, replace=False)]
        elif candidates:
            return [candidates[i] for i in np.random.choice(len(candidates), n, replace=True)]
        else:
            return self.valid_indices[
                np.random.randint(0, len(self.valid_indices), size=n)
            ].tolist()

    def __getitem__(self, idx):
        """Return numpy arrays — fast, zero-copy when batched."""
        refs = self.refs[idx]

        # Truncate and split references
        if len(refs) > self.max_refs:
            perm = np.random.permutation(len(refs))[:self.max_refs]
            refs = [refs[i] for i in perm]

        n_mask = max(1, int(len(refs) * self.mask_ratio))
        perm = np.random.permutation(len(refs))
        history_refs = [refs[perm[i]] for i in range(n_mask, len(refs))]
        positive_candidates = [refs[perm[i]] for i in range(n_mask)]

        # Hard negatives
        paper_idx = int(self.valid_indices[idx])
        exclude_set = set(history_refs + positive_candidates + [paper_idx])
        neg_candidates = self._sample_hard_negatives(idx, exclude_set, self.n_negatives)

        candidate_indices = positive_candidates + neg_candidates
        candidate_labels = [1.0] * len(positive_candidates) + [0.0] * len(neg_candidates)

        # Truncate candidates
        if len(candidate_indices) > self.max_candidates:
            keep = np.random.choice(len(candidate_indices), self.max_candidates, replace=False)
            candidate_indices = [candidate_indices[i] for i in keep]
            candidate_labels = [candidate_labels[i] for i in keep]

        # Assemble full index sequence
        all_indices = [paper_idx] + history_refs + candidate_indices
        n_user_history = 1 + len(history_refs)
        n_candidates = len(candidate_indices)
        seq_len = n_user_history + n_candidates

        all_idx = np.array(all_indices, dtype=np.int64)

        # Vectorized feature gather (numpy — much faster than Python loop + torch.stack)
        scalar = self.features_np[all_idx]         # (seq_len, 5)
        gnn_emb = self.gnn_np[all_idx]             # (seq_len, gnn_dim)
        concept_idx = [self.concept_indices[i] for i in all_indices]  # list of lists

        # Labels
        labels = np.zeros(seq_len, dtype=np.float32)
        labels[n_user_history:] = np.array(candidate_labels, dtype=np.float32)

        # Attention mask (candidate isolation)
        attn_mask = np.zeros((seq_len, seq_len), dtype=np.float32)
        attn_mask[:n_user_history, :n_user_history] = 1.0
        for i in range(n_user_history, seq_len):
            attn_mask[i, :n_user_history] = 1.0
            attn_mask[i, i] = 1.0

        return {
            "scalar": scalar,
            "concept_indices": concept_idx,
            "gnn_emb": gnn_emb,
            "attention_mask": attn_mask,
            "labels": labels,
            "n_user_history": n_user_history,
            "n_candidates": n_candidates,
        }


def collate_variable_length(batch):
    """Fast collate: numpy→torch conversion at batch level (not per-sample)."""
    max_len = max(b["scalar"].shape[0] for b in batch)
    scalar_dim = batch[0]["scalar"].shape[1]
    gnn_dim = batch[0]["gnn_emb"].shape[1]

    max_n_concepts = max(
        len(c) for b in batch for c in b["concept_indices"]
    ) if batch else 1
    max_n_concepts = max(max_n_concepts, 1)

    B = len(batch)
    batch_scalar = torch.zeros(B, max_len, scalar_dim)
    batch_concept_idx = torch.zeros(B, max_len, max_n_concepts, dtype=torch.long)
    batch_gnn = torch.zeros(B, max_len, gnn_dim)
    batch_mask = torch.zeros(B, max_len, max_len)
    batch_labels = torch.full((B, max_len), -100.0)
    batch_nuh = []
    batch_nc = []

    for i, b in enumerate(batch):
        seq_len = b["scalar"].shape[0]
        # Numpy → Torch (fast path for contiguous arrays)
        batch_scalar[i, :seq_len] = torch.from_numpy(b["scalar"])
        batch_gnn[i, :seq_len] = torch.from_numpy(b["gnn_emb"])
        batch_mask[i, :seq_len, :seq_len] = torch.from_numpy(b["attention_mask"])
        batch_labels[i, :seq_len] = torch.from_numpy(b["labels"])
        batch_nuh.append(b["n_user_history"])
        batch_nc.append(b["n_candidates"])
        for j, cidx in enumerate(b["concept_indices"]):
            for k, c in enumerate(cidx):
                batch_concept_idx[i, j, k] = c

    # Fix: padded positions must attend to at least themselves to avoid
    # NaN from softmax(-inf) when all mask entries are 0.
    # Padded positions have label=-100 so they don't affect the loss.
    for b in range(B):
        for i in range(max_len):
            if batch_mask[b, i, i] == 0 and batch_mask[b, i].sum() == 0:
                batch_mask[b, i, i] = 1.0

    return {
        "scalar": batch_scalar,
        "concept_indices": batch_concept_idx,
        "gnn_emb": batch_gnn,
        "attention_mask": batch_mask,
        "labels": batch_labels,
        "n_user_history": batch_nuh,
        "n_candidates": batch_nc,
    }
