"""Phoenix Transformer for citation propagation prediction.

Architecture (NO ID embeddings, features only):
  Paper token = Linear([year_norm, log_degrees, bridge, entropy]) ⊕ Linear(concepts_multi_hot) ⊕ gnn_embedding

  Transformer with candidate isolation mask:
    User+History: causal attention
    Candidates: attend to User+History, NOT to each other

  Training objective: sequence completion
    Mask 20% of paper's references, predict them from remaining refs + hard negatives
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim, max_len=4096, theta=10000.0):
    """Precompute RoPE frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary position embedding."""
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    seq_len = x.size(-2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class MultiHeadAttention(nn.Module):
    """Grouped Query Attention with optional RoPE and logit capping."""

    def __init__(self, dim, n_heads, n_kv_heads, max_len=512, use_rope=False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.use_rope = use_rope

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        if use_rope:
            self.rope_cos, self.rope_sin = precompute_rope(self.head_dim, max_len)

    def _repeat_kv(self, x):
        """Repeat KV heads to match Q heads."""
        if self.n_rep == 1:
            return x
        bs, n_kv, seq, d = x.shape
        return x[:, :, None, :, :].expand(bs, n_kv, self.n_rep, seq, d).reshape(
            bs, self.n_heads, seq, d)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE (optional)
        if self.use_rope:
            device = x.device
            cos = self.rope_cos.to(device)
            sin = self.rope_sin.to(device)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # Repeat KV heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Logit capping (Phoenix style)
        attn = 30.0 * torch.tanh(attn / 30.0)

        if mask is not None:
            # mask: (B, S, S) → (B, 1, S, S) for broadcasting over n_heads
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward (Phoenix style)."""

    def __init__(self, dim, widening_factor=2):
        super().__init__()
        hidden = int(dim * widening_factor * 2 / 3)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, dropout=0.1, max_len=512, use_rope=False):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, n_kv_heads, max_len, use_rope)
        self.ff_norm = RMSNorm(dim)
        self.ff = SwiGLU(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.attn_norm(x), mask))
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x


class PhoenixCitationTransformer(nn.Module):
    """
    Phoenix-style Transformer for citation propagation.

    Input: paper features (no IDs!)
    Output: score for each candidate paper (logits)

    Architecture:
      1. Feature projection: [scalar_features, concept_multi_hot, gnn_emb] → D-dim token
      2. Transformer blocks with candidate isolation mask
      3. Mean-pool candidate tokens → score
    """

    def __init__(
        self,
        scalar_dim=5,          # year_norm, log_in_deg, log_out_deg, bridge, entropy
        concept_vocab_size=500, # number of level 0-2 concepts
        concept_emb_dim=32,    # concept projection dimension
        gnn_emb_dim=64,        # GNN structural embedding dimension
        d_model=128,           # Transformer hidden dimension
        n_heads=4,
        n_kv_heads=4,
        n_layers=4,
        dropout=0.1,
        max_seq_len=128,       # max user + history + candidates
        output_multiplier=0.125,  # Phoenix output scaling
    ):
        super().__init__()
        self.d_model = d_model
        self.output_multiplier = output_multiplier
        self.max_seq_len = max_seq_len

        # Feature projection: scalar + concept + GNN → d_model
        total_input_dim = scalar_dim + concept_emb_dim + gnn_emb_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(total_input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Concept projection (Linear over multi-hot, NOT nn.Embedding)
        self.concept_proj = nn.Linear(concept_vocab_size, concept_emb_dim, bias=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, dropout, max_seq_len, use_rope=False)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # Scoring head
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def make_candidate_isolation_mask(self, n_user_history, n_candidates, device):
        """
        Create candidate isolation attention mask (Phoenix style).

        Layout: [user_history (n_uh), candidates (n_c)]

        User+History: full bidirectional attention (set mode — no positional order)
        Candidates: can attend to ALL user+history + themselves (position-wise),
                    but NOT to other candidates

        Returns: (1, S, S) mask where 1=attend, 0=block
        """
        S = n_user_history + n_candidates
        mask = torch.zeros(S, S, device=device, dtype=torch.float32)

        # User+History: full attention (set mode — no positional order)
        mask[:n_user_history, :n_user_history] = 1.0

        # Candidates: can attend to all user+history + self, but NOT other candidates
        for i in range(n_user_history, S):
            # Attend to all user+history
            mask[i, :n_user_history] = 1.0
            # Attend to self
            mask[i, i] = 1.0
            # Block other candidates
            # (already 0 by default)

        return mask.unsqueeze(0)  # (1, S, S)

    def forward(self, scalar_features, concept_indices, gnn_emb, attention_mask):
        """
        Args:
            scalar_features: (B, S, scalar_dim) — year, degrees, bridge, entropy
            concept_indices: (B, S, max_n_concepts) — padded concept vocab indices (long)
            gnn_emb: (B, S, gnn_emb_dim) — pre-trained GNN embeddings
            attention_mask: (B, S, S) — candidate isolation mask

        Returns:
            logits: (B, S) — per-position logits (only candidates are meaningful)
        """
        # Convert concept indices → multi-hot on GPU via scatter
        B, S, K = concept_indices.shape
        concept_multi_hot = torch.zeros(B, S, self.concept_proj.in_features,
                                        device=concept_indices.device)
        concept_multi_hot.scatter_(2, concept_indices.clamp(min=0), 1.0)

        # Project concepts (Linear over multi-hot, not embedding lookup)
        concept_emb = self.concept_proj(concept_multi_hot)  # (B, S, concept_emb_dim)

        # Concatenate all features
        x = torch.cat([scalar_features, concept_emb, gnn_emb], dim=-1)

        # Project to d_model
        x = self.feature_proj(x)  # (B, S, d_model)

        # Transformer
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.final_norm(x)

        # Score each position
        logits = self.score_head(x).squeeze(-1)  # (B, S)
        logits = logits * self.output_multiplier

        return logits


class CitationDataset(torch.utils.data.Dataset):
    """
    Training dataset for sequence completion.

    For each paper:
      1. Take its reference list as "history"
      2. Mask 20% of refs → these become "candidates" (positive)
      3. Sample hard negatives from 2-3 hop neighbors (negative candidates)
      4. Construct token sequence: [user_paper, history_refs..., candidates...]
    """

    def __init__(
        self,
        paper_indices,     # list of paper indices for this split
        ref_lists,         # list of list: ref_lists[i] = [ref indices for paper i]
        features,          # (n, scalar_dim) scalar features
        concepts,          # list of lists: concepts[i] = [vocab_idx, ...] (nonzero indices)
        concept_vocab_size,# size of concept vocabulary
        gnn_embeddings,    # (n, gnn_dim) pre-trained GNN embeddings
        adj,               # dict: adj[i] = [neighbor indices] (for negative sampling)
        mask_ratio=0.2,
        n_negatives=5,
        max_refs=50,       # truncate long reference lists
        max_candidates=10, # max positive + negative candidates
    ):
        self.paper_indices = paper_indices
        self.ref_lists = ref_lists
        self.features = features
        self.concepts = concepts
        self.concept_vocab_size = concept_vocab_size
        self.gnn_embeddings = gnn_embeddings
        self.adj = adj
        self.mask_ratio = mask_ratio
        self.n_negatives = n_negatives
        self.max_refs = max_refs
        self.max_candidates = max_candidates

        # Filter to papers with enough references
        self.valid_indices = [
            i for i in paper_indices
            if len(ref_lists[i]) >= 3
        ]
        print(f"Dataset: {len(self.valid_indices):,} papers with ≥3 refs "
              f"(out of {len(paper_indices):,})")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        paper_idx = self.valid_indices[idx]
        refs = self.ref_lists[paper_idx]

        # Truncate references
        if len(refs) > self.max_refs:
            refs = list(np.random.choice(refs, self.max_refs, replace=False))

        # Split: mask_ratio become candidates, rest are history
        n_mask = max(1, int(len(refs) * self.mask_ratio))
        perm = torch.randperm(len(refs))
        mask_indices = perm[:n_mask]
        history_indices = perm[n_mask:]

        history_refs = [refs[i] for i in history_indices]
        positive_candidates = [refs[i] for i in mask_indices]

        # Sample hard negatives from 2-3 hop neighbors
        neg_candidates = self._sample_hard_negatives(
            paper_idx, history_refs + positive_candidates, self.n_negatives)

        # All candidates: positives + negatives
        candidate_indices = positive_candidates + neg_candidates
        candidate_labels = [1] * len(positive_candidates) + [0] * len(neg_candidates)

        # Truncate candidates
        if len(candidate_indices) > self.max_candidates:
            keep = list(np.random.choice(
                len(candidate_indices), self.max_candidates, replace=False))
            candidate_indices = [candidate_indices[i] for i in keep]
            candidate_labels = [candidate_labels[i] for i in keep]

        # Build feature tensors
        # Sequence: [paper, history_1, ..., history_n, cand_1, ..., cand_m]
        all_indices = [paper_idx] + history_refs + candidate_indices
        n_user_history = 1 + len(history_refs)
        n_candidates = len(candidate_indices)
        seq_len = n_user_history + n_candidates

        scalar = torch.stack([self.features[i] for i in all_indices])
        # Return concept indices as list (will be batched in collate, converted on GPU)
        concept_indices = [self.concepts[i] for i in all_indices]
        gnn = torch.stack([self.gnn_embeddings[i] for i in all_indices])

        # Labels: 0 for user+history, label for candidates
        labels = torch.zeros(seq_len)
        labels[n_user_history:] = torch.tensor(candidate_labels, dtype=torch.float32)

        # Attention mask: candidate isolation
        attn_mask = self._make_attn_mask(n_user_history, n_candidates)

        return {
            "scalar": scalar,
            "concept_indices": concept_indices,  # list of lists, collated separately
            "gnn_emb": gnn,
            "attention_mask": attn_mask,
            "labels": labels,
            "n_user_history": n_user_history,
            "n_candidates": n_candidates,
        }

    def _sample_hard_negatives(self, paper_idx, exclude, n):
        """Sample hard negatives from 2-3 hop neighborhood."""
        exclude_set = set(exclude)
        exclude_set.add(paper_idx)

        candidates = set()
        # 1-hop neighbors of refs
        for ref in self.ref_lists[paper_idx][:20]:  # limit for speed
            for neighbor in self.adj.get(ref, [])[:20]:
                if neighbor not in exclude_set:
                    candidates.add(neighbor)

        # 2-hop: neighbors of neighbors
        for ref in self.ref_lists[paper_idx][:10]:
            for neighbor in self.adj.get(ref, [])[:10]:
                for nn in self.adj.get(neighbor, [])[:10]:
                    if nn not in exclude_set:
                        candidates.add(nn)

        candidates = list(candidates)
        if len(candidates) >= n:
            return list(np.random.choice(candidates, n, replace=False))
        elif candidates:
            return candidates + list(np.random.choice(candidates, n - len(candidates), replace=True))
        else:
            # Fallback: random papers
            return list(np.random.choice(len(self.ref_lists), n, replace=False))

    def _make_attn_mask(self, n_uh, n_c):
        """Create candidate isolation mask (set mode for user+history)."""
        S = n_uh + n_c
        mask = torch.zeros(S, S, dtype=torch.float32)
        # User+History: full attention (set, no ordering)
        mask[:n_uh, :n_uh] = 1.0
        for i in range(n_uh, S):
            mask[i, :n_uh] = 1.0
            mask[i, i] = 1.0
        return mask


# Need numpy for sampling
import numpy as np


def collate_variable_length(batch):
    """Collate function for variable-length sequences with padding.
    Concept indices are padded to (B, max_len, max_n_concepts) long tensor —
    converted to multi-hot on GPU in model.forward() for speed."""
    max_len = max(b["scalar"].size(0) for b in batch)

    scalar_dim = batch[0]["scalar"].size(-1)
    gnn_dim = batch[0]["gnn_emb"].size(-1)

    # Find max concept count per token in this batch
    max_n_concepts = max(
        len(c) for b in batch for c in b["concept_indices"]
    ) if batch else 1
    max_n_concepts = max(max_n_concepts, 1)

    batch_scalar = torch.zeros(len(batch), max_len, scalar_dim)
    batch_concept_idx = torch.zeros(len(batch), max_len, max_n_concepts, dtype=torch.long)
    batch_gnn = torch.zeros(len(batch), max_len, gnn_dim)
    batch_mask = torch.zeros(len(batch), max_len, max_len)
    batch_labels = torch.full((len(batch), max_len), -100.0)  # -100 = ignore in loss
    batch_nuh = []
    batch_nc = []

    for i, b in enumerate(batch):
        seq_len = b["scalar"].size(0)
        batch_scalar[i, :seq_len] = b["scalar"]
        batch_gnn[i, :seq_len] = b["gnn_emb"]
        batch_mask[i, :seq_len, :seq_len] = b["attention_mask"]
        # PAD positions: self-attention to avoid NaN in softmax
        if seq_len < max_len:
            batch_mask[i, seq_len:, seq_len:] = torch.eye(max_len - seq_len)
        batch_labels[i, :seq_len] = b["labels"]
        batch_nuh.append(b["n_user_history"])
        batch_nc.append(b["n_candidates"])
        # Pad concept indices
        for j, cidx in enumerate(b["concept_indices"]):
            for k, c in enumerate(cidx):
                batch_concept_idx[i, j, k] = c

    return {
        "scalar": batch_scalar,
        "concept_indices": batch_concept_idx,  # (B, max_len, max_n_concepts)
        "gnn_emb": batch_gnn,
        "attention_mask": batch_mask,
        "labels": batch_labels,
        "n_user_history": batch_nuh,
        "n_candidates": batch_nc,
    }
