"""
PENCIL v5 model: SubGraphEncoder + Phoenix-style Transformer with candidate isolation.

Architecture:
  1. SubGraphEncoder: compress 32×68 subgraph features → 256d vector per candidate
  2. Sequence: [u_token, v1_emb, v2_emb, ..., vK_emb]
  3. Candidate isolation mask: candidates attend to u + self only
  4. Training: softmax cross-entropy (find positive among K candidates)
  5. Inference: encode all candidates → single Transformer pass → rank by score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SubGraphEncoder(nn.Module):
    """Compress 32×68 subgraph features → single d_model vector.

    Preserves PENCIL's graph message passing:
      1. Node MLP: project 68d → hidden
      2. N_MSG_LAYERS of: LN → graph_msg_passing(adj_norm @ x → MLP → residual)
      3. Mean-pool over valid nodes
      4. Project to output_dim

    Adjacency matrix is extracted from features[:, 32:64] (columns 32-63).
    """

    def __init__(self, input_dim=68, hidden_dim=256, output_dim=256, n_msg_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.msg_layers = nn.ModuleList()
        for _ in range(n_msg_layers):
            self.msg_layers.append(nn.ModuleDict({
                'ln': nn.LayerNorm(hidden_dim),
                'mp': nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU()),
            }))
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B*K, 32, 68) subgraph features
        Returns:
            (B*K, output_dim) embeddings
        """
        # Extract adjacency matrix from features (columns 32:64)
        adj = x[:, :32, 32:64]  # (B*K, 32, 32)
        deg = adj.sum(dim=2, keepdim=True).clamp(min=1)
        adj_norm = adj / deg

        # Node projection
        h = self.node_proj(x)  # (B*K, 32, hidden)

        # Graph message passing layers
        for layer in self.msg_layers:
            h_ln = layer['ln'](h)
            h_mp = layer['mp'](torch.bmm(adj_norm, h_ln))
            h = h + h_mp

        # Masked mean-pool
        mask = (x[:, :, :65].abs().sum(dim=2, keepdim=True) > 0).float()
        h = h * mask
        h = h.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.pool_proj(h)


class PencilV5Model(nn.Module):
    """Phoenix-style Transformer with candidate isolation mask.

    Sequence layout: [source_u, candidate_1, ..., candidate_K]
    - source_u token: learned query (all zeros input → learned embedding)
    - candidate tokens: SubGraphEncoder output
    - Mask: candidates attend to source + self, NOT to other candidates

    Training: softmax cross-entropy over K candidates
    Inference: rank candidates by Transformer output logits
    """

    def __init__(self, d_model=256, n_heads=8, n_layers=4, dropout=0.1,
                 subgraph_input_dim=68, max_candidates=5001):
        super().__init__()
        self.d_model = d_model
        self.max_candidates = max_candidates

        self.encoder = SubGraphEncoder(subgraph_input_dim, d_model, d_model)

        # Learned source token (replaces zeros input)
        self.source_token = nn.Parameter(torch.randn(d_model) * 0.02)

        # Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Scoring head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def make_cand_isolation_mask(self, n_cands, device):
        """Create candidate isolation mask.

        Returns (1, 1+n_cands, 1+n_cands) mask:
          - Source (pos 0): attends to all (full bidirectional)
          - Candidate i (pos 1+i): attends to source (pos 0) + self (pos 1+i)
        """
        S = 1 + n_cands
        mask = torch.zeros(S, S, device=device)
        # Source attends to everything
        mask[0, :] = 1.0
        # Candidates attend to source + self
        for i in range(n_cands):
            mask[1 + i, 0] = 1.0       # attend to source
            mask[1 + i, 1 + i] = 1.0   # attend to self
        return mask.unsqueeze(0)        # (1, S, S)

    def forward(self, inp, node_ids, n_cands_list=None):
        """
        Args:
            inp: (B, 1+K, 32, 68) — source zeros + K candidate subgraph features
            node_ids: (B, 1+K, 32) — node IDs (unused in forward, kept for compat)
            n_cands_list: list of actual candidate counts per sample (for padding mask)
        Returns:
            logits: (B, 1+K) — per-position logits (position 0 = source, positions 1+K = candidates)
        """
        B, S, N, D = inp.shape
        n_cands = S - 1

        # Encode all subgraphs: (B*S, 32, 68) → (B*S, d_model) → (B, S, d_model)
        flat = inp.reshape(B * S, N, D)
        emb = self.encoder(flat).reshape(B, S, self.d_model)

        # Replace source token (position 0) with learned token
        emb[:, 0] = self.source_token.unsqueeze(0).expand(B, -1)

        # Candidate isolation mask
        attn_mask = self.make_cand_isolation_mask(n_cands, inp.device)

        # Padding mask for padded candidates
        pad_mask = None
        if n_cands_list is not None:
            pad_mask = torch.zeros(B, S, device=inp.device, dtype=torch.bool)
            for i, nc in enumerate(n_cands_list):
                pad_mask[i, 1 + nc:] = True

        # Transformer
        x = emb
        for block in self.blocks:
            x = block(x, attn_mask, pad_mask)
        x = self.norm(x)

        # Score each position
        logits = self.head(x).squeeze(-1)  # (B, S)
        return logits

    def encode_candidates(self, inp):
        """Encode candidates only (for large-candidate inference).

        Args:
            inp: (K, 32, 68) — K candidate subgraph features
        Returns:
            (K, d_model) — candidate embeddings
        """
        return self.encoder(inp)

    def score_with_prefilled(self, source_emb, cand_embs):
        """Score candidates with pre-encoded source and candidates.

        Args:
            source_emb: (1, d_model) — source embedding
            cand_embs: (K, d_model) — candidate embeddings
        Returns:
            (K,) — logits for each candidate
        """
        K = cand_embs.shape[0]
        # Build sequence: (1, 1+K, d_model)
        seq = torch.cat([source_emb.unsqueeze(1), cand_embs.unsqueeze(0)], dim=1)
        attn_mask = self.make_cand_isolation_mask(K, seq.device)
        x = seq
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.norm(x)
        logits = self.head(x).squeeze(-1)  # (1, 1+K)
        return logits[0, 1:]  # (K,)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, pad_mask=None):
        h = self.ln1(x)
        # Combine masks: pad_mask blocks entire rows/cols, attn_mask controls attention pattern
        key_padding_mask = pad_mask  # (B, S) True=ignore
        h_attn, _ = self.attn(h, h, h, attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask)
        x = x + self.drop(h_attn)
        x = x + self.ffn(self.ln2(x))
        return x
