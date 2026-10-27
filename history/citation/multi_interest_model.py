"""Multi-Interest Network (MIND-style) for LBD bridge prediction.

User tower: title embedding sequence → CapsuleLayer → K interest vectors
Candidate tower: paper features → MLP → single embedding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(x, dim=-1):
    norm_sq = (x ** 2).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(norm_sq + 1e-9)
    return norm_sq / (1 + norm_sq) * x / norm


class CapsuleLayer(nn.Module):
    """Dynamic routing: (batch, seq_len, in_dim) → (batch, k_max, out_dim)."""

    def __init__(self, in_dim, out_dim, k_max=3, n_iter=3):
        super().__init__()
        self.k_max = k_max
        self.n_iter = n_iter
        self.bilinear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, seq_len):
        """
        x: (B, L, in_dim)  — padded behavior embeddings
        seq_len: (B,)       — actual length per sample
        Returns: (B, k_max, out_dim)  — K interest vectors
        """
        B, L, _ = x.shape
        device = x.device

        # (B, L, out_dim)
        mapped = self.bilinear(x)
        mapped_sg = mapped.detach()

        # mask: (B, L, 1, 1)
        arange = torch.arange(L, device=device).unsqueeze(0)
        mask = (arange < seq_len.unsqueeze(1)).float().unsqueeze(-1).unsqueeze(-1)

        # routing logits: (B, L, k_max, 1)
        logits = torch.randn(B, L, self.k_max, 1, device=device) * 0.01

        for i in range(self.n_iter):
            w = F.softmax(logits, dim=2) * mask  # (B, L, k, 1)
            w_squeezed = w.squeeze(-1)           # (B, L, k)
            if i < self.n_iter - 1:
                # z[b, k, out] = sum_l w[b,l,k] * mapped[b,l,out]
                z = torch.einsum('blk,ble->bke', w_squeezed, mapped_sg)  # (B, k, out)
                caps = squash(z, dim=-1)
                # agreement: logits[b,l,k,1] += caps[b,k,out] @ mapped[b,l,out]
                logits = logits + torch.einsum('bke,ble->blk', caps, mapped_sg).unsqueeze(-1)
            else:
                z = torch.einsum('blk,ble->bke', w_squeezed, mapped)
                caps = squash(z, dim=-1)

        return caps  # (B, k_max, out_dim)


class MultiInterestModel(nn.Module):
    """Two-tower: multi-interest user + single-vector candidate."""

    def __init__(self, title_dim=384, d_model=128, k_max=3):
        super().__init__()
        self.k_max = k_max

        # User tower: title embeddings → project → capsule routing
        self.user_proj = nn.Sequential(
            nn.Linear(title_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.capsule = CapsuleLayer(d_model, d_model, k_max=k_max, n_iter=3)

        # Candidate tower: title embedding → project
        self.cand_proj = nn.Sequential(
            nn.Linear(title_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def user_forward(self, title_seq, seq_len):
        """
        title_seq: (B, L, 384) — padded title embeddings of references
        seq_len:   (B,)         — number of refs per sample
        Returns:   (B, k_max, d_model) — K interest vectors, L2-normalized
        """
        h = self.user_proj(title_seq)       # (B, L, d_model)
        interests = self.capsule(h, seq_len) # (B, k_max, d_model)
        return F.normalize(interests, p=2, dim=-1)

    def cand_forward(self, title_emb):
        """
        title_emb: (B, 384) — single paper's title embedding
        Returns:   (B, d_model) — L2-normalized candidate vector
        """
        v = self.cand_proj(title_emb)
        return F.normalize(v, p=2, dim=-1)

    def forward(self, user_titles, seq_len, cand_titles):
        """
        user_titles: (B, L, 384)
        seq_len:     (B,)
        cand_titles: (B, 384)
        Returns:     (B, B) similarity matrix
        """
        interests = self.user_forward(user_titles, seq_len)  # (B, K, d)
        cands = self.cand_forward(cand_titles)                # (B, d)

        B = interests.size(0)
        flat = interests.reshape(B * self.k_max, -1)  # (B*K, d)
        sims_all = flat @ cands.T                      # (B*K, B)
        sims = sims_all.reshape(B, self.k_max, B)     # (B, K, B)

        # Max over interests: (B, B)
        return sims.max(dim=1).values
