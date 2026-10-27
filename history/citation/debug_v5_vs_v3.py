"""Quick debug: test v3 and v5 checkpoints on first 10 val anchors."""
import time
import numpy as np
import scipy.sparse as sp
from numba import njit
import re

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32
EMBED_DIM = 68
DIST_BINS = 10

@njit(cache=True)
def _reconstruct_features(nids, u, v, dist_matrix, dist_u, dist_v, out):
    n = 0
    for i in range(MAX_NODES):
        if nids[i] != 0 or i == 0: n = i + 1
    for i in range(MAX_NODES):
        for j in range(EMBED_DIM): out[i, j] = 0.0
    for i in range(n):
        out[i, i] = 1.0
        out[i, 2 * MAX_NODES] = 1.0
        out[i, 2 * MAX_NODES + 1] = min(dist_u[nids[i]], float(DIST_BINS)) / float(DIST_BINS)
        out[i, 2 * MAX_NODES + 2] = min(dist_v[nids[i]], float(DIST_BINS)) / float(DIST_BINS)
        for j in range(n):
            if dist_matrix[nids[i], nids[j]] == 1:
                out[i, MAX_NODES + j] = 1.0

def _extract_val_pairs(data_dir):
    pos_pairs = []
    with open(f'{data_dir}/train.jsonl') as f:
        for line in f:
            if '"label": 1' not in line and '"label":1' not in line: continue
            src_m = re.search(r'"src_idx"\s*:\s*(\d+)', line)
            dst_m = re.search(r'"dst_idx"\s*:\s*(\d+)', line)
            if not src_m or not dst_m: continue
            src_idx = int(src_m.group(1))
            dst_idx = int(dst_m.group(1))
            nid_m = re.search(r'"node_ids"\s*:\s*\[([^\]]+)\]', line)
            if not nid_m: continue
            nids = [int(x) for x in nid_m.group(1).split(',')]
            pos_pairs.append((nids[src_idx], nids[dst_idx]))
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(pos_pairs))
    n_train = int(len(pos_pairs) * 0.8)
    return [pos_pairs[i] for i in perm[n_train:]]

import torch
import torch.nn as nn

class PencilModel(nn.Module):
    def __init__(self, scibert_embeddings):
        super().__init__()
        self.register_buffer('scibert_table', scibert_embeddings)
        self.scibert_proj = nn.Linear(768, 128)
        self.struct_proj = nn.Linear(68, 128)
        self.dist_embed = nn.Embedding(12 * 12, 32)
        self.combine = nn.Linear(128 + 128 + 32, 256)
        self.layers = nn.ModuleList()
        for _ in range(4):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(256, 8, batch_first=True, dropout=0.1),
                'ln1': nn.LayerNorm(256),
                'ffn': nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256), nn.Dropout(0.1)),
                'ln2': nn.LayerNorm(256),
                'mp': nn.Sequential(nn.Linear(256, 256), nn.GELU()),
            }))
        self.ln_f = nn.LayerNorm(256)
        self.head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, inp, node_ids):
        B, N, _ = inp.shape
        adj = inp[:, :N, 32:32 + N]
        deg = adj.sum(dim=2, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        dist_u_bin = (inp[:, :, 65] * 10).long().clamp(0, 11)
        dist_v_bin = (inp[:, :, 66] * 10).long().clamp(0, 11)
        dist_emb = self.dist_embed(dist_u_bin * 12 + dist_v_bin)
        x_raw = inp.clone(); x_raw[:, :, 65] = 0; x_raw[:, :, 66] = 0
        struct_emb = self.struct_proj(x_raw)
        nids_flat = node_ids.reshape(-1).clamp(0, self.scibert_table.shape[0] - 1)
        scibert_emb = self.scibert_proj(self.scibert_table[nids_flat].reshape(B, N, 768))
        pad_mask = (inp[:, :, :65].abs().sum(dim=2) == 0)
        pm3 = pad_mask.unsqueeze(-1)
        scibert_emb = scibert_emb.masked_fill(pm3, 0.0)
        dist_emb = dist_emb.masked_fill(pm3, 0.0)
        struct_emb = struct_emb.masked_fill(pm3, 0.0)
        x = self.combine(torch.cat([struct_emb, scibert_emb, dist_emb], dim=-1))
        for layer in self.layers:
            h = layer['ln1'](x)
            h, _ = layer['attn'](h, h, h, key_padding_mask=pad_mask)
            x = x + h
            x[:, :N] = x[:, :N] + layer['mp'](torch.bmm(adj_norm, x[:, :N]))
            h = layer['ln2'](x)
            x = x + layer['ffn'](h)
        x = self.ln_f(x)
        return self.head(torch.cat([x[:, 0], x[:, 1]], -1)).squeeze(-1)

if __name__ == '__main__':
    device = torch.device('cuda')
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)

    dist_matrix = np.load(f'{DATA_DIR}/dist_matrix.npy')
    val_pairs = _extract_val_pairs(DATA_DIR)

    all_nids = np.load(f'{DATA_DIR}/eval_subgraph_nids.npy', mmap_mode='r')
    cands_len = np.load(f'{DATA_DIR}/eval_cands_len.npy')

    # Compile Numba
    tmp_out = np.zeros((MAX_NODES, EMBED_DIM), dtype=np.float32)
    _reconstruct_features(all_nids[0, 0], 0, 1, dist_matrix, dist_matrix[0], dist_matrix[1], tmp_out)

    # Test 5 anchors
    for ckpt_name, ckpt_path in [('v5', f'{DATA_DIR}/pencil_v5_best.pt'),
                                   ('v3', f'{DATA_DIR}/pencil_v3_best.pt')]:
        model = PencilModel(scibert).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()

        print(f"\n=== {ckpt_name} checkpoint ===")
        for ai in range(5):
            u, v = val_pairs[ai]
            NC = int(cands_len[ai])
            nids_ai = all_nids[ai]

            inp = np.zeros((NC, MAX_NODES, EMBED_DIM), dtype=np.float32)
            for j in range(NC):
                _reconstruct_features(nids_ai[j], u, v, dist_matrix,
                                     dist_matrix[u], dist_matrix[v], tmp_out)
                inp[j] = tmp_out

            pos_idx = -1
            for j in range(NC):
                if nids_ai[j, 0] == u and nids_ai[j, 1] == v:
                    pos_idx = j; break

            # Score in batches
            scores = np.zeros(NC, dtype=np.float32)
            BS = 500
            for start in range(0, NC, BS):
                end = min(start + BS, NC)
                inp_t = torch.from_numpy(inp[start:end]).to(device)
                nids_t = torch.from_numpy(nids_ai[start:end]).to(device)
                with torch.inference_mode():
                    scores[start:end] = model(inp_t, nids_t).cpu().numpy()

            pos_score = scores[pos_idx]
            rank = int((scores > pos_score).sum() + 1)

            # Stats
            print(f"  anchor {ai}: u={u}, v={v}, pos_idx={pos_idx}, rank={rank}/{NC}, "
                  f"pos_score={pos_score:.4f}, "
                  f"score_range=[{scores.min():.4f}, {scores.max():.4f}], "
                  f"mean={scores.mean():.4f}, std={scores.std():.4f}")
