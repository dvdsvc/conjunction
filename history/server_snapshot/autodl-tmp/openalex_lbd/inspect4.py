"""
Critical test: Is the model just using candidate position (source vs non-source)?
If so, it's a trivial classifier, not learning bridge patterns.
"""
import numpy as np, json, re, torch, torch.nn as nn

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES = 32; EMBED_DIM = 68; SCIBERT_DIM = 768; HIDDEN = 256
N_LAYERS = 4; N_HEADS = 8; DIST_BINS = 10; DIST_EMBED_DIM = 32

class PencilV5(nn.Module):
    def __init__(self, scibert_embeddings):
        super().__init__()
        self.register_buffer('scibert_table', scibert_embeddings)
        self.scibert_proj = nn.Linear(SCIBERT_DIM, HIDDEN // 2)
        self.struct_proj = nn.Linear(EMBED_DIM, HIDDEN // 2)
        self.dist_embed = nn.Embedding((DIST_BINS + 2) * (DIST_BINS + 2), DIST_EMBED_DIM)
        self.combine = nn.Linear(HIDDEN // 2 + HIDDEN // 2 + DIST_EMBED_DIM, HIDDEN)
        self.cand_pos_embed = nn.Embedding(64, HIDDEN)
        self.layers = nn.ModuleList()
        for _ in range(N_LAYERS):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(HIDDEN, N_HEADS, batch_first=True, dropout=0.1),
                'ln1': nn.LayerNorm(HIDDEN),
                'ffn': nn.Sequential(nn.Linear(HIDDEN, HIDDEN * 4), nn.GELU(), nn.Linear(HIDDEN * 4, HIDDEN), nn.Dropout(0.1)),
                'ln2': nn.LayerNorm(HIDDEN),
                'mp': nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU()),
            }))
        self.ln_f = nn.LayerNorm(HIDDEN)
        self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))
    def _build_node_features(self, inp, node_ids):
        B, G, N, _ = inp.shape
        flat_inp = inp.reshape(B * G * N, -1)
        flat_nids = node_ids.reshape(B * G * N)
        x_raw = flat_inp.clone()
        x_raw[:, 2 * MAX_NODES + 1] = 0
        x_raw[:, 2 * MAX_NODES + 2] = 0
        struct_emb = self.struct_proj(x_raw)
        nids_flat = flat_nids.clamp(0, self.scibert_table.shape[0] - 1)
        scibert_emb = self.scibert_proj(self.scibert_table[nids_flat])
        dist_u_bin = (flat_inp[:, 2 * MAX_NODES + 1] * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        dist_v_bin = (flat_inp[:, 2 * MAX_NODES + 2] * DIST_BINS).long().clamp(0, DIST_BINS + 1)
        dist_idx = dist_u_bin * (DIST_BINS + 2) + dist_v_bin
        dist_emb = self.dist_embed(dist_idx)
        x = self.combine(torch.cat([struct_emb, scibert_emb, dist_emb], dim=-1))
        cand_idx = torch.arange(G, device=inp.device).unsqueeze(1).expand(-1, N).reshape(-1)
        cand_idx = cand_idx.unsqueeze(0).expand(B, -1).reshape(-1)
        x = x + self.cand_pos_embed(cand_idx)
        return x.reshape(B, G * N, -1)
    def _build_iso_mask(self, n_cands, n_nodes, device):
        S = (1 + n_cands) * n_nodes
        mask = torch.full((S, S), -10000.0, device=device)
        src_end = n_nodes
        mask[:src_end, :] = 0.0
        for k in range(1 + n_cands):
            start = k * n_nodes; end = (k + 1) * n_nodes
            mask[start:end, :src_end] = 0.0
            mask[start:end, start:end] = 0.0
        return mask
    def forward(self, inp, node_ids):
        B, G, N, _ = inp.shape
        device = inp.device
        x = self._build_node_features(inp, node_ids)
        adj = inp[:, :, :, MAX_NODES:MAX_NODES + N]
        deg = adj.sum(dim=3, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        pad_mask = (inp[:, :, :, :2 * MAX_NODES + 1].abs().sum(dim=3) == 0)
        pad_mask_flat = pad_mask.reshape(B, G * N)
        iso_mask = self._build_iso_mask(G - 1, N, device)
        for layer in self.layers:
            h = layer['ln1'](x)
            h, _ = layer['attn'](h, h, h, attn_mask=iso_mask, key_padding_mask=pad_mask_flat)
            x = x + h
            x_cand = x.reshape(B, G, N, -1)
            h_mp = layer['mp'](torch.einsum('bgnh,bgnm->bgmh', x_cand, adj_norm))
            x_cand = x_cand + h_mp
            x = x.reshape(B, G * N, -1)
            x[:, :] = x_cand.reshape(B, G * N, -1)
            h = layer['ln2'](x)
            x = x + layer['ffn'](h)
        x = self.ln_f(x)
        x_cand = x.reshape(B, G, N, -1)
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (x_cand * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)
        return self.head(pooled).squeeze(-1)

def _extract_val_pairs(data_dir):
    pos_pairs = []
    with open(f'{data_dir}/train.jsonl') as f:
        for line in f:
            if '"label": 1' not in line and '"label":1' not in line: continue
            src_m = re.search(r'"src_idx"\s*:\s*(\d+)', line)
            dst_m = re.search(r'"dst_idx"\s*:\s*(\d+)', line)
            if not src_m or not dst_m: continue
            src_idx = int(src_m.group(1)); dst_idx = int(dst_m.group(1))
            nid_m = re.search(r'"node_ids"\s*:\s*\[([^\]]+)\]', line)
            if not nid_m: continue
            nids = [int(x) for x in nid_m.group(1).split(',')]
            pos_pairs.append((nids[src_idx], nids[dst_idx]))
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(pos_pairs))
    n_train = int(len(pos_pairs) * 0.8)
    return [pos_pairs[i] for i in perm[n_train:]]

if __name__ == '__main__':
    device = torch.device('cuda')
    scibert = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    model = PencilV5(scibert).to(device)
    model.load_state_dict(torch.load(f'{DATA_DIR}/pencil_v5_iso_best.pt', weights_only=True))
    model.eval()

    concept_ids = json.load(open(f'{DATA_DIR}/concept_ids.json'))
    concept_names = json.load(open(f'{DATA_DIR}/concept_names.json'))
    def get_name(idx):
        if idx < 0 or idx >= len(concept_ids): return f"?{idx}"
        return concept_names.get(concept_ids[idx], concept_ids[idx])

    val_pairs = _extract_val_pairs(DATA_DIR)
    val_data = np.load(f'{DATA_DIR}/val_precomp_v3.npz')
    val_inp = torch.from_numpy(val_data['inp'].astype(np.float32)).reshape(-1, 6, 32, 68)
    val_nids = torch.from_numpy(val_data['node_ids']).reshape(-1, 6, 32)

    N = 200

    # ===== Test A: Candidate position embedding analysis =====
    print("=== cand_pos_embed values ===")
    with torch.no_grad():
        cpe = model.cand_pos_embed.weight[:6].cpu().numpy()
    for i in range(6):
        print(f"  cand_pos[{i}]: norm={np.linalg.norm(cpe[i]):.4f}, mean={cpe[i].mean():.6f}")

    # ===== Test B: Swap source and negative (put neg at position 0) =====
    print(f"\n=== Test B: Put Neg at Source Position, Pos at Candidate Position ===")
    print(f"If model just scores source high, neg at pos[0] should get high score")
    
    # Build: [neg_subgraph, pos_subgraph] — swap which is source
    inp_swap_pos = val_inp[:N].clone()
    nids_swap_pos = val_nids[:N].clone()
    # Swap position 0 and position 1
    tmp_inp = inp_swap_pos[:, 0].clone()
    tmp_nids = nids_swap_pos[:, 0].clone()
    inp_swap_pos[:, 0] = inp_swap_pos[:, 1]
    inp_swap_pos[:, 1] = tmp_inp
    nids_swap_pos[:, 0] = val_nids[:N, 1]
    nids_swap_pos[:, 1] = tmp_nids

    with torch.inference_mode():
        orig = model(val_inp[:N].to(device), val_nids[:N].to(device)).cpu().numpy()
        swapped = model(inp_swap_pos.to(device), nids_swap_pos.to(device)).cpu().numpy()

    print(f"  Original:    source(=pos)={orig[:,0].mean():.3f}  cand(=neg)={orig[:,1].mean():.3f}")
    print(f"  Swapped:     source(=neg)={swapped[:,0].mean():.3f}  cand(=pos)={swapped[:,1].mean():.3f}")
    print(f"  After swap, source > cand: {(swapped[:, 0] > swapped[:, 1]).mean():.4f}")
    print(f"  -> If source always wins regardless of content, this should be ~1.0")

    # ===== Test C: Source = negative pair, candidates = pos + more negs =====
    print(f"\n=== Test C: Negative as Source, Score Pos Candidate ===")
    print(f"Use neg candidate as source, and test if pos gets ranked #1")
    # Build: [neg_cand_as_src, pos_as_cand, neg2, neg3, neg4, neg5]
    inp_neg_src = val_inp[:N].clone()
    nids_neg_src = val_nids[:N].clone()
    # Put neg (idx 1) at source position
    inp_neg_src[:, 0] = val_inp[:N, 1]
    nids_neg_src[:, 0] = val_nids[:N, 1]
    # Keep pos at position 1
    inp_neg_src[:, 1] = val_inp[:N, 0]
    nids_neg_src[:, 1] = val_nids[:N, 0]

    with torch.inference_mode():
        neg_src_scores = model(inp_neg_src.to(device), nids_neg_src.to(device)).cpu().numpy()

    # Source should get high score (it's always source=candidate 0)
    # Candidate 1 (which is now the POSITIVE) should get... what?
    print(f"  source(=neg): {neg_src_scores[:, 0].mean():.3f}")
    print(f"  cand1(=pos):  {neg_src_scores[:, 1].mean():.3f}")
    print(f"  cand2-5(=neg): {neg_src_scores[:, 2:].mean():.3f}")
    print(f"  pos score > other negs: {(neg_src_scores[:, 1] > neg_src_scores[:, 2:].max(axis=1)).mean():.4f}")

    # ===== Test D: Pure zero input — what score does source get? =====
    print(f"\n=== Test D: Zero Input (only cand_pos_embed) ===")
    zero_inp = torch.zeros(2, 6, 32, 68, device=device)
    zero_nids = torch.zeros(2, 6, 32, dtype=torch.long, device=device)
    # Make node 0 non-padded so it gets processed
    zero_inp[:, :, 0, 0] = 1.0  # minimal feature
    with torch.inference_mode():
        zero_scores = model(zero_inp, zero_nids).cpu().numpy()
    for g in range(6):
        print(f"  cand[{g}] score: {zero_scores[0, g]:.4f}")

    # ===== Test E: Same input for all candidates (no structure signal) =====
    print(f"\n=== Test E: Identical Subgraphs for All Candidates ===")
    print(f"Same features everywhere -> does source still score higher?")
    same_inp = val_inp[:N].clone()
    same_nids = val_nids[:N].clone()
    for g in range(1, 6):
        same_inp[:, g] = same_inp[:, 0]
        same_nids[:, g] = same_nids[:, 0]
    with torch.inference_mode():
        same_scores = model(same_inp.to(device), same_nids.to(device)).cpu().numpy()
    print(f"  All identical subgraphs:")
    for g in range(6):
        print(f"    cand[{g}]: {same_scores[:20, g].mean():.4f}")
    print(f"  source - cand1 diff: {(same_scores[:, 0] - same_scores[:, 1]).mean():.4f}")

    # ===== Test F: Completely random features, no node_ids =====
    print(f"\n=== Test F: Random Noise Features ===")
    rng = np.random.RandomState(42)
    rand_inp = torch.from_numpy(rng.randn(N, 6, 32, 68).astype(np.float32)).to(device)
    rand_nids = torch.from_numpy(rng.randint(0, 1000, (N, 6, 32)).astype(np.int64)).to(device)
    with torch.inference_mode():
        rand_scores = model(rand_inp, rand_nids).cpu().numpy()
    print(f"  Random noise scores:")
    for g in range(6):
        print(f"    cand[{g}]: {rand_scores[:, g].mean():.4f}")

    print(f"\nDone.")
