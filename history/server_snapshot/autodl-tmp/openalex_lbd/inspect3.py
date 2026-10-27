"""
Precise structural comparison:
1. Element-wise adjacency diff between pos and neg subgraphs
2. Targeted ablation: swap node 1 between pos/neg -> does score flip?
3. Ablate everything except node 1 -> still works?
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

def count_nn(feat):
    return int(np.greater(np.abs(feat[:, :65]).sum(axis=1), 0).sum())

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

    N = 200  # test on first 200 anchors

    # ===== Test 1: Exact adjacency difference =====
    print("=== Test 1: Element-wise Adjacency Difference (pos[0] vs neg[1]) ===")
    adj_diffs = []
    for i in range(N):
        pos_nn = count_nn(val_inp[i, 0].numpy())
        neg_nn = count_nn(val_inp[i, 1].numpy())
        nn = max(pos_nn, neg_nn)
        pos_adj = val_inp[i, 0, :nn, 32:32+nn].numpy()
        neg_adj = val_inp[i, 1, :nn, 32:32+nn].numpy()
        diff = np.abs(pos_adj - neg_adj).sum()
        total = pos_adj.sum() + neg_adj.sum()
        adj_diffs.append(diff / max(total, 1))
    print(f"  Adj diff ratio: mean={np.mean(adj_diffs):.4f}, median={np.median(adj_diffs):.4f}")
    print(f"  (0.0 = identical, 1.0 = completely different)")

    # ===== Test 2: Swap node 1 between pos and neg =====
    print(f"\n=== Test 2: Swap Node 1 (candidate v) Between Pos and Neg ===")
    print(f"If model only checks node 1 semantics -> score should flip")
    print(f"If structure matters -> score won't simply flip")
    
    inp_swap = val_inp[:N].clone()
    # Swap: put neg's node 1 features into pos candidate, and vice versa
    # Pos candidate (index 0): swap node 1 with neg candidate (index 1) node 1
    tmp_feat = inp_swap[:, 0, 1, :].clone()
    tmp_nids = val_nids[:N, 0, 1].clone()
    inp_swap[:, 0, 1, :] = inp_swap[:, 1, 1, :]
    inp_swap[:, 1, 1, :] = tmp_feat
    
    nids_swap = val_nids[:N].clone()
    nids_swap[:, 0, 1] = val_nids[:N, 1, 1]
    nids_swap[:, 1, 1] = tmp_nids
    
    with torch.inference_mode():
        orig_scores = model(val_inp[:N].to(device), val_nids[:N].to(device)).cpu().numpy()
        swap_scores = model(inp_swap.to(device), nids_swap.to(device)).cpu().numpy()
    
    orig_pos = orig_scores[:, 0]; orig_neg = orig_scores[:, 1]
    swap_pos = swap_scores[:, 0]; swap_neg = swap_scores[:, 1]
    
    print(f"  Original: pos={orig_pos.mean():.3f}, neg={orig_neg.mean():.3f}")
    print(f"  Swapped:  pos(now has neg's v)={swap_pos.mean():.3f}, neg(now has pos's v)={swap_neg.mean():.3f}")
    print(f"  Score flip rate (pos_score < neg_score after swap): {(swap_pos < swap_neg).mean():.4f}")

    # ===== Test 3: Zero out everything except node 0 and node 1 =====
    print(f"\n=== Test 3: Keep Only Node 0 (u) and Node 1 (v) ===")
    print(f"Zero out nodes 2-31 -> how much does score change?")
    inp_only2 = val_inp[:N].clone()
    inp_only2[:, :, 2:, :] = 0  # zero out nodes 2-31
    nids_only2 = val_nids[:N].clone()
    nids_only2[:, :, 2:] = 0
    
    with torch.inference_mode():
        only2_scores = model(inp_only2.to(device), nids_only2.to(device)).cpu().numpy()
    
    print(f"  Full subgraph: pos={orig_pos.mean():.3f}, neg={orig_neg.mean():.3f}, margin={orig_pos.mean()-orig_neg.mean():.3f}")
    print(f"  Only u+v:      pos={only2_scores[:,0].mean():.3f}, neg={only2_scores[:,1].mean():.3f}, margin={only2_scores[:,0].mean()-only2_scores[:,1].mean():.3f}")
    print(f"  Still ranks pos > neg: {(only2_scores[:, 0] > only2_scores[:, 1].max(axis=1)).mean():.4f}")

    # ===== Test 4: Zero out only node 1, keep everything else =====
    print(f"\n=== Test 4: Zero Out Only Node 1 (v) ===")
    print(f"Remove v's features but keep subgraph structure -> how much does score drop?")
    inp_no1 = val_inp[:N].clone()
    inp_no1[:, :, 1, :] = 0
    nids_no1 = val_nids[:N].clone()
    nids_no1[:, :, 1] = 0
    
    with torch.inference_mode():
        no1_scores = model(inp_no1.to(device), nids_no1.to(device)).cpu().numpy()
    
    print(f"  Full:      pos={orig_pos.mean():.3f}, neg={orig_neg.mean():.3f}")
    print(f"  No node1:  pos={no1_scores[:,0].mean():.3f}, neg={no1_scores[:,1].mean():.3f}")

    # ===== Test 5: Shuffle nodes 2-31 (keep same set, random order) =====
    print(f"\n=== Test 5: Shuffle Nodes 2-31 (same nodes, random positions) ===")
    print(f"If model relies on per-node structure -> score should change")
    print(f"If model only uses aggregate features -> score stays same")
    inp_shuf = val_inp[:N].clone()
    nids_shuf = val_nids[:N].clone()
    rng = np.random.RandomState(99)
    for i in range(N):
        for g in range(6):
            perm = rng.permutation(30) + 2  # shuffle indices 2-31
            inp_shuf[i, g, 2:] = inp_shuf[i, g, perm]
            nids_shuf[i, g, 2:] = nids_shuf[i, g, perm]
            # Also shuffle adjacency rows and columns
            old_adj = inp_shuf[i, g, 2:, 32:62].clone()
            inp_shuf[i, g, 2:, 32:62] = old_adj[perm - 2]
    
    with torch.inference_mode():
        shuf_scores = model(inp_shuf.to(device), nids_shuf.to(device)).cpu().numpy()
    
    print(f"  Original: pos={orig_pos.mean():.3f}, margin={orig_pos.mean()-orig_neg.mean():.3f}")
    print(f"  Shuffled: pos={shuf_scores[:,0].mean():.3f}, margin={shuf_scores[:,0].mean()-shuf_scores[:,1].mean():.3f}")

    # ===== Test 6: Replace subgraph nodes with random concepts =====
    print(f"\n=== Test 6: Replace Nodes 2-31 With Random Concepts ===")
    print(f"Keep node 0 (u) and node 1 (v), replace rest with random SciBERT vectors")
    inp_rand = val_inp[:N].clone()
    nids_rand = val_nids[:N].clone()
    n_concepts = len(concept_ids)
    rng2 = np.random.RandomState(42)
    for i in range(N):
        for g in range(6):
            for j in range(2, 32):
                rand_idx = rng2.randint(0, n_concepts)
                # Set to zero (effectively removing)
                inp_rand[i, g, j, :] = 0
                nids_rand[i, g, j] = 0
    
    with torch.inference_mode():
        rand_scores = model(inp_rand.to(device), nids_rand.to(device)).cpu().numpy()
    
    print(f"  Full:      pos={orig_pos.mean():.3f}, margin={orig_pos.mean()-orig_neg.mean():.3f}")
    print(f"  Only u+v:  pos={rand_scores[:,0].mean():.3f}, margin={rand_scores[:,0].mean()-rand_scores[:,1].mean():.3f}")

    print(f"\nDone.")
