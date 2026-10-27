"""Fast eval for v15: multi-K recall/precision. Runs on CPU to not interrupt training."""
import os, json, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
TEMP_DIR = f'{DATA_DIR}/temporal_l3'

MAX_NODES = 64
MAX_MERGED = 256
EMBED_DIM = 2 * MAX_NODES + 1
SCIBERT_DIM = 768
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
GNN_HIDDEN = 128
PAIR_DIM = 21
FUSION_DIM = 256
N_CANDIDATES = 100
PENCIL_GROUP = 20

CONCEPT_IDS_FILE = f'{DATA_DIR}/concept_ids_l3.json'
OLD_CONCEPT_IDS_FILE = f'{DATA_DIR}/concept_ids.json'


def load_scibert_remapped(concept_ids, n_concepts):
    scibert_old = torch.load(f'{DATA_DIR}/concept_scibert.pt', weights_only=True)
    with open(OLD_CONCEPT_IDS_FILE) as f:
        old_ids = json.load(f)
    old_id_to_idx = {cid: i for i, cid in enumerate(old_ids)}
    scibert = torch.zeros(n_concepts, SCIBERT_DIM)
    for new_idx, cid in enumerate(concept_ids):
        if cid in old_id_to_idx:
            scibert[new_idx] = scibert_old[old_id_to_idx[cid]]
    return scibert


def compute_ndcg_at_k(scores, labels, k=10):
    ranked = np.argsort(-scores)[:k]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum((2.0 ** labels[ranked] - 1.0) / discounts)
    ideal = np.sort(labels)[::-1][:k]
    idcg = np.sum((2.0 ** ideal - 1.0) / discounts)
    return dcg / max(idcg, 1e-10)


# ── Model (same as train script) ──────────────────────────────────────

class PairMLP(nn.Module):
    def __init__(self, in_dim=PAIR_DIM, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, out_dim), nn.GELU())
    def forward(self, x):
        B, K, D = x.shape
        return self.net(x.reshape(B*K, D)).reshape(B, K, -1)

class SubgraphGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = GNN_HIDDEN
        self.node_proj = nn.Linear(SCIBERT_DIM + 1, GNN_HIDDEN)
        self.w_self = nn.ModuleList([nn.Linear(GNN_HIDDEN, GNN_HIDDEN) for _ in range(3)])
        self.w_neigh = nn.ModuleList([nn.Linear(GNN_HIDDEN, GNN_HIDDEN) for _ in range(3)])
        self.lns = nn.ModuleList([nn.LayerNorm(GNN_HIDDEN) for _ in range(3)])
        self.readout = nn.Sequential(nn.Linear(3*GNN_HIDDEN, 2*GNN_HIDDEN), nn.GELU(), nn.Dropout(0.1), nn.Linear(2*GNN_HIDDEN, GNN_HIDDEN), nn.GELU())
    def forward(self, scibert_table, merged_nodes, edge_idx, edge_val, u_local, v_local):
        BK, M = merged_nodes.shape
        device = merged_nodes.device
        B = edge_idx.shape[0]
        K_per_sample = BK // B
        batch_offsets = torch.arange(B, device=device).unsqueeze(1).unsqueeze(2) * (K_per_sample * M)
        edge_idx_global = edge_idx + batch_offsets
        flat_idx = edge_idx_global.reshape(2, -1)
        flat_val = edge_val.reshape(-1)
        keep = flat_val != 0
        adj_sparse = torch.sparse_coo_tensor(flat_idx[:, keep], flat_val[keep], size=(BK*M, BK*M), device=device)
        deg = torch.sparse.sum(adj_sparse, dim=1).to_dense().unsqueeze(1).clamp(min=1)
        row_deg = deg[flat_idx[0, keep]]
        adj_norm = torch.sparse_coo_tensor(flat_idx[:, keep], flat_val[keep]/row_deg.squeeze(), size=(BK*M, BK*M), device=device)
        nids = merged_nodes.clamp(0, scibert_table.shape[0]-1)
        emb = scibert_table[nids]
        log_deg = torch.log1p(deg.reshape(BK, M, 1))
        x = self.node_proj(torch.cat([emb, log_deg], dim=2))
        x_flat = x.reshape(BK*M, -1)
        pad_mask = (merged_nodes == 0)
        for i in range(3):
            x_neigh = torch.sparse.mm(adj_norm, x_flat)
            x_self = self.w_self[i](x_flat) + self.w_neigh[i](x_neigh)
            x_flat = self.lns[i](x_flat + F.gelu(x_self))
        x = x_flat.reshape(BK, M, -1).masked_fill(pad_mask.unsqueeze(-1), 0.0)
        BKM = torch.arange(BK, device=device)
        u_emb = x[BKM, u_local]; v_emb = x[BKM, v_local]
        valid = (~pad_mask).unsqueeze(-1).float()
        mean_pool = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return self.readout(torch.cat([u_emb, v_emb, mean_pool], dim=1))

class PencilEncoder(nn.Module):
    def __init__(self, scibert_embeddings):
        super().__init__()
        self.register_buffer('scibert_table', scibert_embeddings)
        self.scibert_proj = nn.Linear(SCIBERT_DIM, HIDDEN//2)
        self.struct_proj = nn.Linear(EMBED_DIM, HIDDEN//2)
        self.combine = nn.Linear(HIDDEN, HIDDEN)
        self.cand_pos_embed = nn.Embedding(128, HIDDEN)
        self.layers = nn.ModuleList()
        for _ in range(N_LAYERS):
            self.layers.append(nn.ModuleDict({
                'qkv': nn.Linear(HIDDEN, HIDDEN*3), 'out_proj': nn.Linear(HIDDEN, HIDDEN),
                'ln1': nn.LayerNorm(HIDDEN), 'ffn': nn.Sequential(nn.Linear(HIDDEN, HIDDEN*4), nn.GELU(), nn.Linear(HIDDEN*4, HIDDEN), nn.Dropout(0.1)),
                'ln2': nn.LayerNorm(HIDDEN), 'mp': nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU()),
            }))
        self.ln_f = nn.LayerNorm(HIDDEN)
    def _build_node_features(self, inp, node_ids):
        B, G, N, _ = inp.shape
        flat_inp = inp.reshape(B*G*N, -1); flat_nids = node_ids.reshape(B*G*N)
        struct_emb = self.struct_proj(flat_inp)
        scibert_emb = self.scibert_proj(self.scibert_table[flat_nids.clamp(0, self.scibert_table.shape[0]-1)])
        x = self.combine(torch.cat([struct_emb, scibert_emb], dim=-1))
        cand_idx = torch.arange(G, device=inp.device).unsqueeze(1).expand(-1,N).reshape(-1).unsqueeze(0).expand(B,-1).reshape(-1)
        x = x + self.cand_pos_embed(cand_idx)
        return x.reshape(B, G*N, -1)
    def _build_iso_mask(self, n_cands, n_nodes, device):
        S = (1+n_cands)*n_nodes; mask = torch.full((S,S), -10000.0, device=device)
        src_end = n_nodes; mask[:src_end, :] = 0.0
        for k in range(1+n_cands):
            s = k*n_nodes; e = (k+1)*n_nodes; mask[s:e, :src_end] = 0.0; mask[s:e, s:e] = 0.0
        return mask
    def forward(self, inp, node_ids):
        B, G, N, _ = inp.shape; device = inp.device
        x = self._build_node_features(inp, node_ids)
        adj = inp[:,:,:,MAX_NODES:MAX_NODES+N]; deg = adj.sum(dim=3,keepdim=True).clamp(min=1); adj_norm = adj/deg
        pad_mask = (inp[:,:,:,:2*MAX_NODES].abs().sum(dim=3)==0); pad_mask_flat = pad_mask.reshape(B, G*N)
        iso_mask = self._build_iso_mask(G-1, N, device)
        bool_mask = ((~pad_mask_flat).unsqueeze(1).expand(B, G*N, G*N) & (iso_mask==0.0).unsqueeze(0)).unsqueeze(1)
        S = G*N
        for layer in self.layers:
            h = layer['ln1'](x); qkv = layer['qkv'](h); q,k,v = qkv.chunk(3, dim=-1)
            q = q.view(B,S,N_HEADS,HIDDEN//N_HEADS).transpose(1,2)
            k = k.view(B,S,N_HEADS,HIDDEN//N_HEADS).transpose(1,2)
            v = v.view(B,S,N_HEADS,HIDDEN//N_HEADS).transpose(1,2)
            attn_out = F.scaled_dot_product_attention(q,k,v,attn_mask=bool_mask,dropout_p=0.0)
            attn_out = attn_out.transpose(1,2).contiguous().view(B,S,HIDDEN)
            x = x + layer['out_proj'](attn_out)
            x_cand = x.reshape(B,-1,MAX_NODES,HIDDEN)
            h_mp = layer['mp'](torch.einsum('bgnh,bgnm->bgmh', x_cand, adj_norm))
            x = (x_cand + h_mp).reshape(x.shape)
            x = x + layer['ffn'](layer['ln2'](x))
        x = self.ln_f(x); x_cand = x.reshape(B,G,N,-1)
        valid = (~pad_mask).unsqueeze(-1).float()
        return (x_cand * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)

class PencilV15(nn.Module):
    def __init__(self, scibert_embeddings):
        super().__init__()
        self.register_buffer('scibert_table', scibert_embeddings)
        self.pair_mlp = PairMLP(PAIR_DIM, 64)
        self.pencil = PencilEncoder(scibert_embeddings)
        self.gnn = SubgraphGNN()
        self.fusion = nn.Sequential(nn.Linear(448, FUSION_DIM), nn.GELU(), nn.Dropout(0.1), nn.Linear(FUSION_DIM, FUSION_DIM), nn.GELU())
        self.head = nn.Sequential(nn.Linear(FUSION_DIM, FUSION_DIM), nn.GELU(), nn.Linear(FUSION_DIM, 1))
    def forward(self, inp, node_ids, pair_feats, merged_nodes, edge_idx, edge_val, u_local, v_local):
        B, G = inp.shape[:2]
        pair_vec = self.pair_mlp(pair_feats)
        pencil_parts = []
        anchor_inp, anchor_nid = inp[:,:1], node_ids[:,:1]
        cand_inp, cand_nid = inp[:,1:], node_ids[:,1:]
        for start in range(0, cand_inp.shape[1], PENCIL_GROUP):
            end = min(start+PENCIL_GROUP, cand_inp.shape[1])
            pencil_parts.append(self.pencil(torch.cat([anchor_inp, cand_inp[:,start:end]],1), torch.cat([anchor_nid, cand_nid[:,start:end]],1)))
        pencil_vec = torch.cat([pencil_parts[0][:,:1]] + [p[:,1:] for p in pencil_parts], dim=1)
        gnn_vec = self.gnn(self.scibert_table, merged_nodes.reshape(B*G, MAX_MERGED), edge_idx, edge_val, u_local.reshape(B*G), v_local.reshape(B*G)).reshape(B, G, GNN_HIDDEN)
        return self.head(self.fusion(torch.cat([pair_vec, pencil_vec, gnn_vec], -1))).squeeze(-1)


# ── Dataset (same as train) ───────────────────────────────────────────

class AnchorDatasetV15(torch.utils.data.Dataset):
    def __init__(self, pairs_by_src, concept_feats, pair_feats, merged_nodes, deltas, u_local, v_local, adj_csr, n_candidates, rng_seed=42):
        self.anchors = [u for u in sorted(pairs_by_src.keys()) if len(pairs_by_src[u]) >= 2]
        self.pairs_by_src = pairs_by_src
        self.concept_feats = concept_feats
        self.pair_feats = pair_feats
        self.merged_nodes = merged_nodes
        self.deltas = deltas
        self.u_local = u_local
        self.v_local = v_local
        self.adj_csr = adj_csr.tocsr().astype(np.float32)
        self.K = n_candidates
        self.rng = np.random.RandomState(rng_seed)
    def __len__(self): return len(self.anchors)
    def __getitem__(self, idx):
        u = self.anchors[idx]; cands = self.pairs_by_src[u]
        idxs = self.rng.choice(len(cands), self.K, replace=(len(cands)<self.K))
        pair_is = [cands[i][0] for i in idxs]; vs = [cands[i][1] for i in idxs]
        dcs = np.array([self.deltas[pi] for pi in pair_is], dtype=np.float32)
        feats = self.concept_feats[vs]
        nids = np.zeros((len(vs), MAX_NODES), dtype=np.int64)
        for ci, v in enumerate(vs): nids[ci,0]=u; nids[ci,1]=v
        pf = self.pair_feats[pair_is]; mn = self.merged_nodes[pair_is]
        ul = self.u_local[pair_is].astype(np.int64); vl = self.v_local[pair_is].astype(np.int64)
        edge_rows=[]; edge_cols=[]; edge_vals=[]; adj=self.adj_csr; uniq_set=set(); nn_per=[]
        for ci in range(self.K):
            nodes=mn[ci]; nn=int(np.count_nonzero(nodes)); nn_per.append(nn)
            if nn>=2: uniq_set.update(int(x) for x in nodes[:nn])
        if uniq_set:
            uniq=np.array(sorted(uniq_set),dtype=np.int64); n2i={int(n):i for i,n in enumerate(uniq)}
            sub=adj[uniq][:,uniq].toarray()
            for ci in range(self.K):
                nn=nn_per[ci]
                if nn<2: continue
                sel=mn[ci,:nn]; idx=np.array([n2i[int(x)] for x in sel],dtype=np.int64)
                la=sub[idx][:,idx]; r,c=np.where(la!=0)
                if len(r)>0:
                    edge_rows.append(ci*MAX_MERGED+r.astype(np.int64))
                    edge_cols.append(ci*MAX_MERGED+c.astype(np.int64))
                    edge_vals.append(la[r,c].astype(np.float32))
        if edge_rows:
            edge_idx=np.stack([np.concatenate(edge_rows),np.concatenate(edge_cols)]); edge_val=np.concatenate(edge_vals)
        else:
            edge_idx=np.zeros((2,0),dtype=np.int64); edge_val=np.zeros(0,dtype=np.float32)
        G=1+self.K
        out_feats=np.zeros((G,MAX_NODES,EMBED_DIM),dtype=np.float32); out_feats[1:]=feats
        out_nids=np.zeros((G,MAX_NODES),dtype=np.int64); out_nids[0,0]=u; out_nids[1:]=nids
        out_dcs=np.zeros(G,dtype=np.float32); out_dcs[1:]=dcs
        out_pf=np.zeros((G,pf.shape[1]),dtype=np.float32); out_pf[1:]=pf
        out_mn=np.zeros((G,MAX_MERGED),dtype=np.int32); out_mn[0,0]=u; out_mn[1:]=mn
        out_ul=np.zeros(G,dtype=np.int64); out_ul[1:]=ul
        out_vl=np.zeros(G,dtype=np.int64); out_vl[1:]=vl
        return (torch.from_numpy(out_feats), torch.from_numpy(out_nids), torch.from_numpy(out_dcs),
                torch.from_numpy(out_pf), torch.from_numpy(out_mn), torch.from_numpy(edge_idx),
                torch.from_numpy(edge_val), torch.from_numpy(out_ul), torch.from_numpy(out_vl))

def collate(batch):
    max_e=max(b[5].shape[1] for b in batch); B=len(batch)
    ei=torch.zeros(B,2,max_e,dtype=torch.int64); ev=torch.zeros(B,max_e,dtype=torch.float32)
    for i,b in enumerate(batch):
        e=b[5].shape[1]
        if e>0: ei[i,:,:e]=b[5]; ev[i,:e]=b[6]
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]), torch.stack([b[3] for b in batch]),
            torch.stack([b[4] for b in batch]), ei, ev,
            torch.stack([b[7] for b in batch]), torch.stack([b[8] for b in batch]))


def main():
    print("Loading...", flush=True)
    with open(CONCEPT_IDS_FILE) as f: concept_ids = json.load(f)
    n_concepts = len(concept_ids)
    scibert_table = load_scibert_remapped(concept_ids, n_concepts)

    adj_val = sp.load_npz(f'{TEMP_DIR}/graph_v2_2021.npz').tocsr()
    ppr_val = np.load(f'{TEMP_DIR}/ppr_v2_2021.npy').astype(np.float32)
    cache_path = f'{TEMP_DIR}/concept_feats_v9_2021.npy'
    concept_feats_val = np.load(cache_path)

    base = f'{TEMP_DIR}/precomputed_v14/fold4'
    pf = np.load(f'{base}/train_pair_feats.npy', mmap_mode='r')
    mn = np.load(f'{base}/train_merged_nodes.npy', mmap_mode='r')
    deltas = np.load(f'{base}/train_deltas.npy')
    u_loc = np.load(f'{base}/train_u_local.npy')
    v_loc = np.load(f'{base}/train_v_local.npy')
    with open(f'{base}/train_pairs_by_src.json') as f:
        pairs_by_src = {int(k): v for k, v in json.load(f).items()}
    print(f"  {len(deltas):,} pairs, {len(pairs_by_src):,} anchors", flush=True)

    val_ds = AnchorDatasetV15(pairs_by_src, concept_feats_val, pf, mn, deltas, u_loc, v_loc, adj_val, N_CANDIDATES, rng_seed=99)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate)

    device = torch.device('cpu')
    model = PencilV15(scibert_table).to(device)
    model_file = f'{TEMP_DIR}/model_v15_rolling/best.pt'
    if not os.path.exists(model_file):
        model_file = f'{TEMP_DIR}/model_v14_rolling/best.pt'
    model.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=True))
    model.eval()
    print(f"  Loaded {model_file}", flush=True)

    t0 = time.time()
    all_ndcg = []; all_mrr = []
    recall_dict = {k: [] for k in [10, 15, 20, 25, 30, 50]}
    prec_dict = {k: [] for k in [10, 15, 20, 25, 30, 50]}

    with torch.no_grad():
        for bi, (inp, node_ids, dcs, pf_b, mn_b, edge_idx, edge_val, ul, vl) in enumerate(val_loader):
            scores = model(inp, node_ids, pf_b, mn_b, edge_idx, edge_val, ul, vl)
            scores_np = scores.numpy(); dcs_np = dcs.numpy()
            for b in range(scores_np.shape[0]):
                ranked = np.argsort(-scores_np[b])
                all_ndcg.append(compute_ndcg_at_k(scores_np[b], dcs_np[b], k=10))
                pos_mask = dcs_np[b] >= 4; n_pos = int(pos_mask.sum())
                if n_pos == 0: continue
                for k_val in [10, 15, 20, 25, 30, 50]:
                    k_actual = min(k_val, len(ranked))
                    hits_k = int(pos_mask[ranked[:k_actual]].sum())
                    recall_dict[k_val].append(hits_k / n_pos)
                    prec_dict[k_val].append(hits_k / k_actual)
                for rank, idx in enumerate(ranked):
                    if pos_mask[idx]: all_mrr.append(1.0/(rank+1)); break
            if (bi+1) % 50 == 0:
                print(f"  batch {bi+1} ({time.time()-t0:.0f}s)", flush=True)

    dt = time.time() - t0
    print(f"\n=== v15 Eval ({model_file}) [{dt:.0f}s] ===", flush=True)
    print(f"  NDCG@10={np.mean(all_ndcg):.4f}  MRR={np.mean(all_mrr):.4f}", flush=True)
    for k_val in [10, 15, 20, 25, 30, 50]:
        r = np.mean(recall_dict[k_val]) if recall_dict[k_val] else 0
        p = np.mean(prec_dict[k_val]) if prec_dict[k_val] else 0
        print(f"  R@{k_val:>2}={r:.3f}  P@{k_val:>2}={p:.3f}", flush=True)

if __name__ == '__main__':
    main()
