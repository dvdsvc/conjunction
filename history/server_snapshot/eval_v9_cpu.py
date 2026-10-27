"""Batch eval on CPU — uses DataLoader batching for ~10x speedup."""
import time, json, sys, os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/autodl-tmp/openalex_data/pencil_data/temporal_l3')
os.chdir('/root/autodl-tmp/openalex_data/pencil_data/temporal_l3')

from train_temporal_v9 import (
    PencilV9, AnchorDatasetV9, anchor_collate_v9,
    load_scibert_remapped, load_precomputed_v9,
    TEMP_DIR, CONCEPT_IDS_FILE, N_CANDIDATES,
)

N_EVAL = 2000
BATCH = 16

def main():
    torch.set_num_threads(8)  # cap threads to leave room for training
    device = torch.device('cpu')
    print('Device: cpu (batch eval)', flush=True)

    with open(CONCEPT_IDS_FILE) as f:
        concept_ids = json.load(f)
    n_concepts = len(concept_ids)
    scibert_table = load_scibert_remapped(concept_ids, n_concepts)

    adj_val = sp.load_npz(f'{TEMP_DIR}/graph_v2_2015.npz').tocsr()
    concept_feats_val = np.load(f'{TEMP_DIR}/concept_feats_v9_2015.npy')
    print('  Loaded cached concept_feats 2015', flush=True)

    (src_val, pf_val, mn_val,
     deltas_val, ul_val, vl_val) = load_precomputed_v9(0, 'val')

    val_ds = AnchorDatasetV9(src_val, concept_feats_val, pf_val,
                             mn_val, deltas_val, ul_val, vl_val,
                             adj_val, N_CANDIDATES, rng_seed=99)

    model = PencilV9(scibert_table).to(device)
    model_file = f'{TEMP_DIR}/model_fold0_v9/best.pt'
    model.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=True))
    model.eval()
    print(f'  Loaded {model_file}', flush=True)

    n_eval = min(N_EVAL, len(val_ds))
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_ds, range(n_eval)),
        batch_size=BATCH, shuffle=False, num_workers=0,
        collate_fn=anchor_collate_v9)

    print(f'  Evaluating {n_eval} anchors, batch_size={BATCH}...', flush=True)

    mrr_sum, hit1, hit3, hit5, n_valid = 0.0, 0, 0, 0, 0
    all_top, all_bot = [], []
    auc_scores, auc_labels = [], []
    t0 = time.time()
    done = 0

    for batch in val_loader:
        feats, nids, dcs, pf, mn, edge_idx, edge_val, ul, vl = [x.to(device) for x in batch]
        B = feats.shape[0]

        with torch.no_grad():
            scores = model(feats, nids, pf, mn, edge_idx, edge_val, ul, vl)

        s_np = scores.cpu().numpy()  # [B, G]
        d_np = dcs.cpu().numpy()     # [B, G]

        for b in range(B):
            s = s_np[b]
            d = d_np[b]
            cand = np.argsort(-s)[1:]
            pos_mask = d[cand] >= 2

            if pos_mask.any():
                mrr_sum += 1.0 / (np.argmax(pos_mask) + 1)
                n_valid += 1
            if d[cand[0]] >= 2: hit1 += 1
            if (d[cand[:3]] >= 2).any(): hit3 += 1
            if (d[cand[:5]] >= 2).any(): hit5 += 1

            K = len(cand)
            top_k = max(1, K // 10)
            all_top.append(d[cand[:top_k]].mean())
            all_bot.append(d[cand[top_k:]].mean())

            for ci in range(1, len(s)):
                if d[ci] >= 2:
                    auc_scores.append(s[ci]); auc_labels.append(1)
                elif d[ci] == 0:
                    auc_scores.append(s[ci]); auc_labels.append(0)

        done += B
        if done % 200 < BATCH:
            elapsed = time.time() - t0
            eta = elapsed / done * (n_eval - done)
            print(f'    {done}/{n_eval} ({elapsed:.0f}s, ETA {eta:.0f}s)', flush=True)

    dt = time.time() - t0
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(auc_labels, auc_scores)

    print(f'\n{"="*50}', flush=True)
    print(f'  Batch eval ({n_eval} anchors, {dt:.0f}s)', flush=True)
    print(f'  MRR:    {mrr_sum / max(n_valid, 1):.4f}', flush=True)
    print(f'  Hit@1:  {hit1 / n_eval:.4f}  ({hit1}/{n_eval})', flush=True)
    print(f'  Hit@3:  {hit3 / n_eval:.4f}  ({hit3}/{n_eval})', flush=True)
    print(f'  Hit@5:  {hit5 / n_eval:.4f}  ({hit5}/{n_eval})', flush=True)
    print(f'  AUC:    {auc:.4f}', flush=True)
    print(f'  Disc:   {np.mean(all_top) / max(abs(np.mean(all_bot)), 1e-8):.3f}', flush=True)
    print(f'  top={np.mean(all_top):.4f}  bot={np.mean(all_bot):.4f}', flush=True)
    print(f'{"="*50}', flush=True)

if __name__ == '__main__':
    main()
