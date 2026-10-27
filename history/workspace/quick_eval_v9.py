"""Quick eval: MRR, Hit@k, AUC on first N validation anchors."""
import time, json, sys
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from train_temporal_v9 import (
    PencilV9, AnchorDatasetV9, anchor_collate_v9,
    build_concept_feature_table, fill_adjacency_features,
    load_scibert_remapped, load_precomputed_v9,
    TEMP_DIR, CONCEPT_IDS_FILE, N_CANDIDATES, BATCH_SIZE,
)

N_EVAL = 2000  # number of anchors to evaluate

def main():
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    # Load concept IDs + scibert
    with open(CONCEPT_IDS_FILE) as f:
        concept_ids = json.load(f)
    n_concepts = len(concept_ids)
    scibert_table = load_scibert_remapped(concept_ids, n_concepts)

    # Load val graph + concept features (cached)
    adj_val = sp.load_npz(f'{TEMP_DIR}/graph_v2_2015.npz').tocsr()
    cache_path = f'{TEMP_DIR}/concept_feats_v9_2015.npy'
    concept_feats_val = np.load(cache_path)
    print(f"  Loaded cached concept_feats 2015", flush=True)

    # Load precomputed val data
    print("  Loading precomputed val data...", flush=True)
    (src_val, pf_val, mn_val,
     deltas_val, ul_val, vl_val) = load_precomputed_v9(0, 'val')

    # Build dataset
    val_ds = AnchorDatasetV9(src_val, concept_feats_val, pf_val,
                             mn_val, deltas_val, ul_val, vl_val,
                             adj_val, N_CANDIDATES, rng_seed=99)

    # Load model
    model = PencilV9(scibert_table).to(device)
    model_file = f'{TEMP_DIR}/model_fold0_v9/best.pt'
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()
    print(f"  Loaded {model_file}", flush=True)

    # Evaluate first N_EVAL anchors (direct, no DataLoader)
    n_eval = min(N_EVAL, len(val_ds))
    print(f"\n  Evaluating {n_eval} anchors...", flush=True)

    mrr_sum, hit1, hit3, hit5, n_valid = 0.0, 0, 0, 0, 0
    all_top, all_bot = [], []
    auc_scores, auc_labels = [], []
    t0 = time.time()

    for i in range(n_eval):
        sample = val_ds[i]  # direct __getitem__
        feats, nids, dcs, pf, mn, ma, ul, vl = [x.unsqueeze(0).to(device) for x in sample]

        with torch.no_grad():
            scores = model(feats, nids, pf, mn, ma, ul, vl)

        s = scores[0].cpu().numpy()   # [G]
        d = dcs[0].cpu().numpy()      # [G]

        # Rank candidates (skip anchor at idx 0)
        cand = np.argsort(-s)[1:]
        pos_mask = d[cand] >= 2

        if pos_mask.any():
            first_pos = np.argmax(pos_mask)
            mrr_sum += 1.0 / (first_pos + 1)
            n_valid += 1

        if d[cand[0]] >= 2:
            hit1 += 1
        if (d[cand[:3]] >= 2).any():
            hit3 += 1
        if (d[cand[:5]] >= 2).any():
            hit5 += 1

        # disc components
        K = len(cand)
        top_k = max(1, K // 10)
        all_top.append(d[cand[:top_k]].mean())
        all_bot.append(d[cand[top_k:]].mean())

        # AUC data
        for ci in range(1, len(s)):
            if d[ci] >= 2:
                auc_scores.append(s[ci]); auc_labels.append(1)
            elif d[ci] == 0:
                auc_scores.append(s[ci]); auc_labels.append(0)

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_eval} ({time.time()-t0:.0f}s)", flush=True)

    dt = time.time() - t0

    # Compute AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(auc_labels, auc_scores)

    print(f"\n{'='*50}", flush=True)
    print(f"  Quick eval ({n_eval} anchors, {dt:.0f}s)", flush=True)
    print(f"  MRR:    {mrr_sum / max(n_valid, 1):.4f}", flush=True)
    print(f"  Hit@1:  {hit1 / n_eval:.4f}  ({hit1}/{n_eval})", flush=True)
    print(f"  Hit@3:  {hit3 / n_eval:.4f}  ({hit3}/{n_eval})", flush=True)
    print(f"  Hit@5:  {hit5 / n_eval:.4f}  ({hit5}/{n_eval})", flush=True)
    print(f"  AUC:    {auc:.4f}", flush=True)
    print(f"  Disc:   {np.mean(all_top) / max(abs(np.mean(all_bot)), 1e-8):.3f}", flush=True)
    print(f"  top={np.mean(all_top):.4f}  bot={np.mean(all_bot):.4f}", flush=True)
    print(f"{'='*50}", flush=True)

if __name__ == '__main__':
    main()
