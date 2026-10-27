"""v13 eval: check if top-ranked candidates are direct neighbors (dist_t2 ≈ 0)."""
import time, json, sys, os
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, '/root/autodl-tmp/openalex_data/pencil_data/temporal_l3')
os.chdir('/root/autodl-tmp/openalex_data/pencil_data/temporal_l3')

from train_temporal_v13 import (
    PencilV9, AnchorDatasetV9, anchor_collate_v9,
    load_scibert_remapped, load_precomputed_v13,
    TEMP_DIR, CONCEPT_IDS_FILE, N_CANDIDATES,
)

N_EVAL = 7800
BATCH = 16

def main():
    device = torch.device('cuda')
    print(f'Device: {device}', flush=True)

    with open(CONCEPT_IDS_FILE) as f:
        concept_ids = json.load(f)
    n_concepts = len(concept_ids)
    scibert_table = load_scibert_remapped(concept_ids, n_concepts).to(device)

    adj_val = sp.load_npz(f'{TEMP_DIR}/graph_v2_2015.npz').tocsr()
    concept_feats_val = np.load(f'{TEMP_DIR}/concept_feats_v9_2015.npy')

    (src_val, pf_val, mn_val,
     deltas_val, ul_val, vl_val) = load_precomputed_v13(0, 'val')

    val_ds = AnchorDatasetV9(src_val, concept_feats_val, pf_val,
                             mn_val, deltas_val, ul_val, vl_val,
                             adj_val, N_CANDIDATES, rng_seed=99)

    model = PencilV9(scibert_table).to(device)
    model_file = f'{TEMP_DIR}/model_fold0_v13/best.pt'
    if not os.path.exists(model_file):
        for f in sorted(os.listdir(f'{TEMP_DIR}/model_fold0_v13')):
            if f.endswith('.pt'):
                model_file = f'{TEMP_DIR}/model_fold0_v13/{f}'
                break
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()
    print(f'  Loaded {model_file}', flush=True)

    n_eval = min(N_EVAL, len(val_ds))
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_ds, range(n_eval)),
        batch_size=BATCH, shuffle=False, num_workers=2,
        collate_fn=anchor_collate_v9, pin_memory=True)

    print(f'  Evaluating {n_eval} anchors, batch_size={BATCH}...', flush=True)

    all_scores = []
    all_dists = []
    t0 = time.time()

    for batch in val_loader:
        feats, nids, dcs, pf, mn, edge_idx, edge_val, ul, vl = [
            x.to(device, non_blocking=True) for x in batch]
        with torch.no_grad():
            scores = model(feats, nids, pf, mn, edge_idx, edge_val, ul, vl)
        all_scores.append(scores.float().cpu().numpy())
        all_dists.append(dcs.cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    all_dists = np.concatenate(all_dists, axis=0)
    print(f'  Inference: {time.time()-t0:.0f}s', flush=True)

    # Exclude self (index 0), rank remaining candidates
    scores_cand = all_scores[:, 1:]
    dists_cand = all_dists[:, 1:]

    ranked_idx = np.argsort(-scores_cand, axis=1)
    ranked_dists = np.take_along_axis(dists_cand, ranked_idx, axis=1)

    # Direct neighbor threshold: dist_t2 < 1 (first-hop-free, low-degree neighbors)
    # More permissive: dist_t2 < 3 (includes most direct neighbors)
    for threshold in [0.5, 1.0, 2.0, 3.0]:
        is_close = ranked_dists < threshold
        has_close = is_close.any(axis=1)
        n_with_close = int(has_close.sum())

        print(f'\n--- Threshold dist_t2 < {threshold} ---', flush=True)
        print(f'  Anchors with >= 1 close candidate: {n_with_close}/{n_eval} ({n_with_close/n_eval*100:.1f}%)', flush=True)

        for K in [1, 3, 5, 10, 20]:
            if K > ranked_dists.shape[1]:
                continue
            hits = is_close[:, :K].any(axis=1).sum()
            prec = is_close[:, :K].sum() / (n_eval * K)
            # Recall: of anchors that have close candidates, how many are found in top-K
            if n_with_close > 0:
                recall = hits / n_with_close
            else:
                recall = 0
            print(f'  Top-{K:2d}: Hit={int(hits):4d}/{n_eval} ({int(hits)/n_eval*100:5.1f}%)  '
                  f'Prec={prec:.4f}  Recall={recall:.4f}', flush=True)

    # Also show the actual dist_t2 of top-1, top-3, top-5
    print(f'\n--- Average dist_t2 of top-K candidates ---', flush=True)
    for K in [1, 3, 5, 10, 20]:
        if K > ranked_dists.shape[1]:
            continue
        avg_d = ranked_dists[:, :K].mean()
        print(f'  Top-{K:2d}: avg dist_t2 = {avg_d:.2f}', flush=True)

    # Median dist_t2 for comparison
    median_d = np.median(dists_cand[dists_cand > 0])
    print(f'\n  Median dist_t2 (all reachable): {median_d:.2f}', flush=True)

if __name__ == '__main__':
    main()
