"""Fast GPU eval — vectorized metrics, DataLoader workers, larger batch."""
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

N_EVAL = 7800  # all anchors
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
    print('  Loaded cached concept_feats 2015', flush=True)

    (src_val, pf_val, mn_val,
     deltas_val, ul_val, vl_val) = load_precomputed_v13(0, 'val')

    val_ds = AnchorDatasetV9(src_val, concept_feats_val, pf_val,
                             mn_val, deltas_val, ul_val, vl_val,
                             adj_val, N_CANDIDATES, rng_seed=99)

    model = PencilV9(scibert_table).to(device)
    model_file = f'{TEMP_DIR}/model_fold0_v13/best.pt'
    if not os.path.exists(model_file):
        # Fall back to any available checkpoint
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
        collate_fn=anchor_collate_v9,
        pin_memory=True)

    print(f'  Evaluating {n_eval} anchors, batch_size={BATCH}...', flush=True)

    all_scores = []
    all_deltas = []
    t0 = time.time()

    for batch in val_loader:
        feats, nids, dcs, pf, mn, edge_idx, edge_val, ul, vl = [
            x.to(device, non_blocking=True) for x in batch]

        with torch.no_grad():
            scores = model(feats, nids, pf, mn, edge_idx, edge_val, ul, vl)

        all_scores.append(scores.float().cpu().numpy())
        all_deltas.append(dcs.cpu().numpy())

        done = sum(s.shape[0] for s in all_scores)
        if done % (BATCH * 10) < BATCH:
            elapsed = time.time() - t0
            eta = elapsed / done * (n_eval - done)
            print(f'    {done}/{n_eval} ({elapsed:.0f}s, ETA {eta:.0f}s)', flush=True)

    # Vectorized metrics on CPU
    all_scores = np.concatenate(all_scores, axis=0)  # [N, G]
    all_deltas = np.concatenate(all_deltas, axis=0)  # [N, G]
    dt = time.time() - t0
    print(f'  Inference done: {dt:.0f}s', flush=True)

    # For each anchor: rank candidates by score (exclude self at idx 0)
    scores_cand = all_scores[:, 1:]  # [N, G-1]
    deltas_cand = all_deltas[:, 1:]

    # Rank by descending score
    ranked_idx = np.argsort(-scores_cand, axis=1)  # [N, G-1]
    ranked_deltas = np.take_along_axis(deltas_cand, ranked_idx, axis=1)

    # Hit@k: does any of top-k have delta >= 2?
    pos_mask = ranked_deltas >= 2  # [N, G-1]
    has_pos = pos_mask.any(axis=1)  # [N]
    n_valid = int(has_pos.sum())

    # MRR: rank of first positive
    first_pos = np.argmax(pos_mask, axis=1)  # [N]
    mrr = float((1.0 / (first_pos[has_pos] + 1)).mean()) if n_valid > 0 else 0

    # Hit@k
    hit1 = int((ranked_deltas[:, 0] >= 2).sum())
    hit3 = int((ranked_deltas[:, :3] >= 2).any(axis=1).sum())
    hit5 = int((ranked_deltas[:, :5] >= 2).any(axis=1).sum())

    # Disc
    K = ranked_deltas.shape[1]
    top_k = max(1, K // 10)
    top_mean = float(ranked_deltas[:, :top_k].mean())
    bot_mean = float(ranked_deltas[:, top_k:].mean())
    disc = top_mean / max(abs(bot_mean), 1e-8)

    # AUC
    from sklearn.metrics import roc_auc_score
    mask = (deltas_cand >= 2) | (deltas_cand == 0)
    auc_scores = scores_cand[mask]
    auc_labels = (deltas_cand[mask] >= 2).astype(int)
    auc = float(roc_auc_score(auc_labels, auc_scores))

    print(f'\n{"="*50}', flush=True)
    print(f'  GPU eval ({n_eval} anchors, {dt:.0f}s)', flush=True)
    print(f'  MRR:    {mrr:.4f}', flush=True)
    print(f'  Hit@1:  {hit1/n_eval:.4f}  ({hit1}/{n_eval})', flush=True)
    print(f'  Hit@3:  {hit3/n_eval:.4f}  ({hit3}/{n_eval})', flush=True)
    print(f'  Hit@5:  {hit5/n_eval:.4f}  ({hit5}/{n_eval})', flush=True)
    print(f'  AUC:    {auc:.4f}', flush=True)
    print(f'  Disc:   {disc:.3f}', flush=True)
    print(f'  top={top_mean:.4f}  bot={bot_mean:.4f}', flush=True)
    print(f'{"="*50}', flush=True)

if __name__ == '__main__':
    main()
