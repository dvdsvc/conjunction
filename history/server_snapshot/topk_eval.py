"""Output top-K candidates per anchor from best.pt for quality inspection."""
import json, sys, os
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, '/root/autodl-tmp/openalex_data/pencil_data/temporal_v2')
os.chdir('/root/autodl-tmp/openalex_data/pencil_data/temporal_v2')

from train_temporal_v9 import (
    PencilV9, AnchorDatasetV9, anchor_collate_v9,
    load_scibert_remapped, load_precomputed_v9,
    TEMP_DIR, CONCEPT_IDS_FILE, N_CANDIDATES,
)

N_EVAL = 200
TOP_K = 10

def main():
    device = torch.device('cpu')
    torch.set_num_threads(8)

    with open(CONCEPT_IDS_FILE) as f:
        concept_ids = json.load(f)
    n_concepts = len(concept_ids)
    concept_id_to_name = {v: k for k, v in concept_ids.items()} if isinstance(concept_ids, dict) else {}
    scibert_table = load_scibert_remapped(concept_ids, n_concepts)

    adj_val = sp.load_npz(f'{TEMP_DIR}/graph_v2_2015.npz').tocsr()
    concept_feats_val = np.load(f'{TEMP_DIR}/concept_feats_v9_2015.npy')

    (src_val, pf_val, mn_val,
     deltas_val, ul_val, vl_val) = load_precomputed_v9(0, 'val')

    val_ds = AnchorDatasetV9(src_val, concept_feats_val, pf_val,
                             mn_val, deltas_val, ul_val, vl_val,
                             adj_val, N_CANDIDATES, rng_seed=99)

    model = PencilV9(scibert_table).to(device)
    model.load_state_dict(torch.load(f'{TEMP_DIR}/model_fold0_v9/best.pt',
                                      map_location='cpu', weights_only=True))
    model.eval()

    n_eval = min(N_EVAL, len(val_ds))
    print(f'Evaluating {n_eval} anchors, outputting top-{TOP_K}...', flush=True)

    # Load concept names if available
    name_file = f'{TEMP_DIR}/concept_names.json'
    if os.path.exists(name_file):
        with open(name_file) as f:
            id2name = json.load(f)
    else:
        id2name = {}

    good, bad, total = 0, 0, 0
    for i in range(n_eval):
        sample = val_ds[i]
        feats, nids, dcs, pf, mn, ma, ul, vl = [x.unsqueeze(0) for x in sample]
        src_id = int(src_val[val_ds.anchor_indices[i]]) if hasattr(val_ds, 'anchor_indices') else i

        with torch.no_grad():
            scores = model(feats, nids, pf, mn, ma, ul, vl)

        s = scores[0].numpy()
        d = dcs[0].numpy()
        nids_np = nids[0].numpy()

        # Skip anchor at idx 0
        cand_order = np.argsort(-s)[1:]
        top_ids = cand_order[:TOP_K]
        top_deltas = d[top_ids]
        top_scores = s[top_ids]

        n_hit = int((top_deltas >= 2).sum())
        total += 1
        if n_hit > 0:
            good += 1
        else:
            bad += 1

        if i < 10:  # print details for first 10
            anchor_name = id2name.get(str(nids_np[0]), f'id={nids_np[0]}')
            print(f'\n--- Anchor {i}: {anchor_name} (src_id={nids_np[0]}) ---')
            print(f'  Total candidates: {len(s)-1}, positives in all: {int((d[1:]>=2).sum())}')
            for rank, ci in enumerate(top_ids):
                cname = id2name.get(str(nids_np[ci]), f'id={nids_np[ci]}')
                dc = int(d[ci])
                label = 'BRIDGE' if dc >= 2 else ('neg' if dc == 0 else f'dc={dc}')
                print(f'  #{rank+1}: {cname}  score={s[ci]:.4f}  [{label}]')

        if (i+1) % 50 == 0:
            print(f'  {i+1}/{n_eval}  hit_rate={good}/{total}', flush=True)

    print(f'\n{"="*50}')
    print(f'  Top-{TOP_K} hit rate: {good}/{total} = {good/max(total,1):.4f}')
    print(f'{"="*50}')

if __name__ == '__main__':
    main()
