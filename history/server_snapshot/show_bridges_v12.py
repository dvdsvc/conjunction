import json, sys, os
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, '/root/autodl-tmp/openalex_data/pencil_data/temporal_l3')
os.chdir('/root/autodl-tmp/openalex_data/pencil_data/temporal_l3')

from train_temporal_v12 import (
    PencilV9, AnchorDatasetV9, anchor_collate_v9,
    load_scibert_remapped, load_precomputed_v12,
    TEMP_DIR, CONCEPT_IDS_FILE, N_CANDIDATES,
)

def main():
    device = torch.device('cpu')
    torch.set_num_threads(8)

    name_map = json.load(open('/root/autodl-tmp/openalex_data/pencil_data/concept_names.json'))
    with open(CONCEPT_IDS_FILE) as f:
        concept_ids = json.load(f)
    id2name = {i: name_map.get(cid, cid) for i, cid in enumerate(concept_ids)}

    n_concepts = len(concept_ids)
    scibert_table = load_scibert_remapped(concept_ids, n_concepts)
    adj_val = sp.load_npz(f'{TEMP_DIR}/graph_v2_2015.npz').tocsr()
    concept_feats_val = np.load(f'{TEMP_DIR}/concept_feats_v9_2015.npy')

    (src_val, pf_val, mn_val,
     deltas_val, ul_val, vl_val) = load_precomputed_v12(0, 'val')

    val_ds = AnchorDatasetV9(src_val, concept_feats_val, pf_val,
                             mn_val, deltas_val, ul_val, vl_val,
                             adj_val, N_CANDIDATES, rng_seed=99)

    model = PencilV9(scibert_table).to(device)
    model.load_state_dict(torch.load(f'{TEMP_DIR}/model_fold0_v12/best.pt',
                                      map_location='cpu', weights_only=True))
    model.eval()

    rng = np.random.RandomState(42)
    indices = rng.choice(len(val_ds), 20, replace=False)

    for idx in indices:
        feats, nids, dcs, pf, mn, edge_idx, edge_val, ul, vl = val_ds[idx]
        # Add batch dim
        feats = feats.unsqueeze(0)
        nids = nids.unsqueeze(0)
        dcs = dcs.unsqueeze(0)
        pf = pf.unsqueeze(0)
        mn = mn.unsqueeze(0)
        edge_idx = edge_idx.unsqueeze(0)  # [1, 2, E]
        edge_val = edge_val.unsqueeze(0)  # [1, E]
        ul = ul.unsqueeze(0)
        vl = vl.unsqueeze(0)

        with torch.no_grad():
            scores = model(feats, nids, pf, mn, edge_idx, edge_val, ul, vl)

        s = scores[0].numpy()
        d = dcs[0].numpy()
        nids_np = nids[0].numpy()

        anchor_local = int(nids_np[0, 0])
        anchor_name = id2name.get(anchor_local, f'id={anchor_local}')

        cand_order = np.argsort(-s)[1:]
        n_pos_total = int((d[1:] >= 2).sum())

        print(f'\n{"="*60}')
        print(f'Anchor: {anchor_name} ({n_pos_total} bridges in {N_CANDIDATES} cands)')
        print(f'{"-"*60}')
        for rank, ci in enumerate(cand_order[:5]):
            cid = int(nids_np[ci, 1])
            cname = id2name.get(cid, f'id={cid}')
            dc = int(d[ci])
            label = '*** BRIDGE ***' if dc >= 2 else f'neg(dc={dc})'
            print(f'  #{rank+1}: {cname:45s} score={s[ci]:+.3f}  [{label}]')

if __name__ == '__main__':
    main()
