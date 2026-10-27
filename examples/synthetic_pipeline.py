"""CPU graph -> features -> retained fusion model -> optimization -> ranking demo."""
from __future__ import annotations
import contextlib
import io
import json
import tempfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from research.temporal import build_ext_years_l3 as graphs
from research.temporal.fixes import precompute_v15_fix as pre
from research.temporal.fixes import train_temporal_v15_fix as train


def run_demo(steps=3):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    torch.sparse.check_sparse_tensor_invariants.enable()
    n = 70
    concept_ids = [f"C{i}" for i in range(n)]
    papers = [{"year": 2010, "concepts": [{"id": f"C{i}", "score": .9, "level": 3}, {"id": f"C{(i+1)%n}", "score": .9, "level": 3}]} for i in range(n)]
    papers += [{"year": 2020, "concepts": [{"id": f"C{i}", "score": .9, "level": 3}, {"id": f"C{(i+2)%n}", "score": .9, "level": 3}]} for i in range(0,n,3)]
    log = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="lbd-synthetic-") as tmp, contextlib.redirect_stdout(log):
        temp = Path(tmp)
        geology = temp / 'geology.jsonl'; climate = temp / 'climate.jsonl'
        geology.write_text(''.join(json.dumps(p)+'\n' for p in papers));climate.write_text('')
        old = (graphs.TEMP_DIR, graphs.PAPERS_GEOL, graphs.PAPERS_CLIM)
        try:
            graphs.TEMP_DIR = str(temp);graphs.PAPERS_GEOL = str(geology);graphs.PAPERS_CLIM = str(climate)
            mapping = {cid:i for i,cid in enumerate(concept_ids)}
            graphs.build_graph(2012,mapping,n);graphs.build_graph(2024,mapping,n)
            a = sp.load_npz(temp/'graph_v2_2012.npz').tocsr();future = sp.load_npz(temp/'graph_v2_2024.npz').tocsr()
        finally:
            graphs.TEMP_DIR,graphs.PAPERS_GEOL,graphs.PAPERS_CLIM = old
        ppr = graphs.compute_ppr(a)
        distance = pre.compute_weighted_bfs_first_hop_free(future,str(temp/'distance.npy'))
        cc = pre.fast_clustering_coefficients(a)
        degree = np.asarray(a.sum(axis=1)).ravel()
        assert pre.process_split('train',str(temp),a,distance,ppr,degree,cc,np.random.RandomState(7))
        pf = np.load(temp/'train_pair_feats.npy');merged = np.load(temp/'train_merged_nodes.npy')
        labels = np.load(temp/'train_deltas.npy');ul = np.load(temp/'train_u_local.npy');vl = np.load(temp/'train_v_local.npy')
        grouped = {int(k):v for k,v in json.loads((temp/'train_pairs_by_src.json').read_text()).items()}
        features,ids = train.build_concept_feature_table(ppr,n,torch.device('cpu'))
        train.fill_adjacency_features(features,ids,a)
        ds = train.AnchorDatasetV15(grouped,features,ids,pf,merged,labels,ul,vl,a,n_candidates=2,rng_seed=7,deterministic=True)
        batch = train.anchor_collate_v15([ds[0]])
        inp,nids,rel,pair,mn,ei,ev,u,v = batch
        # Random vectors only exercise the 768-d semantic input contract; no SciBERT encoding is claimed.
        semantic = torch.randn(n,train.SCIBERT_DIM)
        model = train.PencilV15(semantic)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        tracked = next(model.parameters()).detach().clone()
        losses=[];gradient_norms=[]
        for _ in range(steps):
            model.train();optimizer.zero_grad(set_to_none=True)
            scores = model(inp,nids,pair,mn,ei,ev,u,v)
            loss = train.lambdarank_loss(scores[:,1:],rel[:,1:])
            assert torch.isfinite(loss)
            loss.backward()
            norm = sum(float(p.grad.detach().norm()) for p in model.parameters() if p.grad is not None)
            assert np.isfinite(norm) and norm>0
            optimizer.step();losses.append(float(loss.detach()));gradient_norms.append(norm)
        model.eval()
        with torch.no_grad():scores=model(inp,nids,pair,mn,ei,ev,u,v)[:,1:].numpy()[0]
        targets=rel[:,1:].numpy()[0];ranked=np.argsort(-scores);positives=targets>=4;hits=int(positives[ranked[:1]].sum())
        rr=next((1/(i+1) for i,k in enumerate(ranked) if positives[k]),0.0)
        report={
            'data_kind':'synthetic graph and random 768-d vectors; pipeline check, not a research benchmark',
            'source_variant':'retained v15_fix scripts, not newly rewritten model code',
            'device':'cpu','concepts':n,'historical_edges':int(a.nnz//2),'future_edges':int(future.nnz//2),
            'precomputed_pairs':len(labels),'anchors':len(grouped),'pair_feature_dim':pf.shape[1],
            'structure_shape':list(features.shape),'semantic_input':'random; not downloaded or encoded SciBERT',
            'model_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'optimizer_steps':steps,'loss_trace':losses,'finite_nonzero_gradients':all(np.isfinite(gradient_norms)),
            'parameter_changed':not torch.equal(tracked,next(model.parameters()).detach()),
            'training_candidate_labels':targets.tolist(),'training_candidate_scores':scores.tolist(),
            'same_training_list_metrics':{'NDCG@2':float(train.compute_ndcg_at_k(scores,targets,2)),
                'Precision@1':hits,'Recall@1':hits/max(int(positives.sum()),1),'MRR':rr},
            'original_functions_used':['build_graph','compute_ppr','process_split','build_concept_feature_table','AnchorDatasetV15','PencilV15','lambdarank_loss','compute_ndcg_at_k'],
        }
        assert report['parameter_changed'] and report['pair_feature_dim']==21
    return report


def main():
    print(json.dumps(run_demo(),indent=2))

if __name__=='__main__':
    main()
