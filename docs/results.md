# Experimental records and validation

[Overview](../README.md) · [Graphs](graph_construction.md) · [Features](features.md) · [Models](models.md) · [Protocol](training_evaluation.md) · [Operation](reproducibility.md) · [Provenance](provenance.md)

## What was actually run for this snapshot

Local environment on 2026-10-26: Python 3.12.13, NumPy 2.5.4, SciPy 1.18.1, PyTorch 2.14.1. Dependency installation accessed the package index; execution then used a locked offline environment.

| Check | Observed result | Meaning |
| --- | --- | --- |
| Added numerical contracts | 6/6 passed | Feature order/width, grade boundaries, endpoint retention, repaired loss invariants and known NDCG orderings |
| CPU synthetic pipeline | 70 concepts; 70 historical and 94 future undirected edges; 140 precomputed pairs; 21 features | Real original graph/precompute functions execute on small constructed input |
| Model execution | 4,224,065 trainable parameters; forward/backward and 3 optimizer steps; finite nonzero gradients; parameters changed | The retained fusion architecture and optimizer path execute, using random 768-d inputs |
| Ranking calculation | NDCG, Precision/Recall and MRR produced on the same two-candidate training list | Pipeline behavior, not validation/test accuracy |
| Dispatch wrapper | Five tasks listed; dry run does not execute; missing-root guard rejects execution | Path portability without automatic server access |

Evidence: [numerical output](validation/numerical-tests.txt), [synthetic pipeline JSON](../examples/outputs/synthetic_pipeline.json), [runtime/command summary](validation/runtime_summary.json). The demo's short loss trace is a synthetic same-list optimization observation. It is not a learning curve from the original research dataset, a verified model improvement, or generalization evidence.

## Local five-window preparation artifacts

The retained server archive includes metadata and source-group candidate index JSON for v15. The following values were counted from the actual local JSON, rather than copied from a README:

| Fold | Window | Saved index anchors | Saved index pair entries |
| --- | --- | --- | --- |
| 0 | 2005→2012 | 3,430 | 844,408 |
| 1 | 2012→2015 | 4,101 | 1,263,445 |
| 2 | 2015→2017 | 4,352 | 1,463,494 |
| 3 | 2017→2021 | 4,481 | 1,582,244 |
| 4 | 2021→2024 | 4,734 | 1,831,632 |

[Archived summary with source hashes](validation/archived_precompute_summary.json) preserves metadata, counts and index extent. These artifacts establish prepared candidate lists/configuration; full feature arrays, graphs, predictions and checkpoints are not bundled. The original grouping/index implementation has consistency issues addressed by the separately retained fix. Index extent and mapped counts differ, so this archive is not presented as a verified training corpus.

## Historical reported results

| Record | Reported value | Current evidence status |
| --- | --- | --- |
| Citation bridge IDF baseline | 464/5,000 target hits at K=5,000 = 9.28% (rounded to 9.3%) | Present in sanitized historical README; arithmetic verified; raw target/rank arrays not available for a full rerun here |
| Temporal v5 index | disc=3.15, top-1=67.8%, Spearman=0.36 | Historical index only; not a newly verified benchmark |
| Old citation retrieval scores | Historical README distinguishes them from the corrected bridge task | Retained for task history, not transferred to temporal or FoS results |

Sources: [citation history](../provenance/citation_history.sanitized.md), [temporal history](../provenance/temporal_history.sanitized.md). A successful predictor or stable improvement is not established by these records. The repository's concrete value is its graph/representation/model/optimization implementation and the ability to inspect its experimental design.

## Protocol audit and next reproducible comparison

The task/candidate distinctions, future-reachability selection, direct-edge inclusion, original ranking weights, graph/group alignment and mixed evaluator paths are documented together in [training/evaluation](training_evaluation.md). The retained fixes are tested as implementation repairs, not claimed retrained gains.

A next full experiment should freeze graph and concept ordering, define candidates without future filtering if novel-link discovery is the target, persist per-anchor candidate IDs/labels/scores and checkpoint identity, and compare structural baselines with the three fusion branches under the same split and candidate universe. That is a proposed next experiment, not completed work. No remote training server, pretrained model download, full-data GPU job or hosted CI was invoked in this polish pass.
