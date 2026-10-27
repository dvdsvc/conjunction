# Training and evaluation protocol

[Overview](../README.md) · [Graphs](graph_construction.md) · [Features](features.md) · [Models](models.md) · [Protocol](training_evaluation.md) · [Results](results.md) · [Operation](reproducibility.md) · [Provenance](provenance.md)


## Training units and objectives

| Variant | Unit/target | Optimization |
| --- | --- | --- |
| Phoenix paper completion | Masked/target cited paper among candidates | Binary cross-entropy logits |
| Static subgraph v5 | Positive bridge and sampled negatives | BPR in the current source |
| Early temporal v5 | Candidates with different distance changes | Pairwise hinge ranking |
| Original temporal v15 | Future weighted-distance relevance grades 0–4 | Original sigmoid/position-weighted function named LambdaRank |
| Retained temporal v15 fix | Same grades, excluding the source group | Prediction-rank-weighted NDCG/RankNet softplus |
| FoS v15 | Graded FoS relationships and 35 features | Its own original v15-style loss |

Original v15 training defaults: batch 8, 100 sampled candidates per anchor, 30 epochs, AdamW at 3e-4 with weight decay 1e-2, cosine annealing and patience 5. Smaller candidate pools sample with replacement, so repeated candidates can occur. FoS has its own defaults (including 20 candidates). No full-data training is executed during this presentation pass.

## Historical windows and checkpoint selection

The rolling temporal script combines folds 0–3 (2005→2012, 2012→2015, 2015→2017, 2017→2021) for training and uses fold 4 (2021→2024) for validation/model selection. It saves `best.pt` by mean validation NDCG and `last.pt` per epoch. This is a multi-window training/validation implementation, not five independent out-of-sample tests. Array filenames still use `train_*` even for the future validation fold.

The standalone `eval_v15_fast.py` loads `precomputed_v14/fold4` and can fall back from a v15 checkpoint to v14. It is preserved as a legacy evaluator; it should not be described as an automatically matched evaluation for repaired v15 arrays. Prefer the matching repair trainer's `--eval` path after regenerating its artifacts.

## Ranking definitions

NDCG uses gain `2^relevance - 1` and log2 rank discount. v15 defines positives as grade ≥4. Precision@K is hits divided by the actual number retrieved; Recall@K divides by positives in that sampled list, not all positive relationships in the global graph. MRR uses the first positive rank. NDCG includes all evaluated lists; the temporal P/R/MRR aggregation skips lists with no positives. These denominator differences matter when comparing experiments.

The citation IDF evaluator instead samples post-2021 source-target bridge records and checks whether that particular target appears in the retrieved list. Its reported Recall@K is a target-hit fraction over those records, not exhaustive multi-positive recall per unique source. IDF and one-hop concept features are built from the loaded data; source-year filtering alone does not establish availability of every feature at an earlier cutoff.

## Candidate and split validity

Original and repaired temporal v15 candidate construction uses historical one-/two-hop neighbors plus future reachability filtering. Historical direct neighbors are included. Scores therefore concern graded future proximity within that selected candidate population. They do not by themselves validate future-blind discovery of first-ever new edges. The citation builder's truncated direct-reference exclusion is documented in [graphs](graph_construction.md).

Do not merge recall from a global paper-retrieval universe, sampled concept lists, or different negative mining protocols. Record data ordering, split years, candidate IDs, sampling seed, label threshold, missing lists and checkpoint identity alongside metrics.

## Checks in this showcase

Added numerical contracts test relevance boundaries, Jaccard column semantics, endpoint retention, joint permutation invariance of the repaired loss, zero-gradient behavior for equal labels and NDCG on known orderings. The CPU example runs the retained architecture at its original widths, with two sampled candidates and three optimizer steps on a small synthetic graph. Its metrics are computed on the same training list and are not held-out performance. [Execution record](results.md).
