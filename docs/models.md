# Models and core mechanisms

[Overview](../README.md) · [Graphs](graph_construction.md) · [Features](features.md) · [Models](models.md) · [Protocol](training_evaluation.md) · [Results](results.md) · [Operation](reproducibility.md) · [Provenance](provenance.md)


## Recommendation-to-scientific-task adaptation

The paper model is a PyTorch `PhoenixCitationTransformer` with scalar, concept multi-hot, supplied GNN and title-vector projections. Its attention mask lets candidates attend to history and themselves, while history excludes candidates. It adapts recommendation-style input roles to papers and references. The CLI defaults to ten layers, while the class default is four; RoPE helpers exist but blocks are constructed with `use_rope=False`. Treat the source model/configuration as the evidence rather than the historical README's generic architecture label. [Paper model](../research/citation/phoenix_model.py) · [Training](../research/citation/train_phoenix.py).

The scientific subgraph models are local PyTorch implementations inspired by PENCIL, not calls that simply run the upstream PENCIL repository. They change scientific representations, neighborhood preparation, candidate organization, scoring and training targets. Upstream ownership is documented in [third-party notices](../THIRD_PARTY_NOTICES.md).

## Temporal v5: semantic/structural subgraph ranking

`PencilV5` combines projected pretrained concept vectors, structure, distance embeddings, candidate-position embeddings, Transformer attention and graph-message updates. A sampled anchor and candidate blocks form the input. The temporal implementation optimizes pairwise hinge ranking on distance change. The static citation v5 script optimizes BPR despite the historical index saying InfoNCE. These are separate training definitions.

The local candidate mask blocks direct candidate-to-candidate attention, but source tokens can attend to all candidates in a group. After multiple layers, information can travel indirectly through source tokens. This is not equivalent to strict independent scoring of each candidate or an exact transplant of the original recommendation mask.

## Temporal v15: three representation branches

| Branch | Input and computation | Output width |
| --- | --- | --- |
| PairMLP | 21 structural/proximity/gravity features, two-layer MLP | 64 |
| PencilEncoder | 768-d scientific vectors + 129-d local structure, projected and combined; four Transformer/message-passing layers | 256 |
| SubgraphGNN | Merged pair graph, semantic vectors + log-degree, three residual message-passing layers; readout of endpoint u, endpoint v and mean pool | 128 |
| Fusion/head | Concatenate 448 dimensions, fusion MLP to 256, scalar score head | 1 score per group |

Candidate encoding is processed in groups of twenty with the source block. The architecture, padding, sparse batching and feature-node alignment are concrete implementation work. The encoder table is a registered buffer rather than an end-to-end SciBERT encoder. [Original v15](../research/temporal/train_temporal_v15.py) · [Retained repair variant](../research/temporal/fixes/train_temporal_v15_fix.py).

## Retained v15 repair variant

An existing follow-up pair of scripts repairs source-group indexing, preservation of both subgraph endpoints, -1 padding for a valid concept zero, selected-neighborhood ID/structure alignment, candidate sparse-edge group offsets and deterministic validation sampling. It excludes the source group from the ranking loss and reported candidate metrics.

Its loss uses current score ranks to weight pairwise RankNet softplus terms by the swap's absolute NDCG change. This differs from the original implementation's sigmoid loss and position-dependent weights. The showcase preserves both versions and uses the retained repair for the CPU example; it does not claim the repair produced verified full-data improvements.

## FoS extension

FoS uses the same fusion idea with 35 pair features and a smaller default candidate list. Its preprocessing reads supplied benchmark embeddings and adds topology features, including core/Fiedler statistics. It retains its own original training and loss implementation, not the temporal repair automatically. FoS preprocessing imports CuPy for CUDA operations; it is optional and not part of the CPU example.
