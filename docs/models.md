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

## Graph computation inside PencilEncoder

Graph structure enters this branch in two places. First, each local node has a 129-wide structural vector: 64 local-position fields, 64 adjacency fields and one normalized log-propagation score. `struct_proj` projects that vector; `scibert_proj` projects the supplied semantic vector. Their concatenation is combined with candidate-position information to produce node tokens. This node-level structure is separate from the 21 scalar pair features passed to PairMLP.

Second, `PencilEncoder.forward` extracts the adjacency fields again and uses them explicitly in every encoder layer. The order is attention with a residual update, local graph-message aggregation with a learned transform and residual update, then the feed-forward residual update. Four layers are followed by masked mean pooling. The retained operations are:

```python
adj = inp[:, :, :, MAX_NODES:MAX_NODES + N]
adj_norm = adj / adj.sum(dim=3, keepdim=True).clamp(min=1)
# After the attention residual update, within every layer:
message_transform = layer['mp']
h_mp = message_transform(torch.einsum('bgnh,bgnm->bgmh', x_cand, adj_norm))
x_cand = x_cand + h_mp
```

For the stored adjacency tensor, the contraction computes `M[m,h] = sum_n H[n,h] * adj_norm[n,m]`, equivalently `adj_norm.T @ H` within each local block. It uses row-normalized stored adjacency; it should not be silently rewritten as a symmetric GCN normalization or `adj_norm @ H`. Each message operation follows direct edges in the selected graph. Attention can mix information across more nodes according to its own mask, so the full encoder is not restricted to one-hop attention or a simple four-hop receptive field. The attention mask controls source/candidate blocks and padding, not just graph-edge membership.

Neighborhood selection and edge aggregation are also distinct. [Temporal v5](../research/temporal/train_temporal_v5.py) selects the center and degree-ranked direct neighbors, and interleaves graph messages with attention. In [v15](../research/temporal/train_temporal_v15.py), `build_concept_feature_table` selects nodes by PPR-like score and `fill_adjacency_features` encodes direct links among those selected nodes. Thus the later local node set is not necessarily limited to the center's one-hop neighbors, while graph-message steps still use direct local edges.

This alternating attention/message-passing design follows the PENCIL reference recorded in [third-party notices](../THIRD_PARTY_NOTICES.md); the scientific input representations, neighborhood preparation, candidate organization and fusion are documented here as local task adaptation. A separate [citation v5 model](../research/citation/pencil_v5_model.py) instead pools graph-message-passing subgraphs before a candidate-level Transformer, so that architecture should not be conflated with the temporal interleaved encoder. The parallel v15 `SubgraphGNN` is a third computation path over a merged endpoint graph, not the graph-message module inside `PencilEncoder`.

## Retained v15 repair variant

An existing follow-up pair of scripts repairs source-group indexing, preservation of both subgraph endpoints, -1 padding for a valid concept zero, selected-neighborhood ID/structure alignment, candidate sparse-edge group offsets and deterministic validation sampling. It excludes the source group from the ranking loss and reported candidate metrics.

Its loss uses current score ranks to weight pairwise RankNet softplus terms by the swap's absolute NDCG change. This differs from the original implementation's sigmoid loss and position-dependent weights. The showcase preserves both versions and uses the retained repair for the CPU example; it does not claim the repair produced verified full-data improvements.

## FoS extension

FoS uses the same fusion idea with 35 pair features and a smaller default candidate list. Its preprocessing reads supplied benchmark embeddings and adds topology features, including core/Fiedler statistics. It retains its own original training and loss implementation, not the temporal repair automatically. FoS preprocessing imports CuPy for CUDA operations; it is optional and not part of the CPU example.
