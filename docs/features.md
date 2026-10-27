# Feature construction and tensor contracts

[Overview](../README.md) · [Graphs](graph_construction.md) · [Models](models.md) · [Protocol](training_evaluation.md) · [Results](results.md) · [Operation](reproducibility.md) · [Provenance](provenance.md)

## Semantic and node features

The concept encoder script uses pretrained `allenai/scibert_scivocab_uncased`, encodes concept names up to 32 tokens and stores the CLS vector (768 dimensions). These are pretrained features, not a fine-tuned scientific language model. ID remapping aligns old embeddings with the L3 concept ordering; missing IDs become zero vectors in the retained loader. The CPU demo uses random vectors of the same shape and makes no SciBERT inference claim.

For v15, each concept receives 64 local positions with a 129-wide structural tensor: 64 positional identity fields, 64 local adjacency fields and one normalized log propagation score. The retained fix passes the actual selected neighborhood IDs with that structure. Merged pair graphs use up to 256 nodes, with explicit local endpoint indices. The original padding value 0 conflicts with a valid concept ID; the retained fix uses -1 for merged-node padding.

These node tensors encode an actual selected graph, not only summary statistics. The adjacency fields participate in the initial structural projection and are also extracted for graph-message updates inside every temporal PENCIL Transformer layer. Early temporal v5 selects direct neighbors; v15 selects nodes by PPR-like score and records direct edges among them. Keep this node-level graph representation separate from the 21-column pair-feature table below and from the merged graph consumed by the parallel SubgraphGNN. See [graph computation inside PencilEncoder](models.md#graph-computation-inside-pencilencoder).

## Temporal v15: exact 21-column order

Source: [compute_all_pair_features and gravity features](../research/temporal/precompute_v15.py). Column names below describe the actual expressions, including duplicates and approximations.

| Index | Feature | Definition |
| --- | --- | --- |
| 0 | `ppr_uv` | Propagation-matrix entry P[u,v] |
| 1 | `ppr_vu` | P[v,u] |
| 2 | `common_neighbors` | (A A)[u,v] after diagonal removal |
| 3 | `ppr_row_sum_sum` | sum(P[u,:]) + sum(P[v,:]); variable name wcn |
| 4 | `jaccard` | CN / max(deg_u+deg_v-CN,1) |
| 5 | `adamic_adar` | A diag(1/log(max(deg,2))) A^T |
| 6 | `degree_u` | Degree of u |
| 7 | `degree_v` | Degree of v |
| 8 | `clustering_u` | Local clustering coefficient of u |
| 9 | `clustering_v` | Local clustering coefficient of v |
| 10 | `average_ppr_u` | P row sum / max(deg_u,1) |
| 11 | `average_ppr_v` | P row sum / max(deg_v,1) |
| 12 | `log_degree_u` | log1p(deg_u) |
| 13 | `log_degree_v` | log1p(deg_v) |
| 14 | `jaccard_repeat` | Exact duplicate of column 4 |
| 15 | `pure_common_neighbors` | Common neighbors below nonzero-degree median |
| 16 | `pure_adamic_adar` | Adamic–Adar restricted to that low-degree set |
| 17 | `neighbor_cohesion` | A3-derived ratio used by the original code |
| 18 | `second_order_overlap` | Jaccard-like overlap of binary two-hop rows |
| 19 | `subgraph_connectivity_proxy` | Degree-sum/local-size heuristic in original code |
| 20 | `resistance_proxy` | log1p(direct weight + two-hop product/2); not effective resistance |

The PPR-labeled builder updates `r <- (1-alpha) M r + alpha r`. Its self-retaining propagation differs from the usual iteration with a fixed restart vector. The showcase preserves and names this implementation explicitly instead of silently replacing it.

## FoS: additional 14 topology columns

Source: [topology feature implementation](../research/fos/precompute_fos_v15.py). Total width is 35 = 15 + 6 + 14. A source header says "8 topology"; the actual returned matrix has fourteen added columns. CuPy is used for the two batched common-neighbor/community operations.

| Index | Feature | Definition |
| --- | --- | --- |
| 21 | `kcore_u` | Core number u |
| 22 | `kcore_v` | Core number v |
| 23 | `fiedler_u` | Fiedler vector component u |
| 24 | `fiedler_v` | Fiedler vector component v |
| 25 | `community_edge_density` | Neighbor cross-edge density heuristic |
| 26 | `neighbor_degree_mean_u` | Mean neighboring degree u |
| 27 | `neighbor_degree_variance_u` | Neighbor degree variance u |
| 28 | `neighbor_degree_mean_v` | Mean neighboring degree v |
| 29 | `neighbor_degree_variance_v` | Neighbor degree variance v |
| 30 | `common_neighbor_kcore_mean` | Mean core number of common neighbors |
| 31 | `triangle_closure_u` | Code-derived closure proxy u |
| 32 | `triangle_closure_v` | Code-derived closure proxy v |
| 33 | `binary_two_hop` | Binary A2 reachability |
| 34 | `binary_two_hop_times_adj` | Binary A2 multiplied by adjacency; not unrestricted 3-walk count |

## Citation features and machine-readable schema

The paper model projects five scalar fields (normalized year, log in-degree, log out-degree, cross-domain bridge score and concept-score entropy), concept multi-hot features, 64-dimensional supplied GNN vectors and 384-dimensional supplied title vectors. Title encoding uses a separate MiniLM script; this is distinct from the concept SciBERT pipeline. [Paper preprocessing](../research/citation/process_data.py) · [Phoenix adapter](../research/citation/phoenix_model.py).

[feature_schema.json](feature_schema.json) records dimensions and column order for inspection. The numerical checks verify the temporal width, Jaccard duplicate, relevance boundaries and endpoint retention. The dictionary describes implemented features, not an ablation result establishing their predictive value.
