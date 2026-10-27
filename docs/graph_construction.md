# Tasks and graph construction

[Overview](../README.md) · [Features](features.md) · [Models](models.md) · [Training/evaluation](training_evaluation.md) · [Results](results.md) · [Operation](reproducibility.md) · [Provenance](provenance.md)

## Task families

| Task | Nodes/relations | Learning or evaluation target | Source |
| --- | --- | --- | --- |
| Citation completion / retrieval | Papers and directed references | Missing or target cited paper among candidates | [process_data](../research/citation/process_data.py), [Phoenix training](../research/citation/train_phoenix.py) |
| Citation bridge retrieval | A→B→C paths, concept overlap filtering | Retrieve a sampled indirect target C from A | [bridge builder](../research/citation/build_real_bridges_v3.py), [IDF evaluation](../research/citation/eval_real_bridges.py) |
| Early temporal concept ranking | Undirected concept co-occurrence snapshots | Historical-to-future distance shortening | [early builder](../research/temporal/build_temporal_data.py), [v5 training](../research/temporal/train_temporal_v5.py) |
| v15 temporal ranking | Historical concept neighborhood candidates | Grades of future weighted proximity | [v15 precompute](../research/temporal/precompute_v15.py), [v15 training](../research/temporal/train_temporal_v15.py) |
| FoS extension | Art_Business benchmark nodes/edges | Same style of graded temporal ranking with topology features | [FoS builder](../research/fos/build_fos_data.py) |

These targets are not interchangeable with prediction of a first-ever future edge.

## Paper preprocessing and citation bridges

`process_data.py` builds concept vocabulary, reference lists, sparse adjacency and paper features. Its defaults split source papers by year (train end 2018, validation end 2021). `build_real_bridges_v3.py` expands references twice and filters concept Jaccard overlap. The default `max_refs=50` truncates references; the code excludes only the truncated direct-reference set from candidates. The implementation therefore does not guarantee exclusion of every direct citation. The historical "real bridge" dataset name is retained as an artifact label, with this boundary explicit.

## Concept snapshot variants

| Builder | Filtering and graph semantics | Saved naming |
| --- | --- | --- |
| `build_temporal_data.py` | Score ≥0.5, level ≤2; cumulative co-occurrence by cutoff | `graph_YEAR.npz` |
| `build_temporal_v2.py` | Score ≥0.7, level ≤3; cumulative streamed construction and propagation | `graph_v2_YEAR.npz` in its configured temporal_v2 directory |
| `build_temporal_l3.py` | Level exactly 3; edges first seen in each cutoff interval, not cumulative snapshots | `graph_YEAR.npz` |
| `build_ext_years_l3.py` | Level exactly 3, score ≥0.7; cumulative graph for supplied years | `graph_v2_YEAR.npz` in temporal_l3 |
| `build_fos_data.py` | Node order from supplied benchmark feature table; undirected cumulative edges by timestamp | `graph_fos_YEAR.npz` |

The original extension builder defaults to years 2005 and 2015. The presentation dispatcher can override year cutoffs as configuration. The v15 loaders expect cumulative `graph_v2_YEAR.npz` and matching `ppr_v2_YEAR.npy`; the first-co-occurrence builder is not silently substituted for them. Match concept ordering and graph semantics before a full run.

## Neighborhoods, distances and supervision

Early graph distances use SciPy `shortest_path(..., method='D')` despite BFS labels in comments. Early candidates are historical reachable pairs at distance at least three, labeled by future distance change. The PPR-oriented builder uses propagation-change supervision as a separate variant.

v15 forms the union of historical one- and two-hop neighbors, then filters by positive historical Adamic–Adar. It reads future reachability while forming candidates. Future weighted distances are produced by symmetrizing/binarizing adjacency, assigning edge cost `log((deg_u+deg_v)/2 + 1)`, running Dijkstra, and subtracting one from the resulting path length. The function name "first hop free" does not literally remove the first edge's weight. Future distances become grades: ≤1→4, (1,2]→3, (2,5]→2, (5,10]→1, larger→0. Callers filter negative unreachable values before grading.

The temporal repair scripts preserve that task definition. They repair source-group indices, endpoint retention and padding; they do not turn the candidate protocol into a future-blind novel-edge benchmark. [Protocol and interpretation](training_evaluation.md).

## Input and output contracts

Paper JSONL records supply `year` and concept objects (`id`, `score`, `level`); citation preparation additionally uses references. Concept ID arrays define row ordering. Precompute output comprises pair features, merged node IDs, local endpoint indices, labels under the legacy `deltas` name, raw future distances and source-group JSON. All arrays and group indices must be regenerated together for the retained fix variant.
