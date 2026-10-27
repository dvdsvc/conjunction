# Research design, source versions and claim boundaries

[Overview](../README.md) · [中文说明](../README.zh-CN.md) · [Graphs](graph_construction.md) · [Features](features.md) · [Models](models.md) · [Protocol](training_evaluation.md)

## Why the expanded source audit matters

The repository is a curated snapshot, not the full history of the original working directory. The 2026-10-27 README revision therefore inspected original root scripts, the paper-model directory, recursive `server_code` backups and the local source archive in addition to the selected 27 research files. The local server backup includes nested duplicate copies and July-dated source snapshots; it does not establish the current state of a live remote training server.

The central design is learning potential scientific relations from temporal graph structure. SciBERT is an input component in selected neural models. It does not describe the whole graph-construction, pair-feature, distance-supervision or ranking work.

## Research goal versus implemented supervision

| Question | Source-backed status |
|---|---|
| Can indirectly related concepts be prioritized for later literature investigation? | Research motivation; graph/ranking implementations explore it. |
| Is strict first-new-co-occurrence prediction the target of every version? | No. [Early code](../research/temporal/build_temporal_data.py) uses distance change; [v15](../research/temporal/precompute_v15.py) uses future weighted-distance grades and includes old neighbors. |
| Is there a retained first-co-occurrence graph builder? | Yes: [build_temporal_l3](../research/temporal/build_temporal_l3.py) stores newly seen edges by cutoff interval. This is a graph definition, not proof of a complete first-edge predictor. |
| Are topology/hub effects only proposed? | No. Degree-weighted path costs, Adamic–Adar, lower-degree-neighbor features and subgraph construction have code implementations. Their predictive benefit remains to be evaluated. |
| Is the system purely semantic or purely topological? | Neither label describes every branch. [Pair features](../research/temporal/precompute_v15.py) are structural; [v15](../research/temporal/train_temporal_v15.py) also uses precomputed semantic vectors in its subgraph branches. |

## Weighting and PMI reconciliation

The identified v11–v15 family constructs edge cost as `log((degree(u)+degree(v))/2+1)` and uses Dijkstra. The selected [v15 implementation](../research/temporal/precompute_v15.py) makes the computation inspectable. Probability-based PMI instead requires co-occurrence/marginal frequency information. The available degree-cost function must not be renamed PMI simply because both involve logarithms or are intended to reduce hub effects.

The owner recalls an implemented PMI-weighted version. Its exact source/call path has not yet been located in the inspected local backups. This remains **source version unresolved**, rather than a conclusion that such work never existed or an unsupported statement that the shown v15 function computes PMI. Historical index notes place PMI under a next-direction heading, but an old note cannot settle the state of a later or unarchived branch.

## Changes from the supplied Catalyst reference

| Reference statement | Treatment in this README |
|---|---|
| “Topology-driven hypothesis generation” | Retain the scientific motivation and working name; describe ranked associations as candidates for verification. |
| “Semantic embeddings are ineffective,” with 0.9414/0.9422 means | The corresponding experiment/prediction record was not located; values are not adopted as results. Similar group means alone would not establish general ineffectiveness. |
| “Zero semantic dependency” | Does not describe the retained v15 architecture. Separate structural representations from semantic inputs and propose a matched ablation. |
| “PMI graph construction implemented” | Reconcile the actual source version; report the traceable degree-cost implementation precisely. |
| “Stages 1–3 implemented” | Candidate selection and ranking code exist, but a validated integrated cascade and dedicated bridge-concept head are not established. |
| “Shortest-path nodes give the formation mechanism” | Structural paths can supply association evidence; causal/scientific mechanism requires separate grounding. Historical `show_bridges` scripts display named candidate pairs and labels, not a trained mechanism head. |
| “Not traditional link prediction” / superiority over named baselines | Describe the actual temporal objective; no categorical exclusion or unrun baseline comparison is claimed. |
| Suggested CLI files and MIT license | Use existing commands and preserve pending license status. |

## Issues retained deliberately

The [protocol page](training_evaluation.md) records candidate future filtering and metric denominators. The [model page](models.md) explains direct versus indirect candidate interactions and the retained v15 repairs. The [feature dictionary](features.md) distinguishes actual expressions from suggestive field names. Those problems are not silently repaired during README writing, and a successful synthetic optimization run is not promoted to a real-data result.

This presentation foregrounds research design and implemented modeling effort while making the unfinished task/implementation boundary explicit. It introduces no new scientific experiment or original-source changes.
