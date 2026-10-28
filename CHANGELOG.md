# Research presentation snapshot — 2026-10-26

## 2026-10-28 — Add the gravitational modeling analogy

Added the author-supplied celestial-mechanics analogy at the end of feature construction in both READMEs. It explains the intuition behind degree, BFS distance, propagation, neighborhood overlap and possible community evolution without claiming a physical force law or a completed community model. Research code and results are unchanged.

## 2026-10-27 — Put graph representations with feature construction

Moved the neural representation explanation into the feature-construction section in both READMEs. Removed the implication that v15 introduced graph-in-Transformer computation or three-branch fusion: temporal v5 already includes interleaved graph updates, and archived v9 explicitly implements all three fusion branches. Kept version-specific dimensions as examples and corrected the implementation-history wording. Research code and experiment records are unchanged.

## 2026-10-27 — Explain graph computation inside PENCIL

Expanded both READMEs and model/feature documentation to distinguish scalar pair features, the graph-message updates interleaved inside temporal PENCIL Transformer layers and the separate merged-subgraph GNN. Documented early one-hop neighbor selection versus later PPR-based node selection, the adjacency input layout and the exact message contraction. Clarified PENCIL method attribution. All research implementations and experiment records remain unchanged.

## 2026-10-27 — Restore the fuller research narrative

Revised both READMEs from the earlier author-directed narrative rather than the compressed summary. Restored motivation and reasoning, and expanded the account of graph sparsity, concept-identity errors, implementation repairs, task-protocol differences and the decision to pause. Retained the staged framework, hub controls, bridge example and quick start. Research code, results and archived versions are unchanged.

## 2026-10-27 — Concise academic README and staged design

Condensed both READMEs into a more formal research overview while retaining the seven-section narrative and fish-oil diagrams. Added the Screen → Rank → Bridge → downstream-validation design, three implemented hub controls and the illustrative Phosphide/Lightning bridge format. Clarified bounded path recovery in the historical three-hop evaluator and distinguished it from unfinished standalone bridge output and scientific mechanism validation. Research code and experiment records remain unchanged.

## 2026-10-27 — Author-directed Catalyst narrative

Rewrote both READMEs around the author's supplied seven-section narrative: literature-discovery motivation, topology-first design, implemented features and neural models, method evolution, reasons for pausing, future directions and repository usage. Added original bilingual SVGs explaining the fish-oil/Raynaud A–B–C literature example, primary-source links and GitHub-compatible math formatting. Updated README structure checks to match the supplied outline. Research code, archived source versions and experiment records are unchanged.

- Preserved 27 scientific Python files byte-for-byte across citation, temporal and FoS branches, including the already-existing v15 repair pair.
- Documented graph variants, exact feature order, model branches, losses, training configuration, candidate protocols and evaluation denominators.
- Added local five-window preparation aggregates and explicitly labeled historical reported metrics; no successful prediction claim or new full-data result was invented.
- Added locked dependencies, a CPU synthetic pipeline, numerical contracts, a dry-run path dispatcher, candidate-file QA and a research CI definition.
- Removed connection details only from copied historical README text; excluded datasets, weights, private state and remote shutdown/control scripts. No publication action taken.

## 2026-10-27 — Research README revision

Expanded both READMEs around the Catalyst research question, temporal targets, graph/weight semantics, topology and semantic inputs, model/training design, unresolved protocol issues and executable entry points. Added a source-version/claim reconciliation page after inspecting the original working tree, nested server backup and code archive. PMI source-version reconciliation remains explicit. Preserved every research implementation and historical result; retained private GitHub visibility.

## 2026-10-27 — Historical source archive

Added 151 deduplicated historical source files (142 Python, 9 shell) with bilingual guides and a complete occurrence/hash index. Reused 25 identical research-core contents instead of duplicating them; retained differing versions and snapshots from four original Git commits. Excluded credential/control material and unrelated inputs, preserved original sources, and retained private GitHub visibility. No historical training or scientific scoring was rerun.
