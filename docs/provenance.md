# Source, changes and publication boundary

[Overview](../README.md) · [Graphs](graph_construction.md) · [Features](features.md) · [Models](models.md) · [Protocol](training_evaluation.md) · [Results](results.md) · [Operation](reproducibility.md) · [Provenance](provenance.md)


## Selected source and version history

Twenty-seven Python files were selected from the local scientific adapters, temporal root scripts, server graph-builder snapshot and the previously retained v15 fix directory. [Source manifest](../provenance/source_manifest.json) records their exact SHA-256 hashes and logical original paths. Code files are byte-identical; two historical READMEs are separately hashed after removing connection/credential content. No credential values are stored in the public manifest or this documentation.

[Git history](../provenance/git_history.json) records four June 23 workspace commits for temporal data/training, wrappers, vectorized graph/evaluation operations and expanded scoring. The current workspace also contains staged/untracked work; a commit description is not evidence that every planned evaluation completed. Source fingerprints and private Git status snapshots are compared before/after this pass. No original checkout, staged work or history is changed.

## Upstream and local adaptation

Local paper models adapt recommendation-style roles and attention into PyTorch scientific inputs. Subgraph models are PENCIL-inspired local implementations with scientific semantic/structural features, ranking targets and training pipelines. FoS is another data adaptation. The base models, libraries and datasets retain their sources. A directory name or commit author does not by itself settle the user's exact personal contribution; owner/collaborator attribution remains for confirmation.

[Third-party notices](../THIRD_PARTY_NOTICES.md) identify the inspected clean upstream checkouts and preserve the Apache license text. This does not assign a project-wide distribution license.

## Presentation-only additions

New content comprises bilingual documentation, feature schema, minimal dependency metadata/lock, synthetic CPU runner, numerical contracts, safe path dispatcher, a source/hash/link scanner, static artwork and a CI definition. The CPU runner uses the original function and model definitions; only sample size, input data and training length are demonstration settings. No algorithmic repair was newly written in this polish pass; the v15 fix files already existed separately.

Excluded material includes original corpora, full candidate indexes, model weights/checkpoints, environments, private configuration, connection credentials, unrelated finance code in the server archive and automatic remote/server-control shell scripts. Public artifacts contain aggregate preparation statistics and explicitly synthetic outputs.

## Local preparation and subsequent private upload

The candidate set is checked for common token/connection patterns, personal paths, syntax, internal links and oversized files. This heuristic check is not an exhaustive security or rights audit. The initial 2026-10-26 preparation was local-only and did not run remote training, download pretrained models or publish a package/site. The reviewed snapshot was subsequently uploaded to a private GitHub repository on 2026-10-27. The current README revision preserves private visibility, disabled Actions and no Pages; it changes documentation only. Confirm author/ownership and a distribution license before any public release.
