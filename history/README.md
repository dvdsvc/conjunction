# LBD historical source and version index

[Main overview](../README.md) · [简体中文](README.zh-CN.md) · [Machine-readable index](source_index.json)

This directory supplements the curated `research/` entry points with historical graph, representation, training, evaluation and diagnostic implementations. It preserves experimental evolution for inspection and possible restoration without replacing the current research entry points.

## Scope

The source inventory covers the original workspace root, `openalex_lbd`, recursive local `server_code` backups, the source-code archive, and script snapshots from four actual workspace Git commits. Of **613 source occurrences** inspected, **597** map to retained code; 16 collection/credential/control occurrences are excluded from the upload. Original files remain untouched.

Deduplication by original SHA-256 retains **176 distinct contents**: 25 already exist under `research/`, and **151 additional files** are archived here (142 Python and 9 shell). Same-name files with different contents remain separate. Occurrence counts are not model or experiment counts.

## Navigation

| Directory | Newly archived files | Source and purpose |
|---|---:|---|
| `workspace/` | 40 | Original graph, topology, preprocessing, training and evaluation scripts |
| `citation/` | 61 | Citation tasks, candidate retrieval, negatives, model comparisons and diagnostics |
| `server_snapshot/` | 44 | Unique/different local server-backup contents with relative structure preserved |
| `archive_only/` | 0 | Archive-only contents; all are represented elsewhere in this pass, so no empty directory is created |
| `git_snapshots/` | 6 | Commit contents not already represented elsewhere |

**Start with `occurrences` in `source_index.json`.** Each entry records a logical source, original hash, retained payload path and treatment. Multiple sources can point to one archived file or an identical file already in `research/`. Nested duplicate backups therefore remain traceable without duplicating their full payload.

Search the index for temporal v6/v7/v8, precompute v11/v12/v13/v14, hard-negative variants, candidate towers, cross-MLP, GraphSAGE and retrieval evaluators. Filenames, headers and configured output directories sometimes disagree; inspect the actual implementation and index together. Version numbers do not establish a complete chronology.

## Preservation and checks

Of the new copies, 149 are byte-identical and 2 replace personal absolute input paths with placeholders only. Original and output hashes are recorded separately. Added Python sources pass syntax parsing; shell files pass `bash -n`. No historical module was imported or executed and no model was trained during archival.

Actual original commit IDs, author dates and subjects appear under `git_commits`. This is a source-snapshot archive, not an import of the mixed workspace's entire `.git` history. No original history is rewritten.

## Execution and research boundaries

Historical modules may load data, change directories, contact services or launch computation at import/run time. Their dependencies, server paths, concept ordering, formats and checkpoints differ. Main README quick-start commands do not execute this archive. Inspect a version's inputs and calls before restoring it in an isolated experiment directory.

Credential-bearing connection/power-control launchers, generic collection scripts, unrelated finance, upstream clones, environments, datasets and weights are excluded. Retained files do not prove all versions ran successfully or establish predictive gains. [Design/status](../docs/design_status.md) and [evaluation protocol](../docs/training_evaluation.md) remain applicable; the PMI source version is still unresolved.
