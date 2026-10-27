# Running and reproducing the code

[Overview](../README.md) · [Graphs](graph_construction.md) · [Features](features.md) · [Models](models.md) · [Protocol](training_evaluation.md) · [Results](results.md) · [Operation](reproducibility.md) · [Provenance](provenance.md)


## Environments

Python 3.11+ is required; local checks target Python 3.12. The default environment contains NumPy/SciPy. `--extra train` adds PyTorch; `--extra encoders` adds Transformers/SentenceTransformers/requests; `--extra legacy` adds pandas, orjson, tqdm, NetworkX and Numba for the selected historical scripts. Optional extras are declared, but only the CPU training environment is installed/validated here. Encoder scripts can download pretrained weights and access name APIs; they are not run by the quick start.

```bash
uv sync --locked --extra train
uv run --locked --extra train python -m tests.test_research_contracts
uv run --locked --extra train python -m examples.synthetic_pipeline
uv run --locked --extra train python -m scripts.check_repository
```

The example builds two synthetic cumulative graphs, applies original distance/candidate/21-feature precompute, constructs Dataset tensors, runs the retained fusion model, performs three AdamW updates and calculates NDCG/P/R/MRR. Random 768-d vectors exercise only the semantic input shape. It writes temporary files outside research inputs and returns JSON; no source dataset or checkpoint is overwritten.

## Full-data input checklist

For temporal v15/fix, supply a `pencil_data` root with `concept_ids_l3.json`, old `concept_ids.json`, `concept_scibert.pt` and a `temporal_l3` subdirectory. Graph/PPR pairs must be named `graph_v2_YEAR.npz` / `ppr_v2_YEAR.npy`, share node ordering and cover required years. Precompute writes pair features, merged nodes, relevance labels, raw future distance, endpoint indexes and source-group JSON per fold.

The local archive contains five-fold metadata/index JSON but not the complete graph arrays, embeddings and checkpoints needed for a full rerun. Do not treat the archive summary as a downloadable training dataset. Full graph/PPR/distance construction can require quadratic memory; original full-data temporal trainers require CUDA.

## Safe path-dispatch wrapper

The new wrapper imports selected guarded original scripts, overrides only their data-path globals and optional year cutoffs, and calls their original `main`. It is a dry run unless `--execute` is explicitly supplied. It does not contact a training server or change the source files.

```bash
uv run --locked --extra train python -m scripts.run_research
uv run --locked --extra train python -m scripts.run_research --task build-extended-l3 --data-root data/pencil_data --papers-root data/papers --years 2005 2012 2015 2017 2021 2024
uv run --locked --extra train python -m scripts.run_research --task precompute-v15-fix --data-root data/pencil_data -- --fold 0
uv run --locked --extra train python -m scripts.run_research --task train-v15-fix --data-root data/pencil_data -- --train
```

These examples print a plan. After preparing the inputs, add `--execute` to run one deliberately. The wrapper rejects missing roots, full temporal training without CUDA, and explicit evaluation without the expected checkpoint. Historical scripts may overwrite caches/checkpoints in their configured data root: use a separate experiment directory. Graph construction requires the supplied paper JSONLs and existing concept ordering.

## Citation and FoS branches

Citation scripts with their own CLI can be inspected without data:

```bash
uv run --locked --extra train --extra legacy python research/citation/train_phoenix.py --help
uv run --locked --extra train python research/citation/eval_real_bridges.py --help
```

Other historical scripts have unguarded top-level data loading; do not import every file indiscriminately. `build_pencil_data.py` is one such archival entry. FoS preprocessing reads an external Art_Business table/embedding layout and imports CuPy; see [requirements-gpu.txt](../requirements-gpu.txt) for the CUDA-specific optional package. The presentation does not automatically install GPU packages, download benchmark data, log in to a server or execute the excluded auto-shutdown scripts.
