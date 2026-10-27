# Third-party methods and sources

| Source | Local source identity inspected | Relationship to this snapshot |
| --- | --- | --- |
| [PENCIL](https://github.com/quang-truong/pencil) | `cd39f1a7ad471c28a434830ee4907db93d1999c4`; tracked checkout clean | Inspiration for local subgraph encoding and link-prediction adaptation; full upstream repository is not vendored |
| [X algorithm / Phoenix](https://github.com/xai-org/x-algorithm) | `0bfc2795d308f90032544322747caacd535f75ae`; tracked checkout clean | Recommendation-inspired attention and input-role reference; scientific PyTorch adapter is a separate local implementation |
| SciBERT | `allenai/scibert_scivocab_uncased` name in original encoder | Pretrained concept-name vectors; no model weights bundled |
| MiniLM | Original title encoder uses a pretrained SentenceTransformers model | Supplied paper-title vectors; distinct from the SciBERT concept pipeline |
| OpenAlex / FoS Art_Business | Source data named by original scientific scripts | External input datasets, not redistributed here |

The inspected PENCIL and X repositories contain Apache 2.0 licenses. [License text](licenses/Apache-2.0.txt) is retained from PENCIL; applicable notices and contribution boundaries must remain respected. A project-wide license for the local scientific work has not been selected on the maintainer's behalf.

NumPy, SciPy, PyTorch and optional encoder/preprocessing libraries retain their own licenses. This snapshot does not claim a new foundation model, a new SciBERT training algorithm or authorship of upstream model designs.
