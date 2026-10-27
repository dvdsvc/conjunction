# Catalyst: Finding Connections Yet to Be Made in Scientific Knowledge Networks

**Literature-Based Discovery · Temporal Knowledge Graphs · Scientific Hypothesis Generation**

**English** · [简体中文](README.zh-CN.md)

> An unfinished research exploration: use the historical evolution of scientific concept networks to identify connections that have not yet been explicitly established, but may be worth investigating.

The project began with a question: **How much knowledge in the scientific literature remains insufficiently connected, and can network structure and its evolution help us identify some of those connections?** The aim was to investigate this question, rather than introduce a new graph neural network or automatically prove scientific hypotheses.

The repository contains literature-network builders, graph features, neural ranking models and training pipelines. It has not produced reliable end-to-end scientific discovery results. Work is currently paused because of data quality, graph sparsity and the time required to address them.

---

## 1. Motivation: Low-Hanging Fruit in Scientific Knowledge

### 1.1 Fish oil and Raynaud's syndrome: undiscovered public knowledge

The initial inspiration came from Don R. Swanson's 1986 paper, [*Fish Oil, Raynaud's Syndrome, and Undiscovered Public Knowledge*](https://pubmed.ncbi.nlm.nih.gov/3797213/). Raynaud's syndrome involves abnormal peripheral circulation: cold or emotional stress can restrict blood flow to the fingers and toes.

Swanson identified two relatively separate bodies of medical literature:

- Studies of fish oil reported effects on blood viscosity, platelet aggregation and vascular reactivity.
- Studies of Raynaud's syndrome connected these physiological factors with the condition's symptoms.

Bringing these findings together suggested a hypothesis worth testing: **Could fish oil alleviate Raynaud's syndrome?**

![A–B–C literature reasoning: fish oil and Raynaud's syndrome are linked through blood viscosity, platelet aggregation and vascular reactivity; the dashed connection is a hypothesis.](assets/fish-oil-raynaud.en.svg)

*A schematic of the literature reasoning in Swanson (1986). Solid lines represent reported connections; the dashed line represents the resulting hypothesis. This is not an experimental result from Catalyst or evidence of established treatment efficacy.*

What interested me was the way the hypothesis emerged. The relevant knowledge was already public; someone still had to connect it. Conventional literature searches begin with a known question, which assumes the researcher knows what to search for. When fields rarely interact, their published findings may remain disconnected even when they could inform one another.

This led me to ask whether we could start from the structure of scientific knowledge itself and actively look for potential connections, without requiring a researcher to formulate each question in advance.

### 1.2 Snowball Earth: different disciplines may hold different parts of an answer

Another inspiration was Snowball Earth. Neoproterozoic strata contain features that were difficult to explain, including evidence of low-latitude glaciation and cap carbonates immediately above glacial deposits. Meanwhile, climate dynamics had established the possibility of an ice–albedo feedback: expanding ice reflects more sunlight, causes further cooling and could drive the planet into a globally frozen state.

Global freezing raised another question: how could Earth become warm again? In his [1992 Snowball Earth proposal](https://authors.library.caltech.edu/records/vaysf-cca32), Joseph Kirschvink connected the geological evidence with a possible escape mechanism. Global ice cover would strongly suppress chemical weathering that consumes atmospheric carbon dioxide, while volcanic outgassing could continue. Accumulating carbon dioxide could eventually produce enough greenhouse warming to end the frozen state.

This connected sedimentary geology, paleomagnetism, climate dynamics, greenhouse warming and the long-term carbon cycle. In his [2009 historical account](https://doi.org/10.1111/j.1365-2451.2009.00716.x), Paul Hoffman described a substantial gap between geologists studying Proterozoic glacial deposits and climate dynamicists studying extreme glaciation before the late 1980s. One community had geological evidence; the other had theoretical tools that could help explain it, but the connection was not widely recognized.

This does not settle every debate about Snowball Earth, or imply that an algorithm could discover the entire theory. It illustrates a possibility that interests me:

**Part of the answer to a persistent problem in one discipline may already exist in another, waiting to be connected.**

I think of these opportunities as the “low-hanging fruit” of scientific knowledge. They need not be new laws of nature. They could be a useful combination of existing findings, a new explanation for an old problem, or a lead worth testing.

### 1.3 From literature retrieval to active discovery

I wanted to explore a different starting point: analyze the structure of the literature, **identify relationships that have not yet been explicitly established but may be forming**, and then let researchers or AI scientist systems investigate and test them.

The intended output is a set of candidate scientific relationships with traceable literature sources, rather than a claim that a model has discovered a scientific law.

---

## 2. Core Design: Looking Beyond Semantic Similarity to Network Structure

### 2.1 Why not predict relationships directly from semantic embeddings?

This was an important design consideration from the beginning. Scientific names, classifications and descriptions already encode existing disciplinary organization: which field a concept belongs to, its place in a hierarchy and which definitions are close to it. I was concerned that relying mainly on semantic similarity could reproduce those existing classifications.

For cross-disciplinary discovery, this creates a tension:

**The concepts we want to connect may be far apart in current disciplinary taxonomies, yet related through scientific mechanisms.**

Similar names do not necessarily make a connection scientifically valuable. Conversely, concepts with very different meanings may be linked through a chain of mechanisms. A predominantly semantic model might find familiar relationships more easily than connections across established boundaries.

The project therefore focused on **how relationships form, as well as how concepts are described**. Semantic information still has a role: the code uses precomputed SciBERT vectors in selected models. In the original design, however, semantics was auxiliary; network topology and its evolution were the main research objects.

### 2.2 A scientific concept network as an evolving system

The initial plan was to use OpenAlex papers, citations, publication dates and concept labels to construct scientific concept networks over time. Papers provide the main observations: when two concepts occur together in a qualifying paper, they can be connected by a co-occurrence edge. Citations supply relationships between papers and support some earlier research routes.

At a historical cutoff $t$, the observable concept graph is:

$$
G_t=(V,E_t)
$$

For concepts $u,v$ without a direct historical connection, the intended prediction is:

$$
P\big((u,v)\in E_{t+\Delta t}\mid G_{\leq t}\big)
$$

The original focus was on pairs with historical BFS distance greater than one: concepts linked indirectly, but not yet directly connected.

One intuition was to imagine concept communities as evolving galaxies. Whether two galaxies approach one another depends on their surroundings, interactions and motion, as well as their current distance. Similarly, future connections between concepts may depend on common neighbors, local communities, hubs and changes in their neighborhoods.

This is a heuristic analogy, not a claim that scientific networks follow a gravitational law. It motivates a concrete question:

**Before two concepts become directly connected, are there recognizable changes in the network around them?**

---

## 3. Features and Research Ideas

The project explored several ways to describe a concept pair.

| Feature | Motivation |
|---|---|
| BFS distance | Current shortest topological distance |
| Bidirectional PPR-like propagation | Relative network position and reachability |
| Ego-graph | Structure and organization of each local neighborhood |
| Common-neighbor count | Number of existing indirect connections |
| Common-neighbor degree distribution | Whether bridges are broad hubs or more specific concepts |
| Adamic–Adar | Downweight highly connected common neighbors |
| Clustering coefficient | Connectivity within a local neighborhood |
| Community membership (planned) | Whether concepts belong to the same knowledge community |
| Temporal neighborhood changes (planned) | Whether the two neighborhoods are gradually approaching one another |
| SciBERT | Semantic representations of concept text; the located encoding script uses concept names |

BFS, propagation quantities, common neighbors, Adamic–Adar, local subgraphs and some structural statistics appear in different code versions. Community membership and continuous temporal change were considered in the design, but do not form a unified, validated implementation.

Another focus was **how to handle hubs**. Two specific concepts might both connect to “climate change,” but that broad connection may offer little useful research detail. A bridge through a less connected concept with a more specific mechanistic relationship could be more informative.

Later code assigns higher path costs to edges involving highly connected nodes:

$$
c(u,v)=\log\left(\frac{\deg(u)+\deg(v)}{2}+1\right)
$$

Together with Adamic–Adar and features based on low-degree common neighbors, this was intended to reduce the influence of generic hubs. These features express research hypotheses; their predictive benefit is not assumed in advance.

---

## 4. Implementation and Method Evolution

The project went through several changes in tasks and models. There was no single final algorithm fixed at the outset.

### 4.1 From paper citations to concept relationships

Early work adapted recommendation and link-prediction methods to scientific paper networks, using citations, semantic representations and candidate ranking to explore potential relationships. The focus then shifted to temporal concept co-occurrence graphs and prediction targets based on changes in historical graph distance. The v15 route added local subgraphs, structural features and multiple representation branches.

Main source locations:

- [Paper citations and early models](research/citation/)
- [Temporal concept graph construction](research/temporal/build_temporal_data.py)
- [Temporal feature precomputation](research/temporal/precompute_v15.py)
- [v15 models and training](research/temporal/train_temporal_v15.py)
- [Later repair versions](research/temporal/fixes/)

### 4.2 Neural models

The later v15 implementation uses three PyTorch representation branches:

| Branch | Main input | Role |
|---|---|---|
| PairMLP | 21 explicit graph features | Learn structural relationships between a node pair |
| PENCIL-style Transformer | Local subgraph structure and precomputed semantic representations | Encode concept neighborhoods and candidate context |
| SubgraphGNN | Two endpoints and their merged subgraph | Model graph connections around a candidate relationship |

The three representations are fused to produce candidate ranking scores. This explored network information at different scales; it does not establish that a more complex model is better than a simpler one. The retained implementations and attribution notes document their relationship to existing recommendation models, graph-learning methods and third-party code.

### 4.3 Training and historical experiments

The research tried classification, pairwise ranking, BPR, distance-change ranking and ranking losses with NDCG-based weights. Results were unsatisfactory and did not establish a consistent, reliable scientific discovery capability.

Temporal experiments used multiple historical windows covering parts of 2005–2024. The repository retains candidate construction, training, evaluation and historical records. Different versions also used different targets: later v15 experiments mainly learned relevance grades derived from future weighted distances, which is not the same task as predicting the first direct connection between concepts.

These versions are retained as stages of the research process. Their metrics are not combined into a single final score for one task.

[Models](docs/models.md) · [Training and evaluation protocol](docs/training_evaluation.md) · [Historical results](docs/results.md)

---

## 5. Why Is the Project Paused?

The project reached several more fundamental obstacles while there were still model ideas to try.

### 5.1 Sparse concept graphs

After restricting the discipline, concept level and historical window, many concepts had few useful connections. For first-time cross-disciplinary connections in particular, positive examples were scarce relative to the candidate space. This limited the training signal and made candidate selection and supervision harder to define.

I tried several loss functions and ranking strategies without obtaining sufficiently strong, stable results. Increasing model complexity did not resolve the underlying problem.

### 5.2 More noise than expected in OpenAlex concept labels

This was the main reason I decided to pause the project. It used the legacy OpenAlex Concepts system, which offered large-scale scientific classifications, but concept labels and paper assignments were not consistently reliable in practice.

For example, during manual inspection I encountered a suspected ambiguity involving `Lightning`: atmospheric lightning and Apple's Lightning connector may have been associated with labels carrying different meanings. This specific example still needs confirmation against the original concept IDs and paper records.

Problems like this directly affect graph construction. Splitting one scientific concept across different nodes, or confusing different meanings of a term, can make graph connectivity diverge from actual relationships in the literature.

**A particularly serious case is when the literature has already connected two concepts, but concept recognition or mapping errors make the constructed graph show no prior connection.**

For a system meant to predict first connections, this can invalidate the task labels themselves. A more complex model may simply learn patterns in an incorrectly constructed network.

OpenAlex has since [deprecated Concepts in favor of Topics](https://help.openalex.org/data/concepts/). Its official explanation discusses accuracy and maintenance problems with the earlier system. This is consistent with the data quality difficulties encountered here, but does not verify any particular erroneous record.

### 5.3 Gaps between historical implementations and the research goal

The many iterations also introduced issues involving candidate indices, node alignment, subgraph padding and ranking calculations. The retained `fixes/` versions address candidate grouping, endpoint preservation and conflicts between node IDs and padding values.

Some later versions also included historically connected nodes as candidates and filtered candidates using future reachability. This differs from the original first-new-connection objective. The repairs mainly address implementation consistency; they do not bring every historical version under the original task definition.

These issues are part of the research record, but prevent interpreting historical ranking scores directly as scientific discovery performance.

### 5.4 A decision about research time

Resuming this work would require prioritizing the concept system, data cleaning and task definition before adding another model component. Given the cost of repairing the data and retraining, I decided to preserve the code and experiment records and turn to other research.

I still find the original question worthwhile. The available data foundation was not sufficient for the intended goal.

---

## 6. Future Directions

If I return to the project, I would start with data sources and concept consistency. One possibility is biomedical literature with a more standardized terminology system, such as MEDLINE's MeSH annotations, together with appropriate concept mapping.

This may offer better control over synonyms, ambiguity and concept hierarchies than directly using legacy OpenAlex Concepts. It would still require inspecting the actual data; it cannot be assumed to solve every problem.

I would then redefine the training task explicitly:

1. Freeze the concept set and existing connections at a historical cutoff.
2. Select only concept pairs without a direct connection at that time.
3. Construct structural, temporal and any necessary semantic features from historical information.
4. Use future publications to identify candidates that form their first direct connection.
5. Trace the underlying papers and intermediate knowledge, then have researchers or AI scientist systems assess the scientific significance.

The aim remains **to find a small number of potential scientific connections worth serious investigation within a large literature**, rather than optimize a link-prediction score alone.

---

## 7. Repository Layout and Running the Code

The repository preserves the main research code, historical versions, entry points and technical documentation.

| Path | Contents |
|---|---|
| [`research/citation/`](research/citation/) | Paper-citation tasks and early models |
| [`research/temporal/`](research/temporal/) | Temporal graphs, features, ranking models and training |
| [`research/temporal/fixes/`](research/temporal/fixes/) | Later repair versions |
| [`research/fos/`](research/fos/) | Related graph-learning experiments |
| [`history/`](history/) | Historical source versions and records |
| [`docs/`](docs/) | Models, data, experiments and implementation details |
| [`examples/`](examples/) | Examples using synthetic data |
| [`tests/`](tests/) | Basic numerical and implementation checks |

A small synthetic example exercises graph construction, feature generation, model forward passes and backpropagation. It checks code execution paths; **it is not a complete scientific discovery demonstration**.

Historical full-data training requires the original paper records, matching concept indices, graph matrices and model environment. Some large datasets and checkpoints are not distributed with this repository.

See [Running and reproduction](docs/reproducibility.md) for commands. Code origins, historical versions and third-party attribution are documented in [Provenance](docs/provenance.md) and [Third-party Notices](THIRD_PARTY_NOTICES.md).

---

## References

1. Swanson, D. R. (1986). [Fish Oil, Raynaud's Syndrome, and Undiscovered Public Knowledge](https://pubmed.ncbi.nlm.nih.gov/3797213/). *Perspectives in Biology and Medicine*.
2. Hoffman, P. F., et al. (1998). [A Neoproterozoic Snowball Earth](https://pubmed.ncbi.nlm.nih.gov/9721097/). *Science*.
3. Hoffman, P. F., & Schrag, D. P. (2000). [Snowball Earth](https://www.scientificamerican.com/article/snowball-earth/). *Scientific American*.
4. Hoffman, P. F. (2009). [Pan-glacial—a third state in the climate system](https://doi.org/10.1111/j.1365-2451.2009.00716.x). *Geology Today*.
5. [OpenAlex — Concepts (deprecated)](https://help.openalex.org/data/concepts/).
6. Kirschvink, J. L. (1992). [Late Proterozoic Low-Latitude Global Glaciation: the Snowball Earth](https://authors.library.caltech.edu/records/vaysf-cca32). In *The Proterozoic Biosphere: A Multidisciplinary Study*.

---

*This repository brings together an earlier research exploration, including code, methods and experiment records from different stages. The project is unfinished and does not claim validated automatic scientific discovery.*
