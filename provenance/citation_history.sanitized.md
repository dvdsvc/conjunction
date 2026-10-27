# LBD (Literature-Based Discovery) — 跨域桥论文预测

## 一句话

从 355 万篇 climate + geology 论文中，预测跨域间接引用（A→B→C，A 不引 C）。

## 核心发现（2026-06-16）

**旧数据定义错误**：`lbd_bridges.pt`（210万对）全部是直接引用，不是 LBD。
IDF cosine R@5000=50.6% 只是"找直接引用"的 trivial 任务。

**已修正**：重新构建真实 2-hop LBD bridge pairs（`lbd_bridges_real.pt`，5174万对）。
A→B→C，A 不引 C，Jaccard(concept_A, concept_C) < 0.1（跨域过滤）。

## 当前最佳结果

IDF-weighted cosine（1hop_concepts），**零训练，真实 LBD 桥数据**：

| K    | Recall | Hits     |
|------|--------|----------|
| 50   | 0.1%   | 5/5000   |
| 100  | 0.3%   | 16/5000  |
| 500  | 1.6%   | 78/5000  |
| 1000 | 2.7%   | 135/5000 |
| 5000 | 9.3%   | 464/5000 |

MRR@5000 = 0.0002。提升空间巨大。

## 数据

- **位置**：服务器 `/root/autodl-tmp/openalex_data/processed/`
- **Climate** 1.54M + **Geology** 2.05M = **3.55M** 论文
- **lbd_bridges_real.pt**（592MB）：51,747,145 对真实 2-hop LBD 桥
  - `source_idx`: int32 tensor — source 论文索引
  - `target_idx`: int32 tensor — target 论文索引（A 不引 C）
  - `jaccard`: float32 tensor — concept Jaccard 重叠度（均值 0.068）
  - Pre-2021 训练集：31,275,117 对
  - Post-2021 测试集：20,472,028 对
- **lbd_bridges.pt**（59MB，已废弃）：旧直接引用对，非 LBD
- **features.pt**：
  - `features`: (3551912, 5) float32 — [year_norm, log_citations, ?, ?, ?]
  - `concepts`: list[list[int]] — 每篇论文的 OpenAlex concept IDs（12785 种）
  - `paper_ids`, `paper_id_to_idx`, `year_range`: (1970, 2026)
- **ref_lists.pt**：list[list[int]] — 每篇论文引用的数据集内论文索引
  - 54.7% 为空，有引用的均值 21.2，中位数 13，max 1787
- **gnn_embeddings.pt**：(3551912, 64) — GNN 预训练 embedding
- **title_embeddings.pt**（5.2GB）：(3551912, 384) — all-MiniLM-L6-v2 编码标题
- **config.json**：模型配置
- **checkpoints/**：phoenix_best.pt（145MB，epoch20 val_auc=0.9919）

## Historical execution environment

Connection details omitted from the presentation copy.
## 代码文件

### 数据构建
- `build_real_bridges_v3.py` — **真实 LBD bridge 构建**（80核并行，75秒，5174万对）
- `build_real_bridges_v2.py` — 单线程版（~20分钟）
- `build_real_bridges.py` — 初版（内存爆炸，已废弃）
- `encode_titles.py` — sentence-transformers 编码 3.55M 标题（384d，12分钟）

### 评估
- `eval_real_bridges.py` — **IDF cosine 评估真实 LBD 桥**（当前 baseline）
- `eval_jaccard.py` — IDF cosine 旧直接引用数据（R@5000=50.6%，已废弃）
- `eval_hybrid.py` — IDF coarse + Phoenix re-rank（Phoenix 反而降低 recall）
- `eval_mask_prediction.py` — 通用引用预测（R@5000=1.8%，无用）
- `eval_retrieval_phoenix.py` — Two-tower Phoenix 全量检索评估

### 模型
- `phoenix_model.py` — PhoenixCitationTransformer（10层 RoPE RMSNorm SwiGLU）
- `compute_candidates_v2.py` — 单 token 候选向量（无数据泄漏）
- `train_phoenix.py` — mask prediction 训练
- `train_candidate_tower_batch.py` — Two-tower candidate tower 训练

### 其他
- `process_data.py` — 数据准备脚本
- `generate_report.py` — 训练报告生成
- `run_all.sh` — 全流程脚本（encode→train→eval→shutdown）

## 实验历史

### 旧数据实验（直接引用，已废弃）

| # | 方法 | R@5000 | 备注 |
|---|------|--------|------|
| 1 | Two-Tower Phoenix | 23.4% | train-eval mismatch 虚高 |
| 2 | CandidateTowerConcept | 0.6% | user tower 坍缩 |
| 3 | IDF cosine(1hop) | 50.6% | 找直接引用，trivial |

### 新数据实验（真实 2-hop LBD 桥）

| # | 方法 | R@5000 | 备注 |
|---|------|--------|------|
| 1 | IDF cosine(1hop) | 9.3% | 新 baseline，零训练 |
| 2 | Phoenix re-rank | — | 未在新数据上测试 |

## 核心教训

1. **任务定义比模型重要**：旧数据全是直接引用，做再多模型优化也是在解错题
2. **Phoenix mask prediction ≠ retrieval**：训练收敛（val_auc=0.9919）但 embedding 无检索价值
3. **单信号有天花板**：IDF cosine 只用概念重叠，对跨域间接连接信号弱（9.3%）
4. **简单方法先跑**：IDF cosine 零训练，先建立 baseline 再考虑复杂模型

## 下一步方向

### 主方向：共被引信号

B 和 C 被同一批论文引过 → B 和 C 学术社群相关 → A 引了 B，可能也该引 C。
共被引 + IDF concept 互补：概念重叠看语义，共被引看社群结构。

### 备选
1. **Contrastive learning**：正样本=真实bridge，hard negative=同域论文
2. **多 signal 融合**：共被引 + concept + title embedding + 时间衰减
3. **Graph embedding**：node2vec / GNN 在 citation graph 上预训练
