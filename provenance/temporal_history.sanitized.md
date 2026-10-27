# 时序 LBD 代码索引

## 数据构建（服务器）
| 脚本 | 用途 | 关键参数 |
|---|---|---|
| `build_temporal_data.py` | 构建时序共现图 + BFS距离矩阵 + 训练对chunks | `MAX_LEVEL=2`（当前限制），`MIN_SCORE=0.5` |

## 训练脚本
| 脚本 | 架构 | 损失函数 | 数据源 | 结果 |
|---|---|---|---|---|
| `train_temporal_v5.py` | v5_iso（iso mask + cand_pos_embed） | pairwise ranking on dc | 时序dc对 | disc=3.15, top-1=67.8%, Spearman=0.36 |
| `train_temporal_lbd.py` | 旧版（无iso mask） | pairwise ranking on dc | 时序dc对 | 有bug已修 |
| `openalex_lbd/train_pencil_v5_iso.py` | v5_iso 原版 | InfoNCE | 静态3-hop bridge | 原始LBD任务 |
| `openalex_lbd/train_v5_mined.py` | v5_iso + model-mined hard neg | InfoNCE | Leiden+A²候选 | 原始LBD任务 |

### 架构差异
- `train_temporal_v5.py` vs `train_temporal_lbd.py`：都用pairwise ranking，v5有iso mask（candidate只能attend source+self）+ cand_pos_embed
- `train_temporal_v5.py` vs `train_pencil_v5_iso.py`：模型架构相同，但数据来源（时序dc vs 静态bridge）和损失函数（ranking vs InfoNCE）不同

### 已知问题
1. **MAX_LEVEL=2**：建图只纳入L0/1/2概念，16,702个L3概念全是孤立节点，dc≥2信号95%来自L2-L2
2. **Hub bias**：Open science(5531x), Balance of trade(2192x)等L2超级节点主导收敛信号
3. **覆盖率低**：模型只评估1.2%的数据对（685K/55.8M），全局recall极低
4. **dc分布极偏**：dc=0:81.5%, dc=1:18.5%, dc≥2:0.1%

## 评估脚本
| 脚本 | 用途 |
|---|---|
| `eval_analysis.py` | 加载eval结果，展示top-K预测+概念名 |
| `eval_recall_k.py` / `eval_recall_k_v4_fast.py` | Recall@K评估 |
| `eval_mixedneg_top5k.py` | 混合负例top-5000评估 |

## 下一步方向
- **Jaccard去边**：`Jaccard(u,v) = |N(u)∩N(v)| / |N(u)∪N(v)|`，阈值以下删边，压制hub
- **PMI加权**：`log(P(u,v) / P(u)P(v))`，天然压制高频概念，改用加权最短路
- **度数归一化**：边权除以 `deg(u)^α * deg(v)^α`，α=0.5

[Connection details removed.]
Python: `/root/miniconda3/bin/python`
数据: `/root/autodl-tmp/openalex_data/pencil_data/temporal_v2/`
