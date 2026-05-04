# CMGDR: Causality-aware Multimodal Graph Debiased Recommendation

基于因果去偏的多模态图神经网络推荐系统

## 项目结构

```
CMGDR/
├── shared_data/              # 模块间共享数据（统一接口）
│   ├── raw/                  # 原始 Amazon 数据
│   ├── processed/            # 预处理后的交互表、特征嵌入、聚类
│   ├── graph/                # 图存储（边表、邻接矩阵）
│   ├── images/               # 下载的商品图片
│   ├── model_outputs/        # 模型 checkpoint 与训练日志
│   │   ├── checkpoints/
│   │   └── logs/
│   └── evaluation/           # 评估结果
│       ├── metrics/
│       └── bias_tables/
│
├── module_A/                 # 成员 A: 数据工程
├── module_B/                 # 成员 B: 模型设计与因果模块
├── module_C/                 # 成员 C: 多模态特征与图结构
├── module_D/                 # 成员 D: 文本建模与实验分析
└── module_E/                 # 成员 E: 工程优化
```

## 数据集

本项目使用 **Amazon Reviews 2023** 数据集（5-core），类目示例：`Sports_and_Outdoors` / `Clothing_Shoes_and_Jewelry` / `Beauty_and_Personal_Care` 等。

- 数据下载与字段说明：<https://amazon-reviews-2023.github.io/>

请从上述地址下载评论文件与商品元数据，置于 `shared_data/raw/`，再运行 `module_A/run.py --step all` 完成清洗、ID 重编号、时间划分与图构建。

> 仓库中**不包含**任何原始数据、预处理产物、商品图片或模型 checkpoint。`shared_data/` 目录的子文件夹（`raw/`、`processed/`、`graph/`、`images/`、`model_outputs/`、`evaluation/`）均会在运行各模块脚本时按需生成。

## 模块分工

| 模块 | 负责人 | 职责 |
|------|--------|------|
| **Module A** | 成员 A | 数据下载、清洗、K-core 过滤、时间划分、图构建 |
| **Module B** | 成员 B | CMGDR 模型（LightGCN + 因果去偏）、训练、消融实验 |
| **Module C** | 成员 C | 图片下载、ViT 视觉特征提取、KMeans 聚类、QA |
| **Module D** | 成员 D | 文本嵌入提取、模型评估、偏差分析 |
| **Module E** | 成员 E | 超参优化、多阶段实验、可视化 |

## 运行顺序

```
Module A → Module C → Module D(文本) → Module B → Module D(评估) → Module E
```

### 端到端运行

```bash
# Step 1: 数据预处理
cd module_A && python run.py --step all && cd ..

# Step 2: 视觉特征提取与聚类
cd module_C && python run.py --step all && cd ..

# Step 3: 文本特征提取
cd module_D && python run_text.py && cd ..

# Step 4: 模型训练
cd module_B && python run_train.py --seed 42 && cd ..

# Step 5: 模型评估
cd module_D && python run_eval.py --split test --seed 42 && cd ..

# Step 6: 超参优化（可选）
cd module_E && python run.py optimization && cd ..
```

## 模块间数据接口

所有模块通过 `shared_data/` 目录交换数据，接口文件格式固定：

| 产出文件 | 格式 | 生产者 | 消费者 |
|----------|------|--------|--------|
| `processed/interactions.parquet` | Parquet | A | B, D |
| `processed/item_multimodal.parquet` | Parquet | A | C, D |
| `processed/item_id_map.json` | JSON | A | B, C, D |
| `graph/edges_train.parquet` | Parquet | A | B |
| `graph/graph_meta.json` | JSON | A | B |
| `graph/edges_item_copurchase.parquet` | Parquet | A | B |
| `processed/item_visual_embeddings.npy` | NumPy | C | B |
| `processed/item_visual_clusters.csv` | CSV | C | B |
| `processed/cluster_prototypes.npy` | NumPy | C | B |
| `processed/item_text_embeddings.npy` | NumPy | D | B |
| `model_outputs/checkpoints/*.pt` | PyTorch | B | D, E |
| `evaluation/metrics/*.json` | JSON | D | E |
