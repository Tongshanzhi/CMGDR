# Module D: Text Modeling & Experiment Analysis

负责人：成员 D（文本建模与实验分析）

## 职责
- 评论文本嵌入提取（SentenceTransformer）
- 文本一致性损失设计
- 模型评估（全量排序 + 采样 1+999 协议）
- 偏差分析（曝光差距、校准差距、聚类探针、反事实偏移）

## 运行
```bash
# 文本特征提取
cd module_D
python run_text.py

# 模型评估（需先完成 Module B 训练）
python run_eval.py --split test --seed 42
```

## 输入
- 来自 Module A: `shared_data/processed/interactions.parquet`, `item_multimodal.parquet`
- 来自 Module B: `shared_data/model_outputs/checkpoints/*.pt`
- 来自 Module C: `shared_data/processed/item_visual_embeddings.npy`, `item_visual_clusters.csv`

## 输出（写入 `shared_data/`）
- `processed/item_text_embeddings.npy` — 文本嵌入
- `evaluation/metrics/*.json` — 评估指标
- `evaluation/bias_tables/*.csv` — 偏差分析表
