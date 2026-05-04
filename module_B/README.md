# Module B: Model Design & Causal Module

负责人：成员 B（模型设计与因果模块）

## 职责
- CMGDR 模型架构（LightGCN 骨干 + 因果去偏机制）
- 视觉原型聚类与因果正则项
- 多模态嵌入层（视觉投影、文本投影、Item-Item 图融合）
- 训练流程与消融实验

## 运行
```bash
cd module_B
python run_train.py --config config/model.yaml --seed 42
python run_ablate.py --config config/model.yaml --seed 42
python run_experiments.py  # LOO 综合实验
```

## 输入（来自 `shared_data/`）
- Module A: `processed/interactions.parquet`, `graph/edges_train.parquet`, `graph/graph_meta.json`
- Module C: `processed/item_visual_embeddings.npy`, `item_visual_clusters.csv`
- Module D: `processed/item_text_embeddings.npy`

## 输出（写入 `shared_data/`）
- `model_outputs/checkpoints/*.pt` — 模型权重
- `model_outputs/logs/*.jsonl` — 训练日志
- `evaluation/metrics/*_train_summary.json` — 训练摘要
