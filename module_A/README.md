# Module A: Data Engineering

负责人：成员 A（项目统筹与数据工程）

## 职责
- Amazon Reviews 2023 数据下载与解压
- 数据清洗（K-core 过滤、评分筛选）
- 用户/物品重编号
- 时间划分（train/valid/test）
- User-Item 图与 Item-Item 共购图构建

## 运行
```bash
cd module_A
python run.py --step all
```

## 输入
- 数据集：Amazon Reviews 2023（5-core）
- 数据下载与字段说明：<https://amazon-reviews-2023.github.io/>

## 输出（写入 `shared_data/`）
- `processed/interactions.parquet` — 清洗后的交互表
- `processed/item_multimodal.parquet` — 物品多模态信息
- `processed/user_id_map.json`, `item_id_map.json` — ID 映射
- `processed/stats.json` — 数据统计
- `graph/edges_train.parquet`, `edges_all.parquet` — 边表
- `graph/adj_train_symmetric.npz` — CSR 邻接矩阵
- `graph/graph_meta.json` — 图元数据
- `graph/edges_item_copurchase.parquet` — Item-Item 共购边
