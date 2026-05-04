# Module C: Multimodal Features & Graph Construction

负责人：成员 C（多模态特征与图结构构建）

## 职责
- 商品图片批量下载
- ViT-B/16 视觉特征提取
- KMeans 视觉原型聚类
- 聚类质量检查与可视化（t-SNE、分布图）

## 运行
```bash
cd module_C
python run.py --step all
```

## 输入（来自 Module A，位于 `shared_data/`）
- `processed/item_multimodal.parquet` — 含 imUrl 的物品表
- `processed/item_id_map.json` — 物品 ID 映射

## 输出（写入 `shared_data/`）
- `images/*.jpg` — 下载的商品图片
- `processed/item_visual_embeddings.npy` — 视觉嵌入 (n_items × 768)
- `processed/item_visual_clusters.csv` — 聚类标签
- `processed/cluster_prototypes.npy` — 聚类原型向量
- `processed/cluster_summary.json` — 聚类统计
