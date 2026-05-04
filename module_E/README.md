# Module E: Engineering Optimization

负责人：成员 E（工程优化与文档呈现）

## 职责
- 训练效率优化（采样策略、批处理配置）
- 超参数搜索与实验自动化
- 多阶段性能推进（Phase 2-4）
- 实验结果可视化与报告整理

## 运行
```bash
cd module_E
python run.py optimization   # Phase 2: 多模态优化
python run.py phase3          # Phase 3: 推进 MGCN 级别
python run.py phase4          # Phase 4: 激进优化（edge dropout, LR schedule 等）
```

## 依赖
- Module B: 训练函数、模型定义、数据加载
- Module D: 采样评估协议

## 输出（写入 `shared_data/`）
- `evaluation/metrics/optimization_results.json`
- `evaluation/metrics/phase3_results.json`
- `evaluation/metrics/phase4_results.json`
