# CMGDR: Causality-aware Multimodal Graph Debiased Recommendation

A multimodal graph recommender that uses **causal debiasing** to lift accuracy
without amplifying visual bias. CMGDR is built on a LightGCN backbone and
combines four modalities — user-item interactions, ViT-B/16 visual embeddings,
SBERT text embeddings, and an item-item co-purchase graph — under a structural
causal model `U → I → Y ← V` with backdoor adjustment.

On Amazon Reviews 2023 — Sports & Outdoors (5-core, 35,598 users / 18,357 items
/ 296,337 interactions), the full configuration improves Recall@10 by **28.4 %**
over LightGCN while reducing the visual exposure gap and counterfactual score
shift versus standard multimodal baselines. See `report/CMGDR_report.pdf` and
`CMGDR_Evaluation_Report.md` for the full evaluation.

## Project layout

```
CMGDR/
├── shared_data/              # shared data exchanged across stages (created at runtime)
│   ├── raw/                  # downloaded Amazon reviews & metadata
│   ├── processed/            # cleaned interactions, features, clusters
│   ├── graph/                # edge tables, adjacency matrices
│   ├── images/               # downloaded product images
│   ├── model_outputs/        # trained checkpoints and training logs
│   │   ├── checkpoints/
│   │   └── logs/
│   └── evaluation/           # metrics and bias tables
│       ├── metrics/
│       └── bias_tables/
│
├── module_A/                 # data engineering: download, clean, k-core, split, graph build
├── module_B/                 # model: LightGCN backbone + causal debiasing, training, ablations
├── module_C/                 # visual: image download, ViT-B/16 features, KMeans prototypes
├── module_D/                 # text & evaluation: SBERT embeddings, ranking metrics, bias probes
├── module_E/                 # optimization: hyper-parameter search, multi-phase experiments
├── new_results/              # additional Section-7 experiment drivers and analysis scripts
└── report/                   # LaTeX paper and figures
```

## Dataset

This project uses **Amazon Reviews 2023** (5-core) — example categories:
`Sports_and_Outdoors`, `Clothing_Shoes_and_Jewelry`, `Beauty_and_Personal_Care`.

- Download and field documentation: <https://amazon-reviews-2023.github.io/>

Place the downloaded reviews + metadata gzipped JSONL files under
`shared_data/raw/`, then run `module_A/run.py --step all` to perform cleaning,
ID re-indexing, temporal split, and graph construction.

> The repository contains **source code only**. Raw datasets, preprocessed
> artefacts, downloaded product images, model checkpoints, and experiment
> outputs are excluded via `.gitignore` — every subdirectory under
> `shared_data/` is generated on demand by the module pipelines.

## End-to-end pipeline

The five modules form a single pipeline:

```
module_A → module_C → module_D (text) → module_B → module_D (eval) → module_E
```

Reproducing the headline numbers from a clean clone:

```bash
# Step 1 — data preprocessing & graph construction
cd module_A && python run.py --step all && cd ..

# Step 2 — visual feature extraction & KMeans clustering
cd module_C && python run.py --step all && cd ..

# Step 3 — text feature extraction
cd module_D && python run_text.py && cd ..

# Step 4 — train CMGDR (and optional baselines)
cd module_B && python run_train.py --seed 42 && cd ..

# Step 5 — evaluation (sampled 1+999 LOO + bias diagnostics)
cd module_D && python run_eval.py --split test --seed 42 && cd ..

# Step 6 — hyper-parameter optimisation (optional)
cd module_E && python run.py optimization && cd ..
```

Each module also has its own `README.md` with finer-grained instructions and
its own `requirements.txt`.

## Inter-module data interface

Modules communicate exclusively through files in `shared_data/`. Formats are
fixed so any module can be re-run independently as long as its inputs exist.

| File | Format | Producer | Consumers |
|------|--------|----------|-----------|
| `processed/interactions.parquet`        | Parquet | module_A | module_B, module_D |
| `processed/item_multimodal.parquet`     | Parquet | module_A | module_C, module_D |
| `processed/item_id_map.json`            | JSON    | module_A | module_B, module_C, module_D |
| `graph/edges_train.parquet`             | Parquet | module_A | module_B |
| `graph/graph_meta.json`                 | JSON    | module_A | module_B |
| `graph/edges_item_copurchase.parquet`   | Parquet | module_A | module_B |
| `processed/item_visual_embeddings.npy`  | NumPy   | module_C | module_B |
| `processed/item_visual_clusters.csv`    | CSV     | module_C | module_B |
| `processed/cluster_prototypes.npy`      | NumPy   | module_C | module_B |
| `processed/item_text_embeddings.npy`    | NumPy   | module_D | module_B |
| `model_outputs/checkpoints/*.pt`        | PyTorch | module_B | module_D, module_E |
| `evaluation/metrics/*.json`             | JSON    | module_D | module_E |

## Evaluation protocol

Following the standard protocol of MMGCN / LATTICE / BM3 / FREEDOM / MGCN /
LGMRec / MENTOR, we use leave-one-out evaluation with sampled scoring
(1 ground-truth positive + 999 random negatives per user). The chronologically
last interaction of each user is held out for testing, the second-to-last for
validation, and the rest for training.

Two families of metrics are reported:

- **Recommendation accuracy** — Recall@K, NDCG@K, HR@K, MRR (K = 10, 20).
- **Debiasing effectiveness** — exposure gap, calibration gap, cluster probe
  accuracy, and counterfactual score shift under visual-style intervention.

## Citation

If you build on this code, please cite the Amazon Reviews 2023 release that
provides the underlying interaction and metadata corpus:

> Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J.
> *Bridging Language and Items for Retrieval and Recommendation.*
> arXiv:2403.03952, 2024.
