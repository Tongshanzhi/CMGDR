"""Scalable User–Item graph: Parquet edges, CSR npz, SQLite for ad-hoc queries."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse


def _load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_homogeneous_indices(n_users: int, item_idx: np.ndarray) -> np.ndarray:
    """Item local idx -> global node id offset by n_users."""
    return item_idx.astype(np.int64) + n_users


def interactions_to_edge_table(df: pd.DataFrame, n_users: int) -> pd.DataFrame:
    """Directed edges user -> item in global id space."""
    dst = build_homogeneous_indices(n_users, df["item_idx"].values)
    out = pd.DataFrame(
        {
            "src": df["user_idx"].astype(np.int64).values,
            "dst": dst,
            "rating": df["overall"].values,
            "time": df["unixReviewTime"].values,
            "split": df["split"].values,
        }
    )
    return out


def train_only_edges(edges: pd.DataFrame) -> pd.DataFrame:
    return edges[edges["split"] == "train"].reset_index(drop=True)


def edges_to_symmetric_biadj(edges: pd.DataFrame, n_nodes: int) -> sparse.csr_matrix:
    """Undirected adjacency for GNN-style message passing (user-item bipartite as one graph)."""
    rows = np.concatenate([edges["src"].values, edges["dst"].values])
    cols = np.concatenate([edges["dst"].values, edges["src"].values])
    data = np.ones(len(rows), dtype=np.float32)
    mat = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    mat.sum_duplicates()
    return mat.tocsr()


def run_graph_build(project_root: Path, config_path: Path | None = None) -> dict:
    cfg_path = config_path or project_root / "config" / "default.yaml"
    cfg = _load_config(cfg_path)
    paths = cfg["paths"]
    gcfg = cfg["graph"]

    proc = project_root / paths["processed_dir"]
    gdir = project_root / paths["graph_dir"]
    gdir.mkdir(parents=True, exist_ok=True)

    rev = pd.read_parquet(proc / "interactions.parquet")
    n_users = int(rev["user_idx"].max()) + 1
    n_items = int(rev["item_idx"].max()) + 1
    n_nodes = n_users + n_items

    edges_all = interactions_to_edge_table(rev, n_users)
    edges_train = train_only_edges(edges_all)

    out: dict[str, Path] = {}

    if gcfg.get("store_parquet", True):
        p_all = gdir / "edges_all.parquet"
        p_tr = gdir / "edges_train.parquet"
        edges_all.to_parquet(p_all, index=False)
        edges_train.to_parquet(p_tr, index=False)
        out["edges_all"] = p_all
        out["edges_train"] = p_tr

    if gcfg.get("store_npz", True):
        adj_train = edges_to_symmetric_biadj(edges_train, n_nodes)
        npz_path = gdir / "adj_train_symmetric.npz"
        sparse.save_npz(npz_path, adj_train)
        out["adj_train_symmetric"] = npz_path
        meta = {
            "n_users": n_users,
            "n_items": n_items,
            "n_nodes": n_nodes,
            "n_edges_train_undirected": int(adj_train.nnz // 2),
            "note": "CSR symmetric adjacency; global ids: user in [0,n_users-1], item in [n_users, n_nodes-1]",
        }
        (gdir / "graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        out["graph_meta"] = gdir / "graph_meta.json"

    if gcfg.get("store_sqlite", True):
        db = gdir / "graph.sqlite"
        if db.exists():
            db.unlink()
        con = sqlite3.connect(db)
        edges_all.to_sql("edges_all", con, index=False)
        edges_train.to_sql("edges_train", con, index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_train_src ON edges_train(src)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_train_dst ON edges_train(dst)")
        con.commit()
        con.close()
        out["sqlite"] = db

    if gcfg.get("export_item_item_copurchase", False):
        item_mm = pd.read_parquet(proc / "item_multimodal.parquet")
        with open(proc / "item_id_map.json", encoding="utf-8") as f:
            asin_to_local = json.load(f)
        valid_asins = set(asin_to_local.keys())
        def _as_list(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return []
            if isinstance(x, (list, tuple)):
                return list(x)
            if hasattr(x, "tolist"):
                return [t for t in x.tolist() if t is not None]
            return []

        rows = []
        for _, r in item_mm.iterrows():
            a = r["asin"]
            if a not in valid_asins:
                continue
            for b in _as_list(r["also_buy"]):
                if b in valid_asins:
                    ia, ib = asin_to_local[a], asin_to_local[b]
                    if ia != ib:
                        rows.append((ia, ib))
        if rows:
            ii = pd.DataFrame(rows, columns=["item_idx_a", "item_idx_b"]).drop_duplicates()
            ii["src_global"] = ii["item_idx_a"].astype(np.int64) + n_users
            ii["dst_global"] = ii["item_idx_b"].astype(np.int64) + n_users
            p_ii = gdir / "edges_item_copurchase.parquet"
            ii.to_parquet(p_ii, index=False)
            out["edges_item_copurchase"] = p_ii

    return out
