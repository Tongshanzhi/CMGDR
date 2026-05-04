"""End-to-end CLI: download -> gunzip -> preprocess -> graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from download import ensure_downloaded, gunzip_if_needed
from graph_builder import run_graph_build
from preprocess import run_preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Amazon Direction-1 data pipeline")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "shared_data")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--step", choices=["all", "download", "preprocess", "graph"], default="all")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    root: Path = args.root
    cfg = args.config or root / "config" / "default.yaml"

    if args.step in ("all", "download"):
        paths = ensure_downloaded(root, cfg, force=args.force_download)
        cat = paths["category"]
        raw_dir = paths["reviews_gz"].parent
        rev_json = raw_dir / f"reviews_{cat}_5.json"
        meta_json = raw_dir / f"meta_{cat}.json"
        gunzip_if_needed(paths["reviews_gz"], rev_json)
        gunzip_if_needed(paths["meta_gz"], meta_json)
        (root / "data" / "raw" / "manifest.json").write_text(
            json.dumps(
                {
                    "category": cat,
                    "reviews_json": str(rev_json),
                    "meta_json": str(meta_json),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if args.step in ("all", "preprocess"):
        man = json.loads((root / "data" / "raw" / "manifest.json").read_text(encoding="utf-8"))
        cat = man["category"]
        raw_dir = root / "data" / "raw"
        if man.get("reviews_json") and man.get("meta_json"):
            rev_json = Path(man["reviews_json"])
            meta_json = Path(man["meta_json"])
            if not rev_json.is_absolute():
                rev_json = root / rev_json
            if not meta_json.is_absolute():
                meta_json = root / meta_json
        else:
            rev_json = raw_dir / f"reviews_{cat}_5.json"
            meta_json = raw_dir / f"meta_{cat}.json"
        res = run_preprocess(root, cfg, rev_json, meta_json)
        print(json.dumps(res["stats"], ensure_ascii=False, indent=2))

    if args.step in ("all", "graph"):
        gout = run_graph_build(root, cfg)
        print("Graph artifacts:", json.dumps({k: str(v) for k, v in gout.items()}, indent=2))


if __name__ == "__main__":
    main()
