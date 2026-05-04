"""End-to-end CLI: download -> extract -> cluster -> qa."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from image_download import run_download
from feature_extract import run_extract
from cluster import run_cluster
from visual_qa import run_qa


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual Modality Lead pipeline")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "shared_data")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--step", choices=["all", "download", "extract", "cluster", "qa"],
                        default="all")
    args = parser.parse_args()
    root = args.root
    cfg = args.config or root / "config" / "visual.yaml"

    if args.step in ("all", "download"):
        print("=" * 60)
        print("STEP 1/4: Download images")
        print("=" * 60)
        manifest = run_download(root, cfg)
        print(json.dumps({k: v for k, v in manifest.items() if k != "failed"}, indent=2))

    if args.step in ("all", "extract"):
        print("\n" + "=" * 60)
        print("STEP 2/4: Extract visual features")
        print("=" * 60)
        summary = run_extract(root, cfg)
        print(json.dumps(summary, indent=2))

    if args.step in ("all", "cluster"):
        print("\n" + "=" * 60)
        print("STEP 3/4: Visual prototype clustering")
        print("=" * 60)
        summary = run_cluster(root, cfg)
        print(json.dumps(summary, indent=2))

    if args.step in ("all", "qa"):
        print("\n" + "=" * 60)
        print("STEP 4/4: Quality assurance")
        print("=" * 60)
        report = run_qa(root, cfg)


if __name__ == "__main__":
    main()
