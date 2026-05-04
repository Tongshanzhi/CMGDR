"""Entry point for text feature extraction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from text_extract import run_text_extract

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "shared_data"
    config = Path(__file__).resolve().parent / "config" / "text.yaml"
    summary = run_text_extract(root, config)
    import json
    print(json.dumps(summary, indent=2))
