"""Entry point for model evaluation."""
import sys
from pathlib import Path
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here / "src"))
sys.path.insert(0, str(_here.parent / "module_B"))
from eval import main

if __name__ == "__main__":
    main()
