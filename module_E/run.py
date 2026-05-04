"""Engineering optimization entry point.

Usage:
    python run.py optimization    # Phase 2 optimization
    python run.py phase3          # Phase 3 push
    python run.py phase4          # Phase 4 aggressive optimization
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here / "src"))
sys.path.insert(0, str(_here.parent / "module_B"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py [optimization|phase3|phase4]")
        sys.exit(1)

    phase = sys.argv[1]
    sys.argv = sys.argv[:1] + sys.argv[2:]  # Remove phase arg

    if phase == "optimization":
        from run_optimization import main
    elif phase == "phase3":
        from run_phase3 import main
    elif phase == "phase4":
        from run_phase4 import main
    else:
        print(f"Unknown phase: {phase}")
        sys.exit(1)

    main()
