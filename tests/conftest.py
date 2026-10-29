"""Make the modules in src/ importable in tests (as run_all.py does)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
