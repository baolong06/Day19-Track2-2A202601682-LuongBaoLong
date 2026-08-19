# noqa: F401
# Adds repo root to sys.path so notebooks can import app.*, scripts.*
from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
