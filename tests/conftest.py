"""Pytest configuration — add src/ to sys.path and set working directory."""

import sys
from pathlib import Path

# Ensure src/ is on path for all tests
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
