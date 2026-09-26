"""Make the hw1 modules importable when pytest is run from repository root."""

from __future__ import annotations

import sys
from pathlib import Path


HW1_ROOT = Path(__file__).resolve().parents[1]
if str(HW1_ROOT) not in sys.path:
    sys.path.insert(0, str(HW1_ROOT))
