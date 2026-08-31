#!/usr/bin/env python3
"""CLI entrypoint for Phase 1 data preparation.

Usage (from repo root, with the project venv active):

    python scripts/prepare_phase1.py
    python scripts/prepare_phase1.py --force-synthetic
    python scripts/prepare_phase1.py --config configs/data.yaml -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prepare_phase1 import main

if __name__ == "__main__":
    main()
