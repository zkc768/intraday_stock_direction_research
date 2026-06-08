"""Compatibility wrapper for the Notebook 08 generator.

The canonical generator now lives at
``scripts/notebooks/generate_deep_sequence_exploration_colab.py``. This wrapper
preserves the legacy command path used by older notes and automation.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.notebooks.generate_deep_sequence_exploration_colab import main


if __name__ == "__main__":
    main()
