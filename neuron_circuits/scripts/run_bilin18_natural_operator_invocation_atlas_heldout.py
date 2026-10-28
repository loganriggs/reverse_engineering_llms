#!/usr/bin/env python3
"""Run the frozen natural-operator atlas on a disjoint FineWeb panel."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bilin18_natural_operator_invocation_atlas as atlas
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


atlas.CONFIG = Path("configs/bilin18_natural_operator_invocation_atlas_heldout.json")
atlas.OUTPUT = TENSOR_ROOT / "bilin18_natural_operator_invocation_atlas_heldout.json"
atlas.REPORT = TENSOR_ROOT / "BILIN18_NATURAL_OPERATOR_INVOCATION_ATLAS_HELDOUT.md"


if __name__ == "__main__":
    atlas.main()
