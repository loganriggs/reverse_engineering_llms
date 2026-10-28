#!/usr/bin/env python3
"""Held-out semantic panel for task-agnostic route modes."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bilin18_unseeded_route_semantic_discovery as discovery
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


discovery.CONFIG = Path("configs/bilin18_unseeded_route_semantic_heldout.json")
discovery.OUTPUT = TENSOR_ROOT / "bilin18_unseeded_route_semantic_heldout.json"
discovery.REPORT = TENSOR_ROOT / "BILIN18_UNSEEDED_ROUTE_SEMANTIC_HELDOUT.md"


if __name__ == "__main__":
    discovery.main()
