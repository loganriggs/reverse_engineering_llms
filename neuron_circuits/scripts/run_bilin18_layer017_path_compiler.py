#!/usr/bin/env python3
"""Run the unchanged path compiler assay through all 18 layers."""

from pathlib import Path

from run_bilin18_layer04_path_compiler import main
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


if __name__ == "__main__":
    main(
        Path("configs/bilin18_layer017_path_compiler.json"),
        TENSOR_ROOT / "layer017_path_compiler.json",
        TENSOR_ROOT / "LAYER017_PATH_COMPILER.md",
    )
