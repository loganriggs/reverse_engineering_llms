#!/usr/bin/env python3
"""Eight-direction replication of the empirical token-difference control."""

from pathlib import Path

from run_bilin12_layer01_chart_empirical_direction_control import main
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


if __name__ == "__main__":
    main(
        config_path=Path("configs/bilin12_layer01_chart_empirical_direction_replication.json"),
        output_path=TENSOR_ROOT / "bilin12_layer01_chart_empirical_direction_replication.json",
        report_path=TENSOR_ROOT / "BILIN12_LAYER01_CHART_EMPIRICAL_DIRECTION_REPLICATION.md",
        figure_path=TENSOR_ROOT / "bilin12_layer01_chart_empirical_direction_replication.png",
    )
