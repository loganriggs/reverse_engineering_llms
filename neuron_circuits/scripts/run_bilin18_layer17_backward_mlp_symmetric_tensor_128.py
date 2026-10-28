#!/usr/bin/env python3
"""Rank-128 scaling red-team for the symmetric final-MLP tensor quotient."""

from pathlib import Path

from run_bilin18_layer17_backward_mlp_symmetric_tensor import main


if __name__ == "__main__":
    main(
        config_path=Path("configs/bilin18_layer17_backward_mlp_symmetric_tensor_128.json"),
        output_path=Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_backward_mlp_symmetric_tensor_128.json"),
        report_path=Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_BACKWARD_MLP_SYMMETRIC_TENSOR_128.md"),
        tensor_output=Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_backward_mlp_symmetric_tensor_128.pt"),
    )
