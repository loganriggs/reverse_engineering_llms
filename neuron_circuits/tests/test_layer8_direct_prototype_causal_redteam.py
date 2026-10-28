import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_bilin18_layer8_direct_prototype_causal_redteam import row_subspace_overlap  # noqa: E402


def test_row_subspace_overlap_ignores_internal_rotation():
    generator = torch.Generator().manual_seed(3)
    rows = torch.randn(4, 20, generator=generator)
    rotation = torch.linalg.qr(torch.randn(4, 4, generator=generator)).Q
    assert row_subspace_overlap(rows, rotation @ rows) > 0.999


def test_direct_redteam_has_native_suffix_and_random_controls():
    source = Path("scripts/run_bilin18_layer8_direct_prototype_causal_redteam.py").read_text()
    for required in ("random_oracle_capture", "finite_mlp_overlap", "finite_suffix_overlap", "finite_energy_ratio", "routed_capture"):
        assert required in source
