import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path("scripts").resolve()))
from run_bilin12_cleaning_fourth_moment_redteam import rank_scores  # noqa: E402


def test_full_rank_quadratic_score_reconstruction():
    generator = torch.Generator().manual_seed(19)
    coordinates = torch.randn(100, 7, generator=generator)
    matrix = torch.randn(7, 7, generator=generator)
    matrix = 0.5 * (matrix + matrix.T)
    rows, energy = rank_scores(coordinates, matrix, [2, 7])
    assert rows[-1]["r2"] > .999999
    assert energy[-1] > .999999


def test_fourth_moment_redteam_has_gaussian_and_natural_controls():
    source = Path("scripts/run_bilin12_cleaning_fourth_moment_redteam.py").read_text()
    for required in ("moment_matched_gaussian", "natural_tail_amplification4", "matrix_energy", "residual_variance_fraction"):
        assert required in source
