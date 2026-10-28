import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_depth_natural_spectrum import gaussian_output_covariance  # noqa: E402


def test_gaussian_moment_covariance_matches_monte_carlo():
    generator = torch.Generator().manual_seed(29)
    output, channels, dimension = 5, 8, 4
    down = torch.randn(output, channels, generator=generator, dtype=torch.float64)
    left = torch.randn(channels, dimension, generator=generator, dtype=torch.float64)
    right = torch.randn(channels, dimension, generator=generator, dtype=torch.float64)
    mean = torch.randn(dimension, generator=generator, dtype=torch.float64)
    factor = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64) / 2
    expected, _ = gaussian_output_covariance(down, left, right, mean, factor)
    z = torch.randn(300_000, dimension, generator=generator, dtype=torch.float64)
    x = mean + z @ factor.T
    write = (left @ x.T) * (right @ x.T)
    write = (down @ write).T
    write = write - write.mean(0, keepdim=True)
    actual = write.T @ write / len(write)
    assert torch.allclose(actual, expected, rtol=0.025, atol=0.025 * expected.abs().max())
