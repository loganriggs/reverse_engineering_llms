import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_hierarchical_router import projection_multiplies  # noqa: E402


def test_hierarchical_projection_count_reuses_coarse_work():
    actual = projection_multiplies(queries=16, modes=512, coarse=64, refine=256, shortlist=64)
    expected = 16 * (512 * 64 + 64 * 192)
    assert actual == expected
    assert actual / (16 * 512 * 256) < 0.35
