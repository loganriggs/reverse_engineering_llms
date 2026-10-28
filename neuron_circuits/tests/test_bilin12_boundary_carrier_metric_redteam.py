import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_boundary_carrier_metric_redteam import scalar_shapley


def test_scalar_shapley_recomposes_game_difference():
    values = [0.0, .1, .3, .5, .2, .7, .8, 1.0]
    result = scalar_shapley(values)
    assert abs(sum(result.values()) - 1.0) < 1e-12


def test_scalar_shapley_recovers_additive_game():
    contributions = [.2, .3, .5]
    values = [sum(contributions[i] for i in range(3) if mask & (1 << i)) for mask in range(8)]
    result = scalar_shapley(values)
    assert all(abs(result[factor] - contributions[index]) < 1e-12 for index, factor in enumerate(result))
