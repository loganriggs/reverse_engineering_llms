import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_router_causal_validation import pearson, repeat_batch, route_indices  # noqa: E402


def test_route_indices_have_expected_shapes_and_fixed_support():
    torch.manual_seed(1)
    queries, dimension = 3, 12
    z = torch.randn(dimension)
    values = torch.randn(queries, dimension)
    vectors = torch.eye(dimension).repeat(queries, 1, 1)
    result = route_indices(z, values, vectors, 2, 3, 6, 4)
    assert set(result) == {"oracle", "flat256", "hierarchical128", "fixed16"}
    assert all(indices.shape == (queries, 2) for indices in result.values())
    assert torch.equal(result["fixed16"], torch.tensor([[0, 1], [0, 1], [0, 1]]))


def test_pearson_handles_identity_and_constant():
    assert abs(pearson([1, 2, 3], [2, 4, 6]) - 1) < 1e-10
    assert pearson([1, 1], [2, 3]) != pearson([1, 1], [2, 3])


def test_repeat_batch_only_repeats_leading_axis():
    value = torch.arange(6).reshape(1, 2, 3)
    repeated = repeat_batch(value, 4)
    assert repeated.shape == (4, 2, 3)
    assert torch.equal(repeated[3], value[0])
