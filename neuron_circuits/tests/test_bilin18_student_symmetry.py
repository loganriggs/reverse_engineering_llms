from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_bilin18_final_mlp_operator import q_inner  # noqa: E402
from train_bilin18_final_mlp_student import NarrowBilinear  # noqa: E402


def test_symmetrized_factor_inner_matches_explicit_matrices():
    torch.manual_seed(0)
    left_a, right_a = torch.randn(3, 5), torch.randn(3, 5)
    left_b, right_b = torch.randn(4, 5), torch.randn(4, 5)
    explicit = torch.empty(3, 4)
    for i in range(3):
        qa = (torch.outer(left_a[i], right_a[i]) +
              torch.outer(right_a[i], left_a[i])) / 2
        for j in range(4):
            qb = (torch.outer(left_b[j], right_b[j]) +
                  torch.outer(right_b[j], left_b[j])) / 2
            explicit[i, j] = (qa * qb).sum()
    torch.testing.assert_close(q_inner(left_a, right_a, left_b, right_b), explicit)


def test_narrow_bilinear_forward_respects_factor_gauges():
    torch.manual_seed(1)
    layer = NarrowBilinear(7, 5)
    torch.nn.init.normal_(layer.left)
    torch.nn.init.normal_(layer.right)
    torch.nn.init.normal_(layer.down)
    x = torch.randn(11, 7)
    expected = layer(x)

    layer.left.data.mul_(3.0)
    layer.right.data.mul_(0.25)
    torch.testing.assert_close(layer(x), expected)

    left = layer.left.data.clone()
    layer.left.data.copy_(layer.right.data)
    layer.right.data.copy_(left)
    torch.testing.assert_close(layer(x), expected)

    layer.left.data.mul_(-1)
    layer.down.data.mul_(-1)
    torch.testing.assert_close(layer(x), expected)
