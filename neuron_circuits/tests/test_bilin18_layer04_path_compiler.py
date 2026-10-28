import sys
from pathlib import Path
from types import SimpleNamespace

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin18_layer04_path_compiler import product_from_pre, product_tangent


class Block:
    def __init__(self):
        self.mlp=SimpleNamespace(
            Left=torch.nn.Linear(5,7,bias=False,dtype=torch.float64),
            Right=torch.nn.Linear(5,7,bias=False,dtype=torch.float64),
        )


def test_product_tangent_matches_finite_difference():
    torch.manual_seed(0); block=Block(); pre=torch.randn(5,dtype=torch.float64); direction=torch.randn(5,dtype=torch.float64); epsilon=1e-6
    predicted=product_tangent(block,pre,direction,5)
    finite=(product_from_pre(block,pre+epsilon*direction,5)-product_from_pre(block,pre-epsilon*direction,5))/(2*epsilon)
    assert torch.allclose(predicted,finite,atol=1e-8,rtol=1e-7)
