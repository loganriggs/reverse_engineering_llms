from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from run_bilin18_layer01_head_to_operand_redteam import permutation_shapley  # noqa: E402


class MLP:
    Left=torch.nn.Linear(4,5,bias=False).double()
    Right=torch.nn.Linear(4,5,bias=False).double()


class Block:
    mlp=MLP()


def test_permutation_shapley_recomposes_factor_endpoints():
    torch.manual_seed(3); base=torch.randn(4); components={"a":torch.randn(4),"b":torch.randn(4),"c":torch.randn(4)}; left,right,error=permutation_shapley(Block(),base,components,12,4)
    assert error<1e-10
    assert left.shape==right.shape==(3,5)
