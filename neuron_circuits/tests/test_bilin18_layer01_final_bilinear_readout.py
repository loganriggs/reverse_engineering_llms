from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from run_bilin18_layer01_final_bilinear_readout import bilinear_terms  # noqa: E402


class MLP:
    Left=torch.nn.Linear(3,4,bias=False)
    Right=torch.nn.Linear(3,4,bias=False)


class Block:
    mlp=MLP()


def test_product_rule_terms_recompose_finite_bilinear_change():
    torch.manual_seed(2); block=Block(); base=torch.randn(3); counter=torch.randn(3); terms=bilinear_terms(block,base,counter)
    expected=block.mlp.Left(counter)*block.mlp.Right(counter)-block.mlp.Left(base)*block.mlp.Right(base)
    assert torch.allclose(sum(terms.values()),expected,atol=1e-6)
