from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from run_bilin18_layer01_semantic_source_factorial import (  # noqa: E402
    shapley,
    write_at_position,
    write_through_position,
)


def test_shapley_recomposes_interacting_two_source_function():
    outputs={(0,0):torch.tensor([0.]),(1,0):torch.tensor([1.]),(0,1):torch.tensor([2.]),(1,1):torch.tensor([5.])}
    values=shapley(outputs)
    assert torch.allclose(sum(values),torch.tensor([5.]))
    assert torch.allclose(values[0],torch.tensor([2.]))
    assert torch.allclose(values[1],torch.tensor([3.]))


def test_write_at_position_ignores_larger_change_at_edited_token():
    delta=torch.zeros(1,4,3)
    delta[0,1]=torch.tensor([100.,200.,300.])
    delta[0,2]=torch.tensor([1.,2.,3.])
    decoder=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
    assert torch.equal(write_at_position(delta,decoder,2),torch.tensor([[1.,2.]]))
    assert torch.equal(
        write_at_position(delta,decoder,2,torch.tensor([1])),
        torch.tensor([[0.,2.]]),
    )
    assert torch.equal(
        write_through_position(delta,decoder,2),
        torch.tensor([[[0.,0.],[100.,200.],[1.,2.]]]),
    )
