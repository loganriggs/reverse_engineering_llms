from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from run_bilin18_layer01_dynamic_neuron_roles import role_statistics  # noqa: E402


def test_role_statistics_are_scale_invariant_and_detect_cancellation():
    left=torch.tensor([8.,1.]); right=torch.tensor([2.,-1.]); quadratic=torch.tensor([0.,.1])
    dominance,side,share,cancellation=role_statistics(left,right,quadratic)
    assert torch.allclose(dominance,torch.tensor([.8,.5]))
    assert side.tolist()==[True,True]
    assert cancellation.tolist()==[False,True]
    assert share[0]==0
