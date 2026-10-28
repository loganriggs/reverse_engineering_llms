import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))

from run_bilin12_attention_relay_topology_audit import group_worlds,vector_shapley


def test_group_worlds_and_generic_shapley_recompose():
    base=torch.randn(2,5);names=["a","b","c"];deltas={name:torch.randn(2,5) for name in names};worlds=group_worlds(base,deltas,names);effects=vector_shapley(names,worlds)
    assert torch.allclose(worlds[-1]-worlds[0],sum(deltas.values()),atol=1e-6)
    assert torch.allclose(sum(effects.values()),worlds[-1]-worlds[0],atol=1e-6)
