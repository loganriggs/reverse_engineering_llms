from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from run_bilin18_layer01_final_attention_routes import project_head_edges,spearman  # noqa: E402


class Attention:
    n_head=2
    head_dim=2
    c_proj=torch.nn.Linear(4,3,bias=False)


class Block:
    attn=Attention()


def test_project_head_edges_matches_manual_head_slices():
    block=Block(); block.attn.c_proj.weight.data.copy_(torch.arange(12,dtype=torch.float32).reshape(3,4))
    values=torch.arange(12,dtype=torch.float32).reshape(2,3,2)
    result=project_head_edges(block,values)
    assert torch.allclose(result[0],values[0]@block.attn.c_proj.weight[:,:2].T)
    assert torch.allclose(result[1],values[1]@block.attn.c_proj.weight[:,2:].T)


def test_spearman_detects_order_and_reverse_order():
    values=torch.tensor([3.,1.,2.])
    assert abs(spearman(values,values)-1)<1e-6
    assert abs(spearman(values,-values)+1)<1e-6
