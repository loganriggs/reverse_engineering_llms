from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from run_bilin18_layer01_final_position_banks import (  # noqa: E402
    final_coordinate_scores,
    jaccard,
    median,
    sequence_coordinate_scores,
)


def test_final_score_excludes_other_positions_while_sequence_score_includes_them():
    delta=torch.tensor([[[2.,0.],[0.,3.]]])
    target=torch.tensor([[[2.,0.],[0.,3.]]])
    decoder=torch.eye(2)
    final=final_coordinate_scores(delta,decoder,target,1)
    sequence=sequence_coordinate_scores(delta,decoder,target,1)
    assert torch.allclose(final,torch.tensor([0.,1.]))
    assert torch.allclose(sequence,torch.tensor([4/13,9/13]))


def test_jaccard():
    assert jaccard(torch.tensor([1,2,3]),torch.tensor([2,3,4])) == 0.5
    assert median(list({"a":1.0,"b":3.0}.values())) == 2.0
