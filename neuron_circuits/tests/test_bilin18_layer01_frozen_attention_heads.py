from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from run_bilin18_layer01_frozen_attention_heads import jaccard,top_mean  # noqa: E402


def test_top_mean_uses_average_importance_and_jaccard_is_set_based():
    rows=[torch.tensor([3.,0.,1.]),torch.tensor([1.,0.,3.])]
    assert set(top_mean(rows,2).tolist())=={0,2}
    assert jaccard([1,2,3],[2,3,4])==.5
