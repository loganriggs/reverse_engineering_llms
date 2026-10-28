from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from run_bilin18_layer01_final_position_bank_redteam import bootstrap_interval,jaccard,permutation_pvalue  # noqa: E402


def test_jaccard_and_bootstrap_are_deterministic():
    assert jaccard([1,2],[2,3])==1/3
    assert bootstrap_interval([1,1,1],100,np.random.default_rng(1))==[1.,1.,1.]


def test_permutation_detects_perfect_two_family_mapping():
    rows=[]
    for index in range(20):
        family="a" if index<10 else "b"
        rows.append({"family":family,"effects":{"a":1. if family=="a" else 0.,"b":1. if family=="b" else 0.},"advantage":1.,"top1":True})
    observed,pvalue,_=permutation_pvalue(rows,1000,np.random.default_rng(2))
    assert observed==1.
    assert pvalue<.01
