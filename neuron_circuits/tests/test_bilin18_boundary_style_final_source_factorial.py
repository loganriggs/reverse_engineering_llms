import itertools
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_bilin18_boundary_style_final_source_factorial import FACTORS,shapley  # noqa: E402


def test_final_source_factorial_is_locked_before_results():
    config=json.loads((ROOT/"configs/bilin18_boundary_style_final_source_factorial.json").read_text())
    assert config["status"].startswith("locked-")
    assert len(config["hypotheses"])==6


def test_generic_final_source_shapley_recomposes():
    torch.manual_seed(9); weights=torch.randn(4,7); outputs={}
    for bits in itertools.product((0,1),repeat=4):
        value=sum(bits[i]*weights[i] for i in range(4)); outputs[bits]=torch.sin(value)+value.square()
    parts=shapley(outputs); torch.testing.assert_close(sum(parts.values()),outputs[(1,1,1,1)]-outputs[(0,0,0,0)])
    assert set(parts)==set(FACTORS)
