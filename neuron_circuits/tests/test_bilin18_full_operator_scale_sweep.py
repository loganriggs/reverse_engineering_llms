import json
from pathlib import Path


def test_scale_sweep_has_local_and_finite_scales():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/"configs/bilin18_full_operator_scale_sweep.json").read_text())
    assert min(config["scales"])<=.05
    assert max(config["scales"])<1
