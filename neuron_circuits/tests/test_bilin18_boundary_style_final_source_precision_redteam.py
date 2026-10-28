import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_precision_redteam_is_locked_after_original_failure():
    config=json.loads((ROOT/"configs/bilin18_boundary_style_final_source_precision_redteam.json").read_text())
    assert config["status"]=="locked-after-float32-recomposition-failure-before-float64-rerun"
    original=json.loads((ROOT/"artifacts/bilin18_tensor_decomposition/bilin18_boundary_style_final_source_factorial.json").read_text())
    assert not original["hypotheses"]["F0_final_source_factorial_recomposes"]["supported"]
    assert config["high_precision"] is True
