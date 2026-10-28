import json
from pathlib import Path


def test_polynomial_split_is_exact_contextual_and_family_ranked():
    config = json.loads(Path("configs/bilin18_layer3_input_polynomial_split.json").read_text())
    assert config["scales"] == [0.1, 1.0]
    assert config["input_rank_by_family"]["increasing_scale"] == 256
    source = Path("scripts/run_bilin18_layer3_input_polynomial_split.py").read_text()
    assert "linear_write + quadratic_write" in source
    assert "frozen_context" in source
    assert "exact_error" in source
    assert "evaluate_writes" in source
