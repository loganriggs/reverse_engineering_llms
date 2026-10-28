import json
from pathlib import Path


def test_fixed_counterfactuals_hold_position_and_distance_constant():
    config = json.loads(Path("configs/bilin18_layer8_semantic_counterfactuals_fixed.json").read_text())
    assert config["causal_position"] - config["edit_position"] == 16
    assert config["sequence_length"] == config["causal_position"] + 1
    assert config["high_effect_contexts_per_pair"] == 2
    assert Path(config["baseline_results"]).exists()
