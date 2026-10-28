import json
from pathlib import Path


def test_behavioral_bottleneck_is_basis_invariant_and_decoded():
    config = json.loads(Path("configs/bilin18_layer8_behavioral_bottleneck.json").read_text())
    assert config["decoded_axes"] >= 4
    assert config["rotation_check_sites"] >= 2
    assert 4 in config["ranks"] and 16 in config["ranks"]
    assert set(config["evaluation"]) == {"fineweb", "wikitext"}
    assert "C8_unweighted_logit_axes_recur" in config["hypotheses"]
    assert "C9_input_verbs_recur" in config["hypotheses"]
