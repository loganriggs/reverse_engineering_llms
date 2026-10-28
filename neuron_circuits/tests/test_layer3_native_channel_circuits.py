import json
from pathlib import Path


def test_layer3_native_channel_test_has_causal_and_null_gates():
    config = json.loads(Path("configs/bilin18_layer3_native_channel_circuits.json").read_text())
    assert config["layer"] == 3
    assert config["recurrence_rank"] == 64
    assert 256 in config["ranks"]
    source = Path("scripts/run_bilin18_layer3_native_channel_circuits.py").read_text()
    assert "channel_scores" in source
    assert "full_mlp_patch" in source
    assert "same_pair_jaccard" in source
    assert "cross_family_null" in source
