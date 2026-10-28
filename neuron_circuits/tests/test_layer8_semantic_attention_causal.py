import json
from pathlib import Path


def test_attention_causal_has_matched_controls_and_two_directions():
    config = json.loads(Path("configs/bilin18_layer8_semantic_attention_causal.json").read_text())
    assert config["oracle_edges"] == 8
    assert config["contexts_per_pair"] >= 1
    assert Path(config["atlas_results"]).exists()
    source = Path("scripts/run_bilin18_layer8_semantic_attention_causal.py").read_text()
    assert "matched_random_edges" in source
    assert '"insertion"' in source
    assert '"removal"' in source
    assert "suffix_from_layer" in source
