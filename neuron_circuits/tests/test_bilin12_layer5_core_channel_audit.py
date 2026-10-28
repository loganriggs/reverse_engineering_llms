import json
from pathlib import Path


def test_config_locks_language_ood_and_exact_causal_controls():
    config = json.loads(Path("configs/bilin12_layer5_core_channel_audit.json").read_text())
    assert "locked" in config["status"]
    assert config["core_channels"] == [196, 202]
    assert len(config["extended_channels"]) == 4
    assert set(config["semantic_evaluation"]) == {"fineweb", "wikitext", "tinystories"}
    assert set(config["causal_evaluation"]) == set(config["semantic_evaluation"])
    assert len(config["hypotheses"]) == 8


def test_script_recomputes_suffix_and_has_matched_controls():
    source = Path("scripts/run_bilin12_layer5_core_channel_audit.py").read_text()
    for required in (
        "run_suffix", "all_positions", "final_position", "matched_pair",
        "core_pair", "exact_kl", "signed_auc",
    ):
        assert required in source
