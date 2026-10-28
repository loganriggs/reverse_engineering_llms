import json
from pathlib import Path


def test_unseen_causal_quotient_uses_disjoint_lexicon_and_finite_patches():
    config = json.loads(Path("configs/bilin18_layer3_unseen_causal_quotient.json").read_text())
    source = Path("scripts/run_bilin18_layer3_unseen_causal_quotient.py").read_text()
    assert "locked" in config["status"]
    assert config["contexts_per_pair"] >= 1
    assert config["minimum_records_per_family"] >= 6
    assert len(config["hypotheses"]) >= 8
    for required in (
        "unseen_config", "lexicon", "seed_ids", "overlap", "forward_layers",
        "evaluate_writes", "wrong_residual", "cross_term", "unique_source_tokens",
        "has_coverage",
    ):
        assert required in source


def test_unseen_causal_panel_has_fixed_positions_and_ood():
    config = json.loads(Path("configs/bilin18_layer3_unseen_causal_quotient.json").read_text())
    source = Path("scripts/run_bilin18_layer3_unseen_causal_quotient.py").read_text()
    assert config["edit_position"] < config["causal_position"] < config["sequence_length"]
    assert "fineweb" in source and "tinystories" in source


def test_context_coverage_sensitivity_is_labeled_posthoc():
    config = json.loads(Path(
        "configs/bilin18_layer3_unseen_causal_quotient_coverage.json"
    ).read_text())
    assert config["status"].startswith("posthoc-")
    assert config["contexts_per_pair"] > 1
    assert "not a new preregistered lexical test" in config["hypotheses_note"]
