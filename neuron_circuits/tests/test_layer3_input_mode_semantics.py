import json
from pathlib import Path


def test_layer3_semantic_labeling_is_prospective_and_controlled():
    config = json.loads(Path("configs/bilin18_layer3_input_mode_semantics.json").read_text())
    source = Path("scripts/run_bilin18_layer3_input_mode_semantics.py").read_text()

    assert "locked" in config["status"]
    assert config["discovery_corpus"] != config["evaluation_corpus"]
    assert config["modes_per_family"] >= 8
    assert config["corpora"]["fineweb"]["rows"] >= 4096
    assert config["corpora"]["tinystories"]["rows"] >= 1200
    assert len(config["hypotheses"]) >= 8
    for required in (
        "input_metric", "token_flags", "random", "wrong", "generic",
        "embedding", "auc", "tail", "evaluation_corpus", "rotated",
    ):
        assert required in source


def test_layer3_semantic_labeling_keeps_signal_types_separate():
    source = Path("scripts/run_bilin18_layer3_input_mode_semantics.py").read_text()
    assert '"signed"' in source
    assert '"magnitude"' in source
    assert "minimum_positive_count" in source
    assert "selected_on" in source
