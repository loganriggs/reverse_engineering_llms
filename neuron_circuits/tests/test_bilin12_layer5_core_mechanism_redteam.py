import json
from pathlib import Path


def test_locked_config_and_independent_slices():
    config = json.loads(Path("configs/bilin12_layer5_core_mechanism_redteam.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["core_channels"] == [196, 202]
    assert config["matched_channels"] == [1232, 1615]
    assert len(config["hypotheses"]) == 7
    for corpus in config["semantic_evaluation"]:
        semantic = config["semantic_evaluation"][corpus]
        substitution = config["substitution_evaluation"][corpus]
        causal = config["causal_evaluation"][corpus]
        assert semantic["start"] + semantic["rows"] <= substitution["start"]
        assert substitution["start"] + substitution["rows"] <= causal["start"]


def test_script_contains_required_redteams():
    source = Path("scripts/run_bilin12_layer5_core_mechanism_redteam.py").read_text()
    assert "factor_semantics" in source
    assert "newline_substitution" in source
    assert "synthetic_additive" in source
    assert "nonlinear_residual_ratio" in source
    assert "fisher_cosine" in source
    assert "activation_quartile_ratio" in source
