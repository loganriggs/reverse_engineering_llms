import json
from pathlib import Path


def test_shared_backbone_quotient_is_weight_defined_and_causal():
    config = json.loads(Path("configs/bilin18_layer3_shared_backbone_quotient.json").read_text())
    source = Path("scripts/run_bilin18_layer3_shared_backbone_quotient.py").read_text()
    assert "locked" in config["status"]
    assert config["family_rank"] == config["shared_rank"] + 24
    assert config["specific_rank"] == config["shared_rank"]
    assert len(config["hypotheses"]) >= 12
    for required in (
        "mean_projector", "shared_like", "family_residual", "cross_term",
        "wrong_residual", "evaluate_writes", "unseen", "random_24",
    ):
        assert required in source


def test_shared_backbone_has_natural_and_causal_controls():
    source = Path("scripts/run_bilin18_layer3_shared_backbone_quotient.py").read_text()
    for required in (
        "tinystories", "wikitext", "fineweb", "maximum_negative_examples",
        "seed_mask", "mean_positive", "mean_negative", "offset",
    ):
        assert required in source
