import json
from pathlib import Path


def test_semantic_subspace_rank_is_rotation_invariant_and_controlled():
    config = json.loads(Path("configs/bilin18_layer3_semantic_subspace_rank.json").read_text())
    source = Path("scripts/run_bilin18_layer3_semantic_subspace_rank.py").read_text()
    assert "locked" in config["status"]
    assert config["ranks"][-1] == 256
    assert config["evaluation_offsets"] == [0, 1]
    assert config["maximum_negative_examples"] >= 20000
    assert len(config["hypotheses"]) >= 8
    for required in (
        "input_metric", "family_subspace", "random_subspace", "wrong",
        "unseen", "seed_ids", "rank_80", "maximum_negative_examples",
    ):
        assert required in source


def test_probe_is_fit_only_on_seed_tokens_and_fineweb():
    source = Path("scripts/run_bilin18_layer3_semantic_subspace_rank.py").read_text()
    assert "discovery_corpus" in source
    assert "seed_mask" in source
    assert "mean_positive" in source
    assert "mean_negative" in source
