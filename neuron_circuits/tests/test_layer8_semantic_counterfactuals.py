import json
from pathlib import Path


def test_semantic_counterfactuals_use_multiple_words_per_family_and_ood():
    config = json.loads(Path("configs/bilin18_layer8_semantic_counterfactuals.json").read_text())
    assert all(len(pairs) >= 3 for pairs in config["families"].values())
    assert config["contexts_per_pair"] >= 3
    assert set(config["evaluation"]) == {"fineweb", "tinystories"}
    assert config["axes"] == 4
