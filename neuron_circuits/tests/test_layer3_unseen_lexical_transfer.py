import json
from pathlib import Path


def test_unseen_lexicon_is_locked_and_disjointly_evaluated():
    config = json.loads(Path("configs/bilin18_layer3_unseen_lexical_transfer.json").read_text())
    source = Path("scripts/run_bilin18_layer3_unseen_lexical_transfer.py").read_text()
    assert "locked" in config["status"]
    assert config["offsets"] == [0, 1, 2, 4]
    assert len(config["lexicon"]) == 5
    assert all(len(pole) >= 10 for family in config["lexicon"].values() for pole in family.values())
    assert len(config["hypotheses"]) >= 9
    for required in (
        "source_results", "representatives", "rotated", "random", "embedding",
        "wrong", "offset", "wikitext", "seed_ids",
    ):
        assert required in source


def test_unseen_lexicon_rejects_multitokens_and_seed_reuse():
    source = Path("scripts/run_bilin18_layer3_unseen_lexical_transfer.py").read_text()
    assert "len(encoded) != 1" in source
    assert "overlaps seed_ids" in source
