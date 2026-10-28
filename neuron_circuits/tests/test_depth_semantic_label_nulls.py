import json
from pathlib import Path


def test_depth_semantic_label_nulls_preserve_natural_writes():
    config = json.loads(Path(
        "configs/bilin18_depth_semantic_label_nulls.json"
    ).read_text())
    source = Path(
        "scripts/run_bilin18_depth_semantic_label_nulls.py"
    ).read_text()
    assert "locked" in config["status"]
    assert config["label_shuffle_replicates"] >= 64
    for required in (
        "pair_writes", "group_sizes", "permutation", "label_shuffle",
        "natural-write", "strict_axes", "pairwise_overlap", "empirical_p",
    ):
        assert required in source
