import json
from pathlib import Path


def test_conditioned_axis_causality_is_locked_and_heldout():
    config = json.loads(Path(
        "configs/bilin18_layer3_conditioned_axis_causality.json"
    ).read_text())
    source = Path(
        "scripts/run_bilin18_layer3_conditioned_axis_causality.py"
    ).read_text()
    assert "locked" in config["status"]
    assert config["contexts_per_pair"] >= 2
    assert 64 in config["spectral_ranks"]
    for required in (
        "canonical_basis", "wrong_family", "random_control", "omit_axis",
        "spectral_64", "pair_balanced", "axis_necessity", "spearman",
        "exact_recomposition_error",
    ):
        assert required in source
