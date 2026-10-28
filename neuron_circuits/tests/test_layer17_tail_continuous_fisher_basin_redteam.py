import json
from pathlib import Path


def test_basin_redteam_is_locked_and_uses_nonprefix_starts():
    config = json.loads(Path(
        "configs/bilin18_layer17_tail_continuous_fisher_basin_redteam.json"
    ).read_text())
    source = Path(
        "scripts/run_bilin18_layer17_tail_continuous_fisher_basin_redteam.py"
    ).read_text()
    assert "locked" in config["status"]
    assert config["optimization_steps"] >= 800
    initializations = [row["initialization"] for row in config["fits"]]
    assert initializations.count("random") >= 3
    assert "orthogonal_to_prefix" in initializations
    for required in (
        "canonical_overlap", "opposite_half", "exact_kl", "frobenius",
    ):
        assert required in source
    names = [row["name"] for row in config["fits"]]
    assert "random_even_a" in names
    assert "orthogonal_even" in names
