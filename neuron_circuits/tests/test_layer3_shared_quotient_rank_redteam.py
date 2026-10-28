import json
from pathlib import Path


def test_rank_redteam_sweeps_cut_and_family_width():
    config = json.loads(Path("configs/bilin18_layer3_shared_quotient_rank_redteam.json").read_text())
    source = Path("scripts/run_bilin18_layer3_shared_quotient_rank_redteam.py").read_text()
    assert "locked" in config["status"]
    assert config["shared_ranks"] == [4, 8, 12, 16]
    assert config["affinity_ranks"] == [16, 32, 64]
    assert len(config["candidate_cluster"]) == 3
    assert len(config["hypotheses"]) >= 8
    for required in (
        "candidate_cluster", "within_cluster", "outside_cluster", "cross_term",
        "shared_like", "family_residual", "evaluate_writes", "eigenvalue_ratio",
    ):
        assert required in source


def test_rank_redteam_uses_both_corpora_and_all_families():
    source = Path("scripts/run_bilin18_layer3_shared_quotient_rank_redteam.py").read_text()
    assert "fineweb" in source
    assert "tinystories" in source
    assert "family_names" in source
