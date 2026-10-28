import json
from pathlib import Path


def test_rank_frontier_is_locked_and_reports_behavioral_metrics():
    config = json.loads(Path(
        "configs/bilin18_layer17_tail_continuous_rank_frontier.json"
    ).read_text())
    source = Path(
        "scripts/run_bilin18_layer17_tail_continuous_rank_frontier.py"
    ).read_text()
    assert "locked" in config["status"]
    assert config["ranks"] == [2, 4, 6, 8, 10, 12, 16, 24, 32]
    assert config["optimization_steps"] >= 600
    for required in (
        "median_exact_kl", "mean_exact_kl", "ce_delta", "top1_agreement",
        "smallest_universal_gate", "opposite_half_fisher",
    ):
        assert required in source
