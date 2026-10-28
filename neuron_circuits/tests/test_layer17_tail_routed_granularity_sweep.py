import json
from pathlib import Path


def test_granularity_sweep_is_locked_and_compute_matched():
    config = json.loads(Path(
        "configs/bilin18_layer17_tail_routed_granularity_sweep.json"
    ).read_text())
    source = Path("scripts/run_bilin18_layer17_tail_routed_granularity_sweep.py").read_text()
    assert "locked" in config["status"]
    assert all(row["experts"] * row["rank"] == config["total_expert_rank"] for row in config["settings"])
    assert [(row["experts"], row["rank"]) for row in config["settings"]] == [
        (1, 32), (2, 16), (4, 8), (8, 4), (16, 2)
    ]
    for required in ("geometric_mean", "opposite_half_fisher", "hard", "soft", "parameter_count"):
        assert required in source
