import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_occupied_outlier_ranking_is_locked_before_results():
    config = json.loads(
        (ROOT / "configs/bilin18_boundary_style_occupied_outlier_ranking.json").read_text()
    )
    assert config["status"].startswith("locked-")
    assert config["folds"] == 3
    assert config["batch_size"] <= 8

