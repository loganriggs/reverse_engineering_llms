import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_weight_outlier_experiment_is_locked_before_results():
    config = json.loads(
        (ROOT / "configs/bilin18_boundary_style_H2_weight_outlier.json").read_text()
    )
    assert config["status"].startswith("locked-")
    assert config["head"] == 2
    assert config["folds"] == 3

