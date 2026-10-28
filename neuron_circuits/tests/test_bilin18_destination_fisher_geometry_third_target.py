import json
from pathlib import Path


def test_third_fisher_target_is_unrelated_and_scattered():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/"configs/bilin18_destination_fisher_geometry_third_target.json").read_text())
    assert config["target"]==[4,1]
    assert config["partner"]==[5,8]
    assert "evaluation_random_seed" in config
