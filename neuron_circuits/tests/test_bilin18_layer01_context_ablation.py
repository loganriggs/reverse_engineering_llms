import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_context_ablation_is_locked_to_source_census():
    config = json.loads((ROOT / "configs/bilin18_layer01_context_ablation.json").read_text())
    assert config["status"].startswith("locked-")
    source = json.loads((ROOT / config["source_config"]).read_text())
    assert len(source["features"]) == 50
    assert source["layers"] == [0, 1]

