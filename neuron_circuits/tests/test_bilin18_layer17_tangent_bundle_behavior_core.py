import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer17_tangent_bundle_behavior_core.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_tangent_bundle_behavior_core.json"


def test_behavior_core_config_is_locked_and_disjoint():
    config = json.loads(CONFIG.read_text())
    assert config["rank"] == 16
    assert config["fit_sites"] == 192
    assert config["objectives"]["scores"] == ["prediction", "target"]
    assert "margin" in config["objectives"]["all"]
    assert len(config["optimization_seeds"]) == 3
    assert config["discovery"]["start"] + config["discovery"]["rows"] <= config["evaluation"]["fineweb"]["start"]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
