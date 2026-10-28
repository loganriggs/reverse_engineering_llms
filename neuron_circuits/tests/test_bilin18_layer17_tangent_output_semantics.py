import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer17_tangent_output_semantics.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_tangent_output_semantics.json"


def test_output_semantics_config_is_locked_and_disjoint():
    config = json.loads(CONFIG.read_text())
    assert config["rank"] == 16
    assert config["rotation_replicates"] == 32
    discovery = config["discovery"]
    fineweb = config["evaluation"]["fineweb"]
    assert fineweb["start"] + fineweb["rows"] <= discovery["start"]
    assert len(config["positions"]) == 16


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
