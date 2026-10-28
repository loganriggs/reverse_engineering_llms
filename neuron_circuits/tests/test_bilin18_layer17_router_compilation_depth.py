import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer17_router_compilation_depth.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_router_compilation_depth.json"


def test_router_compilation_depth_is_locked_and_disjoint():
    config = json.loads(CONFIG.read_text())
    assert config["prediction_boundaries"] == list(range(17))
    assert config["causal_boundaries"][-1] == 16
    assert set(config["causal_positions"]).issubset(config["prediction_positions"])
    fineweb = config["prediction_evaluation"]["fineweb"]
    discovery = config["discovery"]
    assert fineweb["start"] + fineweb["rows"] <= discovery["start"]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
