import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer17_context_operator_atlas.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_context_operator_atlas.json"


def test_context_operator_atlas_is_locked_and_disjoint():
    config = json.loads(CONFIG.read_text())
    assert config["rank"] == 16
    assert config["context_budgets"][-1] == 16
    assert config["rotation_replicates"] == 32
    assert config["evaluation"]["fineweb"]["start"] + config["evaluation"]["fineweb"]["rows"] <= config["discovery"]["start"]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
