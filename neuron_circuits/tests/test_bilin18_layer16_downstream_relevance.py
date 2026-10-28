import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_downstream_relevance.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_downstream_relevance.json"


def test_downstream_test_targets_declared_final_boundary():
    config = json.loads(CONFIG.read_text())
    assert config["target_layer"] == 16
    assert config["final_layer"] == 17
    assert config["endpoint_spectral_budgets"][-1] == config["operator_modes"] == 64


def test_output_matches_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text()); payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
