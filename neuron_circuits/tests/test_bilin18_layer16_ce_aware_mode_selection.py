import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_ce_aware_mode_selection.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_ce_aware_mode_selection.json"


def test_ce_aware_selection_is_fixed_and_not_oracle():
    config = json.loads(CONFIG.read_text())
    assert config["operator_modes"] == config["budgets"][-1] == 256
    assert config["selection_discovery"]["start"] != config["evaluation"]["fineweb"]["start"]
    assert config["random_replicates"] >= 64


def test_output_matches_hypotheses_if_present():
    if not OUTPUT.exists(): return
    config = json.loads(CONFIG.read_text()); payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
    assert len(payload["ce_aware_mode_order"]) == config["operator_modes"]
