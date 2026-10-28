import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_state_conditioned_suffix_tensor.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_state_conditioned_suffix_tensor.json"
MODEL = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_state_conditioned_suffix_tensor.pt"


def test_state_conditioned_tensor_is_locked_and_disjoint():
    config = json.loads(CONFIG.read_text())
    assert config["train_sites"] + config["validation_sites"] == 768
    assert config["training_seeds"] == [11, 29, 47]
    assert config["query_dimension"] == 256
    assert config["state_dimension"] == 64
    assert config["cp_rank"] == 64


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
    assert MODEL.exists()
