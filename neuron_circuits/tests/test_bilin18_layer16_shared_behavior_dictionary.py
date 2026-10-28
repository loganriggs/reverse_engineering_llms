import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_shared_behavior_dictionary.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_shared_behavior_dictionary.json"


def test_shared_dictionary_has_disjoint_train_validation_and_eval():
    config = json.loads(CONFIG.read_text())
    train = config["dictionary_train"]; validation = config["dictionary_validation"]
    assert train["start"] + train["rows"] <= validation["start"]
    assert validation["start"] + validation["rows"] <= config["evaluation"]["fineweb"]["start"]
    assert config["dictionary_atoms"] > config["rank"]


def test_output_matches_hypotheses_if_present():
    if not OUTPUT.exists(): return
    config = json.loads(CONFIG.read_text()); payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
    assert len(payload["dictionary_atoms"]) == config["dictionary_atoms"]
