import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_behavior_subspace.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_behavior_subspace.json"
BASIS = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_behavior_subspaces.pt"


def test_behavior_subspace_uses_disjoint_selection_and_evaluation():
    config = json.loads(CONFIG.read_text())
    train = config["subspace_train"]; validation = config["subspace_validation"]
    assert train["start"] + train["rows"] <= validation["start"]
    assert validation["start"] + validation["rows"] <= config["evaluation"]["fineweb"]["start"]
    assert config["subspace_ranks"] == [32, 64, 128]


def test_output_matches_hypotheses_if_present():
    if not OUTPUT.exists(): return
    config = json.loads(CONFIG.read_text()); payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
    assert BASIS.exists()
