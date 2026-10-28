import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer17_backward_mlp_symmetric_tensor.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_backward_mlp_symmetric_tensor.json"
TENSOR = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_backward_mlp_symmetric_tensor.pt"


def test_symmetric_tensor_quotient_is_disjoint_and_locked():
    config = json.loads(CONFIG.read_text())
    discovery = config["discovery"]
    fineweb = config["evaluation"]["fineweb"]
    assert discovery["start"] + discovery["rows"] <= fineweb["start"]
    assert config["input_dimension"] == config["output_dimension"] == 64
    assert config["mode_budgets"][-1] == 64


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
    assert TENSOR.exists()
