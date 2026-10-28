import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_jacobian_causal_subspace.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_jacobian_causal_subspace.json"


def test_jacobian_causal_subspace_is_locked_and_disjoint():
    config = json.loads(CONFIG.read_text())
    assert config["sensitivity_ranks"] == [16, 32, 64, 128]
    assert config["edit_scales"] == [0.1, 1.0]
    assert config["rank"] == 16
    fineweb = config["evaluation"]["fineweb"]
    discovery = config["discovery"]
    assert fineweb["start"] + fineweb["rows"] <= discovery["start"]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
