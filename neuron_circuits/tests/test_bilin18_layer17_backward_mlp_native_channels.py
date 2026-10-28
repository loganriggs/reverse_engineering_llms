import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer17_backward_mlp_native_channels.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_backward_mlp_native_channels.json"


def test_final_mlp_native_channel_audit_is_disjoint_and_locked():
    config = json.loads(CONFIG.read_text())
    discovery = config["selection_discovery"]
    fineweb = config["evaluation"]["fineweb"]
    assert discovery["start"] + discovery["rows"] <= fineweb["start"]
    assert config["budgets"][-1] == 4608
    assert config["fit_components"] == ["prediction", "target", "margin"]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
