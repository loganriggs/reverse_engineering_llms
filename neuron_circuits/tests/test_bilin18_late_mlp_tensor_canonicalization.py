import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_late_mlp_tensor_canonicalization.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_late_mlp_tensor_canonicalization.json"


def test_canonicalization_is_nested_inside_fixed_atlas():
    config = json.loads(CONFIG.read_text())
    assert config["operator_modes"] == 64
    assert config["inner_ranks"][-1] == config["rank"] == 32
    assert config["layers"] == [13, 14, 15, 16]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
