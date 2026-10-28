import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer17_tangent_bundle_core_stability.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_tangent_bundle_core_stability.json"


def test_core_stability_config_is_locked():
    config = json.loads(CONFIG.read_text())
    assert config["base_rank"] == 64
    assert config["core_ranks"] == [8, 16, 32, 64]
    assert len(config["splits"]) == 4


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
