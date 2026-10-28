import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_repeated_embedding_weight_jvp_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_repeated_embedding_weight_jvp_redteam.json"


def test_weight_jvp_redteam_is_locked():
    config = json.loads(CONFIG.read_text())
    assert config["status"].startswith("hypotheses-locked")
    assert set(config["position_groups"]["original"]).isdisjoint(config["position_groups"]["unseen"])
    assert config["donor_shifts"] == [1, 2, 3]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
