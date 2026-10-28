import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer17_tangent_bundle_split_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_tangent_bundle_split_redteam.json"


def test_split_redteam_is_locked_and_disjoint_from_external_fineweb():
    config = json.loads(CONFIG.read_text())
    assert config["rank"] == 64
    assert len(config["splits"]) == 4
    discovery = config["discovery"]
    external = config["evaluation"]["fineweb"]
    assert discovery["start"] + discovery["rows"] <= external["start"]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
