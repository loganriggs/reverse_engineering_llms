import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_repeated_embedding_path_factorial.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_repeated_embedding_path_factorial.json"


def test_repeated_embedding_factorial_is_locked():
    config = json.loads(CONFIG.read_text())
    assert config["status"].startswith("hypotheses-locked")
    layers = []
    for start, stop in config["layer_groups"]:
        layers.extend(range(start, stop + 1))
    assert layers == list(range(18))
    assert config["positions"][-1] == 127


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
