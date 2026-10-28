import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_mlp_path_symmetry_replication.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_mlp_path_symmetry_replication.json"


def test_suffix_specifies_four_consecutive_mlp_layers():
    config = json.loads(CONFIG.read_text())
    assert config["path_layers"] == list(range(config["layer"] + 1, config["final_layer"]))
    assert len(config["path_layers"]) == 4
    assert config["rank"] == 32


def test_output_matches_locked_config_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert payload["parameter_count"] > 500_000_000
    assert len(payload["group_order"]) == 16
