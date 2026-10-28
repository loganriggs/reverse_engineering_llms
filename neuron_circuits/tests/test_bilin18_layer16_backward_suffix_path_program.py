import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_backward_suffix_path_program.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_backward_suffix_path_program.json"
PATHS = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_backward_suffix_paths.pt"


def test_backward_path_program_uses_disjoint_discovery_and_evaluation():
    config = json.loads(CONFIG.read_text())
    discovery = config["path_discovery"]
    fineweb = config["evaluation"]["fineweb"]
    assert discovery["start"] + discovery["rows"] <= fineweb["start"]
    assert config["layer"] == 16
    assert config["final_layer"] == 17
    assert config["fit_components"] == ["prediction", "target", "margin"]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
    assert PATHS.exists()
