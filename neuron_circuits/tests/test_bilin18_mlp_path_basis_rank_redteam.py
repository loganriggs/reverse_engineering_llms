import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_mlp_path_basis_rank_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_mlp_path_basis_rank_redteam.json"


def test_basis_rank_design_is_factorial():
    config = json.loads(CONFIG.read_text())
    assert config["ranks"] == [32, 64]
    assert config["basis_types"] == ["pca", "random"]


def test_output_matches_locked_config_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert set(payload["results"]) == {"pca32", "pca64", "random32", "random64"}
