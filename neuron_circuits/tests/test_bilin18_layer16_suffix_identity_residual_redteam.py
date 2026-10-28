import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_suffix_identity_residual_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_suffix_identity_residual_redteam.json"
MODEL = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_suffix_identity_residual_redteam.pt"


def test_identity_residual_redteam_extends_rank_and_preserves_split():
    config = json.loads(CONFIG.read_text())
    assert config["train_sites"] + config["validation_sites"] == 768
    assert config["subspace_ranks"][-2:] == [512, 768]
    assert config["fit_components"] == ["prediction", "target", "margin"]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
    assert MODEL.exists()
