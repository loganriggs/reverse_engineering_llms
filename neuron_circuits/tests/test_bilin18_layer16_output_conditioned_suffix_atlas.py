import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_output_conditioned_suffix_atlas.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_output_conditioned_suffix_atlas.json"
MODEL = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_output_conditioned_suffix_atlas.pt"


def test_output_conditioned_atlas_has_disjoint_fit_and_validation():
    config = json.loads(CONFIG.read_text())
    assert config["train_sites"] == 576
    assert config["validation_sites"] == 192
    assert config["subspace_ranks"] == [32, 64, 128, 256]
    assert config["fit_components"] == ["prediction", "target", "margin"]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
    assert MODEL.exists()
