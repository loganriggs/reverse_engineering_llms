import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_forward_adjoint_atlas.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_forward_adjoint_atlas.json"
BASIS = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_forward_adjoint_bases.pt"


def test_forward_adjoint_atlas_has_shared_rank_budgets():
    config = json.loads(CONFIG.read_text())
    assert config["subspace_ranks"][-1] == 256
    assert config["rank"] == 32


def test_output_matches_hypotheses_if_present():
    if not OUTPUT.exists(): return
    config = json.loads(CONFIG.read_text()); payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
    assert BASIS.exists()
    assert set(payload["discovery_spectra"]) == {"forward", "state", "write_adjoint", "effective_adjoint"}
