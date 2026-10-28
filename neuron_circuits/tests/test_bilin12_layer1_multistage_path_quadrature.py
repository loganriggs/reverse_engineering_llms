import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multistage_quadrature_is_locked_and_prospective():
    config = json.loads((ROOT / "configs/bilin12_layer1_multistage_path_quadrature.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["corpus"] == "fineweb"
    assert config["layer"] == 1
    assert config["orders"] == [4, 8, 16, 32, 64]
    assert config["sites"] >= 64
    assert config["cosine_disagreement_threshold"] >= 0.99

