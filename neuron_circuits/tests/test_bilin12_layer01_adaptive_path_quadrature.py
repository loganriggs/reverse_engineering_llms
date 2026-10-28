import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_adaptive_quadrature_is_fineweb_first_and_locked():
    config = json.loads((ROOT / "configs/bilin12_layer01_adaptive_path_quadrature.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["corpus"] == "fineweb"
    assert config["orders"] == [4, 8, 16]
    assert config["sites"] >= 64
