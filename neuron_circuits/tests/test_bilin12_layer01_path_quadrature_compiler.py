import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin12_layer01_path_quadrature_compiler import gauss_integral


def test_quadrature_config_is_locked():
    config = json.loads((ROOT / "configs/bilin12_layer01_path_quadrature_compiler.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["quadrature_orders"] == [1, 2, 3, 4]
    assert config["sites_per_corpus"] >= 64


def test_four_node_gauss_integrates_degree_seven_derivative():
    base = torch.tensor([0.2, -0.4], dtype=torch.float64)
    direction = torch.tensor([0.3, 0.1], dtype=torch.float64)
    function = lambda value: value ** 8
    predicted = gauss_integral(function, base, direction, 4)
    exact = function(base + direction) - function(base)
    assert torch.allclose(predicted, exact, atol=1e-12)
