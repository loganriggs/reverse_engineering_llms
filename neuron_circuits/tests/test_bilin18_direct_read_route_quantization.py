import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_direct_read_route_quantization.py"
spec = importlib.util.spec_from_file_location("direct_read_route_quantization", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_quantization_levels_and_signs():
    values = torch.tensor([-1.0, -0.2, 0.3, 0.9])
    result = module.quantize_symmetric(values, 2)
    assert set(result.tolist()) <= {-1.0, 0.0, 1.0}
    assert result[0] < 0 and result[-1] > 0


def test_eight_bit_quantization_is_close():
    values = torch.linspace(-1, 1, 31)
    result = module.quantize_symmetric(values, 8)
    assert float((result - values).abs().max()) <= 1 / 127 + 1e-6
