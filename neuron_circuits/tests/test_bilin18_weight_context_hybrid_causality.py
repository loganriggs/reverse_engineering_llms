import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_weight_context_hybrid_causality.py"
spec = importlib.util.spec_from_file_location("hybrid_causality", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_missing_semantic_centers_yield_zero():
    routes = torch.tensor([[[1.0, -2.0]]])
    assert torch.count_nonzero(module.semantic_reconstruction(routes, None, .7)) == 0
