import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_ce_residual_incremental_causality.py"
spec = importlib.util.spec_from_file_location("ce_residual_incremental", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_random_complement_is_orthogonal_to_generic():
    generic = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = module.random_complement(generic, 1, np.random.default_rng(3))
    assert np.allclose(result @ generic.T, 0, atol=1e-7)
    assert np.allclose(result @ result.T, 1, atol=1e-7)


def test_projection_exact_on_basis():
    values = torch.tensor([[[2.0, 0.0]]]); basis = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(module.project(values, basis), values)


def test_missing_known_semantic_modes_yield_zero_component():
    values = torch.tensor([[[2.0, -1.0]]])
    assert torch.count_nonzero(module.semantic_reconstruction(values, None, .7)) == 0
