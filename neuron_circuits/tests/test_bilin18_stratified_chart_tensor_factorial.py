import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_stratified_chart_tensor_factorial.py"
spec = importlib.util.spec_from_file_location("stratified_chart_tensor_factorial", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_align_basis_resolves_rotation():
    reference = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    other = np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    aligned = module.align_basis(reference, other)
    assert np.allclose(aligned, reference)


def test_factorized_transform_reconstructs_projector():
    routes = torch.tensor([[1.0, 2.0, 3.0]])
    basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert torch.allclose(module.factorized_transform(routes, basis, basis), torch.tensor([[1.0, 2.0, 0.0]]))
