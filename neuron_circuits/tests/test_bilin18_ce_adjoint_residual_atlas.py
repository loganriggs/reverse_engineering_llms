import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_ce_adjoint_residual_atlas.py"
spec = importlib.util.spec_from_file_location("ce_residual_atlas", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_semantic_reconstruction_without_centers_is_zero():
    routes = torch.tensor([[[1.0, 2.0]]])
    result, selected = module.semantic_reconstruction(routes, None, .7)
    assert torch.count_nonzero(result) == 0
    assert not bool(selected.any())


def test_subspace_cosine_is_rotation_invariant():
    first = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    second = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    assert np.isclose(module.subspace_cosine(first, second), 1.0)


def test_reconstruction_fidelity_exact():
    values = np.asarray([[1.0, 2.0]])
    assert np.isclose(module.reconstruction_fidelity(values, values), 1.0)


def test_residual_forward_does_not_use_inference_attention_helper():
    source = SCRIPT.read_text()
    assert "differentiable_attention_parts" in source
    assert "from run_bilin18_qkov_causal_edges import attention_details" not in source
