import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_unseeded_route_end_to_end_causality.py"
spec = importlib.util.spec_from_file_location("route_e2e", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_reconstruction_helpers():
    routes = torch.tensor([[[2.0, 0.0]]])
    eye = torch.eye(2)
    assert torch.allclose(module.reconstruct_pca(routes, eye), routes)
    assert torch.allclose(module.reconstruct_centroid(routes, eye), routes)
    assert torch.allclose(module.reconstruct_gated(routes, eye[:1], .9), routes)


def test_effect_metric_is_exact_for_native():
    native = torch.tensor([[1.0, 2.0]])
    deleted = torch.zeros_like(native)
    metric = module.empty_metric()
    module.update_metric(metric, native, deleted, native, torch.tensor([1]))
    result = module.finish(metric)
    assert np.isclose(result["causal_fidelity"], 1.0)
    assert np.isclose(result["causal_cosine"], 1.0)
    assert np.isclose(result["ce_effect_fidelity"], 1.0)
    assert np.isclose(result["ce_effect_cosine"], 1.0)
