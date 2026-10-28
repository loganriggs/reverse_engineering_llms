import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_stratified_route_finite_causality.py"
spec = importlib.util.spec_from_file_location("stratified_route_finite", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_projection_is_exact_in_full_basis():
    values = torch.tensor([[1.0, 2.0]])
    assert torch.allclose(module.project(values, torch.eye(2)), values)


def test_runtime_distance_mask():
    raw = np.arange(10)[None, :]
    classes = np.asarray(["other"] * 20)
    mask = module.runtime_mask("distance:local1-4", raw, 8, np.asarray([1.0]), classes, np.asarray([1.0]))
    assert mask.shape == (1, 9)
    assert mask.sum() == 4


def test_runtime_loss_mask_uses_discovery_quartiles():
    raw = np.arange(8)[None, :]
    classes = np.asarray(["other"] * 20)
    discovery = np.arange(8, dtype=float)
    mask = module.runtime_mask("loss:hard", raw, 6, np.asarray([8.0]), classes, discovery)
    assert mask.all()
