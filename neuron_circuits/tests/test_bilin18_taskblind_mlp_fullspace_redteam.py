import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_taskblind_mlp_fullspace.py"
spec = importlib.util.spec_from_file_location("taskblind_mlp_fullspace", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_centered_unit_uses_discovery_mean_and_normalizes():
    discovery = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    replication = np.asarray([[2.0, 0.0], [0.0, 2.0]])
    first, second = module.centered_unit(discovery, replication)
    assert np.allclose(np.linalg.norm(first, axis=1), 1)
    assert np.allclose(np.linalg.norm(second, axis=1), 1)


def test_cuda_kmeans_helper_shape_if_cuda_available():
    import torch
    if not torch.cuda.is_available():
        return
    rng = np.random.default_rng(4)
    values = np.r_[rng.normal(-1, .05, (12, 3)), rng.normal(1, .05, (12, 3))]
    assignment, centers, own, margin = module.spherical_kmeans_cuda(values, 2, 2, 20, 3)
    assert assignment.shape == (24,) and centers.shape == (2, 3)
    assert own.shape == margin.shape == (24,)
