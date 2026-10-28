import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_taskblind_mlp_multiview_atlas.py"
spec = importlib.util.spec_from_file_location("taskblind_mlp_multiview", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_canonical_channel_scale_is_function_invariant():
    down = np.asarray([[1.0, -2.0], [3.0, 1.0]])
    product = np.asarray([[2.0, 5.0]])
    scale = module.canonical_channel_scale(down)
    factors = np.asarray([-4.0, 3.0])
    transformed_product = product * factors
    transformed_down = down / factors
    transformed = module.canonical_channel_scale(transformed_down)
    assert np.allclose(product * scale, transformed_product * transformed)


def test_dual_pca_coordinates_capture_low_rank_signal():
    rng = np.random.default_rng(3)
    left = rng.normal(size=(20, 2)); right = rng.normal(size=(2, 8))
    discovery = left @ right; replication = rng.normal(size=(10, 2)) @ right
    _, _, _, energy = module.dual_pca_coordinates(discovery, replication, 2)
    assert energy["discovery"] > .999999 and energy["replication"] > .999999


def test_center_matching_recovers_permutation():
    first = np.eye(3); second = first[[2, 0, 1]]
    permutation, cosines = module.match_centers(first, second)
    assert permutation.tolist() == [1, 2, 0]
    assert np.allclose(cosines, 1)


def test_frozen_cluster_fidelity_is_exact_on_centers():
    centers = np.eye(2)
    values = centers[[0, 1, 0]]
    assert np.isclose(module.frozen_cluster_fidelity(values, centers), 1)
