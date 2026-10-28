import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_unseeded_route_importance_coverage.py"
spec = importlib.util.spec_from_file_location("route_coverage", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_full_rank_projection_is_exact():
    routes = np.asarray([[1.0, -2.0, 3.0]], dtype=np.float32)
    assert np.allclose(module.project_subspace(routes, np.eye(3, dtype=np.float32)), routes)


def test_centroid_quantization_preserves_norm_and_exact_center():
    centers = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    routes = np.asarray([[0.0, 3.0]], dtype=np.float32)
    result = module.quantize_centroids(routes, centers)
    assert np.allclose(result, routes)


def test_gated_modes_zeroes_nonmembers():
    routes = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    reconstructed, selected = module.gated_modes(routes, [[1.0, 0.0]], .8)
    assert selected.tolist() == [True, False]
    assert np.allclose(reconstructed, [[1.0, 0.0], [0.0, 0.0]])


def test_contextual_edge_metric_detects_exact_reconstruction():
    routes = np.asarray([[[1.0, 2.0], [-1.0, 0.5]]])
    values = np.arange(12, dtype=np.float64).reshape(1, 2, 2, 3)
    stats = module.empty_stats()
    module.update_stats(stats, routes, routes.copy(), values, np.ones((1, 2), dtype=bool))
    result = module.finish(stats)
    assert np.isclose(result["route_fidelity"], 1.0)
    assert np.isclose(result["edge_fidelity"], 1.0)
    assert np.isclose(result["write_fidelity"], 1.0)
    assert np.isclose(result["assigned_edge_energy_fraction"], 1.0)


def test_quadratic_edge_metric_matches_explicit_vectors():
    rng = np.random.default_rng(7)
    routes = rng.normal(size=(5, 3))
    reconstructed = rng.normal(size=(5, 3))
    values = rng.normal(size=(5, 3, 4))
    grams = np.einsum("nik,njk->nij", values, values)
    explicit = module.empty_stats()
    module.update_stats(explicit, routes[:, None], reconstructed[:, None], values[:, None])
    quadratic = module.empty_stats()
    module.update_quadratic_edge_stats(quadratic, routes, reconstructed, grams)
    assert np.isclose(explicit["edge_target"], quadratic["edge_target"])
    assert np.isclose(explicit["edge_error"], quadratic["edge_error"])
    assert np.isclose(explicit["edge_dot"], quadratic["edge_dot"])
    assert np.isclose(explicit["edge_approx"], quadratic["edge_approx"])


def test_mode_count_fields_do_not_collide_with_method_names():
    source = SCRIPT.read_text()
    assert '"atlas_mode_count"' in source
    assert '"semantic_mode_count"' in source
    assert 'row = {"layer": layer, "atlas_modes"' not in source


def test_threshold_redteam_reports_model_wide_and_conditional_coverage():
    source = SCRIPT.read_text()
    assert '"model_wide_assigned_edge_energy_fraction"' in source
    assert '"conditional_layers"' in source
