import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_ce_residual_pole_semantic_discovery.py"
spec = importlib.util.spec_from_file_location("ce_pole_semantics", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_profile_features_are_predeclared():
    assert "punctuation" in module.PROFILE_FEATURES
    assert "capitalized" in module.PROFILE_FEATURES


def test_replication_reuses_discovery_fitted_axes():
    source = SCRIPT.read_text()
    assert 'discovery_complement = normalized_complement(arrays[f"discovery_residual_routes_L{layer}"]' in source
    assert "novel_basis = fit_basis(discovery_complement" in source
