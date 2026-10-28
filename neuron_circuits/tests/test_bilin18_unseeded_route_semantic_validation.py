import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_bilin18_unseeded_route_semantics.py"
spec = importlib.util.spec_from_file_location("unseeded_route_semantic_validation", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_normalized():
    assert module.normalized(" New-York ") == "newyork"


def test_hypothesis_candidate_name_is_not_inferred_by_string_splitting():
    # Regression guard: semantic suffixes such as ``capitalized_output_mode``
    # are not candidate identifiers. The validator uses an explicit map.
    source = SCRIPT.read_text()
    assert '"S0_L17C1_capitalized_output_mode": "L17C1"' in source
    assert 'name.split("_", 2)' not in source
