import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_bilin18_natural_operator_semantics.py"
spec = importlib.util.spec_from_file_location("natural_operator_semantic_validation", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_normalized_and_capitalized():
    assert module.normalized(" New-York ") == "newyork"
    assert module.capitalized(" Alberta")
    assert not module.capitalized(" alberta")


def test_role_profile_counts_predicates():
    role = {"examples": [{
        "next_token": " is", "distance": 1, "source_token": " Cat",
        "output_positive": [" Cat", " dog"], "source_equals_next": False,
    }]}
    profile = module.role_profile(role, {"is"})
    assert profile["auxiliary_rate"] == 1
    assert profile["local_rate"] == 1
    assert profile["source_echo_rate"] == 1
    assert profile["capitalized_output_fraction"] == .5
