import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_bilin18_natural_operator_recursive_generalization.py"
spec = importlib.util.spec_from_file_location("natural_operator_recursive_generalization", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_predicate_rate_uses_all_output_hints():
    role = {"examples": [{"output_positive": [" is", " dog"]}, {"output_positive": [" was", " cat"]}]}
    assert module.predicate_rate(role, {"is", "was"}) == .5
