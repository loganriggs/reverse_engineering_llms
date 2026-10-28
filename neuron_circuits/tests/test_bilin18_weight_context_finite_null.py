import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_weight_context_finite_null.py"
spec = importlib.util.spec_from_file_location("weight_context_finite_null", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_uses_fresh_panel():
    import json
    config = json.loads((ROOT / "configs" / "bilin18_weight_context_finite_null.json").read_text())
    assert config["start_row"] >= 5120
