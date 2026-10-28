import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_l17_semantic_mean_binding.py"
spec = importlib.util.spec_from_file_location("l17_semantic_binding", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_rank_order_is_increasing():
    assert module.rank_order({"b": 2, "a": 1, "c": 3}) == ["a", "b", "c"]
