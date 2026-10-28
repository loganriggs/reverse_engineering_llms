import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_direct_read_layouts.py"
spec = importlib.util.spec_from_file_location("direct_read_layouts", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_insert_before_destination_preserves_edits_and_moves_destination():
    pair = {"base_ids": [1, 2, 3], "source_ids": [1, 8, 3]}
    result = module.insert_before_destination(pair, [6, 7])
    assert result["base_ids"] == [1, 2, 6, 7, 3]
    assert result["source_ids"] == [1, 8, 6, 7, 3]


def test_cosine():
    assert abs(module.cosine([1, 0], [2, 0]) - 1) < 1e-12
