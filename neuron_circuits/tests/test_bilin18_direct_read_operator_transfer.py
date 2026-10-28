import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_direct_read_operator_transfer.py"
spec = importlib.util.spec_from_file_location("direct_read_operator_transfer", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_edited_positions_excludes_destination():
    pair = {"base_ids": [1, 2, 3, 4], "source_ids": [1, 8, 3, 9]}
    assert module.edited_positions(pair) == [1]


def test_edited_positions_supports_repeated_payload_roles():
    pair = {"base_ids": [1, 2, 3, 2, 4], "source_ids": [1, 8, 3, 8, 4]}
    assert module.edited_positions(pair) == [1, 3]
