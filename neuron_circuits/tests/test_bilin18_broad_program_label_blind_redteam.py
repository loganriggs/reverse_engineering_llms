import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_broad_program_label_blind_prediction.py"
spec = importlib.util.spec_from_file_location("broad_program_label_blind_redteam", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_cyclic_sources_preserves_bases_and_rotates_sources():
    pairs = [
        {"base_ids": [1, 2], "source_ids": [1, 3]},
        {"base_ids": [4, 5], "source_ids": [4, 6]},
        {"base_ids": [7, 8], "source_ids": [7, 9]},
    ]
    result = module.cyclic_sources(pairs)
    assert [row["base_ids"] for row in result] == [[1, 2], [4, 5], [7, 8]]
    assert [row["source_ids"] for row in result] == [[4, 6], [7, 9], [1, 3]]


def test_random_replacements_preserve_positions_and_reject_true_token():
    pair = {"base_ids": [1, 2, 3], "source_ids": [1, 9, 3]}
    result = module.random_replacements([pair], [8, 9, 10], 3)[0]
    assert result["source_ids"][0] == 1
    assert result["source_ids"][2] == 3
    assert result["source_ids"][1] in (8, 10)


def test_jaccard():
    assert module.jaccard([1, 2, 3], [2, 3, 4]) == 0.5
