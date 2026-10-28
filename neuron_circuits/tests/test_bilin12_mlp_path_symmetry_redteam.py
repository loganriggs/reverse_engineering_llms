import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts/run_bilin12_mlp_path_symmetry_redteam.py"
CONFIG = ROOT / "configs/bilin12_mlp_path_symmetry_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_mlp_path_symmetry_redteam.json"

spec = importlib.util.spec_from_file_location("path_symmetry", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_symmetry_groups_partition_all_paths():
    counts = [0] * len(module.GROUPS)
    for path in module.PATHS:
        counts[module.group_index(path)] += 1
    assert counts == [2 ** group.count("B") for group in module.GROUPS]
    assert sum(counts) == 81


def test_collapse_preserves_exact_sum():
    generator = torch.Generator().manual_seed(91)
    raw = torch.randn(3, 81, 4, 4, generator=generator)
    grouped = module.collapse(raw)
    torch.testing.assert_close(grouped.sum(1), raw.sum(1))


def test_output_matches_locked_config_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert len(payload["group_order"]) == 16
    assert len(set(payload["group_order"])) == 16
