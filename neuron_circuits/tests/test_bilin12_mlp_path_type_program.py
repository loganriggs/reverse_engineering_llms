import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts/run_bilin12_mlp_path_type_program.py"
CONFIG = ROOT / "configs/bilin12_mlp_path_type_program.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_mlp_path_type_program.json"

spec = importlib.util.spec_from_file_location("mlp_path_types", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_path_inventory_is_complete_and_unique():
    assert len(module.PATHS) == 81
    assert len(set(module.PATHS)) == 81
    assert all(len(path) == 4 and set(path) <= set(module.TERMS) for path in module.PATHS)


def test_greedy_gram_program_matches_direct_residual():
    generator = torch.Generator().manual_seed(19)
    paths = torch.randn(5, 81, 4, 4, generator=generator)
    gram, dot, target2 = module.aggregate_gram(paths)
    ordering = module.greedy_order(gram, dot, target2)
    for budget in (0, 1, 8, 32, 81):
        selected = ordering[:budget]
        target = paths.double().sum(1)
        prediction = paths.double()[:, selected].sum(1) if selected else torch.zeros_like(target)
        direct = float((target - prediction).square().sum())
        assert abs(module.error_from_gram(gram, dot, target2, selected) - direct) <= 1e-8 * max(direct, 1.0)


def test_output_matches_locked_config_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    expected = set(json.loads(CONFIG.read_text())["hypotheses"])
    assert set(payload["hypotheses"]) == expected
    assert len(payload["path_order"]) == 81
    assert len(set(payload["path_order"])) == 81
