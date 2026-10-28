import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts/run_bilin12_mlp_path_order_gain_redteam.py"
CONFIG = ROOT / "configs/bilin12_mlp_path_order_gain_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_mlp_path_order_gain_redteam.json"

spec = importlib.util.spec_from_file_location("path_gain", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_gain_scales_paths_by_interaction_degree():
    raw = torch.ones(2, 16, 3, 3)
    value = module.scaled(raw, 2.0)
    for index, group in enumerate(module.GROUPS):
        assert torch.all(value[:, index] == 2 ** group.count("B"))


def test_effective_support_bounds():
    generator = torch.Generator().manual_seed(3)
    paths = torch.randn(5, 16, 2, 2, generator=generator)
    support = module.effective_support(paths)
    assert 1 <= support <= 16


def test_output_matches_locked_config_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert len(payload["discovery_pair_order"]) == 6
