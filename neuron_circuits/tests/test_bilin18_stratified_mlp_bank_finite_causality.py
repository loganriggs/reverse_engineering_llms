import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_stratified_mlp_bank_finite_causality.py"
spec = importlib.util.spec_from_file_location("stratified_mlp_bank_finite", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_retain_keeps_only_selected_channels():
    product = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(module.retain(product, [1]), torch.tensor([[0.0, 2.0, 0.0]]))


def test_runtime_next_mask():
    raw = np.asarray([[0, 1, 2]])
    classes = np.asarray(["other", "function", "punctuation"])
    mask = module.runtime_site_mask("next:punctuation", raw, 1, np.asarray([1.0]), classes, np.asarray([1.0]))
    assert mask.tolist() == [True]
