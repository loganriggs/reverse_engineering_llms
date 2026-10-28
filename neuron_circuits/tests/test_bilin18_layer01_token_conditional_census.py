import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    path = ROOT / "scripts/run_bilin18_layer01_token_conditional_census.py"
    spec = importlib.util.spec_from_file_location("conditional_census", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conditional_auc_uses_only_same_token_pairs():
    module = load_module()
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=bool)
    tokens = np.array([10, 10, 20, 20, 30, 40])
    scores = np.array([0., 1., 1., 0., 100., -100.])
    result = module.within_token_auc(labels, scores, tokens)
    assert result["pairs"] == 2
    assert result["groups"] == 2
    assert result["auc"] == 0.5


def test_conditional_census_is_locked():
    config = json.loads((ROOT / "configs/bilin18_layer01_token_conditional_census.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["minimum_within_token_pairs"] >= 500

