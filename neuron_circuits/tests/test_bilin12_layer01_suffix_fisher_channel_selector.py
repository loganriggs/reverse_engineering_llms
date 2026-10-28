import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin12_layer01_suffix_fisher_channel_selector import fisher_cotangent


def test_suffix_selector_is_locked_and_prospective():
    config = json.loads((ROOT / "configs/bilin12_layer01_suffix_fisher_channel_selector.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["sites_per_corpus"] >= 64
    assert config["rank"] == 256
    assert "prospective" in config["hypotheses"]["F8_selector_is_prospective"].lower()


def test_fisher_cotangent_is_centered():
    logits = torch.tensor([1.0, -0.2, 0.5])
    tangent = torch.tensor([0.3, -0.4, 0.1])
    value = fisher_cotangent(logits, tangent)
    assert abs(float(value.sum())) < 1e-6
