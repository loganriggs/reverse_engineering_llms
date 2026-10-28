import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin12_layer01_taylor_failure_diagnosis import spearman


def test_diagnosis_is_locked_to_prior_behavior_sites():
    config = json.loads((ROOT / "configs/bilin12_layer01_taylor_failure_diagnosis.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["source_results"].endswith("frozen_taylor_behavior.json")
    assert config["maximum_order"] == 5
    assert config["rank"] == 256


def test_spearman_detects_monotone_order():
    x = np.asarray([1.0, 4.0, 2.0, 3.0])
    assert abs(spearman(x, x) - 1.0) < 1e-12
    assert abs(spearman(x, -x) + 1.0) < 1e-12
