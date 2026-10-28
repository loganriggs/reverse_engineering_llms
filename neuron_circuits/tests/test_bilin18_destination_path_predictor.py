import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from run_bilin18_destination_path_predictor import cosine, fidelity, median_angle_partner


def test_destination_path_predictor_is_prospectively_locked():
    config = json.loads((ROOT / "configs/bilin18_destination_path_predictor.json").read_text())
    assert config["status"] == "locked-before-prospective-destination-path-predictor-run"
    assert config["targets"] == [[0, 0], [2, 2], [5, 3]]
    assert config["finite_scale"] <= 0.01
    assert config["evaluation_rows"] >= 16


def test_response_fidelity_and_cosine_are_exact_on_identity():
    assert fidelity(7.0, 7.0, 7.0) == 1.0
    assert cosine(7.0, 7.0, 7.0) == 1.0
    assert fidelity(4.0, 0.0, 0.0) == 0.0


def test_median_angle_partner_uses_only_gram_geometry():
    gram = np.asarray([[4.0, 0.2, 0.8, 1.6], [0.2, 1.0, 0, 0],
                       [0.8, 0, 1.0, 0], [1.6, 0, 0, 1.0]])
    norms = np.sqrt(np.diag(gram))
    # Absolute target cosines are .1, .4, .8; the exact median is candidate 2.
    assert median_angle_partner(gram, norms, 0) == 2
