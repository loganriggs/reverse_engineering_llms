import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_full_operator_transplant import matched_operator_controls


def test_operator_controls_match_target_norm_and_partner_angle():
    gram = np.asarray([[4.0, 1.0, .2], [1.0, 9.0, .1], [.2, .1, 1.0]])
    norms = np.sqrt(np.diag(gram))
    scale, controls, cosine = matched_operator_controls(gram, norms, 0, 1, 1, 5)
    assert abs(scale - norms[0] / norms[1]) < 1e-9
    row = controls[0]; coefficients = np.asarray([row["target_coefficient"], 0.0, row["other_coefficient"]])
    norm = np.sqrt(coefficients @ gram @ coefficients)
    actual = abs(coefficients @ gram[:, 0] / (norm * norms[0]))
    assert abs(norm - norms[0]) < 1e-8
    assert abs(actual - cosine) < 1e-8
