import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_semantic_source_path_integral import quadrature_rules


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_semantic_source_path_integral.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_semantic_source_path_integral.json"


def test_quadrature_rules_integrate_constants_and_linear_functions():
    for name, (nodes, weights) in quadrature_rules().items():
        assert np.isclose(sum(weights), 1.0), name
        if name != "base_jvp":
            assert np.isclose(sum(node * weight for node, weight in zip(nodes, weights)), 0.5), name


def test_preregistration_is_locked():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"] == "locked-after-tangent-specificity-before-path-integration"
    assert len(payload["hypotheses"]) == 10


def test_completed_output_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
