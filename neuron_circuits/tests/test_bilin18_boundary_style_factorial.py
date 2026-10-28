import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_bilin18_boundary_style_factorial import FACTORS, shapley  # noqa: E402


def test_style_factorial_is_locked_before_results():
    config = json.loads((ROOT / "configs/bilin18_boundary_style_factorial.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["folds"] == 3
    assert config["random_contrasts"] >= 31
    assert len(config["hypotheses"]) == 7


def test_vector_shapley_exactly_recomposes_nonlinear_worlds():
    torch.manual_seed(3)
    weights = torch.randn(len(FACTORS), 5)
    outputs = {}
    for bits in __import__("itertools").product((0, 1), repeat=len(FACTORS)):
        value = sum(bits[index] * weights[index] for index in range(len(FACTORS)))
        outputs[bits] = value.square() + .3 * value.roll(1)
    pieces = shapley(outputs)
    target = outputs[(1, 1, 1, 1)] - outputs[(0, 0, 0, 0)]
    torch.testing.assert_close(sum(pieces.values()), target)


def test_script_declares_non_shapley_direct_arms():
    source = (ROOT / "scripts/run_bilin18_boundary_style_factorial.py").read_text()
    assert '"current_value_only", (0, 0, 1, 0)' in source
    assert '"qk_only", (1, 1, 0, 0)' in source


def test_completed_factorial_recomposes_and_has_all_cells():
    result = json.loads((ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_boundary_style_factorial.json").read_text())
    assert len(result["results"]) == 18
    assert result["hypotheses"]["F0_factorial_recomposes"]["supported"]
    assert max(max(row["local_recomposition_error"], row["output_recomposition_error"]) for row in result["results"]) < 1e-5
