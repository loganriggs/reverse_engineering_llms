import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_bilin18_boundary_style_two_hop_circuit import matched_cue_controls  # noqa: E402


def test_two_hop_circuit_is_locked_before_results():
    config = json.loads((ROOT / "configs/bilin18_boundary_style_two_hop_circuit.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["random_contrasts"] >= 31
    assert len(config["hypotheses"]) == 7


def test_matched_control_never_selects_cue_or_boundary():
    tokens = np.asarray([[3, 4, 5, 6, 7, 8, 9, 10]])
    counts = np.arange(20) + 1
    boundary_flags = np.zeros(20, dtype=bool); boundary_flags[9] = True
    control = matched_cue_controls(tokens, np.asarray([2]), np.asarray([6]), counts, boundary_flags)[0]
    assert control not in (2, 6)
    assert control < 6


def test_completed_native_two_hop_circuit_passes_locked_gates():
    result = json.loads((ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_boundary_style_two_hop_circuit.json").read_text())
    assert result["hypotheses"]["C6_native_two_hop_circuit_is_supported"]["supported"]
    assert result["hypotheses"]["C0_native_branch_recomposes"]["max_logit_error"] < 1e-4
