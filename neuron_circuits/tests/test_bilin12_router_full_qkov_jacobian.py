import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin12_router_full_qkov_jacobian import rms_jvp, rotary


CONFIG = ROOT / "configs/bilin12_router_full_qkov_jacobian.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_router_full_qkov_jacobian.json"
MODE_OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_router_jacobian_mode_finite_reconstruction.json"


def test_rms_jvp_matches_autograd():
    generator = torch.Generator().manual_seed(83)
    x = torch.randn(2, 3, 8, generator=generator)
    tangents = torch.randn(2, 5, 3, 8, generator=generator)
    expected = []
    for index in range(tangents.shape[1]):
        _, tangent = torch.func.jvp(lambda value: F.rms_norm(value, (value.shape[-1],)),
                                    (x,), (tangents[:, index],))
        expected.append(tangent)
    torch.testing.assert_close(rms_jvp(x, tangents), torch.stack(expected, 1), rtol=1e-5, atol=1e-6)


def test_rotary_is_linear_in_its_input():
    generator = torch.Generator().manual_seed(84)
    first = torch.randn(2, 3, 4, 8, generator=generator)
    second = torch.randn(2, 3, 4, 8, generator=generator)
    angle = torch.randn(1, 3, 1, 4, generator=generator)
    cos, sin = angle.cos(), angle.sin()
    torch.testing.assert_close(rotary(first + second, cos, sin),
                               rotary(first, cos, sin) + rotary(second, cos, sin))


def test_full_qkov_output_obeys_decomposition_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    for row in payload["results"].values():
        assert abs(sum(row["target_projection"].values()) - 1) < 1e-5
        assert row["full_vs_native"]["coordinate_fidelity"] > row["mlp_analytic_vs_native_coordinate_fidelity"]
        assert row["mlp_analytic_vs_finite_mlp_coordinate_fidelity"] > 0.99


def test_operator_subspace_rotation_invariance_if_present():
    if not MODE_OUTPUT.exists():
        return
    payload = json.loads(MODE_OUTPUT.read_text())
    row = payload["hypotheses"]["B4_causal_sufficiency_is_rotation_invariant"]
    assert row["supported"]
    assert row["maximum_error"] <= 1e-5
