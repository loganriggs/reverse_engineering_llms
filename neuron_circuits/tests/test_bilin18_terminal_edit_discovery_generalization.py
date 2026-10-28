import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_terminal_edit_discovery_protocol_was_locked():
    config = json.loads(
        (ROOT / "configs/bilin18_terminal_edit_discovery_generalization.json").read_text()
    )
    assert config["status"].startswith("locked-")
    assert config["random_rank1_controls"] >= 8
    assert len(config["control_pairs"]) >= 16


def test_rms_vjp_matches_autograd():
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_bilin18_terminal_edit_discovery_generalization import rms_vjp

    torch.manual_seed(3)
    state = torch.randn(5, 17, dtype=torch.float64, requires_grad=True)
    cotangent = torch.randn_like(state)
    normalized = torch.nn.functional.rms_norm(state, (state.shape[-1],))
    (normalized * cotangent).sum().backward()
    predicted = rms_vjp(state.detach(), cotangent.detach())
    torch.testing.assert_close(predicted, state.grad, rtol=1e-10, atol=1e-10)
