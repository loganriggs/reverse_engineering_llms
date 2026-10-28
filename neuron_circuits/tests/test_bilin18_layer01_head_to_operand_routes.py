from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from run_bilin18_layer01_head_to_operand_routes import rmsnorm_jvp  # noqa: E402


def test_rmsnorm_jvp_matches_autograd():
    torch.manual_seed(4); state=torch.randn(7); directions=torch.randn(3,7)
    expected=torch.stack([torch.func.jvp(lambda x:F.rms_norm(x,(7,)),(state,),(direction,))[1] for direction in directions])
    assert torch.allclose(rmsnorm_jvp(state,directions),expected,atol=2e-6,rtol=2e-6)
