import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_fisher_cancellation_redteam import fisher_energy  # noqa: E402


def test_fisher_energy_ignores_constant_logit_shift():
    logits = torch.tensor([0.0, 1.0, -1.0])
    assert fisher_energy(logits, torch.ones(3) * 4) < 1e-12
