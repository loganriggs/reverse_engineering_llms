import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_learned_router import SupportRouter  # noqa: E402


def test_support_router_output_shape():
    router = SupportRouter(7, 11, 3, 5)
    output = router(torch.randn(4, 7))
    assert output.shape == (4, 3, 5)
