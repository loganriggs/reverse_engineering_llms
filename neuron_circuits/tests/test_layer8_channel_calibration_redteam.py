import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path("scripts").resolve()))
from run_bilin18_layer8_channel_calibration_redteam import (  # noqa: E402
    apply_coordinate_map,
    fit_coordinate_map,
    fit_scalar,
)


def test_scalar_and_coordinate_calibration_recover_known_maps():
    generator = torch.Generator().manual_seed(17)
    prediction = torch.randn(6, 4, 9, generator=generator)
    assert torch.allclose(fit_scalar(prediction, 2.5 * prediction), torch.tensor(2.5), atol=1e-5)
    mapping = torch.randn(4, 4, generator=generator)
    target = apply_coordinate_map(prediction, mapping)
    recovered = fit_coordinate_map(prediction, target)
    assert torch.allclose(recovered, mapping, atol=1e-4, rtol=1e-4)
