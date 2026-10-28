import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_attention_head_semantic_increment_redteam import fit_decoder, predict_decoder


def test_decoder_fits_discovery_direction_and_applies_to_evaluation():
    torch.manual_seed(1)
    labels = torch.tensor([0] * 40 + [1] * 40)
    values = torch.randn(80, 4)
    values[labels.bool(), 2] += 3
    model = fit_decoder(values, labels)
    prediction = predict_decoder(model, values)
    assert prediction[labels.bool()].mean() > prediction[~labels.bool()].mean() + 1
