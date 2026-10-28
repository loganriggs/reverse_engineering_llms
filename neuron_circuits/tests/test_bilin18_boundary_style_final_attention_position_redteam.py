import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_bilin18_boundary_style_final_attention_position_redteam import (  # noqa: E402
    BINS,
    position_masks,
)


def test_position_redteam_is_locked_before_results():
    config = json.loads(
        (ROOT / "configs/bilin18_boundary_style_final_attention_position_redteam.json").read_text()
    )
    assert config["status"].startswith("locked-")
    assert config["quadrature_points"] >= 32
    assert config["random_contrasts"] >= 31


def test_position_bins_partition_prefix_for_near_and_far_boundaries():
    cue = torch.tensor([2, 10, 3])
    boundary = torch.tensor([5, 110, 127])
    masks = position_masks(128, cue, boundary, 127)
    total = sum(mask.long() for mask in masks.values())
    torch.testing.assert_close(total, torch.ones_like(total))
    assert tuple(masks) == BINS
    assert masks["post_1"][0, 126]
    assert masks["post_9_16"][1, 111]
    assert masks["boundary"][2, 127]
    assert not masks["destination"][2, 127]

