import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin12_layer5_backbone_recurrence_redteam import profile_statistics  # noqa: E402


def test_profile_statistics_detect_identical_profiles():
    profile = torch.tensor([0.7, 0.2, 0.1, 0.0])
    rows = profile.repeat(4, 1)
    stats = profile_statistics(rows, topk=2)
    assert stats["rank1_energy_fraction"] > 0.999
    assert stats["mean_pairwise_cosine"] > 0.999
    assert stats["mean_topk_jaccard"] > 0.999
    assert stats["common_topk_channels"] == [0, 1]


def test_config_locks_coordinate_and_matched_group_nulls():
    config = json.loads(Path("configs/bilin12_layer5_backbone_recurrence_redteam.json").read_text())
    assert "locked" in config["status"]
    assert config["permutation_nulls"] >= 256
    assert len(config["hypotheses"]) == 6
