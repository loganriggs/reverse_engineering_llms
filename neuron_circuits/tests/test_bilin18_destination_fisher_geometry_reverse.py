import json
from pathlib import Path


def test_reverse_fisher_target_swaps_pair():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/"configs/bilin18_destination_fisher_geometry_reverse.json").read_text())
    assert config["target"]==[2,5]
    assert config["partner"]==[3,8]
