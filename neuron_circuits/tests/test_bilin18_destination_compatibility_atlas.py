import json
from pathlib import Path


def test_compatibility_atlas_uses_local_scale_and_fixed_pair():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/"configs/bilin18_destination_compatibility_atlas.json").read_text())
    assert config["target"]==[3,8]
    assert config["partner"]==[2,5]
    assert config["intervention_scale"]<=.05
