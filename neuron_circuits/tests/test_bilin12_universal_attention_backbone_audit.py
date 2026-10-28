import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))

from run_bilin12_universal_attention_backbone_audit import CONFIG


def test_universal_backbone_config_has_disjoint_split_capacity():
    import json
    row=json.loads(CONFIG.read_text())
    assert row["selection_rows"]+row["validation_rows"]<=min(spec["rows"] for spec in row["evaluation"].values())
    assert set(row["fixed_candidates"]["middle_core_4568"])=={4,5,6,8}
