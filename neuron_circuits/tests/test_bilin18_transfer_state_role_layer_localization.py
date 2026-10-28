import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_transfer_state_role_layer_localization import localization_specs  # noqa: E402


def test_role_localization_is_locked_after_replication():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_role_layer_localization.json").read_text())
    assert config["status"] == "locked-after-signed-role-replication-before-layer-localization"
    assert config["boundary"] == 6
    assert config["widths"] == [2, 4, 8]


def test_near_and_far_layer_sets_are_distinct_and_sized():
    specs = localization_specs(6, 17, [2, 4, 8])
    assert specs["kv_near4"]["layers"] == {7, 8, 9, 10}
    assert specs["kv_far4"]["layers"] == {14, 15, 16, 17}
    assert len(specs["qv_near8"]["layers"]) == 8
    assert specs["kv_all"]["layers"] == set(range(7, 18))
