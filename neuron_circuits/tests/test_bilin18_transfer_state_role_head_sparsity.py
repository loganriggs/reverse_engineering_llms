import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_transfer_state_role_head_sparsity import (  # noqa: E402
    head_map_from_labels,
    invocation_labels,
)


def test_head_sparsity_is_locked_after_layer_localization():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_role_head_sparsity.json").read_text())
    assert config["status"] == "locked-after-four-layer-qv-sufficiency-before-head-sparsity-test"
    assert config["role_layers"] == [7, 8, 9, 10]
    assert config["ranks"][-1] == 36
    assert config["discovery_rows"] == 2


def test_invocation_labels_and_head_map():
    labels = invocation_labels([7, 8], 3)
    assert labels == [(7, 0), (7, 1), (7, 2), (8, 0), (8, 1), (8, 2)]
    assert head_map_from_labels([(7, 0), (7, 2), (8, 1)]) == {7: {0, 2}, 8: {1}}
