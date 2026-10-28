import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_destination_attention_order import first_passing


def test_attention_order_followup_was_locked_after_first_order_failure():
    config = json.loads((ROOT / "configs/bilin18_destination_attention_order.json").read_text())
    assert config["status"] == "locked-after-first-order-path-falsification-before-order-sweep"
    assert config["max_attention_order"] >= 6
    assert config["fidelity_threshold"] == .90


def test_first_passing_handles_nonmonotonic_frontier():
    rows = {
        "0": {"response_fidelity": .2},
        "1": {"response_fidelity": .95},
        "2": {"response_fidelity": .8},
    }
    assert first_passing(rows, .9) == 1
    assert first_passing(rows, .99) is None
