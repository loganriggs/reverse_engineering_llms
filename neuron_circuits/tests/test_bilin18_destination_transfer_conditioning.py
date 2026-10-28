import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_destination_transfer_conditioning import summarize


def test_transfer_conditioning_was_locked_after_exact_order_audit():
    config = json.loads((ROOT / "configs/bilin18_destination_transfer_conditioning.json").read_text())
    assert config["status"] == "locked-after-exact-high-order-cancellation-before-transfer-conditioning-atlas"


def test_profile_summary_preserves_layers_and_medians():
    records = [
        [{"layer": 2, "gain": 1., "attention_to_identity": .2, "branch_cosine": -.5, "cancellation_condition": 2.}],
        [{"layer": 2, "gain": 3., "attention_to_identity": .4, "branch_cosine": -.1, "cancellation_condition": 4.}],
    ]
    row = summarize(records)[0]
    assert row["layer"] == 2
    assert row["gain"] == 2
    assert row["cancellation_condition"] == 3
