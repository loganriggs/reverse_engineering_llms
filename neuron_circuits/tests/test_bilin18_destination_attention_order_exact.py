import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_exact_order_audit_is_post_failure_and_uses_fixed_rows():
    config = json.loads((ROOT / "configs/bilin18_destination_attention_order_exact.json").read_text())
    assert config["status"] == "locked-after-truncated-order-explosion-before-exact-recomposition-audit"
    assert config["evaluation_rows"] == 8
    source = json.loads((ROOT / config["source_config"]).read_text())
    assert source["evaluation_rows"] >= config["evaluation_rows"]
    assert len(source["targets"]) == 3
