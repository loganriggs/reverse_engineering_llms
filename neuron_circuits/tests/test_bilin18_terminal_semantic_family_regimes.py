import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_semantic_family_regime_protocol_is_locked():
    config = json.loads(
        (ROOT / "configs/bilin18_terminal_semantic_family_regimes.json").read_text()
    )
    families = json.loads((ROOT / config["family_config"]).read_text())["families"]
    assert config["status"].startswith("locked-")
    assert set(families) == {"sentiment", "gender", "tense", "increasing_scale", "polarity"}
    assert all(len(pairs) >= 3 for pairs in families.values())
    assert config["minimum_families"] == 4
