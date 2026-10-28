import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lexical_echo_redteam_is_locked_and_uses_prior_manifest():
    config = json.loads(
        (ROOT / "configs/bilin18_boundary_style_H2_value_echo_lexical_redteam.json").read_text()
    )
    lexical = json.loads((ROOT / config["lexical_config"]).read_text())
    assert config["status"].startswith("locked-")
    assert len(lexical["new_spelling_pairs"]) == 11
    assert config["ranks"][0] == 1

