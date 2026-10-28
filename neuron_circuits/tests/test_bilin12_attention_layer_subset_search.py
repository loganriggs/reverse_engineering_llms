import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))

from run_bilin12_attention_layer_subset_search import CONFIG


def test_subset_search_config_exists_and_is_locked():
    import json
    row=json.loads(CONFIG.read_text())
    assert row["status"].startswith("locked-")
    assert row["selection_rows"]+row["validation_rows"]<=min(spec["rows"] for spec in row["evaluation"].values())
