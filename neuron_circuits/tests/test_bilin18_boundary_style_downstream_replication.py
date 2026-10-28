import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_downstream_replication_is_locked_to_discovery_selection():
    config=json.loads((ROOT/"configs/bilin18_boundary_style_downstream_replication.json").read_text())
    assert config["status"].startswith("locked-")
    text=" ".join(config["hypotheses"].values())
    for selected in ("L17H2","Layer-17 destination","L11H6"):
        assert selected in text


def test_replication_excludes_discovery_cell_in_source():
    source=(ROOT/"scripts/run_bilin18_boundary_style_downstream_replication.py").read_text()
    assert "if fold==0 and split==0: continue" in source
