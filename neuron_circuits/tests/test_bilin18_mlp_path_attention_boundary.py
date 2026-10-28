import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_mlp_path_attention_boundary.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_mlp_path_attention_boundary.json"


def test_attention_panel_is_disjoint_from_highcount_source():
    config = json.loads(CONFIG.read_text())
    source = json.loads((ROOT / config["source_config"]).read_text())
    for corpus, spec in config["evaluation"].items():
        old = source["evaluation"][corpus]
        if spec["path"] == old["path"]:
            assert old["start"] + old["rows"] <= spec["start"] or spec["start"] + spec["rows"] <= old["start"]


def test_output_matches_locked_config_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert len(payload["frozen_top4_groups"]) == 4
