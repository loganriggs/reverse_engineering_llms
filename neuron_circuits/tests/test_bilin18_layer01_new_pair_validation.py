from pathlib import Path
import json


def test_new_pair_panel_is_disjoint_from_original_panel():
    root = Path(__file__).resolve().parents[1]
    new = json.loads((root / "configs/bilin18_layer8_semantic_route_prospective.json").read_text())["new_families"]
    old = json.loads((root / "configs/bilin18_layer8_semantic_counterfactuals.json").read_text())["families"]
    new_pairs = {tuple(pair) for pairs in new.values() for pair in pairs}
    old_pairs = {tuple(pair) for pairs in old.values() for pair in pairs}
    assert len(new_pairs) == 17
    assert new_pairs.isdisjoint(old_pairs)
