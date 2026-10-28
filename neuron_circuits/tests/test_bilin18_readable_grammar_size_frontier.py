import json
from pathlib import Path


def test_readable_frontier_is_locked_and_increases_variable_count_at_fixed_route_budget():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/bilin18_readable_grammar_size_frontier.json").read_text())
    assert config["leaf_counts"] == [21, 32, 42]
    assert config["methods"] == ["fine128"]
    assert config["status"].startswith("locked-after-21-leaf")
