import json
from pathlib import Path


def test_scaled_budget_pairs_increase_together():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/"configs/bilin18_readable_grammar_scaled_budget.json").read_text())
    assert config["leaf_budget_pairs"]==[[21,128],[32,192],[42,256]]
    assert config["support_budgets"]==[128,192,256]
