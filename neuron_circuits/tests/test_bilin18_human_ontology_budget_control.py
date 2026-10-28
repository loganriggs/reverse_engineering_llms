import json
from pathlib import Path


def test_budget_control_matches_scaled_grammar_budgets():
    root = Path(__file__).resolve().parents[1]
    control = json.loads((root / "configs/bilin18_human_ontology_budget_control.json").read_text())
    scaled = json.loads((root / "configs/bilin18_readable_grammar_scaled_budget.json").read_text())
    assert control["support_budgets"] == [128, 192, 256]
    assert control["selection_start"] == scaled["selection_start"]
    assert control["evaluation_start"] == scaled["evaluation_start"]
    assert control["positions"] == scaled["positions"]
