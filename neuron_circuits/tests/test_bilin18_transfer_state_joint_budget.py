import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_joint_budget_sweep_and_analysis_are_locked():
    sweep = json.loads((ROOT / "configs/bilin18_transfer_state_joint_fisher_budget.json").read_text())
    analysis = json.loads((ROOT / "configs/bilin18_transfer_state_joint_budget_analysis.json").read_text())
    assert sweep["status"] == "locked-after-same-rank-joint-chart-failure-before-coordinate-budget-sweep"
    assert sweep["ranks"] == [1, 4, 16, 64, 128]
    assert analysis["status"] == "locked-before-coordinate-budget-results-inspection"
