import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_stratified_chart_cross_finite.py"
spec = importlib.util.spec_from_file_location("stratified_chart_cross_finite", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_config_targets_have_matching_families():
    import json
    config = json.loads((ROOT / "configs/bilin18_stratified_chart_cross_finite_redteam.json").read_text())
    for row in config["targets"]:
        assert row["target"].split(":", 1)[0] == row["alternative"].split(":", 1)[0]
