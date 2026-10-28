import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_broad_regime_construction.py"
spec = importlib.util.spec_from_file_location("broad_regime_redteam", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_first_crossing_uses_layer_order_not_json_order():
    summary = {
        "10": {"destination": {"projection_share": .8}},
        "2": {"destination": {"projection_share": .6}},
        "1": {"destination": {"projection_share": .2}},
    }
    assert module.first_crossing(summary, .5) == 2


def test_locked_redteam_uses_exhaustive_three_head_null():
    config = json.loads((ROOT / "configs" / "bilin18_broad_regime_construction_redteam.json").read_text())
    assert config["status"] == "locked-after-broad-construction-before-materiality-and-subset-null"
    assert config["subset_size"] == 3
    assert len(list(module.itertools.combinations(range(9), config["subset_size"]))) == 84
    assert config["attention_native_rms_ratio_gate"] >= .1
