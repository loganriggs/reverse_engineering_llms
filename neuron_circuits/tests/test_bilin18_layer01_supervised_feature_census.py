import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/run_bilin18_layer01_supervised_feature_census.py"
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("supervised_census", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feature_census_is_locked_and_has_fifty_unique_features():
    config = json.loads((ROOT / "configs/bilin18_layer01_supervised_feature_census.json").read_text())
    assert config["status"].startswith("locked-")
    assert len(config["features"]) == len(set(config["features"])) == 50
    assert config["layers"] == [0, 1]
    assert config["discovery_rows"] == config["heldout_rows"]


def test_vocabulary_features_match_declared_schema():
    module = load_module()
    assert module.vocabulary_features("\n")[module.FEATURES.index("newline")]
    assert module.vocabulary_features(" running")[module.FEATURES.index("suffix_ing")]
    assert module.vocabulary_features("True")[module.FEATURES.index("boolean")]
    assert module.vocabulary_features(" def")[module.FEATURES.index("programming_keyword")]
    assert len(module.vocabulary_features(" example")) == 50
