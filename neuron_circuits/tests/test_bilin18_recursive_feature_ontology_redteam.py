import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_recursive_feature_ontology.py"
spec = importlib.util.spec_from_file_location("recursive_feature_ontology", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_interaction_design_is_one_hot():
    classes = np.asarray(module.CLASSES)
    metadata = {"current_id": np.asarray([0, 1]), "next_id": np.asarray([1, 2])}
    design, names = module.interaction_design(metadata, classes)
    assert design.shape == (2, len(module.CLASSES) ** 2)
    assert np.all(design.sum(1) == 1)
    assert names[np.argmax(design[0])] == "other->continuation"


def test_best_interaction_recovers_binary_split():
    design = np.zeros((40, 2)); design[:20, 0] = 1; design[20:, 1] = 1
    row = module.best_interaction(np.r_[np.ones(20), -np.ones(20)], design, ["first", "second"])
    assert row["auroc"] == 1.0
