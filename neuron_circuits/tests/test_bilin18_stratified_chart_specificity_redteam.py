import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_stratified_chart_specificity.py"
spec = importlib.util.spec_from_file_location("stratified_chart_specificity", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_panel_reconstructs_classes():
    class Dummy(dict):
        pass
    arrays = Dummy()
    for key in ("residuals", "gradients"):
        arrays[f"discovery_{key}_L0"] = np.zeros((2, 9))
    for key in ("full_effect", "semantic_effect", "distance", "source_id", "next_id", "loss"):
        arrays[f"discovery_{key}_L0"] = np.zeros(2, dtype=int)
    classes = np.asarray(["other"])
    result = module.panel(arrays, "discovery", 0, classes)
    assert result["source_class"].tolist() == ["other", "other"]
