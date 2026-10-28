import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_mlp_weight_vs_data_scaling.py"
spec = importlib.util.spec_from_file_location("mlp_weight_data_scaling", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_origin_pca_reconstructs_rank_two_data():
    rng = np.random.default_rng(8); values = rng.normal(size=(20, 2)) @ rng.normal(size=(2, 9))
    basis = module.origin_pca(values, 2)
    recovered = (values @ basis.T) @ basis
    assert np.allclose(values, recovered, atol=1e-5)
