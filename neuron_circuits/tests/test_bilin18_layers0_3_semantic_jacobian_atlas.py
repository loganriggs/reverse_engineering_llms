import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_layers0_3_semantic_jacobian_atlas import (  # noqa: E402
    kernel_prediction, make_kernel,
)


def test_semantic_jacobian_atlas_is_locked_to_shared_states_and_all_objectives():
    config = json.loads(
        (ROOT / "configs/bilin18_layers0_3_semantic_jacobian_atlas.json").read_text()
    )
    assert config["status"] == "locked-after-sparse-chart-atlas-failure-before-full-semantic-jacobian-atlas"
    assert config["contexts_per_position"] == 8
    assert config["source_layers"] == [0, 1, 2, 3]
    assert config["primary_alpha"] == 0.1


def test_kernel_router_preserves_target_mean_for_zero_query_similarity():
    signatures = np.eye(3, dtype=np.float32)
    targets = np.asarray([[1.0, 2.0], [3.0, 0.0], [2.0, 4.0]], dtype=np.float32)
    model = make_kernel(signatures)
    # The centered signature mean has zero norm and hence zero cross-kernel.
    prediction = kernel_prediction(model, signatures.mean(0), targets, 0.1)
    np.testing.assert_allclose(prediction, targets.mean(0), atol=1e-5)
