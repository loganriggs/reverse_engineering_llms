import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_semantic_mlp_subspace_depth import differentiable_mlp_trace


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_semantic_mlp_subspace_depth.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_semantic_mlp_subspace_depth.json"


def test_preregistration_is_locked():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"].startswith("locked-after-layer0-pair-specific")
    assert payload["path_points"] == 65
    assert len(payload["hypotheses"]) == 10


def test_completed_output_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert len(payload["layers"]) == 12
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
