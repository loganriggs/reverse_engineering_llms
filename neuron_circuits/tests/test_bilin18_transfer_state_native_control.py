import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_transfer_state_native_control import native_states


def test_native_control_is_locked_after_static_vocabulary_failure():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_native_control.json").read_text())
    assert config["status"] == "locked-after-static-tangent-vocabulary-failure-before-native-state-control"


def test_native_state_helper_declares_inference_only():
    assert getattr(native_states, "__wrapped__", None) is not None
