import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_layers0_3_local_operator_state_swap import (  # noqa: E402
    source_for, specifications,
)


def test_state_swap_is_locked_to_same_position_atlas_controls():
    config = json.loads(
        (ROOT / "configs/bilin18_layers0_3_local_operator_state_swap.json").read_text()
    )
    assert config["status"] == "locked-after-semantic-jacobian-atlas-failure-before-local-operator-state-swap"
    assert config["near_widths"] == [4, 8]
    assert config["sites_per_feature"] == 2


def test_posthoc_omission_controls_do_not_silently_use_reference_states():
    specs = specifications(1, 18, [4, 8])
    assert source_for(specs["exact_mlp_only"], "mlp", 7) == "target"
    assert source_for(specs["exact_mlp_only"], "attention", 7) == "zero"
    assert source_for(specs["exact_attention_only"], "mlp", 7) == "zero"
    assert source_for(specs["exact_attention_only"], "attention", 7) == "target"
