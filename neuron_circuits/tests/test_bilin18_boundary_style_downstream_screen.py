import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_bilin18_boundary_style_downstream_screen import accumulator  # noqa: E402


def test_downstream_screen_is_locked_before_results():
    config=json.loads((ROOT/"configs/bilin18_boundary_style_downstream_screen.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["layers"]==list(range(9,18))
    assert config["random_contrasts"]>=31


def test_screen_accumulator_has_signed_and_random_fields():
    assert set(accumulator())=={"native","frozen","mediated","random_mediated"}


def test_contrast_uses_checkpoint_output_vocabulary():
    source=(ROOT/"scripts/run_bilin18_boundary_style_downstream_screen.py").read_text()
    assert "model.lm_head.weight.shape[0]" in source
    assert "make_contrasts(len(flags)" not in source
