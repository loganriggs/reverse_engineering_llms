import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_midpoint_jvp_numerical_redteam import CONFIG


def test_numerical_redteam_config_exists():
    assert CONFIG.exists()
