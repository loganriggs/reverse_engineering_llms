import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_suffix_jvp_numerical_redteam import relative_scalar_error


def test_relative_scalar_error_is_symmetric():
    assert relative_scalar_error(2.0, 1.0) == relative_scalar_error(1.0, 2.0) == 0.5
    assert relative_scalar_error(0.0, 0.0) == 0.0
