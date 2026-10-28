import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_ontology_mdl_frontier import log2_choose


def test_enumerative_code_is_symmetric_and_zero_at_boundaries():
    assert math.isclose(log2_choose(10, 3), log2_choose(10, 7))
    assert abs(log2_choose(10, 0)) < 1e-10
    assert abs(log2_choose(10, 10)) < 1e-10
