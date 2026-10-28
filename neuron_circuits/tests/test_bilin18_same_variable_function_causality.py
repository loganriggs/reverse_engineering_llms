import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_same_variable_function_causality import variant_specs


def test_variant_specs_have_one_native_and_three_scopes_per_head():
    rows = variant_specs([2, 5])
    assert rows[0]["name"] == "native"
    assert [row["name"] for row in rows[1:]] == ["H2_all", "H2_near", "H2_far", "H5_all", "H5_near", "H5_far"]
