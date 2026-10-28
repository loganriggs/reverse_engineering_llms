import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_typed_attention_edge_joint_composition import build_variant_specs


def test_joint_variant_specs_have_one_native_and_each_prefix_method():
    specs = build_variant_specs([0, 1, 2], ["fine32", "fine128"])
    assert specs[0] == {"name": "native", "prefix_end": -1, "method": "native"}
    assert len(specs) == 1 + 3 * 3
    assert {row["name"] for row in specs} == {
        "native",
        "none_0_0", "fine32_0_0", "fine128_0_0",
        "none_0_1", "fine32_0_1", "fine128_0_1",
        "none_0_2", "fine32_0_2", "fine128_0_2",
    }
