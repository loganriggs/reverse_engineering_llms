import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_upstream_factorial_is_locked_before_results():
    config=json.loads((ROOT/"configs/bilin18_boundary_style_L8H2_factorial.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["random_contrasts"]>=31
    assert len(config["hypotheses"])==7


def test_upstream_factorial_uses_all_native_tensor_roles():
    source=(ROOT/"scripts/run_bilin18_boundary_style_L8H2_factorial.py").read_text()
    for name in ("qk1","qk2","current_value","persistent_bus"):
        assert f'"{name}"' in source
    assert "for world in WORLDS" in source


def test_completed_upstream_factorial_records_falsifiers():
    result=json.loads((ROOT/"artifacts/bilin18_tensor_decomposition/bilin18_boundary_style_L8H2_factorial.json").read_text())
    assert result["hypotheses"]["P0_factorial_recomposes"]["supported"]
    assert not result["hypotheses"]["P3_value_is_a_cancellation_code"]["supported"]
    assert not result["hypotheses"]["P4_QK_change_opposes_payload"]["supported"]
