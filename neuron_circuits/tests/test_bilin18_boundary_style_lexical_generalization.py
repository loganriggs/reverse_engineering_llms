import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_bilin18_boundary_style_lexical_generalization import token_pairs  # noqa: E402


def test_lexical_generalization_is_locked_before_results():
    config=json.loads((ROOT/"configs/bilin18_boundary_style_lexical_generalization.json").read_text())
    assert config["status"].startswith("locked-")
    assert len(config["new_spelling_pairs"])==11
    assert config["random_contrasts"]>=31


def test_new_pairs_do_not_overlap_old_pairs():
    new=json.loads((ROOT/"configs/bilin18_boundary_style_lexical_generalization.json").read_text())["new_spelling_pairs"]
    old=json.loads((ROOT/"configs/bilin18_boundary_dialect_state_test.json").read_text())["spelling_pairs"]
    assert not ({word for pair in new for word in pair}&{word for pair in old for word in pair})


def test_completed_lexical_generalization_passes_locked_gates():
    result=json.loads((ROOT/"artifacts/bilin18_tensor_decomposition/bilin18_boundary_style_lexical_generalization.json").read_text())
    assert len(result["results"])==4
    assert result["hypotheses"]["G5_typed_circuit_generalizes_new_lexemes"]["supported"]
