import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_broad_task_competence.py"
spec = importlib.util.spec_from_file_location("broad_task_competence", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_declared_suite_spans_computational_regimes():
    probes = module.declared_probes()
    regimes = {item.regime for item in probes}
    families = {item.family for item in probes}
    assert {"lexical", "binding", "retrieval", "composition", "algorithmic", "formal", "factual_memory"} <= regimes
    assert len(families) >= 16
    assert all(item.answer in item.candidates for item in probes)
    assert all(len(set(item.candidates)) == len(item.candidates) for item in probes)


def test_config_is_locked_before_results_and_has_behavioral_gates():
    config = json.loads((ROOT / "configs" / "bilin18_broad_task_competence.json").read_text())
    assert config["status"] == "locked-before-broad-fineweb-model-competence-screen"
    assert config["minimum_usable_examples"] >= 8
    assert config["normalized_accuracy_gate"] >= 0.5
    assert set(config["hypotheses"]) == {
        "C0_tokenization_coverage", "C1_broad_competence", "C2_nonlexical_competence",
        "C3_intervention_ready", "C4_instruction_bias_is_not_assumed",
    }


def test_hamming_pair_requires_aligned_token_sequences():
    first = {"prompt_ids": [1, 2, 3]}
    assert module.hamming_pair(first, {"prompt_ids": [1, 4, 3]}) == 1
    assert module.hamming_pair(first, {"prompt_ids": [1, 2]}) is None


def test_wilson_lower_is_bounded_and_monotone_at_fixed_total():
    values = [module.wilson_lower(successes, 20) for successes in range(21)]
    assert all(0 <= value <= 1 for value in values)
    assert values == sorted(values)
