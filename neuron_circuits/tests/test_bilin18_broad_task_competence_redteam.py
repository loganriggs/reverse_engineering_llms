import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_broad_task_competence.py"
spec = importlib.util.spec_from_file_location("broad_task_redteam", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_semantic_classes_are_declared_before_redteam():
    probes = module.declared_probes()
    sentiment = next(item for item in probes if item.family == "sentiment_classification" and item.answer == " positive")
    groups = module.semantic_classes(sentiment)
    assert " Good" in groups[sentiment.candidates.index(" positive")]
    agreement = next(item for item in probes if item.family == "agreement_attractor" and item.answer == " is")
    groups = module.semantic_classes(agreement)
    assert " was" in groups[agreement.candidates.index(" is")]


def test_redteam_config_distinguishes_candidate_and_operational_gates():
    config = json.loads((ROOT / "configs" / "bilin18_broad_task_competence_redteam.json").read_text())
    assert config["status"] == "locked-after-candidate-screen-before-semantic-output-redteam"
    assert config["semantic_top1_gate"] >= 0.5
    assert config["median_semantic_rank_gate"] <= 2


def test_permutation_p_detects_perfect_balanced_binary_scores():
    scores = [[4.0, 0.0], [0.0, 4.0]] * 20
    labels = [0, 1] * 20
    assert module.permutation_p(module.np.asarray(scores), module.np.asarray(labels), 500, 7) < .01
