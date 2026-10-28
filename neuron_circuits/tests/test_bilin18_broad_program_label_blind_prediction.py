import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_broad_program_label_blind_prediction.py"
spec = importlib.util.spec_from_file_location("broad_program_label_blind_prediction", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_top_subset_is_sorted_and_scale_ordered():
    assert module.top_subset([0.1, 4.0, 2.0, 3.0], 2) == [1, 3]


def test_percentile_includes_observed_tie():
    assert module.percentile([0.0, 1.0, 1.0, 2.0], 1.0) == 0.75


def test_motif_null_preserves_layer_counts_and_detects_extreme_closure():
    # Three selected heads from each of two layers all belong to one component.
    # Random matched triples very rarely attain the same closure.
    components = [[0, 1, 2, 9, 10, 11]]
    result = module.motif_statistics(
        [0, 1, 2, 9, 10, 11], components, [0, 1], heads=9,
        subset_size=3, replicates=2000, seed=7,
    )
    assert result["motif_members"] == 6
    assert result["component_pairs"] == 15
    assert result["component_pair_one_sided_p"] < 0.01
