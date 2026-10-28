import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_late_read_function_atlas import (  # noqa: E402
    frequency_categories, program_predictions, scaled_fidelity, shuffled_categories, top_jaccard,
)


def test_program_predictions_recover_simple_tables():
    distance = np.arange(4.0)[:, None]
    category = np.arange(5.0)[None, :]
    additive = distance + category
    predictions = program_predictions(additive)
    assert scaled_fidelity(predictions["additive"], additive) > .999999
    separable = (distance + 1) * (category + 1)
    predictions = program_predictions(separable)
    assert scaled_fidelity(predictions["rank1"], separable) > .999999


def test_frequency_controls_are_well_formed():
    corpus = np.array([[0, 0, 1, 2], [0, 2, 3, 4]])
    frequency, counts = frequency_categories(corpus, 8, 4)
    human = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    shuffled = shuffled_categories(human, counts, 7)
    assert frequency.min() == 0 and frequency.max() == 3
    assert sorted(shuffled.tolist()) == sorted(human.tolist())
    # Within each log-frequency stratum the label multiset is preserved.
    strata = np.floor(np.log2(counts + 1)).astype(int)
    for level in np.unique(strata):
        mask = strata == level
        assert sorted(shuffled[mask].tolist()) == sorted(human[mask].tolist())


def test_jaccard_and_scaled_fidelity():
    assert top_jaccard([1, 4, 3], [0, 5, 2], 1) == 1.0
    assert scaled_fidelity(np.array([1, 2]), np.array([3, 6])) > .999999
