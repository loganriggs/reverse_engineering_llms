"""Regression: rare-seen cases must be selected with a Boolean mask."""

import numpy as np

from analyze_dual_output_language import score
from train_dual_output_language import exhaustive


def test_rare_seen_selects_all_nonheldout_swap_cases():
    seq, targets, attrs = exhaustive()
    zero_table = np.zeros((2, 6, 2), dtype=np.float64)
    scores, _ = score([zero_table], seq.numpy(), targets.numpy(),
                      np.zeros(2), attrs)
    assert scores['rare_seen']['n'] == 36
    assert scores['rare_seen']['joint_accuracy'] == 1 / 3
    assert scores['rare_heldout']['n'] == 12
    assert scores['rare_heldout']['joint_accuracy'] == 0


if __name__ == '__main__':
    test_rare_seen_selects_all_nonheldout_swap_cases()
    print('Dual-output score masks passed')
