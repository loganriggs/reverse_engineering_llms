import unittest

import numpy as np

from neuron_circuits.metrics import compute_fidelity, forward_kl


class MetricTests(unittest.TestCase):
    def test_identical_logits_have_full_fidelity(self) -> None:
        logits = np.array([0.1, 2.0, -1.0])
        result = compute_fidelity(logits, logits.copy(), answer_index=1)
        self.assertAlmostEqual(result.margin_retention, 1.0)
        self.assertAlmostEqual(result.output_kl, 0.0)
        self.assertTrue(result.answer_preserved)
        self.assertTrue(result.meets(1.0))

    def test_answer_margin_uses_best_competitor(self) -> None:
        full = np.array([1.0, 4.0, 3.0])
        masked = np.array([1.0, 2.5, 2.0])
        result = compute_fidelity(full, masked, answer_index=1)
        self.assertEqual(result.intact_margin, 1.0)
        self.assertEqual(result.masked_margin, 0.5)
        self.assertTrue(result.meets(0.5))

    def test_forward_kl_is_nonnegative(self) -> None:
        value = forward_kl(np.array([4.0, 1.0]), np.array([1.0, 4.0]))
        self.assertGreater(value, 0.0)


if __name__ == "__main__":
    unittest.main()
