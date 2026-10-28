import unittest

import numpy as np

from neuron_circuits.arithmetic import arithmetic_examples, semantic_outcome


class ArithmeticTests(unittest.TestCase):
    def test_examples_have_single_digit_answers(self) -> None:
        examples = arithmetic_examples()
        self.assertGreater(len(examples), 100)
        self.assertTrue(all(0 <= example.answer <= 9 for example in examples))

    def test_semantic_outcome_distinguishes_near_far_and_non_numeric(self) -> None:
        number_ids = {number: number for number in range(10)}
        nearby = semantic_outcome(np.eye(11)[2] * 10, 3, number_ids)
        far = semantic_outcome(np.eye(11)[9] * 10, 3, number_ids)
        nonnumeric = semantic_outcome(np.eye(11)[10] * 10, 3, number_ids)
        self.assertEqual(nearby.failure_kind, "nearby")
        self.assertEqual(far.failure_kind, "far")
        self.assertEqual(nonnumeric.failure_kind, "non_numeric")
        self.assertLess(nearby.expected_distance, far.expected_distance)


if __name__ == "__main__":
    unittest.main()
