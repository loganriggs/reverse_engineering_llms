import unittest

import numpy as np

from neuron_circuits.masks import StageAMask


class StageAMaskTests(unittest.TestCase):
    def test_shape_and_counts(self) -> None:
        mask = StageAMask.full((2, 3, 5))
        self.assertEqual(mask.retained_count, 30)
        self.assertEqual(mask.retained_fraction, 1.0)
        np.testing.assert_array_equal(mask.layer_position_counts(), np.full((2, 3), 5))

    def test_ranked_selection_deduplicates(self) -> None:
        mask = StageAMask.from_ranked_indices((1, 2, 3), [4, 4, 1, 2], retained=3)
        self.assertEqual(mask.retained_count, 3)
        self.assertTrue(mask.values.reshape(-1)[[1, 2, 4]].all())

    def test_wrong_dtype_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            StageAMask(np.ones((1, 2, 3)))

    def test_zero_retention_makes_empty_mask(self) -> None:
        mask = StageAMask.from_ranked_indices((1, 2, 3), [5, 4, 3], retained=0)
        self.assertEqual(mask.retained_count, 0)


if __name__ == "__main__":
    unittest.main()
