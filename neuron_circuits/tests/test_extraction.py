import unittest

import torch

from neuron_circuits.extraction import mask_from_ranking, retained_counts


class ExtractionTests(unittest.TestCase):
    def test_mask_from_ranking(self) -> None:
        ranking = torch.tensor([3, 1, 0, 2])
        mask = mask_from_ranking(ranking, 2, (1, 2, 2))
        self.assertEqual(int(mask.sum()), 2)
        self.assertEqual(mask.reshape(-1).tolist(), [0.0, 1.0, 0.0, 1.0])

    def test_inverted_mask_removes_top_units(self) -> None:
        ranking = torch.tensor([3, 1, 0, 2])
        mask = mask_from_ranking(ranking, 2, (1, 2, 2), invert=True)
        self.assertEqual(mask.reshape(-1).tolist(), [1.0, 0.0, 1.0, 0.0])

    def test_retained_counts_include_endpoints(self) -> None:
        self.assertEqual(retained_counts(100, [0.01, 0.1]), [0, 1, 10, 100])


if __name__ == "__main__":
    unittest.main()
