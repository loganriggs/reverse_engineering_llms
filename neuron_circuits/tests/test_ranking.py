import unittest

import numpy as np

from neuron_circuits.ranking import (
    descending_ranking,
    gradient_activation_scores,
    layer_stratified_ranking,
    write_magnitude_scores,
)


class RankingTests(unittest.TestCase):
    def test_write_magnitude_includes_projection_norm(self) -> None:
        activations = np.array([[[2.0, -3.0]]])
        projection = np.array([[3.0, 4.0], [0.0, 2.0]])
        np.testing.assert_allclose(write_magnitude_scores(activations, projection), [[[10.0, 6.0]]])

    def test_gradient_activation_uses_absolute_product(self) -> None:
        result = gradient_activation_scores(np.array([-2.0, 3.0]), np.array([4.0, -0.5]))
        np.testing.assert_allclose(result, [8.0, 1.5])

    def test_descending_ranking_is_stable(self) -> None:
        ranking = descending_ranking(np.array([2.0, 2.0, 3.0]))
        np.testing.assert_array_equal(ranking, [2, 0, 1])

    def test_layer_stratification_alternates_layers(self) -> None:
        scores = np.array([[[10.0, 9.0]], [[2.0, 1.0]]])
        np.testing.assert_array_equal(layer_stratified_ranking(scores), [0, 2, 1, 3])


if __name__ == "__main__":
    unittest.main()
