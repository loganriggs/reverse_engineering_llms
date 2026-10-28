import torch

from neuron_circuits.restrictions import eligible_instance_mask, restricted_ranking


def test_final_layer_exclusion():
    allowed = eligible_instance_mask((5, 3, 2), "exclude_final_layer", "cpu")
    assert allowed[:4].all()
    assert not allowed[4].any()


def test_combined_exclusion():
    allowed = eligible_instance_mask(
        (6, 3, 2), "exclude_final_four_layers_and_position", "cpu"
    )
    assert allowed[:2, :2].all()
    assert not allowed[2:].any()
    assert not allowed[:, 2].any()


def test_restricted_ranking_never_returns_excluded_indices():
    scores = torch.tensor([1.0, 9.0, 4.0, 8.0])
    allowed = torch.tensor([True, False, True, True])
    assert restricted_ranking(scores, allowed).tolist() == [3, 2, 0]
