import torch

from neuron_circuits.attention_patching import mask_from_unit_ranking


def test_attention_unit_mask_uses_first_ranked_units():
    ranking = torch.tensor([5, 1, 3, 0, 2, 4])
    mask = mask_from_unit_ranking(ranking, 3, (1, 2, 3))
    assert mask.dtype == torch.bool
    assert int(mask.sum()) == 3
    assert mask.reshape(-1).nonzero().reshape(-1).tolist() == [1, 3, 5]


def test_attention_unit_mask_zero_patch():
    ranking = torch.arange(8)
    assert not mask_from_unit_ranking(ranking, 0, (2, 2, 2)).any()
