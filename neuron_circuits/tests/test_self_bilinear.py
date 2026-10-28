import pytest
import torch

from neuron_circuits.self_bilinear import DeepSelfBilinear, SelfBilinearLayer


def test_layer_matches_analytic_quadratic_form_and_decoder_bias():
    torch.manual_seed(3)
    layer = SelfBilinearLayer(4, 2, 5, decoder_bias=True)
    inputs = torch.randn(7, 4)
    output, activations = layer(inputs)
    analytic = torch.einsum("bi,hij,bj->bh", inputs, layer.neuron_matrices(), inputs)
    expected = torch.nn.functional.linear(
        analytic, layer.decoder.weight / 5**0.5, layer.decoder.bias
    )
    assert torch.allclose(activations, analytic, atol=1e-6)
    assert torch.allclose(output, expected, atol=1e-6)


def test_residual_is_exact_identity_addition():
    torch.manual_seed(4)
    plain = SelfBilinearLayer(4, 4, 5, residual=False)
    residual = SelfBilinearLayer(4, 4, 5, residual=True)
    residual.load_state_dict(plain.state_dict())
    inputs = torch.randn(7, 4)
    assert torch.allclose(residual(inputs)[0], inputs + plain(inputs)[0])
    with pytest.raises(ValueError):
        SelfBilinearLayer(4, 2, 5, residual=True)


def test_decoder_bias_ablation_removes_all_constant_writes():
    model = DeepSelfBilinear(6, 4, 2, residual_stream=False, decoder_bias=False)
    assert all(layer.decoder.bias is None for layer in model.layers)
    assert torch.equal(model(torch.zeros(3, 6)), torch.zeros(3, 2))


def test_masks_are_per_example_and_per_layer():
    torch.manual_seed(5)
    model = DeepSelfBilinear(6, 4, 2, residual_stream=False, decoder_bias=False)
    inputs = torch.randn(3, 6)
    all_on = [torch.ones(3, 4), torch.ones(3, 4)]
    intact = model(inputs)
    assert torch.allclose(intact, model(inputs, all_on))
    all_off = [torch.zeros(3, 4), torch.zeros(3, 4)]
    assert torch.equal(model(inputs, all_off), torch.zeros(3, 2))
