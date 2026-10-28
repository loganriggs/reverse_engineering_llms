"""Residual self-bilinear networks: x <- x + D((Lx) * (Rx)) + b."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SelfBilinearLayer(nn.Module):
    """One bilinear write, optionally added to an identity residual stream."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        neurons: int,
        *,
        residual: bool = False,
        decoder_bias: bool = True,
    ) -> None:
        super().__init__()
        if residual and input_dim != output_dim:
            raise ValueError("An identity residual requires input_dim == output_dim")
        # L and R are bias-free so decoder bias is the only source of a constant
        # write and its ablation has an unambiguous meaning.
        self.left = nn.Linear(input_dim, neurons, bias=False)
        self.right = nn.Linear(input_dim, neurons, bias=False)
        self.decoder = nn.Linear(neurons, output_dim, bias=decoder_bias)
        self.neurons = neurons
        self.residual = residual

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        activations = self.left(inputs) * self.right(inputs)
        if mask is not None:
            activations = activations * mask
        # Scale the neuron write, not the independently learned decoder bias.
        output = F.linear(
            activations,
            self.decoder.weight / math.sqrt(self.neurons),
            self.decoder.bias,
        )
        if self.residual:
            output = inputs + output
        return output, activations

    def neuron_matrices(self) -> Tensor:
        """Symmetric matrices Q_j satisfying h_j(x)=x^T Q_j x."""
        left, right = self.left.weight, self.right.weight
        return 0.5 * (
            torch.einsum("hi,hj->hij", left, right)
            + torch.einsum("hi,hj->hij", right, left)
        )


class DeepSelfBilinear(nn.Module):
    """A constant-width bilinear residual stream and bias-free output readout."""

    def __init__(
        self,
        input_dim: int,
        neurons: int,
        depth: int,
        outputs: int = 2,
        *,
        residual_stream: bool = True,
        decoder_bias: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            SelfBilinearLayer(
                input_dim, input_dim, neurons,
                residual=residual_stream, decoder_bias=decoder_bias,
            )
            for _ in range(depth)
        )
        # Bias is deliberately absent here: the decoder-bias switch refers exactly
        # to the D maps in the bilinear blocks.
        self.readout = nn.Linear(input_dim, outputs, bias=False)
        self.neurons = neurons
        self.depth = depth
        self.residual_stream = residual_stream
        self.decoder_bias = decoder_bias

    def forward(
        self,
        inputs: Tensor,
        masks: list[Tensor | None] | None = None,
        *,
        return_activations: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        activations = []
        hidden = inputs
        if masks is None:
            masks = [None] * self.depth
        for layer, mask in zip(self.layers, masks):
            hidden, neurons = layer(hidden, mask)
            activations.append(neurons)
        logits = self.readout(hidden)
        return (logits, activations) if return_activations else logits
