"""Factorized bilinear key-value memory used by the compositional toy experiment."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


LOOKUP = 0
PROPERTY = 1
MAXIMUM = 2
MINIMUM = 3
TASK_NAMES = ("lookup", "property", "maximum", "minimum")


class BilinearMemory(nn.Module):
    """One shared bank of symmetric, rank-two bilinear neurons.

    Learned key embeddings are the memory addresses. Unary tasks pair an address with a
    learned query; binary tasks pair two addresses. Task-specific linear decoders make the
    neurons' output roles explicit while leaving all nonlinear computation in the shared
    bilinear layer.
    """

    def __init__(
        self,
        memories: int,
        values: int,
        embedding_dim: int,
        neurons: int,
    ) -> None:
        super().__init__()
        self.memories = memories
        self.values = values
        self.embedding_dim = embedding_dim
        self.neurons = neurons
        self.embedding = nn.Embedding(memories, embedding_dim)
        self.unary_queries = nn.Parameter(torch.randn(2, embedding_dim) / math.sqrt(embedding_dim))
        self.left = nn.Linear(embedding_dim, neurons, bias=False)
        self.right = nn.Linear(embedding_dim, neurons, bias=False)
        self.decoder = nn.Parameter(torch.randn(4, values, neurons) / math.sqrt(neurons))
        self.decoder_bias = nn.Parameter(torch.zeros(4, values))

    def bilinear_activations(self, first: Tensor, second: Tensor) -> Tensor:
        return 0.5 * (
            self.left(first) * self.right(second)
            + self.left(second) * self.right(first)
        )

    def logits_from_embeddings(self, first: Tensor, second: Tensor, task: Tensor) -> Tensor:
        hidden = self.bilinear_activations(first, second)
        return torch.einsum("bvh,bh->bv", self.decoder[task], hidden) + self.decoder_bias[task]

    def forward(self, first_key: Tensor, second_key: Tensor, task: Tensor) -> Tensor:
        first = self.embedding(first_key)
        second = torch.empty_like(first)
        unary = task < 2
        second[unary] = self.unary_queries[task[unary]]
        second[~unary] = self.embedding(second_key[~unary])
        return self.logits_from_embeddings(first, second, task)

    def hidden(self, first_key: Tensor, second_key: Tensor, task: Tensor) -> Tensor:
        first = self.embedding(first_key)
        second = torch.empty_like(first)
        unary = task < 2
        second[unary] = self.unary_queries[task[unary]]
        second[~unary] = self.embedding(second_key[~unary])
        return self.bilinear_activations(first, second)

    def neuron_matrices(self) -> Tensor:
        """Return symmetric Q_j with h_j(a,b)=a^T Q_j b."""
        left = self.left.weight
        right = self.right.weight
        return 0.5 * (
            torch.einsum("hi,hj->hij", left, right)
            + torch.einsum("hi,hj->hij", right, left)
        )


def balanced_random_values(memories: int, values: int, *, generator: torch.Generator) -> Tensor:
    labels = torch.arange(memories) % values
    return labels[torch.randperm(memories, generator=generator)]


def held_out_pair(first: Tensor, second: Tensor, modulus: int = 5) -> Tensor:
    """Deterministic 20% composition split, independent of assigned values."""
    return ((first * 1009 + second * 9176) % modulus) == 0


def task_targets(value_by_key: Tensor, first: Tensor, second: Tensor, task: Tensor) -> Tensor:
    first_value = value_by_key[first]
    second_value = value_by_key[second]
    target = first_value.clone()
    target[task == PROPERTY] = (first_value[task == PROPERTY] >= value_by_key.max().add(1) // 2).long()
    target[task == MAXIMUM] = torch.maximum(first_value[task == MAXIMUM], second_value[task == MAXIMUM])
    target[task == MINIMUM] = torch.minimum(first_value[task == MINIMUM], second_value[task == MINIMUM])
    return target
