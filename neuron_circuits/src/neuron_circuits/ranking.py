"""Baseline importance scores and deterministic ranking utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.floating]


def activation_scores(activations: FloatArray) -> FloatArray:
    return np.abs(np.asarray(activations, dtype=np.float64))


def gradient_activation_scores(
    activations: FloatArray, gradients: FloatArray
) -> FloatArray:
    activations = np.asarray(activations, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)
    if activations.shape != gradients.shape:
        raise ValueError("activations and gradients must have the same shape")
    return np.abs(activations * gradients)


def write_magnitude_scores(
    activations: FloatArray, output_columns: FloatArray
) -> FloatArray:
    """Score |h_k| ||d_k|| for arrays ending in the MLP channel axis."""
    activations = np.asarray(activations, dtype=np.float64)
    output_columns = np.asarray(output_columns, dtype=np.float64)
    if output_columns.ndim != 2:
        raise ValueError("output_columns must have shape [channel, output_width]")
    if activations.shape[-1] != output_columns.shape[0]:
        raise ValueError("activation and output projection channel counts differ")
    return np.abs(activations) * np.linalg.norm(output_columns, axis=1)


def descending_ranking(scores: FloatArray) -> npt.NDArray[np.int64]:
    """Return stable flattened indices in descending score order."""
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(flat).all():
        raise ValueError("importance scores must be finite")
    return np.argsort(-flat, kind="stable").astype(np.int64, copy=False)


def random_ranking(size: int, seed: int) -> npt.NDArray[np.int64]:
    if size < 0:
        raise ValueError("size must be non-negative")
    return np.random.default_rng(seed).permutation(size)


def layer_stratified_ranking(scores: FloatArray) -> npt.NDArray[np.int64]:
    """Round-robin layers while preserving each layer's local score ranking."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3:
        raise ValueError("scores must have shape [layer, position, channel]")
    layers, positions, channels = scores.shape
    per_layer = [
        descending_ranking(scores[layer]) + layer * positions * channels
        for layer in range(layers)
    ]
    return np.asarray(
        [per_layer[layer][rank] for rank in range(positions * channels) for layer in range(layers)],
        dtype=np.int64,
    )
