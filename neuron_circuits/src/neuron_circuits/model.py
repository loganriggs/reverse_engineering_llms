"""Model-facing contract and validation for Stage A interventions.

Adapters should implement :class:`StageAModel`. The contract deliberately asks
for a new ``forward`` call for every evaluation; an adapter must rebuild all
downstream activations and autoregressive caches inside that call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from .masks import StageAMask


FloatArray = npt.NDArray[np.floating]
AblationBaseline = Literal["zero", "mean", "corrupted"]


@dataclass(frozen=True)
class ForwardTrace:
    """Outputs from one independently computed masked forward pass."""

    logits: FloatArray
    # Post-nonlinearity, pre-output-projection activations, one [position,
    # channel] array per layer. Values reflect the current masked upstream state
    # and are captured before applying the current layer's mask.
    mlp_activations: tuple[FloatArray, ...]


class StageAModel(Protocol):
    @property
    def mask_shape(self) -> tuple[int, int, int]: ...

    def forward(
        self,
        token_ids: Sequence[int],
        mask: StageAMask | None = None,
        *,
        baseline: AblationBaseline = "zero",
    ) -> ForwardTrace: ...


@dataclass(frozen=True)
class MaskValidationReport:
    all_one_max_logit_error: float
    all_one_matches: bool
    upstream_probe: tuple[int, int, int] | None
    downstream_changed: bool | None


def validate_stage_a_adapter(
    model: StageAModel,
    token_ids: Sequence[int],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> MaskValidationReport:
    """Check all-one equivalence and evidence of downstream recomputation.

    The second check ablates the largest-magnitude activation before the final
    layer. It is a smoke test, not a proof: a model can have a genuinely dead
    upstream activation, so adapters should run it across several prompts.
    """
    intact = model.forward(token_ids, None)
    all_one = model.forward(token_ids, StageAMask.full(model.mask_shape))
    if intact.logits.shape != all_one.logits.shape:
        raise ValueError("masked and intact forwards returned different logit shapes")
    max_error = float(np.max(np.abs(intact.logits - all_one.logits)))
    matches = bool(np.allclose(intact.logits, all_one.logits, atol=atol, rtol=rtol))

    layer_count, positions, channels = model.mask_shape
    if len(intact.mlp_activations) != layer_count:
        raise ValueError("trace activation count does not match mask layer count")
    if layer_count < 2:
        return MaskValidationReport(max_error, matches, None, None)

    candidates = np.stack(
        [np.abs(np.asarray(layer)) for layer in intact.mlp_activations[:-1]], axis=0
    )
    expected_shape = (layer_count - 1, positions, channels)
    if candidates.shape != expected_shape:
        raise ValueError(
            f"trace activations have shape {candidates.shape}; expected {expected_shape}"
        )
    probe = tuple(int(i) for i in np.unravel_index(candidates.argmax(), candidates.shape))
    values = np.ones(model.mask_shape, dtype=np.bool_)
    values[probe] = False
    ablated = model.forward(token_ids, StageAMask(values))
    changed = not np.allclose(
        intact.mlp_activations[-1], ablated.mlp_activations[-1], atol=atol, rtol=rtol
    )
    return MaskValidationReport(max_error, matches, probe, bool(changed))
