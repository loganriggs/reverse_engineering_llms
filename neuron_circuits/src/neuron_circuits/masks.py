"""Mask representations shared by extraction and evaluation code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt


BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class StageAMask:
    """A binary MLP activation mask with [layer, position, channel] axes."""

    values: BoolArray

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 3:
            raise ValueError(
                "Stage A masks must have shape [layer, position, channel]; "
                f"got {values.shape}"
            )
        if values.dtype != np.bool_:
            raise TypeError("Stage A masks must use the boolean dtype")

    @classmethod
    def full(cls, shape: tuple[int, int, int]) -> "StageAMask":
        return cls(np.ones(shape, dtype=np.bool_))

    @classmethod
    def empty(cls, shape: tuple[int, int, int]) -> "StageAMask":
        return cls(np.zeros(shape, dtype=np.bool_))

    @classmethod
    def from_ranked_indices(
        cls,
        shape: tuple[int, int, int],
        ranked_indices: Iterable[int],
        retained: int,
    ) -> "StageAMask":
        """Retain the first `retained` unique flattened indices in a ranking."""
        if retained < 0 or retained > int(np.prod(shape)):
            raise ValueError("retained must be between zero and the mask size")

        flat = np.zeros(int(np.prod(shape)), dtype=np.bool_)
        if retained == 0:
            return cls(flat.reshape(shape))
        chosen = 0
        for index in ranked_indices:
            index = int(index)
            if index < 0 or index >= flat.size:
                raise IndexError(f"ranked mask index {index} is out of bounds")
            if not flat[index]:
                flat[index] = True
                chosen += 1
                if chosen == retained:
                    break
        if chosen != retained:
            raise ValueError("ranking does not contain enough unique indices")
        return cls(flat.reshape(shape))

    @property
    def retained_count(self) -> int:
        return int(self.values.sum())

    @property
    def retained_fraction(self) -> float:
        return self.retained_count / self.values.size

    def layer_position_counts(self) -> npt.NDArray[np.int64]:
        return self.values.sum(axis=2, dtype=np.int64)

    def for_layer(self, layer: int) -> BoolArray:
        """Return the [position, channel] mask applied after an MLP activation."""
        return self.values[layer]
