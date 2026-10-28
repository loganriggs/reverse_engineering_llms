"""Tools for extracting and evaluating native-neuron circuits."""

from .masks import StageAMask
from .metrics import FidelityMetrics, compute_fidelity
from .model import ForwardTrace, MaskValidationReport, StageAModel, validate_stage_a_adapter

__all__ = [
    "FidelityMetrics",
    "ForwardTrace",
    "MaskValidationReport",
    "StageAMask",
    "StageAModel",
    "compute_fidelity",
    "validate_stage_a_adapter",
]
