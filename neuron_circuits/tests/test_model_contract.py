import unittest

import numpy as np

from neuron_circuits.model import ForwardTrace, validate_stage_a_adapter


class RecomputingToyModel:
    mask_shape = (2, 1, 2)

    def forward(self, token_ids, mask=None, *, baseline="zero"):
        del baseline
        x = np.array([[float(sum(token_ids)), 1.0]])
        first = x * np.array([[2.0, -1.0]])
        first_write = first if mask is None else first * mask.for_layer(0)
        residual = x + first_write
        second = residual * np.array([[0.5, 3.0]])
        second_write = second if mask is None else second * mask.for_layer(1)
        logits = (residual + second_write).reshape(-1)
        return ForwardTrace(logits=logits, mlp_activations=(first, second))


class CachedDownstreamToyModel(RecomputingToyModel):
    def forward(self, token_ids, mask=None, *, baseline="zero"):
        trace = super().forward(token_ids, mask, baseline=baseline)
        if mask is not None and not mask.values.all():
            intact = super().forward(token_ids, None, baseline=baseline)
            return ForwardTrace(trace.logits, (trace.mlp_activations[0], intact.mlp_activations[1]))
        return trace


class ModelContractTests(unittest.TestCase):
    def test_valid_adapter_passes_both_checks(self) -> None:
        report = validate_stage_a_adapter(RecomputingToyModel(), [2, 3])
        self.assertTrue(report.all_one_matches)
        self.assertTrue(report.downstream_changed)

    def test_cached_downstream_activation_is_detected(self) -> None:
        report = validate_stage_a_adapter(CachedDownstreamToyModel(), [2, 3])
        self.assertTrue(report.all_one_matches)
        self.assertFalse(report.downstream_changed)


if __name__ == "__main__":
    unittest.main()
