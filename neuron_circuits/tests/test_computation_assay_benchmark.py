import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_computation_assay_benchmark.py"
SPEC = importlib.util.spec_from_file_location("assay_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_rotation_preserves_effects():
    rng = np.random.default_rng(4)
    hidden = rng.normal(size=(32, 16)).astype(np.float32)
    decoder = rng.normal(size=(16, 7)).astype(np.float32)
    rotation = MODULE.orthogonal(rng, 16)
    rotated, rotated_decoder = MODULE.rotate_representation(hidden, decoder, rotation)
    assert np.allclose(MODULE.effects(hidden, decoder), MODULE.effects(rotated, rotated_decoder), atol=2e-5)


def test_s90_recovers_one_channel_effect():
    hidden = np.zeros((8, 16), dtype=np.float32)
    hidden[:, 3] = 2
    decoder = np.zeros((16, 4), dtype=np.float32)
    decoder[3, 1] = 1
    assert np.all(MODULE.s90_counts(hidden, decoder) == 1)


def test_operation_subspace_recovers_rotated_rank_two_code():
    rng = np.random.default_rng(3)
    operations = np.repeat(np.arange(3), 64)
    hidden = np.zeros((len(operations), 12), dtype=np.float32)
    for operation in range(3):
        keep = operations == operation
        hidden[keep, 2 * operation:2 * operation + 2] = rng.normal(size=(keep.sum(), 2))
    decoder = rng.normal(size=(12, 5)).astype(np.float32)
    rotation = MODULE.orthogonal(rng, 12)
    rotated, rotated_decoder = MODULE.rotate_representation(hidden, decoder, rotation)
    fidelity = MODULE.operation_subspace_fidelity(
        rotated, operations, rotated, operations, rotated_decoder, 2,
    )
    assert fidelity > .9999


def test_constraint_positive_control_orders_taut_and_diffuse():
    result = MODULE.constraint_system(0)
    assert result["taut"]["finite_edit_origin_r2"] > .9999
    assert result["diffuse"]["finite_edit_origin_r2"] > .9999
    assert result["taut"]["mean_absolute_cosine"] > result["diffuse"]["mean_absolute_cosine"]
    assert result["diffuse"]["effective_support"] > result["taut"]["effective_support"]
