import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin12_layer11_bottleneck_tail_replication import (  # noqa: E402
    effective_rank,
    project_complement,
)


def test_projection_decomposition_is_exact_and_orthogonal():
    rng = np.random.default_rng(11)
    values = rng.normal(size=(41, 13))
    basis, _ = np.linalg.qr(rng.normal(size=(13, 4)))
    kept, residual = project_complement(values, basis)
    assert np.allclose(kept + residual, values, atol=1e-10)
    assert np.allclose(residual @ basis, 0, atol=1e-10)


def test_effective_rank_bounds():
    assert np.isclose(effective_rank(np.ones(8)), 8)
    assert np.isclose(effective_rank(np.asarray([3.0, 0.0, 0.0])), 1)


def test_config_locks_cross_corpus_tail_and_behavior_redteam():
    config = json.loads(Path("configs/bilin12_layer11_bottleneck_tail_replication.json").read_text())
    assert "locked" in config["status"]
    assert config["layer"] == 11
    assert config["discovery"]["rows"] * len(config["positions"]) >= 8192
    assert set(config["evaluation"]) == {"fineweb", "wikitext", "tinystories"}
    assert set(config["behavior"]) == set(config["evaluation"])
    assert 16 in config["tail_ranks"] and 64 in config["tail_ranks"]
    assert len(config["hypotheses"]) == 8


def test_script_uses_post_bias_mean_and_exact_final_layer_path():
    source = Path("scripts/run_bilin12_layer11_bottleneck_tail_replication.py").read_text()
    assert "post_bias_mean = pre_bias_mean + block.mlp.Down_bias.float()" in source
    assert "final_layer_variant_logits" in source
    assert "raw4_tail16" in source
    assert "natural20" in source
    assert 'row["behavior"]["full"]["median_absolute_kl"] <= 1e-7' in source
