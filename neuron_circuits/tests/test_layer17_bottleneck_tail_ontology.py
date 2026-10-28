import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer17_bottleneck_tail_ontology import (  # noqa: E402
    fit_lda_readout,
    make_site_labels,
    project_complement,
)


def test_projection_complement_is_exact_and_orthogonal():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(23, 9))
    basis, _ = np.linalg.qr(rng.normal(size=(9, 3)))
    kept, residual = project_complement(values, basis)
    assert np.allclose(kept + residual, values, atol=1e-10)
    assert np.allclose(residual @ basis, 0, atol=1e-10)


def test_lda_readout_recovers_heldout_signal():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(500, 4))
    labels = train[:, 0] + 0.3 * train[:, 1] > 0
    readout = fit_lda_readout(train, labels, ridge=0.1)
    test = rng.normal(size=(500, 4))
    test_labels = test[:, 0] + 0.3 * test[:, 1] > 0
    scores = (test - readout["mean"]) / readout["scale"] @ readout["weight"]
    assert np.corrcoef(scores, test_labels.astype(float))[0, 1] > 0.7


def test_site_labels_align_current_and_next_tokens():
    rows = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]])
    flags = np.eye(8, dtype=np.uint8)
    labels = make_site_labels(rows, [0, 2], flags, eos_id=7, category_names=[f"c{i}" for i in range(8)])
    assert labels["current/c0"].tolist() == [True, False, False, False]
    assert labels["next/c1"].tolist() == [True, False, False, False]
    assert labels["next/eos"].tolist() == [False, False, False, True]


def test_config_locks_large_ood_and_causal_redteam():
    config = json.loads(Path("configs/bilin18_layer17_bottleneck_tail_ontology.json").read_text())
    assert "locked" in config["status"]
    assert config["discovery"]["rows"] >= 1000
    assert set(config["evaluation"]) == {"fineweb", "wikitext", "tinystories"}
    assert 16 in config["tail_ranks"]
    assert len(config["hypotheses"]) >= 8
