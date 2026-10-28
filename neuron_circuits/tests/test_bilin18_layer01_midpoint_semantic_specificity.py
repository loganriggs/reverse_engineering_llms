import sys
from pathlib import Path
from types import SimpleNamespace

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin18_layer01_midpoint_semantic_specificity import matched_targets


def test_matched_targets_exclude_same_family():
    embedding = torch.nn.Embedding(8, 4)
    model = SimpleNamespace(embedding=embedding, config=SimpleNamespace(n_embd=4))
    families = {"a": [[0, 1], [2, 3]], "b": [[4, 5], [6, 7]]}
    result = matched_targets(model, families)
    assert result[(0, 1)]["target"] in {5, 7}
    assert result[(2, 3)]["target"] in {5, 7}
