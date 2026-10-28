import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_final_destination_head_program import make_write_worlds, mask_heads


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_final_destination_head_program.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_final_destination_head_program.json"


def test_mask_worlds_recompose_heads():
    torch.manual_seed(0)
    base = torch.randn(2, 5, 7)
    deltas = torch.randn(3, 2, 7)
    worlds = make_write_worlds(base, deltas, [0, 1, 3, 7], 4)
    assert torch.equal(worlds[0], base)
    assert torch.allclose(worlds[1, :, 4] - base[:, 4], deltas[0])
    assert torch.allclose(worlds[2, :, 4] - base[:, 4], deltas[:2].sum(0))
    assert torch.allclose(worlds[3, :, 4] - base[:, 4], deltas.sum(0))
    assert mask_heads(5, 3) == [0, 2]


def test_preregistration_is_locked():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"].startswith("locked-before")
    assert payload["edit_positions"] == [63, 95, 119]
    assert payload["layers"] == [5, 6, 8, 10]
    assert len(payload["hypotheses"]) == 9


def test_completed_output_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
    assert payload["hypotheses"]["H1_head_recomposition_is_exact"]["supported"] is False
    for corpus in payload["evaluation"].values():
        for position in corpus.values():
            for row in position.values():
                # The locked 1e-6 gate failed from float32 reassociation. Keep
                # the failure while guarding against a materially broken sum.
                assert row["relative_state_recomposition_error"] <= 1e-4
                assert "63" in row["masks"]
