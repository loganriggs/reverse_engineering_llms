import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer17_tangent_bundle_quotient.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_tangent_bundle_quotient.json"


def test_tangent_bundle_config_has_disjoint_selection_and_evaluation():
    config = json.loads(CONFIG.read_text())
    discovery = config["discovery"]
    fineweb = config["evaluation"]["fineweb"]
    discovery_sites = discovery["rows"] * len(config["positions"])
    assert config["fit_sites"] < discovery_sites
    assert discovery["start"] + discovery["rows"] <= fineweb["start"]
    assert config["tensor_rank"] in config["ranks"]
    assert config["random_rank"] in config["ranks"]


def test_role_tensor_identity_on_small_random_problem():
    generator = torch.Generator().manual_seed(7)
    d, c, ro, rd, rx = 11, 13, 4, 5, 6
    left = torch.randn(c, d, generator=generator)
    right = torch.randn(c, d, generator=generator)
    down = torch.randn(d, c, generator=generator)
    output = torch.linalg.qr(torch.randn(d, ro, generator=generator)).Q
    tangent = torch.linalg.qr(torch.randn(d, rd, generator=generator)).Q
    context = torch.linalg.qr(torch.randn(d, rx, generator=generator)).Q
    tensor = torch.einsum(
        "oc,ci,cj->oij", output.T @ down,
        left @ tangent, right @ context,
    ) + torch.einsum(
        "oc,ci,cj->oij", output.T @ down,
        right @ tangent, left @ context,
    )
    x = torch.randn(9, d, generator=generator)
    delta = torch.randn(9, 3, d, generator=generator)
    x_projected = (x @ context) @ context.T
    delta_projected = (delta @ tangent) @ tangent.T
    direct = torch.einsum(
        "bnc,bc,dc->bnd",
        torch.einsum("bnd,cd->bnc", delta_projected, left)
        * (x_projected @ right.T)[:, None]
        + torch.einsum("bnd,cd->bnc", delta_projected, right)
        * (x_projected @ left.T)[:, None],
        torch.ones(9, c), down,
    ) @ output
    contracted = torch.einsum(
        "bni,oij,bj->bno", delta @ tangent, tensor, x @ context)
    torch.testing.assert_close(contracted, direct, atol=2e-5, rtol=2e-5)


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
