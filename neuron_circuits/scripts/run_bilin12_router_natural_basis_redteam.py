#!/usr/bin/env python3
"""Test whether the compact quadratic-router basis needs edit supervision."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_boundary_bilinear_router import causal
from run_bilin12_boundary_output_interface import token_id
from run_bilin12_boundary_state_contraction import paired_panel
from run_bilin12_pooled_router_tensor_canonicalization import coefficients
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_natural_basis_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_natural_basis_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_NATURAL_BASIS_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin12_router_natural_basis_redteam.png"


def pca(states):
    _, _, vh = torch.linalg.svd(states.float(), full_matrices=False)
    return vh.T.contiguous()


def overlap(left, right, rank):
    return float((left[:, :rank].T @ right[:, :rank]).square().sum() / rank)


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    target = token_id(tokenizer, config["target"])
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": config["output_rank"],
                                                   "discovery": config["router_discovery"]})
    block = model.blocks[11]; dimension = model.config.n_embd
    channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(),
                       block.mlp.Right.weight.float(), channel_to_output)
    forms = (raw + raw.transpose(-1, -2)) / 2
    discovery = paired_panel(model, config["router_discovery"], 11, target, config["batch_size"])
    paired_states = torch.cat((F.rms_norm(discovery[0], (dimension,)),
                               F.rms_norm(discovery[3], (dimension,))))
    base_states = F.rms_norm(discovery[0], (dimension,))
    natural_a, _ = collect_states_writes(model, 11, config["natural_a"],
                                          config["positions"], config["batch_size"])
    natural_b, _ = collect_states_writes(model, 11, config["natural_b"],
                                          config["positions"], config["batch_size"])
    bases = {"paired": pca(paired_states), "base_only": pca(base_states),
             "natural_a": pca(natural_a), "natural_b": pca(natural_b)}
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    bases["random"] = torch.linalg.qr(torch.randn(dimension, max(config["input_ranks"]),
                                                   generator=generator, device="cuda")).Q
    overlaps = {str(rank): overlap(bases["natural_a"], bases["natural_b"], rank)
                for rank in config["input_ranks"]}
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        panel = paired_panel(model, spec, 11, target, config["batch_size"])
        base = F.rms_norm(panel[0], (dimension,)).float()
        changed = F.rms_norm(panel[3], (dimension,)).float(); edit = changed - base
        target_coefficients = (panel[4] - panel[1]).float() @ write_basis
        oracle = causal(model, panel, target_coefficients @ write_basis.T)
        frontier = {}
        for rank in config["input_ranks"]:
            rows = {}
            for name, basis in bases.items():
                predicted = coefficients(forms, basis[:, :rank], base, edit)
                metric = causal(model, panel, predicted @ write_basis.T)
                projected_edit = (edit @ basis[:, :rank]) @ basis[:, :rank].T
                rows[name] = {**metric, "edit_energy_fraction": float(
                    (projected_edit.square().sum(-1) / edit.square().sum(-1).clamp_min(1e-30)).median())}
            frontier[str(rank)] = rows
        results[corpus] = {"oracle": oracle, "frontier": frontier}
    gate = lambda row, oracle: row["fidelity"] >= .70 and row["fidelity"] >= oracle["fidelity"] - .15
    n0 = all(gate(row["frontier"]["16"]["natural_a"], row["oracle"]) for row in results.values())
    n1 = all(row["frontier"]["16"]["natural_a"]["fidelity"]
             >= row["frontier"]["16"]["paired"]["fidelity"] - .05 for row in results.values())
    n2 = all(abs(row["frontier"]["16"]["natural_a"]["fidelity"]
                 - row["frontier"]["16"]["natural_b"]["fidelity"]) <= .05
             for row in results.values())
    n3 = all(row["frontier"][str(rank)][basis]["fidelity"]
             >= row["frontier"][str(rank)]["random"]["fidelity"] + .20
             for row in results.values() for rank in (8,16,32)
             for basis in ("natural_a", "natural_b"))
    n4 = overlaps["16"] >= .80
    n5 = any(row["frontier"]["16"]["natural_a"]["fidelity"]
             >= row["frontier"]["16"]["natural_a"]["edit_energy_fraction"] + .20
             for row in results.values())
    hypotheses = {
        "N0_natural_rank16_is_sufficient": {"supported": n0},
        "N1_target_conditioning_is_unnecessary": {"supported": n1},
        "N2_natural_basis_replicates_across_slices": {"supported": n2},
        "N3_natural_basis_beats_random": {"supported": n3},
        "N4_natural_basis_is_geometrically_stable": {"supported": n4, "overlaps": overlaps},
        "N5_behavioral_compression_exceeds_state_energy": {"supported": n5},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "overlaps": overlaps, "results": results}, indent=2, allow_nan=False))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
    for axis, (corpus, row) in zip(axes, results.items()):
        for name in ("paired", "base_only", "natural_a", "natural_b", "random"):
            axis.plot(config["input_ranks"], [row["frontier"][str(rank)][name]["fidelity"]
                                               for rank in config["input_ranks"]], marker="o", label=name)
        axis.axhline(row["oracle"]["fidelity"], color="black", lw=.8, label="oracle")
        axis.set_xscale("log", base=2); axis.set_title(corpus); axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    lines = ["# Natural-basis red team of the quadratic router", "",
             "The exact quadratic router is held fixed. Its shared input space is now learned either from newline-paired states, base states only, ordinary unedited states at four positions on either of two disjoint FineWeb slices, or a random control. No natural basis sees token-edit labels or write deltas.", "",
             "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Natural basis red team](bilin12_router_natural_basis_redteam.png)", "",
              "| Corpus | Oracle | Rank-16 paired/base/natural-A/natural-B/random | Natural-A edit energy |",
              "|---|---:|---:|---:|"]
    for corpus, row in results.items():
        values = "/".join(f"{row['frontier']['16'][name]['fidelity']:.3f}" for name in
                          ("paired", "base_only", "natural_a", "natural_b", "random"))
        lines.append(f"| {corpus} | {row['oracle']['fidelity']:.3f} | {values} | "
                     f"{row['frontier']['16']['natural_a']['edit_energy_fraction']:.3f} |")
    lines += ["", f"Natural slice A/B rank-16 overlap: `{overlaps['16']:.3f}`.", "",
              "If unedited-state bases match the paired basis, the compactness is a property of the naturally occupied state manifold rather than leakage from the chosen newline edit. This still does not make individual PCA axes semantic or unique; the stable object is the subspace plus the exact quadratic operator restricted to it."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
