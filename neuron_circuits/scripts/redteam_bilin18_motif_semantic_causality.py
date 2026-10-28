#!/usr/bin/env python3
"""Magnitude-equalized finite causal red team for frozen semantic motifs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_qkov_causal_edges import forward_prefix, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


SOURCE = TENSOR_ROOT / "bilin18_unsupervised_motif_semantic_validation.json"
OUTPUT = TENSOR_ROOT / "bilin18_unsupervised_motif_semantic_causal_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_UNSUPERVISED_MOTIF_SEMANTIC_CAUSAL_REDTEAM.md"


def kl(reference, changed):
    log_p = F.log_softmax(reference, -1); p = log_p.exp()
    return float((p * (log_p - F.log_softmax(changed, -1))).sum())


@torch.inference_mode()
def main():
    source = json.loads(SOURCE.read_text())
    config = source["config"]
    corpus = np.load(config["corpus"], mmap_mode="r")
    panel = corpus[config["start"]:config["start"] + config["rows"]]
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.float().eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    records = {}; families = {}
    for family in config["families"]:
        family_rows = []
        for key, rows in source["causal_records"].items():
            if not key.startswith(family + "/"): continue
            updated = []
            for row in rows:
                local_row = row["row"] - config["start"]
                raw = np.asarray(panel[local_row, :row["destination"] + 2], dtype=np.int64)
                ids = torch.as_tensor(raw[:row["destination"] + 1], device="cuda")[None]
                base, _ = forward_prefix(model, ids, row["layer"], 0)
                removal_fraction = min(1., row["control_magnitude"] / max(row["magnitude"], 1e-30))
                changed, _ = forward_prefix(
                    model, ids, row["layer"], 0,
                    edge=("scale", row["head"], row["destination"], row["source"], 1 - removal_fraction),
                )
                target = int(raw[row["destination"] + 1])
                equalized_kl = kl(base, changed)
                equalized_logprob = float(F.log_softmax(changed, -1)[target] - F.log_softmax(base, -1)[target])
                item = {**row, "semantic_removal_fraction": removal_fraction,
                        "equalized_semantic_kl": equalized_kl,
                        "equalized_semantic_target_logprob_change": equalized_logprob,
                        "equalized_kl_control_ratio": equalized_kl / max(row["control_kl"], 1e-30)}
                updated.append(item); family_rows.append(item)
            records[key] = updated
        ratios = np.asarray([row["equalized_kl_control_ratio"] for row in family_rows])
        semantic = np.asarray([row["equalized_semantic_kl"] for row in family_rows])
        control = np.asarray([row["control_kl"] for row in family_rows])
        families[family] = {
            "sites": len(family_rows),
            "median_removal_fraction": float(np.median([row["semantic_removal_fraction"] for row in family_rows])),
            "median_equalized_semantic_kl": float(np.median(semantic)),
            "median_control_kl": float(np.median(control)),
            "ratio_of_medians": float(np.median(semantic) / max(np.median(control), 1e-30)),
            "median_paired_ratio": float(np.median(ratios)),
            "paired_win_fraction": float(np.mean(semantic > control)),
            "median_abs_target_logprob_change": float(np.median(np.abs([
                row["equalized_semantic_target_logprob_change"] for row in family_rows]))),
        }
    hypotheses = {
        "E1_recurrence_survives_equalization": {
            "supported": families["recurrence_reference"]["median_paired_ratio"] >= 1.5
            and families["recurrence_reference"]["paired_win_fraction"] >= .75,
            **families["recurrence_reference"],
        },
        "E2_both_local_composition_motifs_survive": {
            "supported": all(families[name]["median_paired_ratio"] >= 1.5
                             and families[name]["paired_win_fraction"] >= .65
                             for name in ("local_composition_a", "local_composition_b")),
        },
        "E3_boundary_survives_equalization": {
            "supported": families["boundary_multiscale"]["median_paired_ratio"] >= 1.5
            and families["boundary_multiscale"]["paired_win_fraction"] >= .65,
            **families["boundary_multiscale"],
        },
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {"source": str(SOURCE), "method": "partial semantic-edge removal matched to the control edge's exact local-write magnitude",
               "hypotheses": hypotheses, "families": families, "records": records}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Magnitude-equalized semantic-motif causal red team", "",
             "Each semantic edge was only partially removed, so the norm of its removed local attention write equaled the fully removed nonsemantic control edge at the same head and destination.", "", "## Outcomes", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| family | sites | semantic fraction removed | equalized semantic KL | control KL | paired ratio | win fraction |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in families.items():
        lines.append(f"| {name} | {row['sites']} | {row['median_removal_fraction']:.3f} | {row['median_equalized_semantic_kl']:.3g} | {row['median_control_kl']:.3g} | {row['median_paired_ratio']:.2f} | {row['paired_win_fraction']:.1%} |")
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "families": families}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
