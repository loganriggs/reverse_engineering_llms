#!/usr/bin/env python3
"""Separate QK address and OV source-write contributions to semantic routing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from validate_bilin18_unsupervised_motif_semantics import CONFIG, masks, safe_enrichment


OUTPUT = TENSOR_ROOT / "bilin18_unsupervised_motif_route_decomposition.json"
REPORT = TENSOR_ROOT / "BILIN18_UNSUPERVISED_MOTIF_ROUTE_DECOMPOSITION.md"


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    boundary_ids = {tokenizer.eos_token_id}
    for text in ("\n", ".", "!", "?", ";", ":"):
        token = tokenizer.encode(text, add_special_tokens=False)
        if len(token) == 1: boundary_ids.add(token[0])
    source = np.load(config["corpus"], mmap_mode="r")
    panel = source[config["start"]:config["start"] + config["rows"]]
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.float().eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    positions = config["positions"]; dimension = model.config.n_embd
    heads = model.config.n_head; head_dim = dimension // heads
    needed_layers = sorted({layer for family in config["families"].values() for layer, _ in family["heads"]})
    masks_by_layer = {layer: sorted({family["mask"] for family in config["families"].values()
                                     if any(target == layer for target, _ in family["heads"])})
                      for layer in needed_layers}
    # component -> [semantic mass, total mass, semantic count, available count]
    stats = {(kind, layer, head, component): [0., 0., 0, 0]
             for layer in needed_layers for kind in masks_by_layer[layer]
             for head in range(heads) for component in ("joint", "address", "write")}
    for start in range(0, len(panel), config["batch_size"]):
        raw = np.asarray(panel[start:start + config["batch_size"], :config["sequence_length"]], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for layer, block in enumerate(model.blocks[:max(needed_layers) + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            bus = details["value_bus"]
            if layer in needed_layers:
                projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
                source_norm = torch.einsum("bshk,hkd->bshd", details["mixed_value"].float(), projection).norm(dim=-1)
                address = details["pattern"][:, :, positions].abs().float().permute(0, 2, 1, 3)
                write = source_norm.permute(0, 2, 1)[:, None].expand_as(address)
                joint = address * write
                causal = torch.arange(config["sequence_length"], device="cuda")[None, :] <= torch.as_tensor(positions, device="cuda")[:, None]
                for kind in masks_by_layer[layer]:
                    mask = masks(kind, ids, positions, boundary_ids)
                    available = int(causal.sum()) * len(ids)
                    count = int(mask.sum())
                    for head in range(heads):
                        for component, values in (("joint", joint[:, :, head]),
                                                  ("address", address[:, :, head]),
                                                  ("write", write[:, :, head])):
                            valid = values * causal[None]
                            row = stats[(kind, layer, head, component)]
                            row[0] += float(values[mask].sum()); row[1] += float(valid.sum())
                            row[2] += count; row[3] += available
            pre = mixed + details["write"]
            normalized = F.rms_norm(pre, (dimension,))
            x = pre + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
    families = {}
    for name, family in config["families"].items():
        result = {}
        for component in ("joint", "address", "write"):
            values = []
            controls = []
            for layer, head in family["heads"]:
                row = stats[(family["mask"], layer, head, component)]
                values.append(safe_enrichment(row[0], row[1], row[2], row[3])[0])
            for layer in sorted(set(layer for layer, _ in family["heads"])):
                for head in range(heads):
                    row = stats[(family["mask"], layer, head, component)]
                    controls.append(safe_enrichment(row[0], row[1], row[2], row[3])[0])
            result[component] = {
                "family_median_enrichment": float(np.median(values)),
                "layer_control_median_enrichment": float(np.median(controls)),
                "advantage": float(np.median(values) - np.median(controls)),
                "per_head": values,
            }
        families[name] = result
    hypotheses = {
        "D1_recurrence_is_address_selective": {"supported": families["recurrence_reference"]["address"]["advantage"] >= 1,
            **families["recurrence_reference"]["address"]},
        "D2_local_composition_is_address_selective": {"supported": all(families[name]["address"]["advantage"] >= 2
            for name in ("local_composition_a", "local_composition_b"))},
        "D3_boundary_is_address_selective": {"supported": families["boundary_multiscale"]["address"]["advantage"] >= .5,
            **families["boundary_multiscale"]["address"]},
        "D4_semantics_require_conjunction": {"supported": any(row["joint"]["advantage"] > row["address"]["advantage"]
            and row["joint"]["advantage"] > row["write"]["advantage"] for row in families.values())},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {"config": config, "hypotheses": hypotheses, "families": families}
    OUTPUT.write_text(json.dumps(payload, indent=2))
    lines = ["# Semantic-route address/write decomposition", "",
             "Held-out semantic enrichment is decomposed into absolute QK1×QK2 address weight, source OV-write norm, and their product.", "", "## Outcomes", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| family | joint | address | write | joint control | address control | write control |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in families.items():
        lines.append(f"| {name} | {row['joint']['family_median_enrichment']:.2f} | {row['address']['family_median_enrichment']:.2f} | {row['write']['family_median_enrichment']:.2f} | {row['joint']['layer_control_median_enrichment']:.2f} | {row['address']['layer_control_median_enrichment']:.2f} | {row['write']['layer_control_median_enrichment']:.2f} |")
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "families": families}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
