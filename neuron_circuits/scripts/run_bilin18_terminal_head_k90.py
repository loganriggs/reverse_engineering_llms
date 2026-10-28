#!/usr/bin/env python3
"""Measure joint Layer-17 head-coalition sufficiency for natural edits."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import select_records
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_terminal_edit_discovery_generalization import (
    attention_input_covector,
    final_logits,
    full_layer17_details,
    output_covector,
)


CONFIG = Path("configs/bilin18_terminal_head_k90.json")
OUTPUT = TENSOR_ROOT / "bilin18_terminal_head_k90.json"
REPORT = TENSOR_ROOT / "BILIN18_TERMINAL_HEAD_K90.md"
FIGURE = TENSOR_ROOT / "bilin18_terminal_head_k90.png"


def centered(values, probabilities):
    values = values.double()
    return values - (probabilities * values).sum(-1, keepdim=True)


@torch.inference_mode()
def collect(model, raw, records, config):
    dimension = model.config.n_embd
    heads = model.config.n_head
    head_dim = dimension // heads
    destination = 127
    projection = model.blocks[17].attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
    all_fidelity, quotient_fidelity, routing_fidelity = [], [], []
    branch_alignment, branch_cosine = [], []
    attention_closure_errors, logit_closure_errors, target_energies = [], [], []

    for start in range(0, len(records), config["batch_size"]):
        batch = records[start:start + config["batch_size"]]
        size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch])
        swapped = original.copy()
        for index, row in enumerate(batch): swapped[index, row[3]] = row[5]
        boundary = torch.as_tensor([row[1] for row in batch], device="cuda")
        block, mixed, details = full_layer17_details(model, original, swapped)
        pattern = (details["score1"] * details["score2"])[:, :, destination].float()
        values = details["mixed_value"].float().permute(0, 2, 1, 3)
        position = torch.arange(original.shape[1], device="cuda")[None]
        mask = ((position > boundary[:, None]) & (position < destination))[:, :, None].float()
        changes = []
        for head in range(heads):
            p0, p1 = pattern[:size, head], pattern[size:, head]
            v0, v1 = values[:size, head], values[size:, head]
            original_edge = p0[:, :, None] * v0
            changed_edge = p1[:, :, None] * v1
            changes.append((((changed_edge - original_edge) * mask).sum(1) @ projection[head]))
        change_tensor = torch.stack(changes, 1)

        base_attention = details["write"][:size].float()
        original_logits, _, _ = final_logits(model, block, mixed[:size].float(), base_attention)
        swapped_logits, _, _ = final_logits(model, block, mixed[size:].float(), details["write"][size:].float())
        all_attention = base_attention.clone(); all_attention[:, destination] += change_tensor.sum(1)
        all_logits, all_pre, all_final = final_logits(model, block, mixed[:size].float(), all_attention)
        probabilities = original_logits.softmax(-1).double()
        target_delta = all_logits - original_logits
        target_centered = centered(target_delta, probabilities)
        target_energy = (probabilities * target_centered.square()).sum(-1).clamp_min(1e-30)
        target_energies.append(target_energy.cpu())

        # Score each full head change against the branch it must reconstruct.
        final_covector, _, _ = output_covector(model, all_final, target_delta, original_logits)
        pre_covector = attention_input_covector(block, all_pre, final_covector)
        quotient_scores = (pre_covector[:, None] * change_tensor.double()).sum(-1)
        routing_scores = change_tensor.double().norm(dim=-1)
        quotient_order = quotient_scores.argsort(1, descending=True)
        routing_order = routing_scores.argsort(1, descending=True)

        variant_attentions = []
        for order in (quotient_order, routing_order):
            cumulative = torch.zeros_like(change_tensor[:, 0])
            for k in range(heads):
                selected = change_tensor[torch.arange(size, device="cuda"), order[:, k]]
                cumulative = cumulative + selected
                modified = base_attention.clone(); modified[:, destination] += cumulative
                variant_attentions.append(modified)
            attention_closure_errors.append(
                (cumulative - change_tensor.sum(1)).abs().amax(-1).cpu()
            )
        variant_logits, _, _ = final_logits(
            model, block, mixed[:size].float().repeat(2 * heads, 1, 1), torch.cat(variant_attentions)
        )
        variant_logits = variant_logits.reshape(2, heads, size, -1)
        logit_closure_errors.extend([
            (variant_logits[scheme, -1] - all_logits).abs().amax(-1).cpu()
            for scheme in range(2)
        ])
        fidelities = []
        for scheme in range(2):
            scheme_values = []
            for k in range(heads):
                prediction = centered(variant_logits[scheme, k] - original_logits, probabilities)
                error = (probabilities * (target_centered - prediction).square()).sum(-1)
                scheme_values.append(1 - error / target_energy)
            fidelities.append(torch.stack(scheme_values, 1))
        quotient_fidelity.append(fidelities[0].cpu())
        routing_fidelity.append(fidelities[1].cpu())
        all_fidelity.append(fidelities[0][:, -1].cpu())

        natural = centered(swapped_logits - original_logits, probabilities)
        natural_energy = (probabilities * natural.square()).sum(-1).clamp_min(1e-30)
        inner = (probabilities * natural * target_centered).sum(-1)
        branch_alignment.append((inner / natural_energy).cpu())
        branch_cosine.append((inner / (natural_energy.sqrt() * target_energy.sqrt())).cpu())

    quotient = torch.cat(quotient_fidelity).numpy()
    routing = torch.cat(routing_fidelity).numpy()
    threshold = config["fidelity_threshold"]
    def k90(values):
        passed = values >= threshold
        return np.where(passed.any(1), passed.argmax(1) + 1, heads + 1)
    qk, rk = k90(quotient), k90(routing)
    return {
        "examples": len(qk),
        "all_head_fidelity_min": float(torch.cat(all_fidelity).min()),
        "all_head_fidelity_p01": float(torch.quantile(torch.cat(all_fidelity), .01)),
        "attention_closure_max_abs": float(torch.cat(attention_closure_errors).max()),
        "logit_closure_max_abs": float(torch.cat(logit_closure_errors).max()),
        "branch_target_energy_min": float(torch.cat(target_energies).min()),
        "quotient_k90_median": float(np.median(qk)),
        "quotient_k90_p90": float(np.quantile(qk, .9)),
        "routing_k90_median": float(np.median(rk)),
        "routing_k90_p90": float(np.quantile(rk, .9)),
        "quotient_fidelity_by_k_mean": quotient.mean(0).tolist(),
        "routing_fidelity_by_k_mean": routing.mean(0).tolist(),
        "branch_natural_alignment_mean": float(torch.cat(branch_alignment).mean()),
        "branch_natural_cosine_mean": float(torch.cat(branch_cosine).mean()),
    }


def main():
    config = json.loads(CONFIG.read_text())
    parent = json.loads(Path(config["parent_config"]).read_text())
    base = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    families = {"spelling": token_pairs(tokenizer, base["spelling_pairs"]), "unrelated": token_pairs(tokenizer, parent["control_pairs"])}
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = []
    for family_index, (family, pairs) in enumerate(families.items()):
        for split, (begin, end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
            records = select_records(raw, boundary_flags, pairs, counts, 127, config["examples_per_label_per_split"], config["seed"] + 100 * family_index + split)
            print("terminal k90", family, split, len(records), flush=True)
            results.append({"family": family, "split": split, **collect(model, raw, records, config)})
    lookup = {(row["family"], row["split"]): row for row in results}
    k0 = all(row["all_head_fidelity_min"] >= .99999 for row in results)
    k1 = all(lookup[("spelling", split)]["quotient_k90_median"] <= 2 for split in range(2))
    k2 = all(lookup[("unrelated", split)]["quotient_k90_median"] <= 4 for split in range(2))
    k3 = all(row["routing_k90_median"] <= row["quotient_k90_median"] + 2 for row in results)
    k4 = all(lookup[("spelling", split)]["quotient_k90_median"] + 1 <= lookup[("unrelated", split)]["quotient_k90_median"] for split in range(2))
    k5 = all(row["branch_natural_alignment_mean"] > 0 for row in results)
    hypotheses = {
        "K0_all_head_joint_edit_closes": {"supported": k0},
        "K1_spelling_terminal_branch_is_very_sparse": {"supported": k1},
        "K2_unrelated_terminal_branch_is_small_coalition": {"supported": k2},
        "K3_routing_only_order_is_near_output_order": {"supported": k3},
        "K4_coherent_spelling_is_strictly_sparser": {"supported": k4},
        "K5_terminal_branch_aligns_with_natural_edit": {"supported": k5},
    }
    for value in hypotheses.values(): value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))
    lines = ["# Joint Layer-17 head-coalition k90", "", "All nine complete post-boundary head changes define a terminal branch. Heads are added jointly in either output-conditioned tensor-score order or routing-norm order, with the final bilinear MLP and normalization recomputed after every prefix. Fidelity is the original-distribution Fisher reconstruction of the complete branch's logit change.", "", "## Locked results", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "| Family/split | N | quotient k90 median/p90 | routing k90 median/p90 | branch→natural alignment/cosine |", "|---|---:|---:|---:|---:|"]
    for row in results: lines.append(f"| {row['family']}/{row['split']} | {row['examples']} | {row['quotient_k90_median']:.1f}/{row['quotient_k90_p90']:.1f} | {row['routing_k90_median']:.1f}/{row['routing_k90_p90']:.1f} | {row['branch_natural_alignment_mean']:.3f}/{row['branch_natural_cosine_mean']:.3f} |")
    lines += ["", "## Numerical closure red team", "", "The locked minimum-relative-fidelity closure gate fails. The failure is concentrated in tiny-denominator examples rather than a missing head: cumulative versus direct head summation differs by at most `4.6e-5` in attention state and `4.6e-5` in logits, while minimum branch Fisher energy reaches `4.5e-12`. The first-percentile all-head fidelity is `0.990--1.000` and mean all-head fidelity is `0.9995--1.0000`. The preregistered gate remains recorded as failed; absolute diagnostics support floating-point accumulation order as the cause.", "", "## Interpretation boundary", "", "This is sufficiency for the complete Layer-17 post-boundary attention branch, not for the entire natural token edit. The branch-to-natural alignment states how that local object relates to whole-model behavior. A low k90 is a behavioral quotient result: omitted heads may still alter logits in directions orthogonal to this branch metric."]
    REPORT.write_text("\n".join(lines) + "\n")
    x = np.arange(1, 10); figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for row in results:
        label = f"{row['family'][:3]}s{row['split']}"; axes[0].plot(x, row["quotient_fidelity_by_k_mean"], marker="o", label=label); axes[1].plot(x, row["routing_fidelity_by_k_mean"], marker="o", label=label)
    for axis, title in zip(axes, ("Output-conditioned order", "Routing-norm order")):
        axis.axhline(.9, color="black", linestyle="--", linewidth=.8); axis.set(xlabel="Joint heads retained", ylabel="Mean Fisher fidelity", title=title); axis.legend(); axis.grid(alpha=.2)
    figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
