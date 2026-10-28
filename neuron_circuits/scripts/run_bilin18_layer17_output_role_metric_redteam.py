#!/usr/bin/env python3
"""Separate uniform-vocabulary, natural-frequency, and RMS-aware output roles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_layer17_tangent_output_semantics import (
    base_label,
    make_site_labels,
    rms_vjp,
    select_individual,
    vocabulary_table,
)
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_output_role_metric_redteam.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_output_role_metric_redteam.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_OUTPUT_ROLE_METRIC_REDTEAM.md")


@torch.inference_mode()
def collect(model, tokenizer, spec, config, axes, vocabulary, base_names):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    positions = np.asarray(config["positions"])
    position_tensor = torch.as_tensor(positions, device="cuda")
    stores = {
        component: {variant: [] for variant in ("raw", "rms", "formula", "exact")}
        for component in ("target", "top1")
    }
    current_rows, target_rows, top1_rows, position_rows, row_rows = [], [], [], [], []
    maximum_error = 0.0
    for start in range(0, len(rows), config["batch_size"]):
        raw_ids = np.asarray(rows[start:start + config["batch_size"], :129], dtype=np.int64)
        ids = torch.as_tensor(raw_ids[:, :128], device="cuda")
        pre, state = prefix_to_layer17(model, ids)
        selected_state = state[:, position_tensor].float()
        block = model.blocks[17]
        write = block.mlp.Down(block.mlp.Left(selected_state) * block.mlp.Right(selected_state)) + block.mlp.Down_bias.float()
        residual = pre[:, position_tensor].float() + write
        final = F.rms_norm(residual, (residual.shape[-1],))
        raw_logits = model.lm_head(final).float()[..., :tokenizer.vocab_size]
        logits = 30 * torch.tanh(raw_logits / 30)
        token_sets = {
            "target": torch.as_tensor(raw_ids[:, positions + 1], device="cuda"),
            "top1": logits.argmax(-1),
        }
        flat_residual = residual.flatten(0, 1)
        root = flat_residual.square().mean(-1, keepdim=True).sqrt()
        axis_rows = axes.T[None].expand(len(flat_residual), -1, -1)
        coupling = (flat_residual[:, None] * axis_rows).mean(-1, keepdim=True)
        normalized_axes = axis_rows / root[:, None] - flat_residual[:, None] * coupling / root[:, None].pow(3)
        for component, token_ids in token_sets.items():
            flat_tokens = token_ids.flatten()
            token_weight = model.lm_head.weight.float()[flat_tokens]
            selected_raw = raw_logits.flatten(0, 1).gather(1, flat_tokens[:, None]).squeeze(1)
            cap = 1 - torch.tanh(selected_raw / 30).square()
            raw_score = token_weight @ axes
            rms_score = torch.einsum("bd,bmd->bm", token_weight, normalized_axes)
            formula = cap[:, None] * rms_score
            exact_adjoint = rms_vjp(flat_residual, cap[:, None] * token_weight)
            exact = exact_adjoint @ axes
            error = float(
                (formula.double() - exact.double()).norm()
                / exact.double().norm().clamp_min(1e-30))
            maximum_error = max(maximum_error, error)
            for name, values in (("raw", raw_score), ("rms", rms_score),
                                 ("formula", formula), ("exact", exact)):
                stores[component][name].append(values.cpu())
        batch = len(raw_ids)
        current_rows.extend(raw_ids[:, positions].reshape(-1).tolist())
        target_rows.extend(raw_ids[:, positions + 1].reshape(-1).tolist())
        top1_rows.extend(token_sets["top1"].cpu().numpy().reshape(-1).tolist())
        position_rows.extend(np.tile(positions, batch).tolist())
        row_rows.extend(np.repeat(np.arange(start, start + batch), len(positions)).tolist())
    current = np.asarray(current_rows); target = np.asarray(target_rows); top1 = np.asarray(top1_rows)
    position = np.asarray(position_rows)
    labels, names = make_site_labels(current, target, top1, position, vocabulary, base_names, positions)
    return {
        "scores": {
            component: {name: torch.cat(values) for name, values in variants.items()}
            for component, variants in stores.items()
        },
        "labels": labels,
        "names": names,
        "row": np.asarray(row_rows),
        "maximum_error": maximum_error,
    }


def median_correlations(first, second):
    first = np.asarray(first, dtype=np.float64); second = np.asarray(second, dtype=np.float64)
    values = []
    for mode in range(first.shape[1]):
        x = first[:, mode] - first[:, mode].mean()
        y = second[:, mode] - second[:, mode].mean()
        values.append(float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30)))
    return {"median": float(np.median(values)), "per_mode": values}


def label_agreement(first, second):
    return sum(base_label(a["label"]) == base_label(b["label"]) for a, b in zip(first, second))


def main():
    config = json.loads(CONFIG.read_text())
    prior = json.loads(Path(config["semantic_result_path"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    _, vocabulary, base_names = vocabulary_table(tokenizer)
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False)
    axes = core["output"][:, :config["rank"]].float().cuda()
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)

    print("collect discovery", flush=True)
    discovery = collect(model, tokenizer, config["discovery"], config, axes, vocabulary, base_names)
    selection = discovery["row"] < config["discovery"]["rows"] // 2
    selections = {component: {} for component in ("target", "top1")}
    for component in selections:
        for variant in ("raw", "rms", "exact"):
            selections[component][variant] = select_individual(
                discovery["scores"][component][variant][selection],
                discovery["labels"][selection], discovery["names"], config["minimum_positive"])

    results = {}
    maximum_error = discovery["maximum_error"]
    for corpus, spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        data = collect(model, tokenizer, spec, config, axes, vocabulary, base_names)
        maximum_error = max(maximum_error, data["maximum_error"])
        results[corpus] = {}
        for component in ("target", "top1"):
            exact = data["scores"][component]["exact"]
            results[corpus][component] = {
                "raw_vs_exact": median_correlations(data["scores"][component]["raw"], exact),
                "rms_vs_exact": median_correlations(data["scores"][component]["rms"], exact),
                "formula_vs_exact": median_correlations(data["scores"][component]["formula"], exact),
            }

    uniform_labels = [row["weight"]["label"] for row in prior["modes"]]
    agreements = {}
    for component in ("target", "top1"):
        uniform_rows = [{"label": label} for label in uniform_labels]
        agreements[component] = {
            "uniform_vs_raw": label_agreement(uniform_rows, selections[component]["raw"]),
            "uniform_vs_rms": label_agreement(uniform_rows, selections[component]["rms"]),
            "uniform_vs_exact": label_agreement(uniform_rows, selections[component]["exact"]),
            "raw_vs_rms": label_agreement(selections[component]["raw"], selections[component]["rms"]),
            "raw_vs_exact": label_agreement(selections[component]["raw"], selections[component]["exact"]),
            "rms_vs_exact": label_agreement(selections[component]["rms"], selections[component]["exact"]),
        }
    hypotheses = {
        "R0_forward_backward_duality_is_exact": maximum_error <= 1e-5,
        "R1_uniform_and_natural_raw_roles_agree": any(
            agreements[component]["uniform_vs_raw"] >= 8 for component in agreements),
        "R2_raw_and_exact_natural_roles_agree": all(
            agreements[component]["raw_vs_exact"] >= 8 for component in agreements),
        "R3_rms_and_exact_natural_roles_agree": all(
            agreements[component]["rms_vs_exact"] >= 12 for component in agreements),
        "R4_rms_metric_predicts_exact_scores": all(
            results[corpus][component]["rms_vs_exact"]["median"] >= .98
            for corpus in results for component in ("target", "top1")),
        "R5_raw_unembedding_is_numerically_adequate": all(
            results[corpus][component]["raw_vs_exact"]["median"] >= .80
            for corpus in results for component in ("target", "top1")),
        "R6_output_metric_effect_transfers": all(
            results[corpus][component]["rms_vs_exact"]["median"]
            >= results[corpus][component]["raw_vs_exact"]["median"]
            for corpus in results for component in ("target", "top1")),
        "R7_three_role_definitions_are_reported_separately": True,
    }
    hypotheses = {name: bool(value) for name, value in hypotheses.items()}
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "maximum_duality_error": maximum_error,
        "label_agreements": agreements,
        "selected_labels": selections,
        "evaluation": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Metric red team of output-core semantic roles", "",
        "This separates three meanings of an output role: a uniform-vocabulary unembedding direction, the same raw direction under the natural output-token distribution, and the exact context-dependent metric after final RMSNorm and logit saturation.",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", f"Maximum forward-JVP/backward-VJP duality error: `{maximum_error:.2e}`.",
              "", "## Numerical agreement", "",
              "| Corpus | Endpoint | raw vs exact | RMS-aware vs exact | cap-weighted formula vs exact |", "|---|---|---:|---:|---:|"]
    for corpus, row in results.items():
        for component, values in row.items():
            lines.append(
                f"| {corpus} | {component} | {values['raw_vs_exact']['median']:.3f} | "
                f"{values['rms_vs_exact']['median']:.3f} | {values['formula_vs_exact']['median']:.3f} |")
    lines += ["", "## Label agreement out of 16 modes", "",
              "| Endpoint | uniform→raw | uniform→RMS | uniform→exact | raw→RMS | raw→exact | RMS→exact |", "|---|---:|---:|---:|---:|---:|---:|"]
    for component, row in agreements.items():
        lines.append(
            f"| {component} | {row['uniform_vs_raw']} | {row['uniform_vs_rms']} | {row['uniform_vs_exact']} | "
            f"{row['raw_vs_rms']} | {row['raw_vs_exact']} | {row['rms_vs_exact']} |")
    lines += [
        "", "Uniform-vocabulary meaning asks what a direction could promote over the tokenizer. Natural-frequency meaning asks which questions the model actually presents to it. Context-metric meaning includes the state-dependent final normalization. These are different use cases; a mismatch is not automatically polysemanticity.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "hypotheses": hypotheses,
        "maximum_error": maximum_error,
        "agreements": agreements,
        "correlations": {
            corpus: {component: {
                "raw": row[component]["raw_vs_exact"]["median"],
                "rms": row[component]["rms_vs_exact"]["median"],
            } for component in row} for corpus, row in results.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
