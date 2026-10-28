#!/usr/bin/env python3
"""Semantic atlas and rotation red team for the stable output tensor core."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import CLASSES, TOKENIZER, classify_token
from discover_bilin18_weight_only_circuits import CATEGORIES, token_flags
from run_bilin12_router_jacobian_mode_semantics import (
    GENERIC,
    evaluate_subspace,
    fit_subspace_decoders,
    generic_flags,
)
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_tangent_output_semantics.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_tangent_output_semantics.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_TANGENT_OUTPUT_SEMANTICS.md")


def vocabulary_table(tokenizer):
    texts = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
    structural = np.stack([
        np.concatenate([token_flags(text), generic_flags(text)]) for text in texts
    ])
    coarse = np.zeros((tokenizer.vocab_size, len(CLASSES)), dtype=np.uint8)
    coarse[np.arange(tokenizer.vocab_size), [classify_token(text) for text in texts]] = 1
    labels = np.concatenate([structural, coarse], 1)
    names = (
        [f"flag:{name}" for name in CATEGORIES]
        + [f"generic:{name}" for name in GENERIC]
        + [f"coarse:{name}" for name in CLASSES]
    )
    return texts, labels, names


def rms_vjp(x, adjoint):
    root = x.square().mean(-1, keepdim=True).sqrt()
    coupling = (x * adjoint).mean(-1, keepdim=True)
    return adjoint / root - x * coupling / root.pow(3)


def rankdata(values):
    """One-sort average ranks; equivalent to the prior tie-aware AUC helper."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2
        start = stop
    return ranks


def auc_matrix(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    positives = labels.sum(0)
    negatives = len(labels) - positives
    values = np.full((scores.shape[1], labels.shape[1]), np.nan)
    valid = (positives > 0) & (negatives > 0)
    for mode in range(scores.shape[1]):
        ranks = rankdata(scores[:, mode])
        rank_sums = labels.T @ ranks
        values[mode, valid] = (
            rank_sums[valid] - positives[valid] * (positives[valid] + 1) / 2
        ) / (positives[valid] * negatives[valid])
    return values


def select_individual(scores, labels, names, minimum):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    positives = labels.sum(0)
    valid = np.flatnonzero((positives >= minimum) & (positives <= len(labels) - minimum))
    matrix = auc_matrix(scores, labels)
    selections = []
    for mode in range(scores.shape[1]):
        raw = matrix[mode, valid]
        oriented = np.maximum(raw, 1 - raw)
        winner = int(np.nanargmax(oriented))
        label = int(valid[winner])
        selections.append({
            "mode": mode,
            "label_index": label,
            "label": names[label],
            "sign": 1 if matrix[mode, label] >= .5 else -1,
            "discovery_auc": float(oriented[winner]),
        })
    return selections


def evaluate_individual(scores, labels, selections):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    matrix = auc_matrix(scores, labels)
    values = []
    for row in selections:
        value = matrix[row["mode"], row["label_index"]]
        if row["sign"] < 0 and np.isfinite(value):
            value = 1 - value
        values.append(None if not np.isfinite(value) else float(value))
    finite = np.asarray([value for value in values if value is not None])
    return {
        "mode_aucs": values,
        "median_auc": float(np.median(finite)),
        "modes_above_080": int((finite >= .80).sum()),
    }


def make_site_labels(current, target, top1, position, vocabulary, base_names, positions):
    arrays, names = [], []
    for prefix, ids in (("current", current), ("target", target), ("top1", top1)):
        arrays.append(vocabulary[ids])
        names.extend([f"{prefix}:{name}" for name in base_names])
    for value in positions:
        arrays.append((position == value)[:, None].astype(np.uint8))
        names.append(f"position:{value}")
    return np.concatenate(arrays, 1), names


@torch.inference_mode()
def collect(model, tokenizer, spec, config, axes, vocabulary, base_names):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    positions = np.asarray(config["positions"])
    position_tensor = torch.as_tensor(positions, device="cuda")
    score_rows = {"target": [], "top1": []}
    current_rows, target_rows, top1_rows, position_rows, row_rows = [], [], [], [], []
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        pre, state = prefix_to_layer17(model, ids)
        selected_state = state[:, position_tensor].float()
        block = model.blocks[17]
        write = (
            block.mlp.Down(block.mlp.Left(selected_state) * block.mlp.Right(selected_state))
            + block.mlp.Down_bias.float())
        residual = pre[:, position_tensor].float() + write
        final = F.rms_norm(residual, (residual.shape[-1],))
        raw_logits = model.lm_head(final).float()[..., :tokenizer.vocab_size]
        logits = 30 * torch.tanh(raw_logits / 30)
        top1 = logits.argmax(-1)
        target = torch.as_tensor(raw[:, positions + 1], device="cuda")
        for component, token_ids in (("target", target), ("top1", top1)):
            selected_raw = raw_logits.gather(-1, token_ids[..., None]).squeeze(-1)
            cap = 1 - torch.tanh(selected_raw / 30).square()
            covector = cap[..., None] * model.lm_head.weight.float()[token_ids]
            adjoint = rms_vjp(residual, covector)
            score_rows[component].append((adjoint @ axes).reshape(-1).cpu())
        batch = len(raw)
        current_rows.extend(raw[:, positions].reshape(-1).tolist())
        target_rows.extend(raw[:, positions + 1].reshape(-1).tolist())
        top1_rows.extend(top1.cpu().numpy().reshape(-1).tolist())
        position_rows.extend(np.tile(positions, batch).tolist())
        row_rows.extend(np.repeat(np.arange(start, start + batch), len(positions)).tolist())
        if start % (config["batch_size"] * 25) == 0:
            print(f"  rows {start + batch}/{len(rows)}", flush=True)
    current = np.asarray(current_rows)
    target = np.asarray(target_rows)
    top1 = np.asarray(top1_rows)
    position = np.asarray(position_rows)
    labels, names = make_site_labels(
        current, target, top1, position, vocabulary, base_names, positions)
    return {
        "scores": {name: torch.cat(values).reshape(-1, axes.shape[1])
                   for name, values in score_rows.items()},
        "labels": labels,
        "label_names": names,
        "current": current,
        "target": target,
        "top1": top1,
        "position": position,
        "row": np.asarray(row_rows),
        "rows": rows,
    }


def rotation(generator, rank):
    return torch.linalg.qr(torch.randn(rank, rank, generator=generator)).Q


def finite_median(row):
    values = [value for value in row["mode_aucs"] if value is not None]
    return float(np.median(values))


def evaluation_bundle(scores, labels, selections):
    return evaluate_individual(np.asarray(scores), labels, selections)


def base_label(name):
    if name.startswith(("current:", "target:", "top1:")):
        return name.split(":", 1)[1]
    return name


def top_contexts(data, tokenizer, scores, sign, count=3):
    values = sign * scores
    chosen = np.argsort(values)[-count:][::-1]
    rows = []
    for index in chosen:
        row = int(data["row"][index]); position = int(data["position"][index])
        raw = data["rows"][row]
        start = max(0, position - 10); stop = min(128, position + 8)
        rows.append({
            "score": float(values[index]),
            "position": position,
            "current": tokenizer.decode([int(data["current"][index])]),
            "target": tokenizer.decode([int(data["target"][index])]),
            "top1": tokenizer.decode([int(data["top1"][index])]),
            "context": tokenizer.decode(raw[start:stop].tolist()).replace("\n", "\\n"),
        })
    return rows


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    texts, vocabulary, base_names = vocabulary_table(tokenizer)
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False)
    axes = core["output"][:, :config["rank"]].float().cuda()
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    unembedding = model.lm_head.weight.float()[:tokenizer.vocab_size]
    vocabulary_scores = (unembedding @ axes).cpu()
    even = np.arange(tokenizer.vocab_size) % 2 == 0
    weight_selection = select_individual(
        vocabulary_scores[even], vocabulary[even], base_names, config["minimum_positive"])
    weight_evaluation = evaluate_individual(
        vocabulary_scores[~even], vocabulary[~even], weight_selection)
    top_tokens = []
    for mode in range(config["rank"]):
        values = vocabulary_scores[:, mode]
        high = torch.topk(values, 6).indices.tolist()
        low = torch.topk(-values, 6).indices.tolist()
        top_tokens.append({
            "promoted": [texts[index] for index in high],
            "suppressed": [texts[index] for index in low],
        })

    generator = torch.Generator().manual_seed(config["seed"])
    rotations = [rotation(generator, config["rank"]) for _ in range(config["rotation_replicates"])]
    maximum_rotation_error = max(
        float((matrix.T @ matrix - torch.eye(config["rank"])).abs().max())
        for matrix in rotations)
    weight_rotation_medians = []
    for matrix in rotations:
        rotated = vocabulary_scores @ matrix
        selected = select_individual(
            rotated[even], vocabulary[even], base_names, config["minimum_positive"])
        weight_rotation_medians.append(finite_median(
            evaluate_individual(rotated[~even], vocabulary[~even], selected)))

    random_axes = torch.linalg.qr(torch.randn(
        axes.shape[0], config["isotropic_replicates"] * config["rank"],
        generator=generator)).Q
    isotropic_scores = (unembedding @ random_axes.cuda()).cpu()
    weight_isotropic_medians = []
    for index in range(config["isotropic_replicates"]):
        scores = isotropic_scores[:, index * config["rank"]:(index + 1) * config["rank"]]
        selected = select_individual(
            scores[even], vocabulary[even], base_names, config["minimum_positive"])
        weight_isotropic_medians.append(finite_median(
            evaluate_individual(scores[~even], vocabulary[~even], selected)))

    print("collect semantic discovery", flush=True)
    discovery = collect(
        model, tokenizer, config["discovery"], config, axes, vocabulary, base_names)
    selection_mask = discovery["row"] < config["discovery"]["rows"] // 2
    confirmation_mask = ~selection_mask
    natural_selections, natural_results, rotation_results, subspace_results = {}, {}, {}, {}
    for component in ("target", "top1"):
        natural_selections[component] = select_individual(
            discovery["scores"][component][selection_mask],
            discovery["labels"][selection_mask], discovery["label_names"],
            config["minimum_positive"])
        natural_results[component] = {
            "confirmation": evaluation_bundle(
                discovery["scores"][component][confirmation_mask],
                discovery["labels"][confirmation_mask], natural_selections[component])
        }
        output_prefix = f"{component}:"
        output_columns = np.asarray([
            index for index, name in enumerate(discovery["label_names"])
            if name.startswith(output_prefix)
        ])
        decoder = fit_subspace_decoders(
            discovery["scores"][component][selection_mask],
            discovery["labels"][selection_mask][:, output_columns],
            config["minimum_positive"], config["ridge"])
        subspace_results[component] = {
            "confirmation": evaluate_subspace(
                discovery["scores"][component][confirmation_mask],
                discovery["labels"][confirmation_mask][:, output_columns], decoder),
            "decoder": decoder,
            "columns": output_columns,
        }
        rotation_results[component] = []
        for matrix in rotations:
            rotated_selection = discovery["scores"][component][selection_mask] @ matrix
            rotated_confirmation = discovery["scores"][component][confirmation_mask] @ matrix
            selected = select_individual(
                rotated_selection, discovery["labels"][selection_mask],
                discovery["label_names"], config["minimum_positive"])
            rotation_results[component].append({
                "selection": selected,
                "confirmation_median_auc": finite_median(evaluate_individual(
                    rotated_confirmation, discovery["labels"][confirmation_mask], selected)),
            })

    evaluations = {}
    atlas_data = {}
    for corpus, spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        data = collect(model, tokenizer, spec, config, axes, vocabulary, base_names)
        atlas_data[corpus] = data
        evaluations[corpus] = {}
        for component in ("target", "top1"):
            individual = evaluation_bundle(
                data["scores"][component], data["labels"], natural_selections[component])
            output_columns = subspace_results[component]["columns"]
            subspace = evaluate_subspace(
                data["scores"][component], data["labels"][:, output_columns],
                subspace_results[component]["decoder"])
            rotated_medians = []
            for matrix, row in zip(rotations, rotation_results[component]):
                rotated_medians.append(finite_median(evaluate_individual(
                    data["scores"][component] @ matrix, data["labels"], row["selection"])))
            evaluations[corpus][component] = {
                "individual": individual,
                "subspace": subspace,
                "rotation_median": float(np.median(rotated_medians)),
                "rotation_p95": float(np.quantile(rotated_medians, .95)),
                "rotation_values": rotated_medians,
            }

    corresponding_counts = {
        component: sum(
            row["label"].startswith(f"{component}:")
            for row in natural_selections[component])
        for component in ("target", "top1")
    }
    agreement = 0
    mode_rows = []
    fineweb = atlas_data["fineweb"]
    for mode in range(config["rank"]):
        weight = weight_selection[mode]
        target = natural_selections["target"][mode]
        top1 = natural_selections["top1"][mode]
        match = base_label(weight["label"]) in (base_label(target["label"]), base_label(top1["label"]))
        agreement += int(match)
        mode_rows.append({
            "mode": mode,
            "weight": {**weight, "heldout_auc": weight_evaluation["mode_aucs"][mode]},
            "target": {
                **target,
                "confirmation_auc": natural_results["target"]["confirmation"]["mode_aucs"][mode],
                "evaluation_auc": {
                    corpus: evaluations[corpus]["target"]["individual"]["mode_aucs"][mode]
                    for corpus in evaluations
                },
            },
            "top1": {
                **top1,
                "confirmation_auc": natural_results["top1"]["confirmation"]["mode_aucs"][mode],
                "evaluation_auc": {
                    corpus: evaluations[corpus]["top1"]["individual"]["mode_aucs"][mode]
                    for corpus in evaluations
                },
            },
            "weight_natural_label_match": match,
            "tokens": top_tokens[mode],
            "fineweb_target_examples": top_contexts(
                fineweb, tokenizer, fineweb["scores"]["target"][:, mode].numpy(),
                target["sign"]),
        })

    weight_native = finite_median(weight_evaluation)
    natural_medians = {
        corpus: {
            component: finite_median(evaluations[corpus][component]["individual"])
            for component in ("target", "top1")
        } for corpus in evaluations
    }
    expected_counts = {
        "discovery": config["discovery"]["rows"] * len(config["positions"]),
        **{
            corpus: spec["rows"] * len(config["positions"])
            for corpus, spec in config["evaluation"].items()
        },
    }
    hypotheses = {
        "S0_large_disjoint_atlas_completed": (
            len(discovery["row"]) == 16000
            and len(atlas_data["fineweb"]["row"]) == 32000
            and len(atlas_data["wikitext"]["row"]) == 12800
            and len(atlas_data["tinystories"]["row"]) == 12800),
        "S1_weight_axes_have_lexical_semantics": weight_native >= .65,
        "S2_weight_axes_are_more_semantic_than_rotations": (
            weight_native >= np.quantile(weight_rotation_medians, .95) + .03),
        "S3_target_adjoint_semantics_transfer": all(
            natural_medians[corpus]["target"] >= .60 for corpus in evaluations),
        "S4_top1_adjoint_semantics_transfer": all(
            natural_medians[corpus]["top1"] >= .60 for corpus in evaluations),
        "S5_output_variables_dominate_axis_labels": (
            corresponding_counts["target"] >= 12 and corresponding_counts["top1"] >= 12),
        "S6_output_subspace_has_transferable_semantics": all(
            evaluations[corpus][component]["subspace"]["median_auc"] >= .65
            for corpus in evaluations for component in ("target", "top1")),
        "S7_weight_and_natural_roles_agree": agreement >= config["rank"] // 2,
        "S8_rotation_control_is_matched": maximum_rotation_error <= 1e-5,
    }
    hypotheses = {name: bool(value) for name, value in hypotheses.items()}
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "sample_counts": expected_counts,
        "maximum_rotation_orthogonality_error": maximum_rotation_error,
        "weight_summary": {
            "native_median_auc": weight_native,
            "rotation_median": float(np.median(weight_rotation_medians)),
            "rotation_p95": float(np.quantile(weight_rotation_medians, .95)),
            "isotropic_median": float(np.median(weight_isotropic_medians)),
            "isotropic_p95": float(np.quantile(weight_isotropic_medians, .95)),
        },
        "natural_confirmation": natural_results,
        "natural_evaluation": evaluations,
        "subspace_confirmation": {
            component: subspace_results[component]["confirmation"]
            for component in ("target", "top1")
        },
        "corresponding_output_label_counts": corresponding_counts,
        "weight_natural_agreement": agreement,
        "modes": mode_rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Semantic atlas of the tangent-bundle output core", "",
        f"The atlas uses {expected_counts['discovery']:,} FineWeb semantic-discovery sites, {expected_counts['fineweb']:,} disjoint FineWeb sites, and {expected_counts['wikitext']:,} sites from each OOD corpus. Weight-only vocabulary labels, natural target/top1 adjoints, and random rotations of the identical physical subspace are evaluated separately.",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += [
        "", "## Summary", "",
        f"Weight-only native-axis median vocabulary AUROC: **{weight_native:.3f}**; rotation p95 **{np.quantile(weight_rotation_medians, .95):.3f}**; isotropic p95 **{np.quantile(weight_isotropic_medians, .95):.3f}**.",
        "", "| Corpus | target axes | target subspace | top1 axes | top1 subspace | target/top1 rotation p95 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus in evaluations:
        row = evaluations[corpus]
        lines.append(
            f"| {corpus} | {finite_median(row['target']['individual']):.3f} | {row['target']['subspace']['median_auc']:.3f} | "
            f"{finite_median(row['top1']['individual']):.3f} | {row['top1']['subspace']['median_auc']:.3f} | "
            f"{row['target']['rotation_p95']:.3f}/{row['top1']['rotation_p95']:.3f} |")
    lines += [
        "", f"Corresponding output-variable selections: target {corresponding_counts['target']}/16; top1 {corresponding_counts['top1']}/16. Weight/natural property agreement: {agreement}/16.",
        "", "## Mode examples", "",
        "| Mode | weight label / AUROC | target label / FW-Wiki-Tiny | top1 label / FW-Wiki-Tiny | promoted → suppressed |", "|---:|---|---|---|---|",
    ]
    def fmt(value): return "NA" if value is None else f"{value:.2f}"
    for row in mode_rows:
        w, t, p = row["weight"], row["target"], row["top1"]
        tauc = "/".join(fmt(t["evaluation_auc"][c]) for c in evaluations)
        pauc = "/".join(fmt(p["evaluation_auc"][c]) for c in evaluations)
        promoted = repr(row["tokens"]["promoted"][0]); suppressed = repr(row["tokens"]["suppressed"][0])
        lines.append(
            f"| {row['mode']} | {w['label']} / {fmt(w['heldout_auc'])} | {t['label']} / {tauc} | "
            f"{p['label']} / {pauc} | {promoted} → {suppressed} |")
    lines += [
        "", "Weight tails describe a direction's potential output effect. Natural adjoint labels describe which output questions read that direction. Agreement is stronger evidence than either alone. Same-subspace rotations determine whether meaning belongs to the reported HOSVD axes or only to their joint span.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "hypotheses": hypotheses,
        "weight": payload["weight_summary"],
        "natural_medians": natural_medians,
        "corresponding": corresponding_counts,
        "agreement": agreement,
    }, indent=2))


if __name__ == "__main__":
    main()
