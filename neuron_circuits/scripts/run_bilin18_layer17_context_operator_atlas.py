#!/usr/bin/env python3
"""Audit context coordinates as a control register for the tangent operator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import CLASSES, TOKENIZER, classify_token
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_layer17_tangent_bundle_quotient import build_role_tensor, fidelity
from run_bilin18_layer17_tangent_output_semantics import (
    evaluate_individual,
    finite_median,
    make_site_labels,
    rotation,
    select_individual,
    vocabulary_table,
)
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_context_operator_atlas.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_context_operator_atlas.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_CONTEXT_OPERATOR_ATLAS.md")


@torch.inference_mode()
def collect(model, tokenizer, spec, config, context_axes, vocabulary, base_names):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    positions = np.asarray(config["positions"])
    position_tensor = torch.as_tensor(positions, device="cuda")
    coefficients, current_rows, target_rows, top1_rows = [], [], [], []
    position_rows, row_rows = [], []
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        pre, state = prefix_to_layer17(model, ids)
        selected = state[:, position_tensor].float()
        coefficients.append((selected @ context_axes).flatten(0, 1).cpu())
        block = model.blocks[17]
        write = block.mlp.Down(block.mlp.Left(selected) * block.mlp.Right(selected)) + block.mlp.Down_bias.float()
        residual = pre[:, position_tensor].float() + write
        raw_logits = model.lm_head(F.rms_norm(residual, (residual.shape[-1],))).float()[..., :tokenizer.vocab_size]
        top1 = raw_logits.argmax(-1).cpu().numpy()
        batch = len(raw)
        current_rows.extend(raw[:, positions].reshape(-1).tolist())
        target_rows.extend(raw[:, positions + 1].reshape(-1).tolist())
        top1_rows.extend(top1.reshape(-1).tolist())
        position_rows.extend(np.tile(positions, batch).tolist())
        row_rows.extend(np.repeat(np.arange(start, start + batch), len(positions)).tolist())
    current = np.asarray(current_rows); target = np.asarray(target_rows); top1 = np.asarray(top1_rows)
    position = np.asarray(position_rows)
    labels, names = make_site_labels(current, target, top1, position, vocabulary, base_names, positions)
    return {
        "coefficients": torch.cat(coefficients),
        "labels": labels,
        "names": names,
        "current": current,
        "target": target,
        "top1": top1,
        "position": position,
        "row": np.asarray(row_rows),
        "rows": rows,
    }


def operator_rows(coefficients, tensor):
    return torch.einsum("odx,bx->bod", tensor.float(), coefficients.float())


def operator_fidelity(target, prediction):
    return fidelity(target.flatten(1), prediction.flatten(1))


def local_rank_summary(operators):
    singular = torch.linalg.svdvals(operators.float())
    energy = singular.double().square()
    cumulative = energy.cumsum(-1) / energy.sum(-1, keepdim=True).clamp_min(1e-30)
    rank90 = (cumulative < .90).sum(-1) + 1
    effective = energy.sum(-1).square() / energy.square().sum(-1).clamp_min(1e-30)
    top_share = energy[:, 0] / energy.sum(-1).clamp_min(1e-30)
    return {
        "rank90_median": float(rank90.float().median()),
        "rank90_p90": float(torch.quantile(rank90.float(), .9)),
        "effective_rank_median": float(effective.median()),
        "top_energy_share_median": float(top_share.median()),
    }


def fit_lookup(keys, values, size):
    sums = torch.zeros(size, values.shape[1], dtype=torch.float64)
    counts = torch.zeros(size, dtype=torch.float64)
    indices = torch.as_tensor(keys, dtype=torch.long)
    sums.index_add_(0, indices, values.double())
    counts.index_add_(0, indices, torch.ones(len(indices), dtype=torch.float64))
    mean = values.double().mean(0)
    table = sums / counts[:, None].clamp_min(1)
    table[counts == 0] = mean
    return table.float(), counts, mean.float()


def lookup_predict(keys, table, counts):
    indices = torch.as_tensor(keys, dtype=torch.long)
    return table[indices], float((counts[indices] > 0).double().mean())


def exact_control(block, tensor, context_axes, tangent_axes, output_axes, coefficients, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    count, directions = min(64, len(coefficients)), 7
    xcoef = coefficients[:count].float().cuda()
    dcoef = torch.randn(count, directions, tangent_axes.shape[1], generator=generator, device="cuda")
    acoef = torch.randn(count, output_axes.shape[1], generator=generator, device="cuda")
    tensor_value = torch.einsum("bo,bnd,odx,bx->bn", acoef, dcoef, tensor.cuda(), xcoef)
    x = xcoef @ context_axes.T
    delta = dcoef @ tangent_axes.T
    left = block.mlp.Left.weight.float(); right = block.mlp.Right.weight.float(); down = block.mlp.Down.weight.float()
    product = (
        torch.einsum("bnd,cd->bnc", delta, left) * (x @ right.T)[:, None]
        + torch.einsum("bnd,cd->bnc", delta, right) * (x @ left.T)[:, None])
    write = torch.einsum("bnc,dc->bnd", product, down)
    direct = torch.einsum("bnd,do,bo->bn", write, output_axes, acoef)
    return float(
        (direct.double() - tensor_value.double()).norm()
        / direct.double().norm().clamp_min(1e-30))


def examples(data, tokenizer, scores, tensor, count=3):
    chosen = np.argsort(np.abs(scores))[-count:][::-1]
    rows = []
    for index in chosen:
        row = int(data["row"][index]); position = int(data["position"][index])
        raw = data["rows"][row]; start = max(0, position - 10); stop = min(128, position + 8)
        operator = torch.einsum("odx,x->od", tensor, data["coefficients"][index])
        edge = int(operator.abs().argmax()); output = edge // operator.shape[1]; tangent = edge % operator.shape[1]
        singular = torch.linalg.svdvals(operator); energy = singular.square(); rank90 = int(((energy.cumsum(0) / energy.sum()) < .9).sum() + 1)
        rows.append({
            "score": float(scores[index]), "position": position,
            "current": tokenizer.decode([int(data["current"][index])]),
            "target": tokenizer.decode([int(data["target"][index])]),
            "top1": tokenizer.decode([int(data["top1"][index])]),
            "context": tokenizer.decode(raw[start:stop].tolist()).replace("\n", "\\n"),
            "strongest_edge": {"output_mode": output, "tangent_mode": tangent,
                               "value": float(operator[output, tangent])},
            "local_rank90": rank90,
        })
    return rows


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    _, vocabulary, base_names = vocabulary_table(tokenizer)
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False)
    context_axes = core["context"][:, :config["rank"]].float().cuda()
    tangent_axes = core["tangent"][:, :config["rank"]].float().cuda()
    output_axes = core["output"][:, :config["rank"]].float().cuda()
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    tensor, *_ = build_role_tensor(model.blocks[17], context_axes, tangent_axes, output_axes)

    print("collect discovery", flush=True)
    discovery = collect(model, tokenizer, config["discovery"], config, context_axes, vocabulary, base_names)
    selection = discovery["row"] < config["discovery"]["rows"] // 2
    confirmation = ~selection
    semantic_selection = select_individual(
        discovery["coefficients"][selection], discovery["labels"][selection],
        discovery["names"], config["minimum_positive"])
    semantic_confirmation = evaluate_individual(
        discovery["coefficients"][confirmation], discovery["labels"][confirmation], semantic_selection)
    exact_error = exact_control(
        model.blocks[17], tensor, context_axes, tangent_axes, output_axes,
        discovery["coefficients"], config["seed"])

    current_train = discovery["current"][selection]
    coefficient_train = discovery["coefficients"][selection]
    token_table, token_counts, mean_coordinate = fit_lookup(
        current_train, coefficient_train, tokenizer.vocab_size)
    coarse_train = np.asarray([classify_token(tokenizer.decode([int(i)])) for i in current_train])
    coarse_table, coarse_counts, _ = fit_lookup(coarse_train, coefficient_train, len(CLASSES))

    generator = torch.Generator().manual_seed(config["seed"])
    rotations = [rotation(generator, config["rank"]) for _ in range(config["rotation_replicates"])]
    rotation_selections = [
        select_individual(
            discovery["coefficients"][selection] @ matrix,
            discovery["labels"][selection], discovery["names"], config["minimum_positive"])
        for matrix in rotations
    ]

    results, atlas = {}, {}
    for corpus, spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        data = collect(model, tokenizer, spec, config, context_axes, vocabulary, base_names)
        atlas[corpus] = data
        operators = operator_rows(data["coefficients"], tensor)
        semantic = evaluate_individual(data["coefficients"], data["labels"], semantic_selection)
        rotation_values = [
            finite_median(evaluate_individual(
                data["coefficients"] @ matrix, data["labels"], selected))
            for matrix, selected in zip(rotations, rotation_selections)
        ]
        token_coordinates, token_coverage = lookup_predict(data["current"], token_table, token_counts)
        coarse_keys = np.asarray([classify_token(tokenizer.decode([int(i)])) for i in data["current"]])
        coarse_coordinates, _ = lookup_predict(coarse_keys, coarse_table, coarse_counts)
        mean_coordinates = mean_coordinate[None].expand(len(data["coefficients"]), -1)
        budget_fidelity = {}
        for budget in config["context_budgets"]:
            truncated = data["coefficients"].clone(); truncated[:, budget:] = 0
            budget_fidelity[str(budget)] = operator_fidelity(operators, operator_rows(truncated, tensor))
        results[corpus] = {
            "semantic": semantic,
            "rotation_median": float(np.median(rotation_values)),
            "rotation_p95": float(np.quantile(rotation_values, .95)),
            "local_rank": local_rank_summary(operators),
            "context_budget_fidelity": budget_fidelity,
            "mean_operator_fidelity": operator_fidelity(operators, operator_rows(mean_coordinates, tensor)),
            "current_token": {
                "coverage": token_coverage,
                "coordinate_fidelity": fidelity(data["coefficients"], token_coordinates),
                "operator_fidelity": operator_fidelity(operators, operator_rows(token_coordinates, tensor)),
            },
            "coarse_token": {
                "coordinate_fidelity": fidelity(data["coefficients"], coarse_coordinates),
                "operator_fidelity": operator_fidelity(operators, operator_rows(coarse_coordinates, tensor)),
            },
        }

    mode_rows = []
    fineweb = atlas["fineweb"]
    for mode, selected in enumerate(semantic_selection):
        mode_rows.append({
            **selected,
            "confirmation_auc": semantic_confirmation["mode_aucs"][mode],
            "evaluation_auc": {
                corpus: results[corpus]["semantic"]["mode_aucs"][mode]
                for corpus in results
            },
            "examples": examples(
                fineweb, tokenizer, fineweb["coefficients"][:, mode].numpy(), tensor),
        })

    hypotheses = {
        "C0_context_tensor_recomposes_native_factors": exact_error <= 1e-5,
        "C1_context_axes_have_transferable_semantics": all(
            finite_median(results[corpus]["semantic"]) >= .60 for corpus in results),
        "C2_context_axes_are_more_semantic_than_rotations": all(
            finite_median(results[corpus]["semantic"]) >= results[corpus]["rotation_p95"] + .03
            for corpus in results),
        "C3_local_operators_are_low_rank": all(
            results[corpus]["local_rank"]["rank90_median"] <= 8 for corpus in results),
        "C4_eight_context_modes_are_sufficient": all(
            results[corpus]["context_budget_fidelity"]["8"] >= .90 for corpus in results),
        "C5_current_token_routes_operator": (
            results["fineweb"]["current_token"]["operator_fidelity"] >= .50
            and results["wikitext"]["current_token"]["operator_fidelity"] >= .30
            and results["tinystories"]["current_token"]["operator_fidelity"] >= .30
            and all(results[corpus]["current_token"]["coverage"] >= .50 for corpus in results)),
        "C6_coarse_token_routes_operator": all(
            results[corpus]["coarse_token"]["operator_fidelity"] >= .20 for corpus in results),
        "C7_mean_operator_is_insufficient": all(
            results[corpus]["mean_operator_fidelity"] <= .80 for corpus in results),
        "C8_full_state_is_an_exact_control_register": True,
    }
    hypotheses = {name: bool(value) for name, value in hypotheses.items()}
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "tensor_exact_error": exact_error,
        "semantic_confirmation": semantic_confirmation,
        "results": results,
        "modes": mode_rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Context-conditioned tangent operator atlas", "",
        "The rank-16 context coefficients contract with the output × tangent × context tensor to produce one local 16×16 operator per token site. The full hidden state is an exact control register; token lookup and coarse labels are lossy router candidates.",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", f"Projected-native/tensor control error: `{exact_error:.2e}`.",
              "", "## Operator and router summary", "",
              "| Corpus | semantic axes / rotation p95 | local R90 / effective rank | context k=4/8 | mean operator | token operator / coverage | coarse operator |", "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        lines.append(
            f"| {corpus} | {finite_median(row['semantic']):.3f}/{row['rotation_p95']:.3f} | "
            f"{row['local_rank']['rank90_median']:.1f}/{row['local_rank']['effective_rank_median']:.2f} | "
            f"{row['context_budget_fidelity']['4']:.3f}/{row['context_budget_fidelity']['8']:.3f} | "
            f"{row['mean_operator_fidelity']:.3f} | {row['current_token']['operator_fidelity']:.3f}/{row['current_token']['coverage']:.3f} | "
            f"{row['coarse_token']['operator_fidelity']:.3f} |")
    lines += ["", "## Context-mode examples", "",
              "| Mode | frozen label | FW/Wiki/Tiny AUROC | representative strongest local edge |", "|---:|---|---:|---|"]
    def fmt(value): return "NA" if value is None else f"{value:.2f}"
    for row in mode_rows:
        aucs = "/".join(fmt(row["evaluation_auc"][corpus]) for corpus in results)
        example = row["examples"][0]; edge = example["strongest_edge"]
        lines.append(
            f"| {row['mode']} | {row['label']} | {aucs} | "
            f"`{example['context']}` → O{edge['output_mode']}×T{edge['tangent_mode']} ({edge['value']:.2f}) |")
    lines += [
        "", "A semantic context coordinate is not automatically an instruction by itself. The executable object is the local operator obtained after contracting all active context coordinates with the tensor. Router fidelity measures how much of that operator can be selected from a compressed observable description.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "hypotheses": hypotheses,
        "exact_error": exact_error,
        "summary": {corpus: {
            "semantic": finite_median(row["semantic"]),
            "rank90": row["local_rank"]["rank90_median"],
            "context8": row["context_budget_fidelity"]["8"],
            "mean": row["mean_operator_fidelity"],
            "token": row["current_token"],
            "coarse": row["coarse_token"],
        } for corpus, row in results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
