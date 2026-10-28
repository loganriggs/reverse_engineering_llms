#!/usr/bin/env python3
"""Causally reconstruct natural final-MLP writes with the sparse router graph."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_boundary_output_interface import token_id
from run_bilin12_boundary_state_contraction import paired_panel
from run_bilin12_router_joint_diagonalization import evaluate_forms, masked_forms
from run_bilin12_router_pair_graph_operations import coordinates, pair_scores
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_graph_natural_causality.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_graph_natural_causality.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_GRAPH_NATURAL_CAUSALITY.md"


@torch.inference_mode()
def final_components(model, ids, positions):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, bus = block.attn(F.rms_norm(mixed, (dimension,)), bus)
        pre = mixed + attention; state = F.rms_norm(pre, (dimension,))
        write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        x = pre + write
        if layer == 11:
            selected_pre = pre[:, positions].float(); selected_state = state[:, positions].float()
            selected_write = write[:, positions].float()
    native = logits(model, selected_pre + selected_write)
    return selected_pre, selected_state, selected_write, native


def logits(model, hidden):
    raw = model.lm_head(F.rms_norm(hidden, (hidden.shape[-1],))).float()
    return 30 * torch.tanh(raw / 30)


def kl(reference, prediction):
    logp = F.log_softmax(reference.float(), -1); logq = F.log_softmax(prediction.float(), -1)
    return (logp.exp() * (logp - logq)).sum(-1)


def empty():
    return {"count": 0, "ablation_kl": 0.0, "method_kl": {}, "ce_delta": {},
            "top1_same": {}, "newline": {"count": 0, "ablation_kl": 0.0, "method_kl": {}},
            "non_newline": {"count": 0, "ablation_kl": 0.0, "method_kl": {}}}


def update(acc, native, ablated, methods, next_ids, current_ids, newline):
    flat_native = native.flatten(0, 1); flat_ablated = ablated.flatten(0, 1)
    targets = next_ids.flatten(); current = current_ids.flatten(); base_kl = kl(flat_native, flat_ablated)
    acc["count"] += len(targets); acc["ablation_kl"] += float(base_kl.sum())
    native_ce = F.cross_entropy(flat_native, targets, reduction="none")
    for name, value in methods.items():
        flat = value.flatten(0, 1); error = kl(flat_native, flat)
        acc["method_kl"][name] = acc["method_kl"].get(name, 0.0) + float(error.sum())
        acc["ce_delta"][name] = acc["ce_delta"].get(name, 0.0) + float(
            (F.cross_entropy(flat, targets, reduction="none") - native_ce).sum())
        acc["top1_same"][name] = acc["top1_same"].get(name, 0) + int(
            (flat.argmax(-1) == flat_native.argmax(-1)).sum())
    for group, mask in (("newline", current == newline), ("non_newline", current != newline)):
        count = int(mask.sum()); acc[group]["count"] += count
        acc[group]["ablation_kl"] += float(base_kl[mask].sum())
        for name, value in methods.items():
            error = kl(flat_native, value.flatten(0, 1))
            acc[group]["method_kl"][name] = acc[group]["method_kl"].get(name, 0.0) + float(error[mask].sum())


def finalize(acc):
    denominator = max(acc["ablation_kl"], 1e-30)
    result = {"sites": acc["count"], "mean_ablation_kl": acc["ablation_kl"] / acc["count"], "methods": {}}
    for name, value in acc["method_kl"].items():
        result["methods"][name] = {"kl_recovery": 1 - value / denominator,
                                     "mean_kl": value / acc["count"],
                                     "mean_ce_delta": acc["ce_delta"][name] / acc["count"],
                                     "top1_agreement": acc["top1_same"][name] / acc["count"]}
    for group in ("newline", "non_newline"):
        row = acc[group]; base = max(row["ablation_kl"], 1e-30)
        result[group] = {"sites": row["count"], "recovery": {
            name: 1 - value / base for name, value in row["method_kl"].items()}}
    return result


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    targets = {name: token_id(tokenizer, name) for name in config["targets"]}; newline = targets["\n"]
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": config["output_rank"],
                                                   "discovery": config["basis_discovery"]})
    block = model.blocks[11]; dimension = model.config.n_embd
    channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(), block.mlp.Right.weight.float(), channel_to_output)
    full_forms = (raw + raw.transpose(-1, -2)) / 2
    states, _ = collect_states_writes(model, 11, config["natural"], config["natural_positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(states.float(), full_matrices=False); natural_basis = vh[:config["input_rank"]].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", natural_basis, full_forms, natural_basis).detach()
    rotation, training = train_rotation(restricted, config); rotation = rotation.detach(); forms = rotation.T[None] @ restricted @ rotation[None]
    with torch.inference_mode():
        operation_scores = {}
        for name, target in targets.items():
            panel = paired_panel(model, config["graph_discovery"], 11, target, config["batch_size"])
            base, edit = coordinates(model, panel, natural_basis, rotation)
            operation_scores[name] = pair_scores(forms, base, edit)
        pairs = list(operation_scores[next(iter(operation_scores))])
        pooled = {pair: float(np.mean([operation_scores[name][pair] for name in targets])) for pair in pairs}
        selected = sorted(pairs, key=pooled.get, reverse=True)[:config["pair_budget"]]
        random = list(pairs); np.random.default_rng(config["seed"]).shuffle(random); random = random[:config["pair_budget"]]
        selected_forms = masked_forms(forms, selected, config["pair_budget"])
        random_forms = masked_forms(forms, random, config["pair_budget"])
        diagonal_forms = masked_forms(forms, [], 0)
        results = {}
        positions = np.asarray(config["positions"], dtype=np.int64)
        for corpus, spec in config["evaluation"].items():
            print("evaluate", corpus, flush=True); acc = empty()
            source = np.load(spec["path"], mmap_mode="r")
            rows = source[spec["start"]:spec["start"] + spec["rows"]]
            for start in range(0, len(rows), config["batch_size"]):
                raw_ids = np.asarray(rows[start:start + config["batch_size"]], dtype=np.int64)
                ids = torch.as_tensor(raw_ids[:, :128], device="cuda")
                pre, state, write, native = final_components(model, ids, positions)
                flat_state = state.flatten(0, 1); coordinates_value = flat_state @ natural_basis @ rotation
                exact_coefficients = ((write - block.mlp.Down_bias) @ write_basis).flatten(0, 1)
                # evaluate_forms(base, edit) computes changes; for absolute x^T S x,
                # pass base=0 and edit=x instead.
                coefficient_methods = {name: evaluate_forms(form, torch.zeros_like(coordinates_value), coordinates_value)
                                       for name, form in (("full", forms), ("selected", selected_forms),
                                                          ("diagonal", diagonal_forms), ("random", random_forms))}
                ablated_write = write - (exact_coefficients @ write_basis.T).reshape_as(write)
                ablated = logits(model, pre + ablated_write)
                method_logits = {}
                for name, coefficients in coefficient_methods.items():
                    replacement = ablated_write + (coefficients @ write_basis.T).reshape_as(write)
                    method_logits[name] = logits(model, pre + replacement)
                current = torch.as_tensor(raw_ids[:, positions], device="cuda")
                next_ids = torch.as_tensor(raw_ids[:, positions + 1], device="cuda")
                update(acc, native, ablated, method_logits, next_ids, current, newline)
            results[corpus] = finalize(acc)
    c0 = all(row["methods"]["full"]["kl_recovery"] >= .80 for row in results.values())
    c1 = all(row["methods"]["selected"]["kl_recovery"] >= .70 for row in results.values())
    c2 = all(row["methods"]["selected"]["kl_recovery"] >= row["methods"]["full"]["kl_recovery"] - .10 for row in results.values())
    c3 = all(row["methods"]["selected"]["kl_recovery"] >= row["methods"]["random"]["kl_recovery"] + .20 for row in results.values())
    c4 = all(row["methods"]["selected"]["top1_agreement"] >= .98 and row["methods"]["selected"]["mean_ce_delta"] <= .02 for row in results.values())
    c5 = all(row["non_newline"]["recovery"]["selected"] >= .60 for row in results.values())
    hypotheses = {
        "C0_full_restricted_program_recovers_natural_effect": {"supported": c0},
        "C1_selected_graph_recovers_natural_effect": {"supported": c1},
        "C2_selected_graph_matches_full_restricted": {"supported": c2},
        "C3_selected_graph_beats_random": {"supported": c3},
        "C4_selected_graph_preserves_predictions": {"supported": c4},
        "C5_boundary_sites_are_not_the_only_use": {"supported": c5},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "training": training, "hypotheses": hypotheses,
                                  "selected_pairs": [list(pair) for pair in selected], "results": results}, indent=2, allow_nan=False))
    lines = ["# Natural causal reconstruction by the sparse router graph", "",
             "At 9,600 natural sites per corpus, the exact pooled rank-32 component of the native Layer-11 MLP write is removed and replaced by the complete restricted quadratic program, its diagonal-plus-32-edge approximation, a diagonal-only program, or a matched random-edge graph. Orthogonal native write components and final RMSNorm are preserved.", "",
             "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Full/selected/diagonal/random KL recovery | Selected top-1 / CE delta | Newline/non-newline selected recovery | Sites |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        recovery = "/".join(f"{row['methods'][name]['kl_recovery']:.3f}" for name in ("full","selected","diagonal","random"))
        lines.append(f"| {corpus} | {recovery} | {row['methods']['selected']['top1_agreement']:.3f} / "
                     f"{row['methods']['selected']['mean_ce_delta']:.4f} | {row['newline']['recovery']['selected']:.3f}/"
                     f"{row['non_newline']['recovery']['selected']:.3f} | {row['sites']} |")
    lines += ["", "KL recovery is normalized against deleting the exact pooled component: 1 means native-equivalent, 0 means no better than deletion, and negative values are worse. This is a replacement test of a declared output component, not a claim that the graph is independently necessary when all other final-layer directions may compensate."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
