#!/usr/bin/env python3
"""Factor the exact attention singleton paths into canonical QK and value roles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin12_mlp_path_symmetry_redteam import GROUPS as MLP_GROUPS, collapse
from run_bilin12_mlp_path_type_program import aggregate_gram, boundary, greedy_order, norm, path_matrices
from run_bilin12_router_finite_transport import fidelity
from run_bilin18_attention_layer_path_program import (
    ATTENTION_PATHS, mlp_post_jvp, native_cache, sequence_rms_jvp,
    two_qk_attention_jvp,
)
from run_bilin18_mlp_path_attention_boundary import dynamic_finite
from run_bilin18_mlp_path_symmetry_replication import build_axes, load_rows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_attention_factor_program.json")
OUTPUT = TENSOR_ROOT / "bilin18_attention_factor_program.json"
REPORT = TENSOR_ROOT / "BILIN18_ATTENTION_FACTOR_PROGRAM.md"


def labels(config):
    result = ["I"]
    for layer in range(config["layer"] + 1, config["final_layer"] + 1):
        result += [f"L{layer}Q", f"L{layer}V"]
    return result


@torch.inference_mode()
def singleton_factor_paths(model, ids, position, config, input_axes, output_axes):
    x, cache = native_cache(model, ids, config); batch, length, dimension = x.shape
    rank_in, rank_out = input_axes.shape[1], output_axes.shape[1]; stores = []
    branch_norms = {"qk1": 0.0, "qk2": 0.0}
    for first in range(0, rank_in, config["tangent_chunk"]):
        axes = input_axes[:, first:first + config["tangent_chunk"]]
        identity = torch.zeros(batch, axes.shape[1], length, dimension, device="cuda")
        identity[:, :, position] = (config["scale"] * x[:, position].square().mean(-1).sqrt()[:, None, None]
                                      * axes.T[None])
        active = identity[:, None]
        for row in cache:
            index = row["index"]; block = model.blocks[index]; path_count = active.shape[1]
            flat = active.reshape(batch, -1, length, dimension)
            mixed = block.lambdas[0].float() * flat
            identity_mixed = block.lambdas[0].float() * active[:, 0]
            input_tangent = sequence_rms_jvp(row["mixed"].float(), identity_mixed)
            _, _, components = two_qk_attention_jvp(
                block.attn, row["attention_input"].float(), row["bus"], input_tangent,
                return_components=True,
            )
            branch_norms["qk1"] += float(components["qk1"].square().sum())
            branch_norms["qk2"] += float(components["qk2"].square().sum())
            if index < config["final_layer"]:
                baseline = mlp_post_jvp(block, row["pre"].float(), row["state"].float(), mixed)
                baseline = baseline.reshape(batch, path_count, axes.shape[1], length, dimension)
                qpath = mlp_post_jvp(block, row["pre"].float(), row["state"].float(), components["qk"])
                vpath = mlp_post_jvp(block, row["pre"].float(), row["state"].float(), components["value"])
                active = torch.cat([baseline, qpath[:, None], vpath[:, None]], 1)
            else:
                baseline = sequence_rms_jvp(row["pre"].float(), mixed)[:, :, position] @ output_axes
                baseline = baseline.reshape(batch, path_count, axes.shape[1], rank_out)
                qpath = sequence_rms_jvp(row["pre"].float(), components["qk"])[:, :, position] @ output_axes
                vpath = sequence_rms_jvp(row["pre"].float(), components["value"])[:, :, position] @ output_axes
                active = torch.cat([baseline, qpath[:, None], vpath[:, None]], 1)
        stores.append(active.cpu())
    return torch.cat(stores, 2), branch_norms


def target_gram(paths, target):
    flat = paths.double().reshape(len(paths), paths.shape[1], -1)
    target_flat = target.double().reshape(len(target), -1)
    gram = torch.einsum("npi,nqi->pq", flat, flat)
    dot = torch.einsum("npi,ni->p", flat, target_flat)
    return gram, dot, target_flat.square().sum()


def error_from_gram(gram, dot, target2, selected):
    if not selected: return float(target2)
    index = torch.as_tensor(selected, dtype=torch.long)
    return float(target2 - 2 * dot[index].sum() + gram[index][:, index].sum())


def greedy(gram, dot, target2):
    chosen = []; remaining = set(range(len(gram)))
    while remaining:
        item = min(remaining, key=lambda value: error_from_gram(gram, dot, target2, chosen + [value]))
        chosen.append(item); remaining.remove(item)
    return chosen


def target_fidelity(target, paths, selected):
    prediction = paths[:, selected].sum(1) if selected else torch.zeros_like(target)
    return fidelity(target, prediction)


def target_oracle(target, paths, budget):
    numerator = denominator = 0.0
    for site_target, site_paths in zip(target, paths):
        gram, dot, target2 = target_gram(site_paths[None], site_target[None])
        selected = greedy(gram, dot, target2)[:budget]
        numerator += error_from_gram(gram, dot, target2, selected); denominator += float(target2)
    return 1 - numerator / max(denominator, 1e-30)


def random_summary(target, paths, budget, replicates, generator):
    values = []
    for _ in range(replicates):
        selected = [0] if budget == 1 else [0] + (torch.randperm(paths.shape[1] - 1, generator=generator)[:budget - 1] + 1).tolist()
        values.append(target_fidelity(target, paths, selected))
    return {"median": float(np.median(values)), "p95": float(np.quantile(values, .95)), "maximum": max(values)}


@torch.inference_mode()
def collect(model, input_axes, output_axes, spec, config, full_order, mlp_order, finite=False):
    source = load_rows(spec); factor_rows = []; target_rows = []; mlp4_rows = []; finite_rows = []
    branch = {"qk1": 0.0, "qk2": 0.0}; maximum_singleton_error = 0.0
    singleton_indices = [ATTENTION_PATHS.index(tuple("A" if offset == layer else "I" for offset in range(5)))
                         for layer in range(5)]
    for start in range(0, len(source), config["batch_size"]):
        ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
        for position in config["positions"]:
            # Reuse the exact full layer paths as the target and singleton audit.
            from run_bilin18_attention_layer_path_program import attention_paths
            full_paths, _, _ = attention_paths(model, ids, position, config, input_axes, output_axes)
            factors, norms = singleton_factor_paths(model, ids, position, config, input_axes, output_axes)
            target_rows.append(full_paths.sum(1)); factor_rows.append(factors)
            for key in branch: branch[key] += norms[key]
            for layer, singleton in enumerate(singleton_indices):
                predicted = factors[:, 1 + 2 * layer] + factors[:, 2 + 2 * layer]
                expected = full_paths[:, singleton]
                relative = ((predicted - expected).flatten(1).norm(dim=1)
                            / expected.flatten(1).norm(dim=1).clamp_min(1e-30))
                maximum_singleton_error = max(maximum_singleton_error, float(relative.max()))
            raw_mlp = path_matrices(model, ids, position, config, input_axes, output_axes)
            grouped = collapse(raw_mlp.cpu()); mlp4_rows.append(grouped[:, mlp_order[:4]].sum(1))
            if finite: finite_rows.append(dynamic_finite(model, ids, position, config, input_axes, output_axes).cpu())
        print(f"  rows {start + len(ids)}/{len(source)}", flush=True)
    return {"paths": torch.cat(factor_rows), "target": torch.cat(target_rows),
            "mlp4": torch.cat(mlp4_rows), "finite": torch.cat(finite_rows) if finite_rows else None,
            "branch_norm": {key: value ** .5 for key, value in branch.items()},
            "maximum_singleton_error": maximum_singleton_error}


@torch.inference_mode(False)
def main():
    raw = json.loads(CONFIG.read_text()); parent = json.loads(Path(raw["source_config"]).read_text())
    base = json.loads(Path(parent["source_config"]).read_text())
    config = {**base, **{key: value for key, value in parent.items()
                        if key not in ("source_config", "source_artifact", "hypotheses")},
              **{key: value for key, value in raw.items()
                 if key not in ("source_config", "source_artifact", "hypotheses")},
              "hypotheses": raw["hypotheses"]}
    full_artifact = json.loads(Path(raw["source_artifact"]).read_text())
    full_order = [ATTENTION_PATHS.index(tuple(label)) for label in full_artifact["path_order"]]
    mlp_artifact = json.loads(Path(parent["source_artifact"]).read_text())
    mlp_order = [MLP_GROUPS.index(tuple(label)) for label in mlp_artifact["group_order"]]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    input_axes, output_axes = build_axes(model, config)
    print("collect discovery", flush=True)
    discovery = collect(model, input_axes, output_axes, config["path_discovery"], config, full_order, mlp_order)
    ordering = greedy(*target_gram(discovery["paths"], discovery["target"]))
    generator = torch.Generator().manual_seed(config["seed"]); results = {}
    maximum_error = discovery["maximum_singleton_error"]
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        data = collect(model, input_axes, output_axes, spec, config, full_order, mlp_order, finite=True)
        maximum_error = max(maximum_error, data["maximum_singleton_error"]); rows = {}
        for budget in config["budgets"]:
            selected = ordering[:budget]; nonidentity = [index for index in selected if index != 0]
            combined = data["mlp4"] + (data["paths"][:, nonidentity].sum(1) if nonidentity else 0)
            rows[str(budget)] = {"frozen_fidelity": target_fidelity(data["target"], data["paths"], selected),
                                 "oracle_fidelity": target_oracle(data["target"], data["paths"], budget),
                                 "random": random_summary(data["target"], data["paths"], budget, config["random_replicates"], generator),
                                 "combined_to_finite_fidelity": fidelity(data["finite"], combined)}
        q_indices = [1 + 2 * layer for layer in range(5)]; v_indices = [2 + 2 * layer for layer in range(5)]
        results[corpus] = {"sites": len(data["target"]), "budgets": rows,
                           "all_singletons_fidelity": target_fidelity(data["target"], data["paths"], list(range(11))),
                           "qk_only_fidelity": target_fidelity(data["target"], data["paths"], [0] + q_indices),
                           "value_only_fidelity": target_fidelity(data["target"], data["paths"], [0] + v_indices),
                           "qk1_qk2_norm_ratio": data["branch_norm"]["qk1"] / max(data["branch_norm"]["qk2"], 1e-30)}
    fineweb4 = results["fineweb"]["budgets"]["4"]["frozen_fidelity"]
    supported = {
        "F0_qk_value_singletons_are_exact": maximum_error <= 1e-5,
        "F1_singletons_explain_attention_program": all(row["all_singletons_fidelity"] >= .995 for row in results.values()),
        "F2_four_factor_paths_are_high_fidelity": all(row["budgets"]["4"]["frozen_fidelity"] >= .98 for row in results.values()),
        "F3_qk_routing_dominates_value_response": all(row["qk_only_fidelity"] >= row["value_only_fidelity"] + .005 for row in results.values()),
        "F4_factor_paths_beat_random": all(row["budgets"]["4"]["frozen_fidelity"] >= row["budgets"]["4"]["random"]["p95"] + .01 for row in results.values()),
        "F5_factor_program_transfers_ood": all(results[name]["budgets"]["4"]["frozen_fidelity"] >= fineweb4 - .03 for name in ("wikitext", "tinystories")),
        "F6_compact_factor_and_mlp_program_predicts_finite": all(row["budgets"]["4"]["combined_to_finite_fidelity"] >= .94 for row in results.values()),
        "F7_context_specific_factor_selection_is_small": all(row["budgets"]["4"]["oracle_fidelity"] < row["budgets"]["4"]["frozen_fidelity"] + .02 for row in results.values()),
        "F8_qk_branch_asymmetry_is_small": all(.5 <= row["qk1_qk2_norm_ratio"] <= 2 for row in results.values()),
    }
    hypotheses = {name: {"supported": bool(value)} for name, value in supported.items()}
    hypotheses["F0_qk_value_singletons_are_exact"]["maximum_relative_error"] = maximum_error
    names = labels(config); payload = {"config": raw, "hypotheses": hypotheses,
                                      "factor_order": [names[index] for index in ordering], "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# QK-versus-value factor program for 546M late attention", "",
             "Every single-layer attention response is split exactly into the symmetry-invariant sum of both normalized two-QK routing branches (`Q`) and current-value/OV response (`V`). The persistent Layer-0 value bus is fixed because the intervention begins after Layer 12. Identity plus ten layer-factor paths is compared with the complete 32-path analytic target, so missing multi-attention-layer interactions remain visible.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum singleton factorization error: `{maximum_error:.2e}`.", "",
              "Frozen factor order: `" + "`, `".join(payload["factor_order"]) + "`.", "",
              "| Corpus | All singletons | Q-only / V-only | K4 frozen / oracle / random-p95 | MLP4 + factor K4→finite | Q1/Q2 norm |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        four = row["budgets"]["4"]
        lines.append(f"| {corpus} | {row['all_singletons_fidelity']:.3f} | {row['qk_only_fidelity']:.3f} / {row['value_only_fidelity']:.3f} | {four['frozen_fidelity']:.3f} / {four['oracle_fidelity']:.3f} / {four['random']['p95']:.3f} | {four['combined_to_finite_fidelity']:.3f} | {row['qk1_qk2_norm_ratio']:.2f} |")
    lines += ["", "`Q` is canonical under exchanging the two QK factor branches; Q1 and Q2 are reported only as a symmetry/red-team diagnostic. A compact Q/V program identifies an operator role, not a human semantic concept or a fixed head circuit."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
