#!/usr/bin/env python3
"""Exact four-path VJP program through the final attention/MLP suffix."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from run_bilin12_mlp_path_type_program import norm
from run_bilin18_attention_layer_path_program import native_cache
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer16_backward_suffix_path_program.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer16_backward_suffix_path_program.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER16_BACKWARD_SUFFIX_PATH_PROGRAM.md")
PATH_OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer16_backward_suffix_paths.pt")
PATHS = (
    "final_residual__mixed_direct",
    "final_residual__attention",
    "final_mlp__mixed_direct",
    "final_mlp__attention",
)
COMPONENTS = ("ce", "prediction", "target", "margin")


def fidelity(target, prediction):
    return 1.0 - float(
        (target.double() - prediction.double()).square().sum()
        / target.double().square().sum().clamp_min(1e-30))


@torch.inference_mode(False)
def suffix_paths(block, row17, position, covectors):
    mixed = row17["mixed"].detach().clone().requires_grad_(True)
    bus = row17["bus"].detach().clone()
    block.attn.rotary.seq_len_cached = None
    block.attn.rotary.cos_cached = None
    block.attn.rotary.sin_cached = None
    attention, _ = block.attn(norm(mixed), bus)
    pre = mixed + attention
    mlp = block.mlp(norm(pre))
    residual = pre + mlp
    final = norm(residual)
    results = {}
    maximum_internal_error = 0.0
    for component, covector in covectors.items():
        full_covector = torch.zeros_like(final)
        full_covector[:, position] = covector.cuda()
        residual_adjoint, = torch.autograd.grad(
            final, residual, full_covector, retain_graph=True)
        mlp_adjoint, = torch.autograd.grad(
            mlp, pre, residual_adjoint, retain_graph=True)
        residual_attention, = torch.autograd.grad(
            attention, mixed, residual_adjoint, retain_graph=True)
        mlp_attention, = torch.autograd.grad(
            attention, mixed, mlp_adjoint, retain_graph=True)
        full_mixed, = torch.autograd.grad(
            final, mixed, full_covector, retain_graph=True)
        scale = block.lambdas[0].float()
        results[component] = torch.stack((
            scale * residual_adjoint[:, position],
            scale * residual_attention[:, position],
            scale * mlp_adjoint[:, position],
            scale * mlp_attention[:, position],
        ), dim=1).detach().cpu()
        full = (scale * full_mixed[:, position]).detach().cpu()
        internal_error = float(
            (results[component].sum(1).double() - full.double()).norm()
            / full.double().norm().clamp_min(1e-30))
        maximum_internal_error = max(maximum_internal_error, internal_error)
    results["ce"] = results["prediction"] - results["target"]
    return results, maximum_internal_error


@torch.inference_mode(False)
def collect_paths(model, source_spec, factor_data, config):
    source = load_rows(source_spec)
    block = model.blocks[17]
    stores = {component: [] for component in COMPONENTS}
    maximum_write_error = 0.0
    maximum_internal_error = 0.0
    factor_offset = source_spec.get("factor_start", 0)
    consumed = 0
    for start in range(0, len(source), config["batch_size"]):
        array = np.asarray(source[start:start + config["batch_size"]])
        ids = torch.as_tensor(array[:, :128].astype(np.int64), device="cuda")
        _, cache = native_cache(model, ids, config)
        row17 = {row["index"]: row for row in cache}[17]
        for position in config["positions"]:
            indices = slice(factor_offset + consumed, factor_offset + consumed + len(ids))
            covectors = {
                component: factor_data["components"][component]["output_covector"][indices]
                for component in ("prediction", "target", "margin")
            }
            paths, internal_error = suffix_paths(block, row17, position, covectors)
            maximum_internal_error = max(maximum_internal_error, internal_error)
            for component in COMPONENTS:
                reference = factor_data["components"][component]["write_adjoint"][indices]
                reconstruction = paths[component].sum(1)
                error = float(
                    (reference.double() - reconstruction.double()).norm()
                    / reference.double().norm().clamp_min(1e-30))
                maximum_write_error = max(maximum_write_error, error)
                stores[component].append(paths[component])
            consumed += len(ids)
        print(f"  rows {start + len(ids)}/{len(source)}", flush=True)
    assert consumed == len(source) * len(config["positions"])
    return ({component: torch.cat(values) for component, values in stores.items()},
            maximum_write_error, maximum_internal_error)


@torch.inference_mode()
def behavioral_paths(factor_data, write_paths, weights):
    state = factor_data["state"].float().cuda()
    forward = factor_data["forward"].float().cuda()
    left_weight = weights["left"].float().cuda()
    right_weight = weights["right"].float().cuda()
    down = weights["down"].float().cuda()
    left = state @ left_weight.T
    right = state @ right_weight.T
    results = {}
    maximum_error = 0.0
    for component, paths in write_paths.items():
        write_adjoint = paths.float().cuda()
        channel_adjoint = torch.einsum("spd,dc->spc", write_adjoint, down)
        effective = (
            torch.einsum("spc,sc,cd->spd", channel_adjoint, right, left_weight)
            + torch.einsum("spc,sc,cd->spd", channel_adjoint, left, right_weight))
        effects = torch.einsum("snd,spd->spn", forward, effective).cpu()
        reference = factor_data["components"][component]["target"]
        error = float(
            (reference.double() - effects.sum(1).double()).norm()
            / reference.double().norm().clamp_min(1e-30))
        maximum_error = max(maximum_error, error)
        results[component] = effects
    return results, maximum_error


def subset_fidelity(effects, targets, selected):
    predictions = []
    references = []
    for component in targets:
        predictions.append(effects[component][:, selected].sum(1))
        references.append(targets[component])
    return fidelity(torch.cat(references), torch.cat(predictions))


def select_subsets(effects, targets, components):
    selected = {}
    scoped_targets = {component: targets[component] for component in components}
    for count in (1, 2, 3):
        candidates = []
        for indices in itertools.combinations(range(len(PATHS)), count):
            value = subset_fidelity(effects, scoped_targets, indices)
            candidates.append((value, indices))
        value, indices = max(candidates)
        selected[str(count)] = {
            "indices": list(indices),
            "paths": [PATHS[index] for index in indices],
            "discovery_fidelity": value,
        }
    return selected


@torch.inference_mode()
def rank90(rows):
    values = rows.float().cuda()
    singular = torch.linalg.svdvals(values)
    energy = singular.double().square()
    shares = energy / energy.sum().clamp_min(1e-30)
    return int(torch.searchsorted(
        shares.cumsum(0), torch.tensor(.90, dtype=shares.dtype, device=shares.device)).item() + 1)


def slice_factors(data, start, count):
    return {
        "state": data["state"][start:start + count],
        "forward": data["forward"][start:start + count],
        "components": {
            component: {
                key: value[start:start + count]
                for key, value in data["components"][component].items()
            } for component in COMPONENTS
        },
    }


def main():
    config = json.loads(CONFIG.read_text())
    factors = torch.load(config["factor_path"], map_location="cpu", weights_only=False)
    weights = factors["layer16_weights"]
    static_result = json.loads(Path(config["static_result_path"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    discovery_spec = config["path_discovery"]
    discovery_count = discovery_spec["rows"] * len(config["positions"])
    discovery_full = factors[discovery_spec["factor_dataset"]]
    discovery = slice_factors(
        discovery_full, discovery_spec.get("factor_start", 0), discovery_count)
    print("collect discovery paths", flush=True)
    discovery_write, discovery_write_error, discovery_internal_error = collect_paths(
        model, discovery_spec, discovery_full, config)
    discovery_effects, discovery_behavior_error = behavioral_paths(
        discovery, discovery_write, weights)
    targets = {
        component: discovery["components"][component]["target"]
        for component in COMPONENTS
    }
    selection = select_subsets(discovery_effects, targets, config["fit_components"])
    spectra = {
        component: {
            PATHS[path]: rank90(discovery_write[component][:, path])
            for path in range(len(PATHS))
        } for component in COMPONENTS
    }

    results = {}
    maximum_write_error = discovery_write_error
    maximum_internal_error = discovery_internal_error
    maximum_behavior_error = discovery_behavior_error
    saved_paths = {"discovery": discovery_write}
    for corpus, spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        full_data = factors["evaluation"][spec["factor_dataset"]]
        count = spec["rows"] * len(config["positions"])
        data = slice_factors(full_data, spec.get("factor_start", 0), count)
        write_paths, write_error, internal_error = collect_paths(model, spec, full_data, config)
        effects, behavior_error = behavioral_paths(data, write_paths, weights)
        maximum_write_error = max(maximum_write_error, write_error)
        maximum_internal_error = max(maximum_internal_error, internal_error)
        maximum_behavior_error = max(maximum_behavior_error, behavior_error)
        saved_paths[corpus] = write_paths
        component_rows = {}
        for component in COMPONENTS:
            target = data["components"][component]["target"]
            programs = {}
            for count_key, selected in selection.items():
                programs[count_key] = fidelity(
                    target, effects[component][:, selected["indices"]].sum(1))
            programs["4"] = fidelity(target, effects[component].sum(1))
            component_rows[component] = {
                "program_fidelity": programs,
                "no_attention_fidelity": fidelity(
                    target, effects[component][:, [0, 2]].sum(1)),
                "no_final_mlp_fidelity": fidelity(
                    target, effects[component][:, [0, 1]].sum(1)),
                "static_behavior_fidelity": static_result["results"][corpus]["components"][component]["shared_behavior_fidelity"],
            }
        aggregate = {}
        aggregate_targets = {
            component: data["components"][component]["target"]
            for component in config["fit_components"]
        }
        for count_key, selected in selection.items():
            aggregate[count_key] = subset_fidelity(
                effects, aggregate_targets, selected["indices"])
        aggregate["4"] = subset_fidelity(
            effects, aggregate_targets, range(4))
        results[corpus] = {
            "sites": count,
            "aggregate_program_fidelity": aggregate,
            "components": component_rows,
        }
    del model
    torch.cuda.empty_cache()

    exact_error = max(maximum_write_error, maximum_behavior_error)
    tested = config["fit_components"]
    hypotheses = {
        "P0_four_path_backward_program_is_exact": exact_error <= 1e-5,
        "P1_two_backward_paths_are_sufficient": all(
            row["components"][component]["program_fidelity"]["2"] >= .90
            for row in results.values() for component in tested),
        "P2_three_backward_paths_are_high_fidelity": all(
            row["components"][component]["program_fidelity"]["3"] >= .95
            for row in results.values() for component in tested),
        "P3_attention_response_is_material": all(
            any(row["components"][component]["no_attention_fidelity"] < .95
                for component in tested) for row in results.values()),
        "P4_final_mlp_response_is_material": all(
            any(row["components"][component]["no_final_mlp_fidelity"] < .95
                for component in tested) for row in results.values()),
        "P5_path_order_transfers_ood": all(
            results[name]["aggregate_program_fidelity"][count]
            >= results["fineweb"]["aggregate_program_fidelity"][count] - .10
            for name in ("wikitext", "tinystories") for count in ("2", "3")),
        "P6_short_program_can_have_broad_path_vectors": any(
            value >= 128 for row in spectra.values() for value in row.values()),
        "P7_context_executed_program_resolves_static_map_failure": all(
            row["components"][component]["program_fidelity"]["4"]
            >= row["components"][component]["static_behavior_fidelity"] + .50
            for row in results.values() for component in tested),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["P0_four_path_backward_program_is_exact"].update({
        "maximum_write_adjoint_error": maximum_write_error,
        "maximum_same_graph_internal_error": maximum_internal_error,
        "maximum_behavioral_error": maximum_behavior_error,
    })
    payload = {
        "config": config,
        "hypotheses": hypothesis_rows,
        "path_names": list(PATHS),
        "selection": selection,
        "discovery_path_rank90": spectra,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    torch.save(saved_paths, PATH_OUTPUT)

    lines = [
        "# Exact backward path program through the final suffix",
        "",
        "The final output covector is propagated through four exact path types: "
        "final residual versus final bilinear MLP, crossed with direct mixed-state "
        "versus attention response. These are evaluated as context-executed weight "
        "programs, not fitted static maps.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypothesis_rows.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        f"Maximum write-adjoint recomposition error: `{maximum_write_error:.2e}`; "
        f"same-graph internal error: `{maximum_internal_error:.2e}`; "
        f"behavioral recomposition error: `{maximum_behavior_error:.2e}`.",
        "",
        "| Path count | Discovery-selected paths | Discovery aggregate fidelity |",
        "|---:|---|---:|",
    ]
    for count in ("1", "2", "3"):
        row = selection[count]
        lines.append(
            f"| {count} | {', '.join(row['paths'])} | {row['discovery_fidelity']:.3f} |")
    lines += [
        "",
        "| Corpus | Component | K1/K2/K3/K4 | No attention | No final MLP | Static fitted map |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        for component in COMPONENTS:
            item = row["components"][component]
            program = item["program_fidelity"]
            lines.append(
                f"| {corpus} | {component} | "
                f"{program['1']:.3f}/{program['2']:.3f}/{program['3']:.3f}/{program['4']:.3f} | "
                f"{item['no_attention_fidelity']:.3f} | "
                f"{item['no_final_mlp_fidelity']:.3f} | "
                f"{item['static_behavior_fidelity']:.3f} |")
    lines += [
        "",
        "| Component | " + " | ".join(PATHS) + " |",
        "|---|" + "---:|" * len(PATHS),
    ]
    for component in COMPONENTS:
        lines.append(
            f"| {component} | "
            + " | ".join(str(spectra[component][path]) for path in PATHS) + " |")
    lines += [
        "",
        "A low path count is algebraic program length. The R90 values show how broad "
        "each path's realized adjoint family remains in residual coordinates. Thus "
        "a short exact program and a broad activation/readout atlas can coexist.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
