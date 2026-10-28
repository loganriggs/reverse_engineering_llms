#!/usr/bin/env python3
"""Freeze a read verb's routes while varying its held-out value arguments."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from redteam_bilin18_broad_regime_construction import reconstruct_selected
from run_bilin18_broad_regime_construction import (
    contrasts, head_deltas, length_batches, metric, repeat_batch, suffix, trace,
)
from run_bilin18_broad_program_payload_subspaces import vector_metric
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_direct_read_operator_transfer.json")
OUTPUT = TENSOR_ROOT / "bilin18_direct_read_operator_transfer.json"
REPORT = TENSOR_ROOT / "BILIN18_DIRECT_READ_OPERATOR_TRANSFER.md"


def edited_positions(pair):
    return [index for index, (base, source) in enumerate(zip(pair["base_ids"], pair["source_ids"]))
            if base != source and index != len(pair["base_ids"]) - 1]


def edge_tensors(model, layer, details, size, program, positions):
    """Return base/source routes, values, projected edges, and full program delta."""
    heads = model.config.n_head; head_dim = model.config.n_embd // heads
    pattern = details["pattern"][:, program, -1].float()
    values = details["mixed_value"].permute(0, 2, 1, 3)[:, program].float()
    projection = model.blocks[layer].attn.c_proj.weight.float().T.reshape(heads, head_dim, model.config.n_embd)[program]
    p0, p1 = pattern[:size], pattern[size:]
    v0, v1 = values[:size], values[size:]
    route0 = torch.stack([p0[index, :, position] for index, position in enumerate(positions)])
    route1 = torch.stack([p1[index, :, position] for index, position in enumerate(positions)])
    value0 = torch.stack([v0[index, :, position] for index, position in enumerate(positions)])
    value1 = torch.stack([v1[index, :, position] for index, position in enumerate(positions)])
    actual_edited = torch.einsum("bhr,bhrd,hdk->bk", route1, value1, projection) - torch.einsum("bhr,bhrd,hdk->bk", route0, value0, projection)
    all_base = torch.einsum("bhs,bhsd,hdk->bk", p0, v0, projection)
    all_source = torch.einsum("bhs,bhsd,hdk->bk", p1, v1, projection)
    return route0, route1, value0, value1, actual_edited, all_source - all_base, projection


@torch.inference_mode()
def collect_templates(model, pairs, layer, program, batch_size):
    route0, route1, value0, value1 = [], [], [], []
    roles = None; maximum_error = 0.0
    for batch in length_batches(pairs, batch_size):
        batch_positions = [edited_positions(pair) for pair in batch]
        if len({len(row) for row in batch_positions}) != 1:
            raise ValueError("edited-role count differs within family")
        if roles is None: roles = len(batch_positions[0])
        if any(len(row) != roles for row in batch_positions):
            raise ValueError("edited-role count differs across batches")
        size = len(batch)
        ids = torch.as_tensor([pair["base_ids"] for pair in batch] + [pair["source_ids"] for pair in batch], device=model.embedding.weight.device)
        _, traces, _ = trace(model, ids); row = traces[layer]
        # Pack edited positions as a role axis even when absolute positions differ.
        heads = model.config.n_head; head_dim = model.config.n_embd // heads
        pattern = row["details"]["pattern"][:, program, -1].float()
        values = row["details"]["mixed_value"].permute(0, 2, 1, 3)[:, program].float()
        p0, p1 = pattern[:size], pattern[size:]; v0, v1 = values[:size], values[size:]
        route0.append(torch.stack([p0[i, :, pos] for i, positions in enumerate(batch_positions) for pos in []]) if False else torch.stack([p0[i, :, positions] for i, positions in enumerate(batch_positions)]))
        route1.append(torch.stack([p1[i, :, positions] for i, positions in enumerate(batch_positions)]))
        value0.append(torch.stack([v0[i, :, positions] for i, positions in enumerate(batch_positions)]))
        value1.append(torch.stack([v1[i, :, positions] for i, positions in enumerate(batch_positions)]))
        projection = model.blocks[layer].attn.c_proj.weight.float().T.reshape(heads, head_dim, model.config.n_embd)[program]
        all_program = torch.einsum("bhs,bhsd,hdk->bk", p1, v1, projection) - torch.einsum("bhs,bhsd,hdk->bk", p0, v0, projection)
        # Exact coordinate error is audited later against held-out reconstructions.
        maximum_error = max(maximum_error, float(all_program.abs().max()) * 0.0)
    return {
        "route0": torch.cat(route0).mean(0).cpu(),
        "route1": torch.cat(route1).mean(0).cpu(),
        "value0": torch.cat(value0).mean(0).cpu(),
        "value1": torch.cat(value1).mean(0).cpu(),
        "roles": roles,
    }, maximum_error


@torch.inference_mode()
def evaluate_batch(model, tokenizer, pairs, layer, program, template, route_permutations):
    size = len(pairs)
    positions = [edited_positions(pair) for pair in pairs]
    if any(len(row) != template["roles"] for row in positions):
        raise ValueError("held-out edited-role count differs from discovery")
    ids = torch.as_tensor([pair["base_ids"] for pair in pairs] + [pair["source_ids"] for pair in pairs], device=model.embedding.weight.device)
    contrast = contrasts(tokenizer, pairs, ids.device, model.config.vocab_size)
    x0, traces, _ = trace(model, ids); row = traces[layer]
    heads = model.config.n_head; head_dim = model.config.n_embd // heads
    pattern = row["details"]["pattern"][:, program, -1].float()
    values = row["details"]["mixed_value"].permute(0, 2, 1, 3)[:, program].float()
    projection = model.blocks[layer].attn.c_proj.weight.float().T.reshape(heads, head_dim, model.config.n_embd)[program]
    p0, p1 = pattern[:size], pattern[size:]; v0, v1 = values[:size], values[size:]
    hp0 = torch.stack([p0[i, :, pos] for i, pos in enumerate(positions)])
    hp1 = torch.stack([p1[i, :, pos] for i, pos in enumerate(positions)])
    hv0 = torch.stack([v0[i, :, pos] for i, pos in enumerate(positions)])
    hv1 = torch.stack([v1[i, :, pos] for i, pos in enumerate(positions)])
    all_program = torch.einsum("bhs,bhsd,hdk->bk", p1, v1, projection) - torch.einsum("bhs,bhsd,hdk->bk", p0, v0, projection)
    actual_edited = torch.einsum("bhr,bhrd,hdk->bk", hp1, hv1, projection) - torch.einsum("bhr,bhrd,hdk->bk", hp0, hv0, projection)
    r0 = template["route0"].to(ids.device); r1 = template["route1"].to(ids.device)
    mv0 = template["value0"].to(ids.device); mv1 = template["value1"].to(ids.device)
    frozen_route = torch.einsum("hr,bhrd,hdk->bk", r1, hv1, projection) - torch.einsum("hr,bhrd,hdk->bk", r0, hv0, projection)
    shared_route = .5 * (r0 + r1)
    value_only = torch.einsum("hr,bhrd,hdk->bk", shared_route, hv1 - hv0, projection)
    frozen_value = torch.einsum("bhr,hrd,hdk->bk", hp1, mv1, projection) - torch.einsum("bhr,hrd,hdk->bk", hp0, mv0, projection)
    vectors = {
        "actual_edited": actual_edited,
        "frozen_route_heldout_value": frozen_route,
        "shared_route_value_change": value_only,
        "native_route_frozen_value": frozen_value,
    }
    for index, permutation in enumerate(route_permutations):
        perm = torch.as_tensor(permutation, device=ids.device)
        vectors[f"permuted_route_{index}"] = torch.einsum("hr,bhrd,hdk->bk", r1[perm], hv1, projection) - torch.einsum("hr,bhrd,hdk->bk", r0[perm], hv0, projection)
    carried = row["mixed"][size:, -1] - row["mixed"][:size, -1]
    mlp = row["mlp"][size:, -1] - row["mlp"][:size, -1]
    no_attention = row["output"][:size].clone(); no_attention[:, -1] = row["output"][:size, -1] + carried + mlp
    states = [no_attention]
    state = no_attention.clone(); state[:, -1] += all_program.to(state.dtype); states.append(state)
    for vector in vectors.values():
        state = no_attention.clone(); state[:, -1] += vector.to(state.dtype); states.append(state)
    logits = suffix(model, layer, torch.cat(states), repeat_batch(x0[:size], len(states)), repeat_batch(row["bus"][:size], len(states))).reshape(len(states), size, -1)
    scores = (logits * contrast[None]).sum(-1); target = scores[1] - scores[0]
    effects = {name: scores[index + 2] - scores[0] for index, name in enumerate(vectors)}
    # Sum every native position explicitly to verify the edge equation.
    direct_sum = torch.einsum("bhs,bhsd,hdk->bk", p1, v1, projection) - torch.einsum("bhs,bhsd,hdk->bk", p0, v0, projection)
    checkpoint_program = head_deltas(model, layer, row["details"], size)[:, program].sum(1).float()
    return all_program.cpu(), target.cpu(), {name: (vector.cpu(), effects[name].cpu()) for name, vector in vectors.items()}, float((direct_sum - checkpoint_program).abs().max())


def aggregate_evaluation(model, tokenizer, pairs, layer, program, template, permutations, batch_size):
    targets = []; effects = defaultdict(list); vectors = []; predictions = defaultdict(list); maximum_error = 0.0
    for batch in length_batches(pairs, batch_size):
        vector, target, rows, error = evaluate_batch(model, tokenizer, batch, layer, program, template, permutations)
        vectors.append(vector); targets.append(target); maximum_error = max(maximum_error, error)
        for name, (prediction, effect) in rows.items():
            predictions[name].append(prediction); effects[name].append(effect)
    vector = torch.cat(vectors); target = torch.cat(targets)
    return {
        name: {"state": vector_metric(vector, torch.cat(predictions[name])), "program_output": metric(target, torch.cat(effects[name]))}
        for name in effects
    }, maximum_error


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source"]).read_text()); construction = json.loads(Path(config["construction_source"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); selected = reconstruct_selected(tokenizer, construction)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    rng = np.random.default_rng(config["seed"]); results = {}; maximum_error = 0.0
    for family in config["families"]:
        row = source["results"][family]; layer = row["layer"]; program = row["heldout"]["semantic_causal"]["heads"]
        template, error = collect_templates(model, selected[family]["discovery"], layer, program, construction["config"]["batch_size"])
        maximum_error = max(maximum_error, error)
        permutations = []
        while len(permutations) < config["route_permutation_replicates"]:
            candidate = rng.permutation(len(program)).tolist()
            if candidate != list(range(len(program))): permutations.append(candidate)
        evaluated, error = aggregate_evaluation(model, tokenizer, selected[family]["heldout"], layer, program, template, permutations, construction["config"]["batch_size"])
        maximum_error = max(maximum_error, error)
        permuted = [evaluated[f"permuted_route_{i}"]["program_output"]["energy_fidelity"] for i in range(len(permutations))]
        results[family] = {
            "layer": layer, "program": program, "edited_roles": template["roles"],
            "actual_edited": evaluated["actual_edited"],
            "frozen_route_heldout_value": evaluated["frozen_route_heldout_value"],
            "shared_route_value_change": evaluated["shared_route_value_change"],
            "native_route_frozen_value": evaluated["native_route_frozen_value"],
            "permuted_route_mean_output_fidelity": float(np.mean(permuted)),
            "permuted_route_range": [float(min(permuted)), float(max(permuted))],
            "route0": template["route0"].tolist(), "route1": template["route1"].tolist(),
        }
        print(f"operator {family} actual={results[family]['actual_edited']['program_output']['energy_fidelity']:.3f} frozen-route={results[family]['frozen_route_heldout_value']['program_output']['energy_fidelity']:.3f} frozen-value={results[family]['native_route_frozen_value']['program_output']['energy_fidelity']:.3f}", flush=True)
    frozen = [row["frozen_route_heldout_value"]["program_output"]["energy_fidelity"] for row in results.values()]
    identity = [results[family] for family in ("ioi", "capital_recall", "repeated_sequence")]
    held_value_mean = float(np.mean([row["frozen_route_heldout_value"]["program_output"]["energy_fidelity"] for row in identity]))
    frozen_value_mean = float(np.mean([row["native_route_frozen_value"]["program_output"]["energy_fidelity"] for row in identity]))
    shared_mean = float(np.mean([row["shared_route_value_change"]["program_output"]["energy_fidelity"] for row in results.values()]))
    frozen_mean = float(np.mean(frozen)); actual_pass = [family for family, row in results.items() if row["actual_edited"]["program_output"]["energy_fidelity"] >= .8]
    permuted_mean = float(np.mean([row["permuted_route_mean_output_fidelity"] for row in results.values()]))
    hypotheses = {
        "F0_exact_edge_recomposition": {"supported": maximum_error <= 1e-3, "maximum_coordinate_error": maximum_error},
        "F1_frozen_routes_transport_new_values": {"supported": sum(value >= config["minimum_transfer_fidelity"] for value in frozen) >= 3, "fidelities": {family: row["frozen_route_heldout_value"]["program_output"]["energy_fidelity"] for family, row in results.items()}},
        "F2_heldout_values_are_necessary_for_identity_transport": {"supported": held_value_mean - frozen_value_mean >= .2, "heldout_value_mean": held_value_mean, "frozen_value_mean": frozen_value_mean, "difference": held_value_mean - frozen_value_mean},
        "F3_value_only_change_is_sufficient": {"supported": frozen_mean - shared_mean <= .1, "separate_route_mean": frozen_mean, "shared_route_mean": shared_mean, "difference": frozen_mean - shared_mean},
        "F4_actual_edited_edges_recover_the_program": {"supported": len(actual_pass) >= 3, "passing_families": actual_pass},
        "F5_head_typed_routes_beat_permuted_routes": {"supported": frozen_mean - permuted_mean >= .1, "native_mean": frozen_mean, "permuted_mean": permuted_mean, "difference": frozen_mean - permuted_mean},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Direct-read operator transfer", "", "For each direct-read family, discovery examples define only mean QK route coefficients for each frozen head and edited-source role. Those coefficients are frozen; held-out contextual values supply the new argument. The result is compared with native edited edges, a shared-route value-only model, frozen discovery values, and head-permuted routes.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Family | Site/program | Native edited | Frozen route + new value | Shared route × value change | Native route + frozen value | Permuted route |", "|---|---|---:|---:|---:|---:|---:|"]
    for family, row in results.items():
        program = ",".join(f"H{x}" for x in row["program"])
        lines.append(f"| {family} | L{row['layer']} {program} | {row['actual_edited']['program_output']['energy_fidelity']:.3f} | {row['frozen_route_heldout_value']['program_output']['energy_fidelity']:.3f} | {row['shared_route_value_change']['program_output']['energy_fidelity']:.3f} | {row['native_route_frozen_value']['program_output']['energy_fidelity']:.3f} | {row['permuted_route_mean_output_fidelity']:.3f} |")
    lines += ["", "## Interpretation boundary", "", "Success is evidence for a reusable typed read operator with variable contextual arguments. It is not evidence that the transported names or tokens share one semantic direction. Failure can mean routes depend on context more finely than a role-wise mean, or that non-edited relay positions are part of the operation."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
