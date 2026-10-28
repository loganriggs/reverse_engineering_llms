#!/usr/bin/env python3
"""Test whether semantic specificity lives in payload directions, not head IDs."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from redteam_bilin18_broad_program_label_blind_prediction import (
    edited_token_pool, random_replacements,
)
from redteam_bilin18_broad_regime_construction import reconstruct_selected
from run_bilin18_broad_regime_construction import (
    contrasts, head_deltas, length_batches, metric, repeat_batch, suffix, trace,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_broad_program_payload_subspaces.json")
OUTPUT = TENSOR_ROOT / "bilin18_broad_program_payload_subspaces.json"
REPORT = TENSOR_ROOT / "BILIN18_BROAD_PROGRAM_PAYLOAD_SUBSPACES.md"


def fit_basis(vectors, rank):
    vectors = torch.as_tensor(vectors, dtype=torch.float32)
    _, singular, vh = torch.linalg.svd(vectors, full_matrices=False)
    effective_rank = min(rank, len(singular), int((singular > singular.max().clamp_min(1e-30) * 1e-6).sum()))
    return vh[:effective_rank].contiguous(), singular.cpu().numpy()


def gaussian_basis(dimension, rank, seed):
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dimension, rank, generator=generator), mode="reduced")
    return q.T.contiguous()


@torch.inference_mode()
def program_vectors_batch(model, pairs, layer, heads):
    size = len(pairs)
    ids = torch.as_tensor(
        [pair["base_ids"] for pair in pairs] + [pair["source_ids"] for pair in pairs],
        device=model.embedding.weight.device,
    )
    _, traces, _ = trace(model, ids)
    delta = head_deltas(model, layer, traces[layer]["details"], size).float()
    attention = traces[layer]["attention"][size:, -1] - traces[layer]["attention"][:size, -1]
    error = float((delta.sum(1) - attention).abs().max())
    return delta[:, heads].sum(1).cpu(), error


def collect_program_vectors(model, pairs, layer, heads, batch_size):
    rows = []; maximum_error = 0.0
    for batch in length_batches(pairs, batch_size):
        vectors, error = program_vectors_batch(model, batch, layer, heads)
        rows.append(vectors); maximum_error = max(maximum_error, error)
    return torch.cat(rows), maximum_error


def vector_metric(target, value):
    target, value = target.double(), value.double()
    energy = target.square().sum().clamp_min(1e-30)
    return {
        "projection_share": float((target * value).sum() / energy),
        "energy_fidelity": float(1 - (target - value).square().sum() / energy),
        "target_rms": float(target.square().mean().sqrt()),
        "value_rms": float(value.square().mean().sqrt()),
    }


@torch.inference_mode()
def evaluate_bases_batch(model, tokenizer, pairs, layer, heads, bases):
    size = len(pairs)
    ids = torch.as_tensor(
        [pair["base_ids"] for pair in pairs] + [pair["source_ids"] for pair in pairs],
        device=model.embedding.weight.device,
    )
    contrast = contrasts(tokenizer, pairs, ids.device, model.config.vocab_size)
    x0, traces, _ = trace(model, ids); row = traces[layer]
    delta = head_deltas(model, layer, row["details"], size).float()
    program = delta[:, heads].sum(1)
    all_attention = delta.sum(1)
    attention_native = row["attention"][size:, -1] - row["attention"][:size, -1]
    error = float((all_attention - attention_native).abs().max())
    carried = row["mixed"][size:, -1] - row["mixed"][:size, -1]
    mlp = row["mlp"][size:, -1] - row["mlp"][:size, -1]
    no_attention = row["output"][:size].clone()
    no_attention[:, -1] = row["output"][:size, -1] + carried + mlp
    states = [no_attention]
    for vector in (program, all_attention):
        state = no_attention.clone(); state[:, -1] += vector.to(state.dtype); states.append(state)
    projected = {}
    for name, basis in bases.items():
        basis = basis.to(program.device, dtype=program.dtype)
        value = (program @ basis.T) @ basis
        projected[name] = value
        state = no_attention.clone(); state[:, -1] += value.to(state.dtype); states.append(state)
    logits = suffix(
        model, layer, torch.cat(states), repeat_batch(x0[:size], len(states)),
        repeat_batch(row["bus"][:size], len(states)),
    ).reshape(len(states), size, -1)
    scores = (logits * contrast[None]).sum(-1)
    base = scores[0]
    program_effect = scores[1] - base
    all_effect = scores[2] - base
    effects = {name: scores[index + 3] - base for index, name in enumerate(bases)}
    return program.cpu(), program_effect.cpu(), all_effect.cpu(), {
        name: (projected[name].cpu(), effects[name].cpu()) for name in bases
    }, error


def evaluate_bases(model, tokenizer, pairs, layer, heads, bases, batch_size):
    vectors = []; program_effects = []; all_effects = []
    projected = defaultdict(list); effects = defaultdict(list); maximum_error = 0.0
    for batch in length_batches(pairs, batch_size):
        vector, program_effect, all_effect, rows, error = evaluate_bases_batch(
            model, tokenizer, batch, layer, heads, bases,
        )
        vectors.append(vector); program_effects.append(program_effect); all_effects.append(all_effect)
        for name, (projection, effect) in rows.items():
            projected[name].append(projection); effects[name].append(effect)
        maximum_error = max(maximum_error, error)
    vector = torch.cat(vectors); program_effect = torch.cat(program_effects); all_effect = torch.cat(all_effects)
    return {
        name: {
            "state": vector_metric(vector, torch.cat(projected[name])),
            "program_output": metric(program_effect, torch.cat(effects[name])),
            "all_attention_output": metric(all_effect, torch.cat(effects[name])),
        }
        for name in bases
    }, {
        "program_vs_all_attention": metric(all_effect, program_effect),
        "maximum_head_sum_abs": maximum_error,
    }


def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source"]).read_text())
    construction = json.loads(Path(config["construction_source"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    selected = reconstruct_selected(tokenizer, construction)
    families = list(source["results"])
    pool = edited_token_pool(selected, families)
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)

    # Fit every true family/rank basis first so cross-family bases can be tested.
    true_bases = {}; fit_info = {}; maximum_error = 0.0
    for family in families:
        row = source["results"][family]
        layer = row["layer"]; heads = row["heldout"]["semantic_causal"]["heads"]
        vectors, error = collect_program_vectors(
            model, selected[family]["discovery"], layer, heads, construction["config"]["batch_size"],
        )
        maximum_error = max(maximum_error, error)
        fit_info[family] = {"examples": len(vectors)}
        for rank in config["ranks"]:
            basis, singular = fit_basis(vectors, rank)
            true_bases[(family, rank)] = basis
            fit_info[family][f"rank_{rank}"] = {
                "effective_rank": len(basis),
                "singular_values": singular.tolist(),
                "discovery_state_fidelity": vector_metric(vectors, (vectors @ basis.T) @ basis)["energy_fidelity"],
            }

    results = {}
    for family_index, family in enumerate(families):
        row = source["results"][family]
        layer = row["layer"]; heads = row["heldout"]["semantic_causal"]["heads"]
        bases = {}
        for rank in config["ranks"]:
            bases[f"true_rank_{rank}"] = true_bases[(family, rank)]
        # Equal-rank bases learned from semantically unrelated token changes at
        # exactly the same locations and through the exact same frozen heads.
        for replicate in range(config["random_replacement_replicates"]):
            random_pairs = random_replacements(
                selected[family]["discovery"], pool,
                config["seed"] + 1009 * family_index + replicate,
            )
            vectors, error = collect_program_vectors(
                model, random_pairs, layer, heads, construction["config"]["batch_size"],
            )
            maximum_error = max(maximum_error, error)
            basis, _ = fit_basis(vectors, config["primary_rank"])
            bases[f"random_edit_{replicate}"] = basis
        for replicate in range(config["gaussian_replicates"]):
            bases[f"gaussian_{replicate}"] = gaussian_basis(
                model.config.n_embd, config["primary_rank"],
                config["seed"] + 7919 * family_index + replicate,
            )
        # Exploratory cross-family residual bases: coordinates are shared but
        # depths differ, so failure means family-depth specificity, not semantics alone.
        for other in families:
            if other != family:
                bases[f"cross_{other}"] = true_bases[(other, config["primary_rank"])]
        evaluated, audit = evaluate_bases(
            model, tokenizer, selected[family]["heldout"], layer, heads, bases,
            construction["config"]["batch_size"],
        )
        maximum_error = max(maximum_error, audit["maximum_head_sum_abs"])
        random_rows = [evaluated[f"random_edit_{i}"] for i in range(config["random_replacement_replicates"])]
        gaussian_rows = [evaluated[f"gaussian_{i}"] for i in range(config["gaussian_replicates"])]
        cross_rows = {other: evaluated[f"cross_{other}"] for other in families if other != family}
        def summarize(rows):
            return {
                key: float(np.mean([value[key]["energy_fidelity"] for value in rows]))
                for key in ("state", "program_output", "all_attention_output")
            }
        results[family] = {
            "layer": layer, "heads": heads, "fit": fit_info[family],
            "true": {f"rank_{rank}": evaluated[f"true_rank_{rank}"] for rank in config["ranks"]},
            "random_edit_mean": summarize(random_rows),
            "random_edit_program_output_range": [
                float(min(value["program_output"]["energy_fidelity"] for value in random_rows)),
                float(max(value["program_output"]["energy_fidelity"] for value in random_rows)),
            ],
            "gaussian_mean": summarize(gaussian_rows),
            "cross_family": cross_rows,
            "best_cross_family_program_output": float(max(value["program_output"]["energy_fidelity"] for value in cross_rows.values())),
            "audit": audit,
        }
        true = evaluated[f"true_rank_{config['primary_rank']}"]
        print(f"payload {family} state={true['state']['energy_fidelity']:.3f} program={true['program_output']['energy_fidelity']:.3f} random={results[family]['random_edit_mean']['program_output']:.3f} gaussian={results[family]['gaussian_mean']['program_output']:.3f}", flush=True)

    primary = f"rank_{config['primary_rank']}"
    true_mean = float(np.mean([row["true"][primary]["program_output"]["energy_fidelity"] for row in results.values()]))
    random_mean = float(np.mean([row["random_edit_mean"]["program_output"] for row in results.values()]))
    gaussian_mean = float(np.mean([row["gaussian_mean"]["program_output"] for row in results.values()]))
    rank2_pass = [family for family, row in results.items() if row["true"][primary]["program_output"]["energy_fidelity"] >= config["minimum_output_fidelity"]]
    rank1_pass = [family for family, row in results.items() if row["true"]["rank_1"]["program_output"]["energy_fidelity"] >= config["minimum_output_fidelity"]]
    output_compact = [family for family, row in results.items() if row["true"][primary]["program_output"]["energy_fidelity"] - row["true"][primary]["state"]["energy_fidelity"] >= .20]
    hypotheses = {
        "S0_exact_head_recomposition": {"supported": maximum_error <= 1e-3, "maximum_head_sum_abs": maximum_error},
        "S1_rank2_true_payload_transfers": {"supported": len(rank2_pass) >= 4, "passing_families": rank2_pass, "mean_program_output_fidelity": true_mean},
        "S2_true_payload_beats_matched_arbitrary_edits": {"supported": true_mean - random_mean >= .10, "true_mean": true_mean, "random_edit_mean": random_mean, "difference": true_mean - random_mean},
        "S3_true_payload_beats_gaussian_subspaces": {"supported": true_mean - gaussian_mean >= .10, "true_mean": true_mean, "gaussian_mean": gaussian_mean, "difference": true_mean - gaussian_mean},
        "S4_rank1_often_suffices": {"supported": len(rank1_pass) >= 4, "passing_families": rank1_pass},
        "S5_payload_is_more_output_compact_than_state_compact": {"supported": len(output_compact) >= 3, "families": output_compact},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Broad-program payload subspaces", "",
             "Head identity failed the arbitrary-edit specificity test. This assay freezes each semantic-causal three-head backbone, learns rank-one and rank-two bases of its discovery payload vectors, and asks whether those directions transfer to held-out examples. Equal-rank bases from arbitrary tokens at the same positions and Gaussian bases are controls.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Family | Site/heads | Rank-1 output | Rank-2 state / output / all-attn | Random-edit output | Gaussian output | Best other-family output |", "|---|---|---:|---:|---:|---:|---:|"]
    for family, row in results.items():
        rank1 = row["true"]["rank_1"]["program_output"]["energy_fidelity"]
        rank2 = row["true"][primary]
        heads = ",".join(f"H{head}" for head in row["heads"])
        lines.append(f"| {family} | L{row['layer']} {heads} | {rank1:.3f} | {rank2['state']['energy_fidelity']:.3f} / {rank2['program_output']['energy_fidelity']:.3f} / {rank2['all_attention_output']['energy_fidelity']:.3f} | {row['random_edit_mean']['program_output']:.3f} | {row['gaussian_mean']['program_output']:.3f} | {row['best_cross_family_program_output']:.3f} |")
    lines += ["", "## Interpretation boundary", "",
              "A transferable payload span is an invocation coordinate system, not automatically a named concept. Its semantic reading additionally depends on the task-defined source relation and output quotient. Cross-family bases come from different depths and therefore test family-depth specificity rather than pure semantic identity."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
