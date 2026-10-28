#!/usr/bin/env python3
"""Factor broad-regime compact head programs by source role and QK/value."""

from __future__ import annotations

from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from redteam_bilin18_broad_task_competence import class_token_ids
from redteam_bilin18_broad_regime_construction import reconstruct_selected
from run_bilin18_broad_regime_construction import (
    TENSOR_ROOT, contrasts, length_batches, metric, repeat_batch, suffix, trace,
)
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_broad_regime_operand_grammar.json")
OUTPUT = TENSOR_ROOT / "bilin18_broad_regime_operand_grammar.json"
REPORT = TENSOR_ROOT / "BILIN18_BROAD_REGIME_OPERAND_GRAMMAR.md"


def role_masks(pair, tokenizer, roles, device):
    base = torch.as_tensor(pair["base_ids"], device=device)
    source = torch.as_tensor(pair["source_ids"], device=device)
    length = len(base)
    destination = torch.zeros(length, dtype=torch.bool, device=device); destination[-1] = True
    edited = (base != source) & ~destination
    base_groups = class_token_ids(tokenizer, pair["base"])
    source_groups = class_token_ids(tokenizer, pair["source"])
    base_index = pair["base"].candidates.index(pair["base"].answer)
    source_index = pair["source"].candidates.index(pair["source"].answer)
    answer_ids = set(base_groups[base_index]) | set(source_groups[source_index])
    answer = torch.as_tensor([int(a) in answer_ids or int(b) in answer_ids for a, b in zip(base, source)], device=device)
    answer &= ~(destination | edited)
    repeated = ((base == base[-1]) | (source == source[-1])) & ~(destination | edited | answer)
    other = ~(destination | edited | answer | repeated)
    mapping = {
        "edited_non_destination": edited,
        "answer_mentions": answer,
        "query_repeats": repeated,
        "other": other,
        "destination": destination,
    }
    assert all(name in mapping for name in roles)
    stacked = torch.stack([mapping[name] for name in roles])
    assert bool((stacked.sum(0) == 1).all())
    return mapping


def shapley(scores, names):
    count = len(names); result = {}
    for factor_index, name in enumerate(names):
        value = torch.zeros_like(next(iter(scores.values())))
        others = [index for index in range(count) if index != factor_index]
        for size in range(count):
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            for subset in itertools.combinations(others, size):
                absent = [0] * count
                for index in subset: absent[index] = 1
                present = absent.copy(); present[factor_index] = 1
                value += weight * (scores[tuple(present)] - scores[tuple(absent)])
        result[name] = value
    return result


def edge_worlds(model, layer, details, size, program, masks, roles):
    heads = model.config.n_head; head_dim = model.config.n_embd // heads
    pattern = details["pattern"][:, program, -1]
    values = details["mixed_value"].permute(0, 2, 1, 3)[:, program]
    projection = model.blocks[layer].attn.c_proj.weight.T.reshape(heads, head_dim, model.config.n_embd)[program]
    p0, p1 = pattern[:size], pattern[size:]
    v0, v1 = values[:size], values[size:]
    def edge(p, v): return torch.einsum("bhsd,hdk->bhsk", p[..., None] * v, projection)
    base, route, value, full = edge(p0, v0), edge(p1, v0), edge(p0, v1), edge(p1, v1)
    delta = full - base
    grouped = {}
    for role in roles:
        mask = torch.stack([item[role] for item in masks])
        grouped[role] = (delta * mask[:, None, :, None]).sum((1, 2))
    qkv = {
        (0, 0): torch.zeros_like(next(iter(grouped.values()))),
        (1, 0): (route - base).sum((1, 2)),
        (0, 1): (value - base).sum((1, 2)),
        (1, 1): delta.sum((1, 2)),
    }
    return grouped, qkv


@torch.inference_mode()
def evaluate_batch(model, tokenizer, pairs, layer, program, roles):
    size = len(pairs)
    ids = torch.as_tensor([pair["base_ids"] for pair in pairs] + [pair["source_ids"] for pair in pairs], device="cuda")
    contrast = contrasts(tokenizer, pairs, ids.device, model.config.vocab_size)
    x0, traces, _ = trace(model, ids); row = traces[layer]
    base_state = row["output"][:size]
    carried = row["mixed"][size:, -1] - row["mixed"][:size, -1]
    mlp = row["mlp"][size:, -1] - row["mlp"][:size, -1]
    no_attention = base_state.clone(); no_attention[:, -1] = base_state[:, -1] + carried + mlp
    masks = [role_masks(pair, tokenizer, roles, ids.device) for pair in pairs]
    groups, qkv = edge_worlds(model, layer, row["details"], size, program, masks, roles)
    state_error = float((sum(groups.values()) - qkv[(1, 1)]).abs().max())

    worlds = tuple(itertools.product((0, 1), repeat=len(roles)))
    scores = {}
    for start in range(0, len(worlds), 8):
        chunk = worlds[start:start + 8]; states = []
        for world in chunk:
            state = no_attention.clone(); state[:, -1] += sum(world[index] * groups[name] for index, name in enumerate(roles)); states.append(state)
        logits = suffix(model, layer, torch.cat(states), repeat_batch(x0[:size], len(states)), repeat_batch(row["bus"][:size], len(states))).reshape(len(states), size, -1)
        for index, world in enumerate(chunk): scores[world] = (logits[index] * contrast).sum(-1)
    group_phi = shapley(scores, roles); group_target = scores[(1,) * len(roles)] - scores[(0,) * len(roles)]
    group_error = float((sum(group_phi.values()) - group_target).norm() / group_target.norm().clamp_min(1e-30))

    qkv_order = ((0, 0), (1, 0), (0, 1), (1, 1)); states = []
    for world in qkv_order:
        state = no_attention.clone(); state[:, -1] += qkv[world]; states.append(state)
    logits = suffix(model, layer, torch.cat(states), repeat_batch(x0[:size], 4), repeat_batch(row["bus"][:size], 4)).reshape(4, size, -1)
    qkv_scores = {world: (logits[index] * contrast).sum(-1) for index, world in enumerate(qkv_order)}
    qkv_phi = shapley(qkv_scores, ("routing", "value")); qkv_target = qkv_scores[(1, 1)] - qkv_scores[(0, 0)]
    qkv_error = float((sum(qkv_phi.values()) - qkv_target).norm() / qkv_target.norm().clamp_min(1e-30))
    return {
        "target": group_target.cpu(),
        "roles": {name: group_phi[name].cpu() for name in roles},
        "qkv": {name: qkv_phi[name].cpu() for name in ("routing", "value")},
        "errors": {"state_max_abs": state_error, "role_shapley_relative": group_error, "qkv_shapley_relative": qkv_error},
    }


def aggregate(rows, roles):
    target = torch.cat([row["target"] for row in rows])
    return {
        "examples": len(target),
        "target_rms": float(target.double().square().mean().sqrt()),
        "roles": {name: metric(target, torch.cat([row["roles"][name] for row in rows])) for name in roles},
        "qkv": {name: metric(target, torch.cat([row["qkv"][name] for row in rows])) for name in ("routing", "value")},
    }


def cosine(first, second):
    first, second = np.asarray(first), np.asarray(second)
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source"]).read_text()); redteam = json.loads(Path(config["redteam_source"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    selected = reconstruct_selected(tokenizer, source)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)

    results, maximum_errors = {}, {"state_max_abs": 0.0, "role_shapley_relative": 0.0, "qkv_shapley_relative": 0.0}
    for family in source["config"]["families"]:
        layer = source["head_summary"][family]["site"]
        program = source["head_summary"][family]["head_order"][:3]
        results[family] = {"layer": layer, "program": program}
        for split in ("discovery", "heldout"):
            rows = []
            for batch in length_batches(selected[family][split], source["config"]["batch_size"]):
                row = evaluate_batch(model, tokenizer, batch, layer, program, config["roles"]); rows.append(row)
                for name in maximum_errors: maximum_errors[name] = max(maximum_errors[name], row["errors"][name])
            results[family][split] = aggregate(rows, config["roles"])
        discovery_profile = [results[family]["discovery"]["roles"][name]["projection_share"] for name in config["roles"]]
        heldout_profile = [results[family]["heldout"]["roles"][name]["projection_share"] for name in config["roles"]]
        results[family]["profile_cosine"] = cosine(discovery_profile, heldout_profile)
        print(f"operands {family} cosine={results[family]['profile_cosine']:.3f}", flush=True)

    material = [family for family, row in redteam["results"].items() if row["material_attention"]]
    relevant = {}
    for family in material:
        relevant[family] = sum(results[family]["heldout"]["roles"][name]["projection_share"] for name in ("edited_non_destination", "answer_mentions", "query_repeats"))
    retrieval_share = results["repeated_sequence"]["heldout"]["roles"]["edited_non_destination"]["projection_share"] + results["repeated_sequence"]["heldout"]["roles"]["query_repeats"]["projection_share"]
    ioi_profile = [results["ioi"]["heldout"]["roles"][name]["projection_share"] for name in config["roles"]]
    agreement_profile = [results["agreement_attractor"]["heldout"]["roles"][name]["projection_share"] for name in config["roles"]]
    binding_cosine = cosine(ioi_profile, agreement_profile)
    dominant_roles = {family: max(config["roles"], key=lambda name: results[family]["heldout"]["roles"][name]["projection_share"]) for family in ("ioi", "agreement_attractor")}
    value_shares = {family: results[family]["heldout"]["qkv"]["value"]["projection_share"] for family in material}
    hypotheses = {
        "O0_tensor_edge_factorials_recompose": {"supported": maximum_errors["state_max_abs"] <= 1e-4 and maximum_errors["role_shapley_relative"] <= 1e-5 and maximum_errors["qkv_shapley_relative"] <= 1e-5, "observed": maximum_errors},
        "O1_operand_profiles_transfer": {"supported": sum(results[family]["profile_cosine"] >= .8 for family in results) >= 5, "cosines": {family: row["profile_cosine"] for family,row in results.items()}},
        "O2_programs_use_relationally_relevant_sources": {"supported": sum(value >= .5 for value in relevant.values()) >= 4, "shares": relevant},
        "O3_retrieval_uses_repetition_structure": {"supported": retrieval_share >= .4, "share": retrieval_share},
        "O4_binding_is_not_one_operand_type": {"supported": binding_cosine < .9 or len(set(dominant_roles.values())) > 1, "cosine": binding_cosine, "dominant_roles": dominant_roles},
        "O5_value_change_is_common_but_not_assumed_universal": {"supported": sum(value >= .6 for value in value_shares.values()) >= 4, "shares": value_shares},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"maximum_errors":maximum_errors,"results":results},indent=2,allow_nan=False))
    lines=["# Broad-regime operand grammar","","Each frozen top-three head program is expanded into exact source-position tensor edges. Five disjoint source roles and a separate QK-routing/value factorial pass through the nonlinear suffix with Shapley allocation.","","## Locked results",""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name,row in hypotheses.items())
    lines += ["","| Family | Site/program | Edited | Answer mention | Query repeat | Other | Destination | Routing | Value | D/H cosine |","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for family,row in results.items():
        held=row["heldout"]
        lines.append(f"| {family} | L{row['layer']} " + ",".join(f"H{h}" for h in row["program"]) + " | " + " | ".join(f"{held['roles'][name]['projection_share']:.3f}" for name in config["roles"]) + f" | {held['qkv']['routing']['projection_share']:.3f} | {held['qkv']['value']['projection_share']:.3f} | {row['profile_cosine']:.3f} |")
    lines += ["","## Guardrail","","These roles are relational partitions of the selected counterfactual prompts, not global semantic labels for a head. A large `other` or destination share can still be a coherent relay. Source-role profiles identify where the invocation reads; they do not by themselves prove the transported state has the proposed semantic meaning."]
    REPORT.write_text("\n".join(lines)+"\n")
    print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
