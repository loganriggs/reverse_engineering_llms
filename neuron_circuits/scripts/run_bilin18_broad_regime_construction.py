#!/usr/bin/env python3
"""Layerwise construction and compact-head assay across broad task regimes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from redteam_bilin18_broad_task_competence import class_token_ids
from run_bilin18_broad_task_competence import COLORS, Probe, declared_probes, tokenize_probe
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_broad_regime_construction.json")
OUTPUT = TENSOR_ROOT / "bilin18_broad_regime_construction.json"
REPORT = TENSOR_ROOT / "BILIN18_BROAD_REGIME_CONSTRUCTION.md"
FACTORS = ("carried", "attention", "mlp")
WORLDS = tuple(itertools.product((0, 1), repeat=3))


def repeated_sequence_pairs():
    rows = []
    candidates = tuple(f" {color}" for color in COLORS)
    for index, first in enumerate(COLORS):
        base_answer = COLORS[(index + 1) % len(COLORS)]
        source_answer = COLORS[(index + 2) % len(COLORS)]
        base = Probe(f"repeat-cf-{first}-{base_answer}", "repeated_sequence", "retrieval", f"{first} {base_answer}; {first} {base_answer}; {first}", f" {base_answer}", candidates, "repeated_pair_cf")
        source = Probe(f"repeat-cf-{first}-{source_answer}", "repeated_sequence", "retrieval", f"{first} {source_answer}; {first} {source_answer}; {first}", f" {source_answer}", candidates, "repeated_pair_cf")
        rows.append((base, source))
    return rows


def token_edits(first, second):
    return None if len(first) != len(second) else sum(a != b for a, b in zip(first, second))


def candidate_pairs(tokenizer, families, maximum_edits, seed):
    by_family = defaultdict(list)
    for item in declared_probes():
        if item.family not in families or item.family == "repeated_sequence":
            continue
        encoded = tokenize_probe(tokenizer, item)
        if encoded is not None:
            by_family[item.family].append((item, *encoded))
    candidates = defaultdict(list)
    for family, rows in by_family.items():
        for first_index, first in enumerate(rows):
            for second in rows[first_index + 1:]:
                first_item, first_ids, first_candidates, first_answer = first
                second_item, second_ids, second_candidates, second_answer = second
                edits = token_edits(first_ids, second_ids)
                if first_candidates != second_candidates or first_answer == second_answer or edits is None or edits > maximum_edits:
                    continue
                candidates[family].append({
                    "base": first_item, "source": second_item,
                    "base_ids": first_ids, "source_ids": second_ids,
                    "token_edits": edits,
                })
    for base, source in repeated_sequence_pairs():
        base_encoded, source_encoded = tokenize_probe(tokenizer, base), tokenize_probe(tokenizer, source)
        if base_encoded is None or source_encoded is None:
            continue
        base_ids, base_candidates, base_answer = base_encoded
        source_ids, source_candidates, source_answer = source_encoded
        edits = token_edits(base_ids, source_ids)
        if base_candidates == source_candidates and base_answer != source_answer and edits is not None and edits <= maximum_edits:
            candidates["repeated_sequence"].append({"base": base, "source": source, "base_ids": base_ids, "source_ids": source_ids, "token_edits": edits})

    rng = random.Random(seed)
    selected = {}
    for family in families:
        rows = candidates[family]
        rng.shuffle(rows)
        rows.sort(key=lambda row: row["token_edits"])
        chosen, used = [], set()
        for row in rows:
            keys = (row["base"].example_id, row["source"].example_id)
            if any(key in used for key in keys):
                continue
            chosen.append(row); used.update(keys)
        selected[family] = chosen
    return selected


@torch.inference_mode()
def trace(model, ids):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, value_bus = x0, None
    layers = []
    for block in model.blocks:
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), value_bus, model.config, model.tt)
        pre = mixed + details["write"]
        normalized = F.rms_norm(pre, (dimension,))
        mlp = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        x = pre + mlp; value_bus = details["value_bus"]
        layers.append({"mixed": mixed, "attention": details["write"], "mlp": mlp, "output": x, "bus": value_bus, "details": details})
    logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float() / 30)
    return x0, layers, logits


@torch.inference_mode()
def suffix(model, layer, state, x0, bus):
    dimension = model.config.n_embd
    x = state
    for block in model.blocks[layer + 1:]:
        x, bus = block(x, bus, x0)
    return 30 * torch.tanh(model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float() / 30)


def contrasts(tokenizer, pairs, device, vocab_size):
    vocab = vocab_size
    output = torch.zeros(len(pairs), vocab, device=device)
    for index, pair in enumerate(pairs):
        base_groups = class_token_ids(tokenizer, pair["base"])
        source_groups = class_token_ids(tokenizer, pair["source"])
        base_index = pair["base"].candidates.index(pair["base"].answer)
        source_index = pair["source"].candidates.index(pair["source"].answer)
        negative, positive = base_groups[base_index], source_groups[source_index]
        output[index, positive] += 1 / len(positive)
        output[index, negative] -= 1 / len(negative)
    return output


def shapley(scores):
    result = {}
    for factor_index, name in enumerate(FACTORS):
        value = torch.zeros_like(next(iter(scores.values())))
        others = [index for index in range(3) if index != factor_index]
        for size in range(3):
            weight = math.factorial(size) * math.factorial(2 - size) / math.factorial(3)
            for subset in itertools.combinations(others, size):
                absent = [0, 0, 0]
                for index in subset: absent[index] = 1
                present = absent.copy(); present[factor_index] = 1
                value += weight * (scores[tuple(present)] - scores[tuple(absent)])
        result[name] = value
    return result


def metric(target, value):
    target, value = target.double(), value.double()
    energy = target.square().sum().clamp_min(1e-30)
    return {
        "projection_share": float((target * value).sum() / energy),
        "energy_fidelity": float(1 - (target - value).square().sum() / energy),
        "direction_agreement": float((target * value > 0).double().mean()),
        "target_rms": float(target.square().mean().sqrt()),
        "value_rms": float(value.square().mean().sqrt()),
    }


def repeat_batch(value, count):
    return torch.cat([value] * count)


def length_batches(pairs, batch_size):
    groups = defaultdict(list)
    for pair in pairs:
        groups[len(pair["base_ids"])].append(pair)
    for length in sorted(groups):
        rows = groups[length]
        for start in range(0, len(rows), batch_size):
            yield rows[start:start + batch_size]


@torch.inference_mode()
def layerwise_batch(model, tokenizer, pairs, layers):
    size = len(pairs)
    ids = torch.as_tensor([pair["base_ids"] for pair in pairs] + [pair["source_ids"] for pair in pairs], device="cuda")
    contrast = contrasts(tokenizer, pairs, ids.device, model.config.vocab_size)
    x0, traces, logits = trace(model, ids)
    native = (logits[size:] * contrast).sum(-1) - (logits[:size] * contrast).sum(-1)
    result, vectors = {}, {}
    maximum = {"state_relative": 0.0, "shapley_relative": 0.0, "base_logit_abs": 0.0}
    for layer in layers:
        row = traces[layer]
        base_state = row["output"][:size]
        components = {
            "carried": row["mixed"][size:, -1] - row["mixed"][:size, -1],
            "attention": row["attention"][size:, -1] - row["attention"][:size, -1],
            "mlp": row["mlp"][size:, -1] - row["mlp"][:size, -1],
        }
        reconstructed = base_state[:, -1] + sum(components.values())
        denominator = (row["output"][size:, -1] - base_state[:, -1]).norm().clamp_min(1e-30)
        maximum["state_relative"] = max(maximum["state_relative"], float((reconstructed - row["output"][size:, -1]).norm() / denominator))
        states = []
        for world in WORLDS:
            state = base_state.clone()
            state[:, -1] = base_state[:, -1] + sum(world[index] * components[name] for index, name in enumerate(FACTORS))
            states.append(state)
        world_logits = suffix(model, layer, torch.cat(states), repeat_batch(x0[:size], len(states)), repeat_batch(row["bus"][:size], len(states))).reshape(len(states), size, -1)
        maximum["base_logit_abs"] = max(maximum["base_logit_abs"], float((world_logits[0] - logits[:size]).abs().max()))
        scores = {world: (world_logits[index] * contrast).sum(-1) for index, world in enumerate(WORLDS)}
        base_score = scores[(0, 0, 0)]
        destination = scores[(1, 1, 1)] - base_score
        phi = shapley(scores)
        maximum["shapley_relative"] = max(maximum["shapley_relative"], float((sum(phi.values()) - destination).norm() / destination.norm().clamp_min(1e-30)))
        result[str(layer)] = {
            "destination": metric(native, destination),
            "branches": {name: metric(native, phi[name]) for name in FACTORS},
        }
        vectors[str(layer)] = {
            "destination": destination.cpu(),
            "branches": {name: phi[name].cpu() for name in FACTORS},
        }
    return native.cpu(), result, maximum, vectors


def head_deltas(model, layer, details, size):
    heads = model.config.n_head; head_dim = model.config.n_embd // heads
    pattern = details["pattern"][:, :, -1]
    values = details["mixed_value"].permute(0, 2, 1, 3)
    attended = torch.einsum("bhs,bhsd->bhd", pattern, values)
    projection = model.blocks[layer].attn.c_proj.weight.T.reshape(heads, head_dim, model.config.n_embd)
    writes = torch.einsum("bhd,hdk->bhk", attended, projection)
    return writes[size:] - writes[:size]


@torch.inference_mode()
def head_program_batch(model, tokenizer, pairs, layer, programs):
    size = len(pairs)
    ids = torch.as_tensor([pair["base_ids"] for pair in pairs] + [pair["source_ids"] for pair in pairs], device="cuda")
    contrast = contrasts(tokenizer, pairs, ids.device, model.config.vocab_size)
    x0, traces, _ = trace(model, ids); row = traces[layer]
    base_state = row["output"][:size]
    carried = row["mixed"][size:, -1] - row["mixed"][:size, -1]
    attention = row["attention"][size:, -1] - row["attention"][:size, -1]
    mlp = row["mlp"][size:, -1] - row["mlp"][:size, -1]
    delta = head_deltas(model, layer, row["details"], size)
    error = float((delta.sum(1) - attention).abs().max())
    no_attention = base_state.clone(); no_attention[:, -1] = base_state[:, -1] + carried + mlp
    states = [no_attention]
    for heads in programs.values():
        state = no_attention.clone()
        if heads: state[:, -1] += delta[:, heads].sum(1)
        states.append(state)
    logits = suffix(model, layer, torch.cat(states), repeat_batch(x0[:size], len(states)), repeat_batch(row["bus"][:size], len(states))).reshape(len(states), size, -1)
    base_score = (logits[0] * contrast).sum(-1)
    effects = {name: (logits[index + 1] * contrast).sum(-1) - base_score for index, name in enumerate(programs)}
    target = effects["all"]
    return target.cpu(), {name: value.cpu() for name, value in effects.items()}, error


def aggregate_layerwise(rows, layers):
    target = torch.cat([row["native"] for row in rows])
    summary = {}
    for layer in layers:
        key = str(layer)
        summary[key] = {"destination": {}, "branches": {name: {} for name in FACTORS}}
        for field, names in (("destination", (None,)), ("branches", FACTORS)):
            for name in names:
                parts = [row["vectors"][key][field] if name is None else row["vectors"][key][field][name] for row in rows]
                summary[key][field if name is None else "branches"]["value" if name is None else name] = metric(target, torch.cat(parts))
        summary[key]["destination"] = summary[key]["destination"].pop("value")
    return summary


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    pair_pool = candidate_pairs(tokenizer, config["families"], config["maximum_token_edits"], config["seed"])
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)

    selected, split_rows, max_errors = {}, defaultdict(list), {"state_relative": 0.0, "shapley_relative": 0.0, "base_logit_abs": 0.0}
    for family_index, family in enumerate(config["families"]):
        candidates = pair_pool[family][:config["maximum_pairs_per_family"]]
        # Endpoint filtering is computed before the discovery/held-out split and
        # uses only whether the intended source-vs-base semantic change is positive.
        retained = []
        for batch in length_batches(candidates, config["batch_size"]):
            native, _, _, _ = layerwise_batch(model, tokenizer, batch, [])
            retained.extend(pair for pair, effect in zip(batch, native.tolist()) if effect > 0)
        retained = retained[:config["maximum_pairs_per_family"]]
        midpoint = max(1, len(retained) // 2)
        selected[family] = {
            "pairs": [{"base": row["base"].example_id, "source": row["source"].example_id, "token_edits": row["token_edits"]} for row in retained],
            "discovery_count": midpoint, "heldout_count": len(retained) - midpoint,
        }
        for split, pairs in (("discovery", retained[:midpoint]), ("heldout", retained[midpoint:])):
            completed = 0
            for batch in length_batches(pairs, config["batch_size"]):
                if not batch: continue
                native, layerwise, errors, vectors = layerwise_batch(model, tokenizer, batch, config["layers"])
                for name in max_errors: max_errors[name] = max(max_errors[name], errors[name])
                split_rows[(family, split)].append({"native": native, "vectors": vectors})
                completed += len(batch)
                print(f"layerwise {family} {split} {completed}/{len(pairs)}", flush=True)

    layer_summary, sites = {}, {}
    for family in config["families"]:
        layer_summary[family] = {}
        for split in ("discovery", "heldout"):
            rows = split_rows[(family, split)]
            layer_summary[family][split] = aggregate_layerwise(rows, config["layers"]) if rows else {}
        discovery = layer_summary[family]["discovery"]
        crossing = next((layer for layer in config["layers"] if discovery[str(layer)]["destination"]["projection_share"] >= config["destination_crossing"]), None)
        if crossing is None and discovery:
            crossing = max(config["layers"], key=lambda layer: discovery[str(layer)]["destination"]["projection_share"])
        sites[family] = crossing

    head_summary = {}
    for family in config["families"]:
        site = sites[family]; rows = pair_pool[family]
        chosen_ids = {(row["base"], row["source"]) for row in selected[family]["pairs"]}
        chosen = [row for row in rows if (row["base"].example_id, row["source"].example_id) in chosen_ids]
        midpoint = selected[family]["discovery_count"]
        discovery_pairs, heldout_pairs = chosen[:midpoint], chosen[midpoint:midpoint + selected[family]["heldout_count"]]
        singleton_programs = {f"head_{head}": [head] for head in range(model.config.n_head)}; singleton_programs["all"] = list(range(model.config.n_head))
        discovery_targets, discovery_effects = [], defaultdict(list); maximum_head_error = 0.0
        for batch in length_batches(discovery_pairs, config["batch_size"]):
            target, effects, error = head_program_batch(model, tokenizer, batch, site, singleton_programs)
            discovery_targets.append(target); maximum_head_error=max(maximum_head_error,error)
            for name,value in effects.items(): discovery_effects[name].append(value)
        target = torch.cat(discovery_targets)
        singleton_metrics={name:metric(target,torch.cat(values)) for name,values in discovery_effects.items()}
        order=sorted(range(model.config.n_head),key=lambda h:abs(singleton_metrics[f"head_{h}"]["projection_share"]),reverse=True)
        programs={f"top_{k}":order[:k] for k in range(1,model.config.n_head+1)}; programs["all"]=list(range(model.config.n_head))
        held_targets, held_effects=[],defaultdict(list)
        for batch in length_batches(heldout_pairs, config["batch_size"]):
            target,effects,error=head_program_batch(model,tokenizer,batch,site,programs)
            held_targets.append(target); maximum_head_error=max(maximum_head_error,error)
            for name,value in effects.items(): held_effects[name].append(value)
        if held_targets:
            held_target=torch.cat(held_targets); held_metrics={name:metric(held_target,torch.cat(values)) for name,values in held_effects.items()}
            k90=next((k for k in range(1,model.config.n_head+1) if held_metrics[f"top_{k}"]["energy_fidelity"]>=.9),None)
        else:
            held_metrics={}; k90=None
        head_summary[family]={"site":site,"head_order":order,"heldout":held_metrics,"heldout_k90":k90,"max_head_sum_abs":maximum_head_error}
        print(f"heads {family} L{site} order={order} k90={k90}",flush=True)

    covered=[f for f,row in selected.items() if row["discovery_count"]>=2 and row["heldout_count"]>=2]
    crossings=[f for f in covered if layer_summary[f]["heldout"] and max(row["destination"]["projection_share"] for row in layer_summary[f]["heldout"].values())>=config["destination_crossing"]]
    dominants={}
    for family in covered:
        site=str(sites[family]); held=layer_summary[family]["heldout"].get(site)
        if held: dominants[family]=max(FACTORS,key=lambda name:held["branches"][name]["projection_share"])
    compact={f:(head_summary[f]["heldout"].get("top_3",{}).get("energy_fidelity",-math.inf)>=config["head_compact_fidelity"]) for f in covered}
    top_heads={f:head_summary[f]["head_order"][0] for f in covered}
    plain=set(head_summary["sentiment_classification"]["head_order"][:3]); negated=set(head_summary["negated_sentiment"]["head_order"][:3])
    hypotheses={
        "B0_broad_endpoint_coverage":{"supported":len(covered)>=6,"families":covered},
        "B1_destination_variables_emerge":{"supported":len(crossings)>=5,"families":crossings},
        "B2_construction_branches_are_not_universal":{"supported":len(set(dominants.values()))>=2,"dominant":dominants},
        "B3_attention_pulses_often_have_compact_head_programs":{"supported":sum(compact.values())>=4,"top3_fidelity":{f:head_summary[f]["heldout"].get("top_3",{}).get("energy_fidelity") for f in covered}},
        "B4_no_single_universal_head":{"supported":len(set(top_heads.values()))>=3,"top_heads":top_heads},
        "B5_negation_is_not_assumed_to_be_plain_sentiment":{"supported":True,"plain_site":sites["sentiment_classification"],"negated_site":sites["negated_sentiment"],"top3_overlap":len(plain&negated)},
    }
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    payload={"config":config,"hypotheses":hypotheses,"selected":selected,"maximum_errors":max_errors,"sites":sites,"layer_summary":layer_summary,"head_summary":head_summary}
    OUTPUT.write_text(json.dumps(payload,indent=2,allow_nan=False))
    lines=["# Broad-regime construction and compact-head assay","","Seven operationally screened families are traced with disjoint discovery and held-out minimal pairs. At each layer, the changed destination state is exactly decomposed into carried state, fresh attention, and realized MLP write; all eight coalitions pass through the untouched suffix. Crossing layers and head rankings are frozen on discovery pairs.","","## Locked results",""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name,row in hypotheses.items())
    lines += ["","| Family | Regime | Pairs D/H | Site | Held destination | Dominant branch | Top head | Top-3 head fidelity | k90 |","|---|---|---:|---:|---:|---|---:|---:|---:|"]
    source=json.loads(Path(config["source"]).read_text())
    for family in config["families"]:
        site=str(sites[family]); held=layer_summary[family]["heldout"].get(site,{}); destination=held.get("destination",{}).get("projection_share"); top3=head_summary[family]["heldout"].get("top_3",{}).get("energy_fidelity")
        lines.append(f"| {family} | {source['summary'][family]['regime']} | {selected[family]['discovery_count']}/{selected[family]['heldout_count']} | {site} | {destination if destination is not None else '—'} | {dominants.get(family,'—')} | H{head_summary[family]['head_order'][0]} | {top3 if top3 is not None else '—'} | {head_summary[family]['heldout_k90'] if head_summary[family]['heldout_k90'] is not None else '—'} |")
    lines += ["","## Guardrail","","A compact head program reconstructs the finite attention branch at one discovery-selected site. It does not show that the head has one global semantic meaning, and it does not imply attention dominates the complete task. Branch dominance, destination sufficiency, and head compactness are reported separately."]
    REPORT.write_text("\n".join(lines)+"\n")
    print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
