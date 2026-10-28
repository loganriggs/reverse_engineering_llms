#!/usr/bin/env python3
"""Tournament cross-model maps on semantic, arbitrary-memory, and generic edits."""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import ActivationPatch
from audit_grammar_semantic_transfer import make_transition_examples
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_gate_cross_init import greedy_match


CONFIG = Path("configs/grammar_operation_family_tournament.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_operation_family_tournament.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_OPERATION_FAMILY_TOURNAMENT.md")


@torch.inference_mode()
def forward(model, prompts, device, patch=None):
    tokens = torch.as_tensor(np.asarray(prompts, dtype=np.int64), device=device)
    logits, activations = model(tokens, patch=patch, return_activations=True)
    return logits[:, -1, list(VOCAB.answers)].float(), [value.float() for value in activations]


def random_facts(rng, per_relation):
    rows = []
    for relation in range(8):
        rows.append(np.column_stack((rng.integers(0, 512, per_relation, dtype=np.int32),
                                     rng.integers(0, 512, per_relation, dtype=np.int32),
                                     np.full(per_relation, relation, dtype=np.int32))))
    return np.concatenate(rows)


def fit_ridge(source, target, relative):
    source = source.double(); target = target.double()
    sm, tm = source.mean(0), target.mean(0); x, y = source - sm, target - tm
    gram = x.T @ x; penalty = relative * torch.trace(gram) / gram.shape[0]
    weight = torch.linalg.solve(gram + penalty * torch.eye(gram.shape[0], dtype=gram.dtype), x.T @ y)
    return {"source_mean": sm.float(), "target_mean": tm.float(), "weight": weight.float()}


def fit_native(source, target):
    x = source - source.mean(0); y = target - target.mean(0)
    x = x / x.norm(dim=0, keepdim=True).clamp_min(1e-30)
    y = y / y.norm(dim=0, keepdim=True).clamp_min(1e-30)
    mapping = greedy_match((x.T @ y).abs())
    raw_x = source - source.mean(0); raw_y = target - target.mean(0)
    paired = raw_y[:, mapping]
    scale = (raw_x * paired).sum(0) / raw_x.square().sum(0).clamp_min(1e-30)
    return {"mapping": mapping, "scale": scale}


def apply_native(delta, fitted):
    output = torch.zeros_like(delta); output[:, fitted["mapping"]] = delta * fitted["scale"]
    return output


def apply_ridge(delta, fitted):
    return delta @ fitted["weight"]


def fit_maps(source_model, target_model, source_anchors, target_anchors, relations, config):
    maps = {}
    for layer in config["layers"]:
        source, target = source_anchors[layer], target_anchors[layer]
        relation = {}
        for value in range(8):
            keep = relations == value
            relation[value] = fit_ridge(source[keep], target[keep], config["ridge_relative"])
        source_write = source @ source_model.blocks[layer].down.weight.detach().cpu().T
        target_write = target @ target_model.blocks[layer].down.weight.detach().cpu().T
        write = fit_ridge(source_write, target_write, config["ridge_relative"])
        target_pinv = torch.linalg.pinv(target_model.blocks[layer].down.weight.detach().cpu())
        maps[layer] = {"native": fit_native(source, target),
                       "global": fit_ridge(source, target, config["ridge_relative"]),
                       "relation": relation, "write": write, "target_down_pinv": target_pinv,
                       "source_down": source_model.blocks[layer].down.weight.detach().cpu()}
    return maps


def map_deltas(delta, relations, fitted):
    relation = torch.empty_like(delta)
    for value in range(8):
        keep = torch.as_tensor(relations == value)
        if bool(keep.any()): relation[keep] = apply_ridge(delta[keep], fitted["relation"][value])
    source_write = delta @ fitted["source_down"].T
    target_write = apply_ridge(source_write, fitted["write"])
    write_gate = target_write @ fitted["target_down_pinv"].T
    return {"native_permutation": apply_native(delta, fitted["native"]),
            "global_ridge": apply_ridge(delta, fitted["global"]),
            "relation_ridge": relation, "write_quotient": write_gate}


def fisher_rows(base, target_effect, predicted_effect):
    probability = torch.softmax(base, -1)
    target = target_effect - (probability * target_effect).sum(-1, keepdim=True)
    predicted = predicted_effect - (probability * predicted_effect).sum(-1, keepdim=True)
    target_energy = (probability * target.square()).sum(-1).clamp_min(1e-30)
    predicted_energy = (probability * predicted.square()).sum(-1).clamp_min(1e-30)
    cosine = (probability * target * predicted).sum(-1) / torch.sqrt(target_energy * predicted_energy).clamp_min(1e-30)
    fidelity = 1 - (probability * (predicted - target).square()).sum(-1) / target_energy
    return target_energy.sqrt(), cosine, fidelity


@torch.inference_mode()
def evaluate_family(source_model, target_model, maps, family, config, device):
    base_prompts = family["base_prompts"]; target_labels = torch.as_tensor(family["target_labels"], device=device)
    source_base, source_base_acts = forward(source_model, base_prompts, device)
    target_base, target_base_acts = forward(target_model, base_prompts, device)
    rows = []
    for layer in config["layers"]:
        source_delta = family["source_delta"](layer, source_model, source_base_acts, device).cpu()
        target_delta = family["target_delta"](layer, target_model, target_base_acts, device).cpu()
        own_logits, _ = forward(target_model, base_prompts, device,
                                ActivationPatch(layer, target_delta.to(device), additive=True))
        own_effect = own_logits - target_base
        energy, _, _ = fisher_rows(target_base, own_effect, own_effect)
        keep = energy >= config["minimum_fisher_effect_norm"]
        methods = {}
        mapped = map_deltas(source_delta, family["relations"], maps[layer])
        for name, values in mapped.items():
            predicted_logits, _ = forward(target_model, base_prompts, device,
                                          ActivationPatch(layer, values.to(device), additive=True))
            predicted_effect = predicted_logits - target_base
            _, cosine, fidelity = fisher_rows(target_base, own_effect, predicted_effect)
            methods[name] = {"median_cosine": float(cosine[keep].median()) if bool(keep.any()) else None,
                             "median_fidelity": float(fidelity[keep].median()) if bool(keep.any()) else None,
                             "target_accuracy": float((predicted_logits[keep].argmax(-1) == target_labels[keep]).float().mean()) if bool(keep.any()) else None}
        source_logits, _ = forward(source_model, base_prompts, device,
                                   ActivationPatch(layer, source_delta.to(device), additive=True))
        source_effect = source_logits - source_base
        _, cosine, fidelity = fisher_rows(target_base, own_effect, source_effect)
        methods["source_effect_copy"] = {"median_cosine": float(cosine[keep].median()) if bool(keep.any()) else None,
                                           "median_fidelity": float(fidelity[keep].median()) if bool(keep.any()) else None,
                                           "target_accuracy": None}
        methods["target_own_delta"] = {"median_cosine": 1.0, "median_fidelity": 1.0,
                                         "target_accuracy": float((own_logits[keep].argmax(-1) == target_labels[keep]).float().mean()) if bool(keep.any()) else None}
        rows.append({"layer": layer, "examples": len(base_prompts), "effectful_examples": int(keep.sum()),
                     "median_own_effect_norm": float(energy.median()), "methods": methods})
    return rows


def semantic_family(source_model, target_model, manifest, config, pair_seed, device):
    rng = np.random.default_rng(pair_seed); permutations_table = np.asarray(manifest["permutations"])
    records = []; discovery = int(config["semantic_examples_per_transition"] * config["semantic_discovery_fraction"])
    for relation in range(4):
        for before in (0, 2, 4, 6):
            base, changed = make_transition_examples(rng, relation, before, before + 1, config["semantic_examples_per_transition"])
            records.append((relation, before, base, changed, int(permutations_table[relation, before + 1])))
    discovery_cache = {}
    base_eval, target_labels, relations = [], [], []
    for index, (relation, before, base, changed, target) in enumerate(records):
        base0, changed0 = encode_prompts(manifest, base[:discovery], 0), encode_prompts(manifest, changed[:discovery], 0)
        _, source_base = forward(source_model, base0, device); _, source_changed = forward(source_model, changed0, device)
        _, target_base = forward(target_model, base0, device); _, target_changed = forward(target_model, changed0, device)
        discovery_cache[index] = ([((source_changed[layer] - source_base[layer]).mean(0)).cpu() for layer in range(4)],
                                  [((target_changed[layer] - target_base[layer]).mean(0)).cpu() for layer in range(4)])
        count = len(base) - discovery; base_eval.append(encode_prompts(manifest, base[discovery:], config["evaluation_template"]))
        target_labels.extend([target] * count); relations.extend([relation] * count)
    per_transition = config["semantic_examples_per_transition"] - discovery
    def deltas(side):
        def function(layer, model, base_acts, device):
            values = [discovery_cache[i][side][layer].repeat(per_transition, 1) for i in range(len(records))]
            return torch.cat(values).to(device)
        return function
    return {"base_prompts": np.concatenate(base_eval), "target_labels": np.asarray(target_labels),
            "relations": np.asarray(relations), "source_delta": deltas(0), "target_delta": deltas(1)}


def paired_family(source_model, target_model, manifest, base_indices, donor_indices, labels, config, device):
    facts = np.asarray(manifest["train_facts"]); template = config["evaluation_template"]
    base_prompts = encode_prompts(manifest, facts[base_indices], template)
    donor_prompts = encode_prompts(manifest, facts[donor_indices], template)
    _, source_donor = forward(source_model, donor_prompts, device)
    _, target_donor = forward(target_model, donor_prompts, device)
    def source_delta(layer, model, base_acts, device): return source_donor[layer] - base_acts[layer]
    def target_delta(layer, model, base_acts, device): return target_donor[layer] - base_acts[layer]
    return {"base_prompts": base_prompts, "target_labels": labels[donor_indices],
            "relations": facts[base_indices, 2], "source_delta": source_delta, "target_delta": target_delta}


def make_pairs(manifest, labels, selected, correct_mask, count, arbitrary, seed):
    facts = np.asarray(manifest["train_facts"]); structured = np.asarray(manifest["train_structured"])
    rng = np.random.default_rng(seed); candidates = np.flatnonzero(correct_mask & (selected if arbitrary else np.ones_like(selected, bool)))
    groups = {}
    for index in candidates:
        key = (int(facts[index, 2]), int(structured[index])) if arbitrary else int(facts[index, 2])
        groups.setdefault(key, []).append(int(index))
    bases, donors = [], []
    for key in sorted(groups, key=str):
        members = np.asarray(groups[key]); rng.shuffle(members)
        for offset in range(0, len(members) - 1, 2):
            first, second = members[offset:offset + 2]
            if labels[first] == labels[second]: continue
            bases.append(first); donors.append(second)
    order = rng.permutation(len(bases))[:count]
    if len(order) < count: raise RuntimeError(("insufficient pairs", arbitrary, len(order), count))
    return np.asarray(bases)[order], np.asarray(donors)[order]


def anchors(model, manifest, facts, config, device):
    values = [[] for _ in range(4)]
    for template in config["anchor_templates"]:
        _, activations = forward(model, encode_prompts(manifest, facts, template), device)
        for layer in range(4): values[layer].append(activations[layer].cpu())
    return [torch.cat(layer) for layer in values]


def summarize_pairs(pair_results, regime, family, layer, method, metric="median_fidelity"):
    rows = [row for row in pair_results if row["regime"] == regime]
    values = []
    for row in rows:
        result = next(item for item in row["families"][family] if item["layer"] == layer)
        value = result["methods"][method][metric]
        if value is not None: values.append(value)
    return float(np.mean(values)) if values else float("nan")


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = load_manifest(Path(config["pure_models"][0]["root"]) / "manifest.npz")
    rng = np.random.default_rng(config["seed"])
    anchor_facts = random_facts(rng, config["anchor_examples_per_relation"])
    anchor_relations = np.tile(np.arange(8), config["anchor_examples_per_relation"])
    # random_facts concatenates relation blocks, whereas tile interleaves.
    anchor_relations = np.tile(
        np.repeat(np.arange(8), config["anchor_examples_per_relation"]),
        len(config["anchor_templates"]),
    )
    models, anchor_data = {}, {}
    for regime, specs, rho in (("pure", config["pure_models"], config["pure_rho"]),
                               ("mixed", config["mixed_models"], config["mixed_rho"])):
        for spec in specs:
            key = (regime, spec["name"]); print("load", key, flush=True)
            model = load_model(Path(spec["root"]) / f"rho_{rho:g}/depth_4/checkpoint.pt", device)
            models[key] = model; anchor_data[key] = anchors(model, manifest, anchor_facts, config, device)
    labels_by_regime = {}; selected_by_regime = {}; correct_by_regime = {}
    facts = np.asarray(manifest["train_facts"])
    for regime, specs, rho in (("pure", config["pure_models"], config["pure_rho"]),
                               ("mixed", config["mixed_models"], config["mixed_rho"])):
        labels, selected, _ = labels_for_rho(manifest, "train", rho)
        common = np.ones(len(facts), bool)
        for spec in specs:
            model = models[(regime, spec["name"])]
            for template in (0, config["evaluation_template"]):
                logits, _ = forward(model, encode_prompts(manifest, facts, template), device)
                common &= logits.argmax(-1).cpu().numpy() == labels
        labels_by_regime[regime] = labels; selected_by_regime[regime] = selected; correct_by_regime[regime] = common
    generic_count = config.get("generic_pair_examples", config["pair_examples"])
    arbitrary_count = config.get("arbitrary_pair_examples", config["pair_examples"])
    family_pairs = {}
    for regime in ("pure", "mixed"):
        family_pairs[(regime, "generic_pair")] = make_pairs(manifest, labels_by_regime[regime], selected_by_regime[regime], correct_by_regime[regime], generic_count, False, config["seed"] + (1 if regime == "pure" else 2))
    family_pairs[("mixed", "arbitrary_swap")] = make_pairs(manifest, labels_by_regime["mixed"], selected_by_regime["mixed"], correct_by_regime["mixed"], arbitrary_count, True, config["seed"] + 3)
    pair_results = []
    for regime, specs in (("pure", config["pure_models"]), ("mixed", config["mixed_models"])):
        for source_spec, target_spec in permutations(specs, 2):
            print("pair", regime, source_spec["name"], "->", target_spec["name"], flush=True)
            source = models[(regime, source_spec["name"])]; target = models[(regime, target_spec["name"])]
            fitted = fit_maps(source, target, anchor_data[(regime, source_spec["name"])],
                              anchor_data[(regime, target_spec["name"])], anchor_relations, config)
            families = {}
            semantic = semantic_family(source, target, manifest, config,
                                       config["seed"] + source_spec["seed"] * 1000 + target_spec["seed"], device)
            families["semantic"] = evaluate_family(source, target, fitted, semantic, config, device)
            for family_name in (("generic_pair",) if regime == "pure" else ("generic_pair", "arbitrary_swap")):
                base, donor = family_pairs[(regime, family_name)]
                family = paired_family(source, target, manifest, base, donor, labels_by_regime[regime], config, device)
                families[family_name] = evaluate_family(source, target, fitted, family, config, device)
            pair_results.append({"regime": regime, "source": source_spec["name"], "target": target_spec["name"], "families": families})
    layers = config["layers"]
    def valid(regime, family, layer):
        return all(next(item for item in row["families"][family] if item["layer"] == layer)["effectful_examples"] >= 64
                   for row in pair_results if row["regime"] == regime)
    pure_global = {layer: summarize_pairs(pair_results, "pure", "semantic", layer, "global_ridge") for layer in layers}
    best_layer = max(layers, key=pure_global.get)
    t0 = any(valid("pure", "semantic", layer) and min(next(item for item in row["families"]["semantic"] if item["layer"] == layer)["methods"]["global_ridge"]["median_fidelity"] for row in pair_results if row["regime"] == "pure") >= .5 for layer in layers)
    t1 = valid("pure", "semantic", best_layer) and pure_global[best_layer] >= summarize_pairs(pair_results, "pure", "semantic", best_layer, "native_permutation") + .15
    t2 = any(valid("pure", "semantic", layer) and min(next(item for item in row["families"]["semantic"] if item["layer"] == layer)["methods"]["write_quotient"]["median_fidelity"] for row in pair_results if row["regime"] == "pure") >= .5 for layer in layers)
    t3 = any(valid("pure", "semantic", layer) and summarize_pairs(pair_results, "pure", "semantic", layer, "relation_ridge") >= pure_global[layer] + .1 for layer in layers)
    mixed_global = {layer: summarize_pairs(pair_results, "mixed", "arbitrary_swap", layer, "global_ridge") for layer in layers}
    mixed_write = {layer: summarize_pairs(pair_results, "mixed", "arbitrary_swap", layer, "write_quotient") for layer in layers}
    t4 = all(valid("mixed", "arbitrary_swap", layer) and mixed_global[layer] < .3 for layer in layers)
    t5 = any(valid("mixed", "arbitrary_swap", layer) and mixed_write[layer] >= .3 and mixed_write[layer] >= mixed_global[layer] + .15 for layer in layers)
    t6 = (valid("pure", "semantic", best_layer) and valid("pure", "generic_pair", best_layer)
          and pure_global[best_layer] >= summarize_pairs(pair_results, "pure", "generic_pair", best_layer, "global_ridge") + .2)
    entering = [(row, family, item) for row in pair_results for family, items in row["families"].items() for item in items]
    t7 = all(item["effectful_examples"] >= 64 for _, _, item in entering)
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (t0, t1, t2, t3, t4, t5, t6, t7))}
    aggregate = {}; aggregate_accuracy = {}
    for regime in ("pure", "mixed"):
        families = ("semantic", "generic_pair") if regime == "pure" else ("semantic", "generic_pair", "arbitrary_swap")
        aggregate[regime] = {}; aggregate_accuracy[regime] = {}
        for family in families:
            aggregate[regime][family] = {str(layer): {method: summarize_pairs(pair_results, regime, family, layer, method)
                                                       for method in ("native_permutation", "global_ridge", "relation_ridge", "write_quotient", "source_effect_copy")}
                                                  for layer in layers}
            aggregate_accuracy[regime][family] = {str(layer): {method: summarize_pairs(pair_results, regime, family, layer, method, "target_accuracy")
                                                                for method in ("native_permutation", "global_ridge", "relation_ridge", "write_quotient", "target_own_delta")}
                                                       for layer in layers}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "best_pure_semantic_layer": best_layer,
                                  "aggregate_fidelity": aggregate, "aggregate_target_accuracy": aggregate_accuracy,
                                  "pairs": pair_results}, indent=2, allow_nan=False))
    lines = ["# Held-out operation-family alignment tournament", "",
             "All maps are fitted from unlabelled anchor activations before intervention outcomes are computed. Native permutation, global gate ridge, relation-conditioned ridge, and a weight-quotiented residual-write map compete on finite target-model patches. Fidelity is measured against the target model's own activation delta, only where that own edit has Fisher norm at least 0.1.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "## Median effect fidelity across ordered model pairs", "", "| Regime/family/layer | Native | Global ridge | Relation ridge | Write quotient | Source effect |", "|---|---:|---:|---:|---:|---:|"]
    for regime, families in aggregate.items():
        for family, layer_rows in families.items():
            for layer, values in layer_rows.items():
                lines.append(f"| {regime} / {family} / L{layer} | {values['native_permutation']:.3f} | {values['global_ridge']:.3f} | {values['relation_ridge']:.3f} | {values['write_quotient']:.3f} | {values['source_effect_copy']:.3f} |")
    lines += ["", "The semantic family uses minimal direct-rule transitions, with delta discovery on template 0 and finite evaluation on disjoint identities in template 6. Arbitrary swaps pair two learned randomized facts with the same relation and structured answer but different stored labels. Generic pairs are unrestricted learned prompt swaps. The own target delta is a causal ceiling, not an input to any alignment fit."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
