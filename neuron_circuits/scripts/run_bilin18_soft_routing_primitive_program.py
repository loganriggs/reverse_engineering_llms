#!/usr/bin/env python3
"""Execute soft NMF source variables in the six-layer typed routing program."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_attention_head_semantic_causality import reference_logits
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_joint_composition import kl
from run_bilin18_typed_attention_edge_vocabulary import distance_ids, rmsnorm_pullback_readers


CONFIG = Path("configs/bilin18_soft_routing_primitive_program.json")
FACTOR_SOURCE = TENSOR_ROOT / "bilin18_routing_primitive_factors.pt"
PRIOR_SOURCE = TENSOR_ROOT / "bilin18_routing_primitive_factorization.json"
OUTPUT = TENSOR_ROOT / "bilin18_soft_routing_primitive_program.json"
REPORT = TENSOR_ROOT / "BILIN18_SOFT_ROUTING_PRIMITIVE_PROGRAM.md"


def normalize_memberships(membership):
    membership = membership.clamp_min(0)
    total = membership.sum(-1, keepdim=True)
    uniform = torch.full_like(membership, 1 / membership.shape[-1])
    return torch.where(total > 0, membership / total.clamp_min(1e-30), uniform)


def support_keep_weights(membership, distance, support, heads, distances, variables):
    """Return [B,Q,H,K] soft keep weights for a set of H×D×V types."""
    batch, keys, _ = membership.shape
    queries = distance.shape[0]
    support_cube = torch.zeros(heads, distances, variables, device=membership.device)
    index = torch.as_tensor(support, device=membership.device)
    head = index // (distances * variables)
    remainder = index % (distances * variables)
    dist = remainder // variables; variable = remainder % variables
    support_cube[head, dist, variable] = 1
    keep = torch.zeros(batch, queries, heads, keys, device=membership.device)
    for distance_index in range(distances):
        mask = (distance == distance_index).float()
        selected = torch.einsum("bkv,hv->bhk", membership, support_cube[:, distance_index])
        keep += selected[:, None] * mask[None, :, None]
    return keep


@torch.inference_mode()
def soft_discovery_scores(model, layer, raw, membership_by_token, config, probes):
    heads = model.config.n_head; dimension = model.config.n_embd; variables = membership_by_token.shape[1]
    distances = len(config["distance_bins"]) - 1
    distance = distance_ids(config["positions"], config["sequence_length"], config["distance_bins"], "cuda")
    onehot_distance = F.one_hot(distance.clamp_min(0), distances).float() * (distance >= 0)[:, :, None]
    scores = torch.zeros(heads, distances, variables, device="cuda", dtype=torch.float64)
    for start in range(0, len(raw), config["selection_batch_size"]):
        ids = torch.as_tensor(raw[start:start + config["selection_batch_size"], :config["sequence_length"]], device="cuda")
        membership = membership_by_token[ids]
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for block_index, block in enumerate(model.blocks[:layer + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = details["value_bus"]
            if block_index == layer:
                pre = mixed[:, config["positions"]].float() + details["write"][:, config["positions"]].float()
                readers = rmsnorm_pullback_readers(pre, probes)
                projection = block.attn.c_proj.weight.float().T.reshape(heads, dimension // heads, dimension)
                head_readers = torch.einsum("bpqo,hdo->bphqd", readers, projection)
                value_effect = torch.einsum("bkhd,bphqd->bphqk", details["mixed_value"].float(), head_readers)
                pattern = details["pattern"][:, :, config["positions"]].permute(0, 2, 1, 3).float()
                edge_effect = value_effect * pattern[:, :, :, None]
                grouped = torch.einsum("bphqk,bkv,pkd->bphdvq", edge_effect, membership, onehot_distance)
                scores += grouped.square().sum((0, 1, 5)).double()
            pre_full = mixed + details["write"]; state = F.rms_norm(pre_full, (dimension,))
            x = pre_full + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    return scores.reshape(-1)


@torch.inference_mode()
def select_soft_supports(model, discovery, membership, config):
    supports = {}
    type_count = model.config.n_head * (len(config["distance_bins"]) - 1) * config["rank"]
    for layer in config["layers"]:
        print("select soft supports layer", layer, flush=True)
        probes, _, _ = operator_probe_vectors(model.blocks[layer], config["operator_probes"], config["seed"] + layer)
        scores = soft_discovery_scores(model, layer, discovery, membership, config, probes)
        order = scores.argsort(descending=True).cpu().numpy()
        generator = np.random.default_rng(config["seed"] + 1000 + layer)
        active = torch.where(scores > 0)[0].cpu().numpy()
        supports[layer] = {f"soft{budget}": order[:budget] for budget in config["budgets"]}
        supports[layer]["random128"] = generator.choice(active, min(128, len(active)), replace=False)
        supports[layer]["all"] = np.arange(type_count)
    return supports


@torch.inference_mode()
def program_logits(model, ids, membership_by_token, supports, methods, config):
    dimension = model.config.n_embd; heads = model.config.n_head
    distances = len(config["distance_bins"]) - 1; variables = membership_by_token.shape[1]
    variants = 2 + len(methods); batch = len(ids); length = config["sequence_length"]
    base_x0 = F.rms_norm(model.embedding(ids), (dimension,)).float()
    x0 = base_x0.unsqueeze(0).repeat(variants, 1, 1, 1).flatten(0, 1); x, bus = x0, None
    membership = membership_by_token[ids]
    distance = distance_ids(range(length), length, config["distance_bins"], "cuda")
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = details["value_bus"]
        if layer in config["layers"]:
            pattern = details["pattern"].reshape(variants, batch, heads, length, length)
            values = details["mixed_value"].reshape(variants, batch, length, heads, -1)
            native = details["write"].reshape(variants, batch, length, dimension)
            writes = [native[0], torch.zeros_like(native[1])]
            for variant, method in enumerate(methods, start=2):
                keep = support_keep_weights(membership, distance, supports[layer][method], heads, distances, variables)
                attended = torch.einsum("bqhk,bkhd->bqhd", pattern[variant].permute(0, 2, 1, 3) * keep, values[variant])
                writes.append(block.attn.c_proj(attended.reshape(batch, length, dimension)).float())
            write = torch.stack(writes).flatten(0, 1)
        else:
            write = details["write"].float()
        pre = mixed + write; state = F.rms_norm(pre, (dimension,))
        x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    raw = model.lm_head(F.rms_norm(x, (dimension,))).float()
    return (30 * torch.tanh(raw / 30)).reshape(variants, batch, length, -1)


@torch.inference_mode()
def evaluate_soft(model, raw, membership, supports, config):
    methods = [f"soft{budget}" for budget in config["budgets"]] + ["random128", "all"]
    accum = {name: {"kl": 0.0, "ce": 0.0, "top1": 0} for name in methods}
    denominator = 0.0; maximum_error = 0.0; count = 0
    positions = torch.as_tensor(config["positions"], device="cuda")
    for start in range(0, len(raw), config["evaluation_batch_size"]):
        batch_raw = raw[start:start + config["evaluation_batch_size"]]
        ids = torch.as_tensor(batch_raw[:, :config["sequence_length"]], device="cuda")
        logits = program_logits(model, ids, membership, supports, methods, config)
        native = logits[0, :, positions]; baseline = logits[1, :, positions]
        target = torch.as_tensor(batch_raw[:, np.asarray(config["positions"]) + 1], device="cuda")
        native_ce = F.cross_entropy(native.flatten(0, 1), target.flatten(), reduction="none").reshape_as(target)
        reference = torch.stack([reference_logits(model, ids, int(position)) for position in config["positions"]], 1)
        maximum_error = max(maximum_error, float((native - reference).abs().max()))
        denominator += float(kl(native, baseline).sum())
        for index, name in enumerate(methods, start=2):
            prediction = logits[index, :, positions]
            accum[name]["kl"] += float(kl(native, prediction).sum())
            predicted_ce = F.cross_entropy(prediction.flatten(0, 1), target.flatten(), reduction="none").reshape_as(target)
            accum[name]["ce"] += float((predicted_ce - native_ce).sum())
            accum[name]["top1"] += int((prediction.argmax(-1) == native.argmax(-1)).sum())
        count += target.numel()
    results = {}
    for name, row in accum.items():
        results[name] = {"kl_recovery": 1 - row["kl"] / max(denominator, 1e-30), "mean_kl_error": row["kl"] / count, "mean_ce_delta": row["ce"] / count, "top1_agreement": row["top1"] / count}
    return results, maximum_error, denominator / count


def main():
    config = json.loads(CONFIG.read_text())
    factors = torch.load(FACTOR_SOURCE, map_location="cpu", weights_only=True)
    membership = normalize_memberships(factors["membership_a"].float()).to("cuda")
    prior = json.loads(PRIOR_SOURCE.read_text())
    hard_recovery = prior["results"]["nmf_a"]["methods"]["fine128"]["kl_recovery"]
    corpus = np.load(config["corpus"], mmap_mode="r")
    discovery = np.asarray(corpus[config["selection_start"]:config["selection_start"] + config["selection_rows"], :config["sequence_length"]], dtype=np.int64)
    evaluation = np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    supports = select_soft_supports(model, discovery, membership, config)
    results, maximum_error, baseline_kl = evaluate_soft(model, evaluation, membership, supports, config)
    recoveries = [results[f"soft{budget}"]["kl_recovery"] for budget in config["budgets"]]
    top1s = [results[f"soft{budget}"]["top1_agreement"] for budget in config["budgets"]]
    hypotheses = {
        "S0_all_soft_types_reproduce_native": {"supported": results["all"]["kl_recovery"] >= .9999 and results["all"]["top1_agreement"] >= .999},
        "S1_soft128_improves_hard128": {"supported": results["soft128"]["kl_recovery"] >= hard_recovery + .03, "difference": results["soft128"]["kl_recovery"] - hard_recovery},
        "S2_soft128_is_prediction_faithful": {"supported": results["soft128"]["top1_agreement"] >= .80},
        "S3_soft256_is_high_fidelity": {"supported": results["soft256"]["top1_agreement"] >= .90},
        "S4_selected_soft128_beats_random": {"supported": results["soft128"]["kl_recovery"] >= results["random128"]["kl_recovery"] + .20, "difference": results["soft128"]["kl_recovery"] - results["random128"]["kl_recovery"]},
        "S5_soft_budget_frontier_is_monotonic": {"supported": all(a < b for a, b in zip(recoveries, recoveries[1:])) and all(a < b for a, b in zip(top1s, top1s[1:]))},
        "native_branch_matches_checkpoint": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
    }
    for value in hypotheses.values(): value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "hard_nmf_a_recovery": hard_recovery, "baseline_kl": baseline_kl, "results": results}, indent=2, allow_nan=False))
    lines = ["# Soft routing-variable program", "", "NMF token memberships are normalized to a partition of unity and executed directly. Each edge is fractionally assigned to variables; selecting a `(head, distance, variable)` type retains that membership fraction. All types must therefore exactly recover native attention. The route supports are selected from soft grouped operator effects on discovery.", "", f"Maximum native-branch error: `{maximum_error:.2e}`. No-attention baseline KL: `{baseline_kl:.3f}`.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "| Program | KL recovery | Mean KL error | CE delta | Top-1 |", "|---|---:|---:|---:|---:|"]
    for name in [f"soft{budget}" for budget in config["budgets"]] + ["random128", "all"]:
        value = results[name]; lines.append(f"| {name} | {value['kl_recovery']:.4f} | {value['mean_kl_error']:.4f} | {value['mean_ce_delta']:+.4f} | {value['top1_agreement']:.4f} |")
    lines += ["", "## Guardrails", "", "Soft types are more expressive than hard types because a token edge can be partially retained by several variables. The comparison holds the number of selected `(head, distance, variable)` predicates fixed, not the number of nonzero edge memberships or arithmetic operations. NMF membership values are real numbers and therefore cost more to describe than a categorical dictionary unless quantized or generated by a shorter rule. High causal fidelity alone does not establish MDL superiority or readability."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
