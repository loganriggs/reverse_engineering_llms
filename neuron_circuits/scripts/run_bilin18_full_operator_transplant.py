#!/usr/bin/env python3
"""Finite transplant test for a complete bus-aware value-side head operator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_output_verb_transplant import checkpoint_logits
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_variable_function_atlas import factored_frobenius_cosines


CONFIG = Path("configs/bilin18_full_operator_transplant.json")
OUTPUT = TENSOR_ROOT / "bilin18_full_operator_transplant.json"
REPORT = TENSOR_ROOT / "BILIN18_FULL_OPERATOR_TRANSPLANT.md"


def build_function_bank(model, layers):
    heads = model.config.n_head; dimension = model.config.n_embd; head_dim = dimension // heads
    rows = []; factors = []
    for layer in layers:
        block = model.blocks[layer].attn; lamb = float(block.lamb.detach())
        for head in range(heads):
            sl = slice(head * head_dim, (head + 1) * head_dim)
            output = block.c_proj.weight[:, sl].detach().float()
            value = block.c_v.weight[sl].detach().float()
            selector = torch.zeros(head_dim, dimension, device=output.device); selector[:, sl] = torch.eye(head_dim, device=output.device)
            combined = torch.cat(((1 - lamb) * value, lamb * selector), 1)
            rows.append({"label": f"L{layer}H{head}", "layer": layer, "head": head, "lamb": lamb, "output": output, "value": value})
            factors.append((output.cpu(), combined.cpu()))
    cosine, norms = factored_frobenius_cosines(factors)
    gram = cosine * (norms[:, None] * norms[None, :])
    return rows, gram, norms


def matched_operator_controls(gram, norms, target, partner, count, seed):
    signed_cosine = gram[target, partner] / (norms[target] * norms[partner])
    sign = 1.0 if signed_cosine >= 0 else -1.0
    cosine = abs(float(signed_cosine))
    rng = np.random.default_rng(seed)
    candidates = [index for index in range(len(norms)) if index not in (target, partner)]
    chosen = rng.choice(candidates, count, replace=False)
    controls = []
    for other in chosen:
        projection = gram[target, other] / gram[target, target]
        orthogonal_norm = np.sqrt(max(gram[other, other] - gram[target, other] ** 2 / gram[target, target], 1e-30))
        beta = np.sqrt(max(1 - cosine ** 2, 0)) * norms[target] / orthogonal_norm
        controls.append({"other": int(other), "target_coefficient": float(cosine - beta * projection), "other_coefficient": float(beta)})
    partner_scale = float(sign * norms[target] / norms[partner])
    return partner_scale, controls, cosine


def apply_function(row, pooled_state, pooled_bus, heads):
    head_dim = pooled_bus.shape[-1] // heads
    sl = slice(row["head"] * head_dim, (row["head"] + 1) * head_dim)
    current = F.linear(pooled_state, row["value"])
    mixed = (1 - row["lamb"]) * current + row["lamb"] * pooled_bus[:, :, sl]
    return F.linear(mixed, row["output"])


@torch.inference_mode()
def intervention_logits(model, ids, target_row, partner_row, partner_scale, controls, bank, positions, intervention_scale=1.0):
    variants = 3 + len(controls)  # native, delete, partner, controls
    batch, length = ids.shape; dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    x0_base = F.rms_norm(model.embedding(ids), (dimension,)).float()
    x0 = x0_base.unsqueeze(0).repeat(variants, 1, 1, 1).flatten(0, 1)
    x, bus = x0, None; factor_error = 0.0
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        normalized = F.rms_norm(mixed, (dimension,))
        details = attention_details(block.attn, normalized, bus, model.config, model.tt)
        bus = details["value_bus"]
        write = details["write"].float()
        if layer == target_row["layer"]:
            base_state = normalized.reshape(variants, batch, length, dimension)[0]
            base_bus = bus.reshape(variants, batch, length, heads, head_dim)[0].reshape(batch, length, dimension)
            pattern = details["pattern"].reshape(variants, batch, heads, length, length)[0, :, target_row["head"]]
            pooled_state = torch.einsum("bqk,bkd->bqd", pattern, base_state)
            pooled_bus = torch.einsum("bqk,bkd->bqd", pattern, base_bus)
            native_head = apply_function(target_row, pooled_state, pooled_bus, heads)
            partner_head = partner_scale * apply_function(partner_row, pooled_state, pooled_bus, heads)
            control_heads = []
            for control in controls:
                other = apply_function(bank[control["other"]], pooled_state, pooled_bus, heads)
                control_heads.append(control["target_coefficient"] * native_head + control["other_coefficient"] * other)
            replacement = torch.stack([native_head, torch.zeros_like(native_head), partner_head] + control_heads)
            delta = intervention_scale * (replacement - native_head[None])
            write = write.reshape(variants, batch, length, dimension) + delta
            write = write.flatten(0, 1)
            attended = torch.einsum("bhqk,bkhd->bhqd", details["pattern"], details["mixed_value"].float())
            sl = slice(target_row["head"] * head_dim, (target_row["head"] + 1) * head_dim)
            direct = F.linear(attended[:, target_row["head"]], block.attn.c_proj.weight[:, sl].float()).reshape(variants, batch, length, dimension)[0]
            factor_error = float((direct - native_head).norm() / direct.norm().clamp_min(1e-30))
        pre = mixed + write
        state = F.rms_norm(pre, (dimension,))
        x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    x = x.reshape(variants, batch, length, dimension)[:, :, positions]
    raw = model.lm_head(F.rms_norm(x, (dimension,))).float()
    return 30 * torch.tanh(raw / 30), factor_error


def main():
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    bank, gram, norms = build_function_bank(model, config.get("layers", [0, 1, 2, 3, 4, 5]))
    labels = [row["label"] for row in bank]
    pair_labels = [f"L{layer}H{head}" for layer, head in config["pair"]]
    pair_indices = [labels.index(label) for label in pair_labels]
    programs = []
    maximum_geometry_error = 0.0
    for direction in range(2):
        target = pair_indices[direction]; partner = pair_indices[1 - direction]
        scale, controls, cosine = matched_operator_controls(gram, norms, target, partner, config["control_count"], config["seed"] + direction)
        for control in controls:
            coefficients = np.zeros(len(bank)); coefficients[target] = control["target_coefficient"]; coefficients[control["other"]] = control["other_coefficient"]
            norm = np.sqrt(coefficients @ gram @ coefficients)
            actual_cosine = abs(coefficients @ gram[:, target] / (norm * norms[target]))
            maximum_geometry_error = max(maximum_geometry_error, abs(actual_cosine - cosine), abs(norm / norms[target] - 1))
        programs.append({"name": f"{pair_labels[direction]}_to_{pair_labels[1-direction]}", "target": target, "partner": partner, "partner_scale": scale, "controls": controls, "cosine": cosine})
    corpus = np.load(config["corpus"], mmap_mode="r")
    raw = np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    halves = {"a": raw[:len(raw)//2], "b": raw[len(raw)//2:]}; positions = torch.as_tensor(config["positions"], device="cuda")
    validation_ids = torch.as_tensor(raw[:1, :config["sequence_length"]], device="cuda")
    reference = checkpoint_logits(model, validation_ids, positions); maximum_native_error = 0.0; maximum_factor_error = 0.0
    results = {}
    for program in programs:
        check, factor_error = intervention_logits(model, validation_ids, bank[program["target"]], bank[program["partner"]], program["partner_scale"], program["controls"], bank, positions)
        maximum_native_error = max(maximum_native_error, float((check[0] - reference).abs().max())); maximum_factor_error = max(maximum_factor_error, factor_error)
        for half, panel in halves.items():
            print("full operator transplant", program["name"], half, flush=True)
            totals = torch.zeros(3 + len(program["controls"]), dtype=torch.float64, device="cuda"); sites = 0
            for start in range(0, len(panel), config["evaluation_batch_size"]):
                ids = torch.as_tensor(panel[start:start + config["evaluation_batch_size"], :config["sequence_length"]], device="cuda")
                logits, factor_error = intervention_logits(model, ids, bank[program["target"]], bank[program["partner"]], program["partner_scale"], program["controls"], bank, positions)
                maximum_factor_error = max(maximum_factor_error, factor_error)
                logp = F.log_softmax(logits[0].float(), -1)
                for variant in range(1, len(logits)):
                    logq = F.log_softmax(logits[variant].float(), -1)
                    totals[variant] += (logp.exp() * (logp - logq)).sum(-1).clamp_min(0).double().sum()
                sites += logits.shape[1] * logits.shape[2]
            means = (totals / sites).cpu().numpy(); controls = means[3:]
            results[f"{program['name']}_{half}"] = {"operator_cosine": program["cosine"], "delete_kl": float(means[1]), "partner_kl": float(means[2]), "control_kl": controls.tolist(), "control_median": float(np.median(controls)), "control_p10": float(np.quantile(controls,.10)), "partner_control_ratio": float(means[2]/max(np.median(controls),1e-30))}
    privileged = all(results[f"{p['name']}_{h}"]["partner_kl"] < results[f"{p['name']}_{h}"]["control_median"] for p in programs for h in halves)
    tail = any(all(results[f"{p['name']}_{h}"]["partner_kl"] < results[f"{p['name']}_{h}"]["control_p10"] for h in halves) for p in programs)
    deletion = all(results[f"{p['name']}_{h}"]["partner_kl"] < results[f"{p['name']}_{h}"]["delete_kl"] for p in programs for h in halves)
    stable = all(abs(results[f"{p['name']}_a"]["partner_control_ratio"]-results[f"{p['name']}_b"]["partner_control_ratio"])<=.25 for p in programs)
    hypotheses = {
        "W1_factored_native_operator_matches_head_write": {"supported": maximum_factor_error <= 1e-4, "maximum_relative_error": maximum_factor_error},
        "W2_controls_match_operator_angle_and_norm": {"supported": maximum_geometry_error <= 1e-5, "maximum_error": maximum_geometry_error},
        "W3_partner_operator_is_privileged": {"supported": privileged},
        "W4_partner_operator_beats_control_tail": {"supported": tail},
        "W5_partner_operator_beats_deletion": {"supported": deletion},
        "W6_operator_advantage_is_split_stable": {"supported": stable},
        "native_branch_matches_checkpoint": {"supported": maximum_native_error <= 1e-4, "maximum_error": maximum_native_error},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results,"control_labels":{p["name"]:[labels[c["other"]] for c in p["controls"]] for p in programs}},indent=2,allow_nan=False))
    lines=["# Finite transplant of a complete bus-aware value operator","",f"The pair {pair_labels[0]}/{pair_labels[1]} maximizes the product of absolute full-operator cosine and replicated natural-response cosine. Each complete current-state/value-bus map is transplanted under the target head's native QK routing and rescaled to the target operator norm. Controls combine the target with one other learned operator to match both operator cosine and norm exactly.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["",f"Maximum factorization error `{maximum_factor_error:.2e}`; native checkpoint error `{maximum_native_error:.2e}`; matched-geometry error `{maximum_geometry_error:.2e}`.","","| Transplant / split | Operator cosine | Partner KL | Control median [p10] | Delete KL | Partner/control |","|---|---:|---:|---:|---:|---:|"]
    for p in programs:
        for h in halves:
            row=results[f"{p['name']}_{h}"]; lines.append(f"| {p['name']} / {h} | {row['operator_cosine']:.3f} | {row['partner_kl']:.3e} | {row['control_median']:.3e} [{row['control_p10']:.3e}] | {row['delete_kl']:.3e} | {row['partner_control_ratio']:.3f} |")
    lines += ["","## Guardrails","","This tests the value-side operator under the target's native QK routing; it does not transplant the router or destination layer. Controls are exact in the raw Frobenius operator metric but downstream curvature can distinguish directions with the same angle. The pair was selected from prior weight and natural-response panels, while the finite evaluation panel is fresh."]
    REPORT.write_text("\n".join(lines)+"\n"); print(json.dumps({"hypotheses":hypotheses,"results":results},indent=2)); print(REPORT)


if __name__ == "__main__": main()
