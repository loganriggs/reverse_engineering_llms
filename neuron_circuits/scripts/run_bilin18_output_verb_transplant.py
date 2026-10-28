#!/usr/bin/env python3
"""Finite transplantation test for a recurring leading output verb."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_output_verb_transplant.json")
MODE_SOURCE = TENSOR_ROOT / "bilin18_bus_aware_function_templates.pt"
OUTPUT = TENSOR_ROOT / "bilin18_output_verb_transplant.json"
REPORT = TENSOR_ROOT / "BILIN18_OUTPUT_VERB_TRANSPLANT.md"


def matched_controls(native, partner, mode_bank, count, seed):
    native = F.normalize(native.float(), dim=0)
    partner = F.normalize(partner.float(), dim=0)
    if torch.dot(native, partner) < 0: partner = -partner
    cosine = torch.dot(native, partner).clamp(-1, 1)
    generator = torch.Generator(device=native.device).manual_seed(seed)
    controls = []
    for _ in range(count):
        coefficient = torch.randn(len(mode_bank), generator=generator, device=native.device)
        orthogonal = coefficient @ mode_bank
        orthogonal = orthogonal - torch.dot(orthogonal, native) * native
        orthogonal = F.normalize(orthogonal, dim=0)
        controls.append(cosine * native + torch.sqrt((1 - cosine.square()).clamp_min(0)) * orthogonal)
    return partner, torch.stack(controls), float(cosine)


@torch.inference_mode()
def intervention_logits(model, ids, target_layer, target_head, native_mode, replacements, positions):
    variants = 2 + len(replacements)  # native, delete, replacements
    batch, length = ids.shape; dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    x0_base = F.rms_norm(model.embedding(ids), (dimension,)).float()
    x0 = x0_base.unsqueeze(0).repeat(variants, 1, 1, 1).flatten(0, 1)
    x, bus = x0, None
    target_stats = None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
        bus = details["value_bus"]
        write = details["write"].float()
        if layer == target_layer:
            pattern = details["pattern"]
            values = details["mixed_value"].float()
            attended = torch.einsum("bhqk,bkhd->bhqd", pattern, values)
            head_value = attended[:, target_head]
            sl = slice(target_head * head_dim, (target_head + 1) * head_dim)
            head_write = F.linear(head_value, block.attn.c_proj.weight[:, sl].float())
            coefficient = torch.einsum("bqd,d->bq", head_write, native_mode).reshape(variants, batch, length)
            native_head = head_write.reshape(variants, batch, length, dimension)[0, :, positions]
            native_coefficient = coefficient[0, :, positions]
            target_stats = {
                "coefficient_square": float(native_coefficient.square().sum()),
                "head_write_square": float(native_head.square().sum()),
                "sites": int(native_coefficient.numel()),
            }
            replacement_bank = torch.cat((native_mode[None], torch.zeros_like(native_mode)[None], replacements), 0)
            delta_direction = replacement_bank - native_mode[None]
            delta = coefficient[:, :, :, None] * delta_direction[:, None, None, :]
            write = write.reshape(variants, batch, length, dimension) + delta
            write = write.flatten(0, 1)
        pre = mixed + write
        state = F.rms_norm(pre, (dimension,))
        x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    x = x.reshape(variants, batch, length, dimension)[:, :, positions]
    raw = model.lm_head(F.rms_norm(x, (dimension,))).float()
    return 30 * torch.tanh(raw / 30), target_stats


@torch.inference_mode()
def checkpoint_logits(model, ids, positions):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
    for block in model.blocks:
        x, bus = block(x, bus, x0)
    raw = model.lm_head(F.rms_norm(x[:, positions], (dimension,))).float()
    return 30 * torch.tanh(raw / 30)


def main():
    config = json.loads(CONFIG.read_text())
    saved = torch.load(MODE_SOURCE, map_location="cpu", weights_only=True)
    labels = saved["labels"]; mode_bank = saved["output_modes"].float().to("cuda")
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    pair_labels = [f"L{layer}H{head}" for layer, head in config["pair"]]
    directions = [mode_bank[labels.index(label)] for label in pair_labels]
    programs = []
    maximum_angle_error = 0.0
    for index, ((layer, head), native, partner) in enumerate(zip(config["pair"], directions, directions[::-1])):
        aligned, controls, cosine = matched_controls(native, partner, mode_bank, config["control_count"], config["seed"] + index)
        actual = F.normalize(controls, dim=1) @ F.normalize(native, dim=0)
        maximum_angle_error = max(maximum_angle_error, float((actual - cosine).abs().max()))
        replacements = torch.cat((aligned[None], controls), 0)
        programs.append({"name": f"{pair_labels[index]}_to_{pair_labels[1-index]}", "layer": layer, "head": head, "native": F.normalize(native, dim=0), "replacements": replacements, "cosine": cosine})
    corpus = np.load(config["corpus"], mmap_mode="r")
    raw = np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    halves = {"a": raw[:len(raw)//2], "b": raw[len(raw)//2:]}
    results = {}
    positions = torch.as_tensor(config["positions"], device="cuda")
    validation_ids = torch.as_tensor(raw[:1, :config["sequence_length"]], device="cuda")
    reference = checkpoint_logits(model, validation_ids, positions)
    maximum_native_error = 0.0
    for program in programs:
        native_check, _ = intervention_logits(model, validation_ids, program["layer"], program["head"], program["native"], program["replacements"], positions)
        maximum_native_error = max(maximum_native_error, float((native_check[0] - reference).abs().max()))
    for program in programs:
        for half, panel in halves.items():
            print("finite verb transplant", program["name"], half, flush=True)
            totals = torch.zeros(2 + len(program["replacements"]), dtype=torch.float64, device="cuda")
            sites = 0
            coefficient_square = 0.0; head_write_square = 0.0
            for start in range(0, len(panel), config["evaluation_batch_size"]):
                ids = torch.as_tensor(panel[start:start + config["evaluation_batch_size"], :config["sequence_length"]], device="cuda")
                logits, stats = intervention_logits(model, ids, program["layer"], program["head"], program["native"], program["replacements"], positions)
                coefficient_square += stats["coefficient_square"]; head_write_square += stats["head_write_square"]
                native = logits[0]; logp = F.log_softmax(native.float(), -1)
                for variant in range(1, len(logits)):
                    logq = F.log_softmax(logits[variant].float(), -1)
                    totals[variant] += (logp.exp() * (logp - logq)).sum(-1).clamp_min(0).double().sum()
                sites += native.shape[0] * native.shape[1]
            means = (totals / sites).cpu().numpy()
            controls = means[3:]
            results[f"{program['name']}_{half}"] = {
                "native_partner_cosine": program["cosine"],
                "delete_kl": float(means[1]), "partner_kl": float(means[2]),
                "control_kl": controls.tolist(), "control_median": float(np.median(controls)),
                "control_p10": float(np.quantile(controls, .10)),
                "partner_control_ratio": float(means[2] / max(np.median(controls), 1e-30)),
                "coefficient_rms": float(np.sqrt(coefficient_square / sites)),
                "leading_component_energy_fraction": float(coefficient_square / max(head_write_square, 1e-30)),
            }
    privileged = all(results[f"{p['name']}_{half}"]["partner_kl"] < results[f"{p['name']}_{half}"]["control_median"] for p in programs for half in halves)
    tail = any(all(results[f"{p['name']}_{half}"]["partner_kl"] < results[f"{p['name']}_{half}"]["control_p10"] for half in halves) for p in programs)
    deletion = all(results[f"{p['name']}_{half}"]["partner_kl"] < results[f"{p['name']}_{half}"]["delete_kl"] for p in programs for half in halves)
    stable = all(abs(results[f"{p['name']}_a"]["partner_control_ratio"] - results[f"{p['name']}_b"]["partner_control_ratio"]) <= .25 for p in programs)
    hypotheses = {
        "V1_controls_are_angle_matched": {"supported": maximum_angle_error <= 1e-5, "maximum_error": maximum_angle_error},
        "V2_partner_is_privileged_on_both_transplants": {"supported": privileged},
        "V3_partner_beats_the_control_tail": {"supported": tail},
        "V4_partner_is_better_than_deleting_the_verb": {"supported": deletion},
        "V5_transplant_advantage_is_split_stable": {"supported": stable},
        "native_branch_matches_checkpoint": {"supported": maximum_native_error <= 1e-4, "maximum_error": maximum_native_error},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = [
        "# Finite transplant of a candidate shared output verb", "",
        "L4H2 and L5H7 are the strongest bus-aware output-direction pair and remain aligned after global whitening. At each target layer, the native scalar coefficient on the leading direction is preserved while that direction is replaced by the partner direction. Eight controls are random combinations of learned head modes, constructed to have exactly the same cosine to the native direction. The full nonlinear suffix is recomputed on fresh FineWeb.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum native-branch checkpoint error: `{maximum_native_error:.2e}`.", "", "| Transplant / split | Native↔partner cosine | Partner KL | Control median [p10] | Delete KL | Partner/control | Coeff RMS / energy fraction |", "|---|---:|---:|---:|---:|---:|---:|"]
    for program in programs:
        for half in halves:
            row = results[f"{program['name']}_{half}"]
            lines.append(f"| {program['name']} / {half} | {row['native_partner_cosine']:.3f} | {row['partner_kl']:.3e} | {row['control_median']:.3e} [{row['control_p10']:.3e}] | {row['delete_kl']:.3e} | {row['partner_control_ratio']:.3f} | {row['coefficient_rms']:.3f} / {row['leading_component_energy_fraction']:.3f} |")
    lines += [
        "", "## Guardrails", "",
        "This tests interchangeability of one output direction, not the complete QK/value function. Controls match angular distance and are drawn from the learned output-mode span, but they are not matched for every downstream curvature direction. The pair was selected after the atlas and matched-null analyses; the evaluation corpus is fresh, but a positive result remains a prospective validation of this pair rather than a population prevalence estimate.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
