#!/usr/bin/env python3
"""Finite suffix validation of natural-stratum native MLP channel banks."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_stratified_natural_mlp_census import bank, site_strata
from run_bilin18_stratified_natural_route_census import token_class_from_flags
from run_bilin18_unseeded_route_end_to_end_causality import empty_metric, finish, full_logits, update_metric


CONFIG = Path("configs/bilin18_stratified_mlp_bank_finite_causality.json")
OUTPUT = TENSOR_ROOT / "bilin18_stratified_mlp_bank_finite_causality.json"
REPORT = TENSOR_ROOT / "BILIN18_STRATIFIED_MLP_BANK_FINITE_CAUSALITY.md"


@torch.inference_mode()
def native_product(model, ids, target_layer):
    dimension = model.config.n_embd; x0 = F.rms_norm(model.embedding(ids), (dimension,)); x = x0; bus = None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0; details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = details["value_bus"]
        pre = mixed + details["write"]; normalized = F.rms_norm(pre, (dimension,)); product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer == target_layer: return product[:, -1].float()
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    raise ValueError(target_layer)


@torch.inference_mode()
def intervene_product(model, ids, target_layer, products_by_condition):
    dimension = model.config.n_embd; x0 = F.rms_norm(model.embedding(ids), (dimension,)); x = x0; bus = None; destination = ids.shape[1] - 1
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0; details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = details["value_bus"]
        pre = mixed + details["write"]; normalized = F.rms_norm(pre, (dimension,)); product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer != target_layer:
            x = pre + block.mlp.Down(product) + block.mlp.Down_bias; continue
        names = list(products_by_condition); states = []
        for name in names:
            modified = product.clone(); modified[:, destination] = products_by_condition[name].to(modified.dtype)
            states.append(pre + block.mlp.Down(modified) + block.mlp.Down_bias)
        count = len(names); x = torch.cat(states); x0 = x0.repeat(count, 1, 1); bus = bus.repeat(count, 1, 1, 1) if bus.ndim == 4 else bus.repeat(count, 1, 1)
        for later in model.blocks[layer + 1:]:
            mixed = later.lambdas[0] * x + later.lambdas[1] * x0; later_details = attention_details(later.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = later_details["value_bus"]
            pre = mixed + later_details["write"]; normalized = F.rms_norm(pre, (dimension,)); x = pre + later.mlp.Down(later.mlp.Left(normalized) * later.mlp.Right(normalized)) + later.mlp.Down_bias
        raw = model.lm_head(F.rms_norm(x, (dimension,))[:, destination]).float(); logits = 30 * torch.tanh(raw / 30)
        return names, logits.reshape(count, len(ids), -1)
    raise ValueError(target_layer)


def runtime_site_mask(stratum, raw, position, losses, token_classes, discovery_loss):
    family, label = stratum.split(":", 1)
    if family == "current": return token_classes[raw[:, position]] == label
    if family == "next": return token_classes[raw[:, position + 1]] == label
    if family == "position": return np.asarray([f"p{position}" == label] * len(raw))
    if family == "loss":
        boundaries = np.quantile(discovery_loss, [.25, .50, .75]); labels = np.asarray(["easy", "medium-low", "medium-high", "hard"])[np.digitize(losses, boundaries)]; return labels == label
    raise ValueError(stratum)


def retain(product, selected):
    output = torch.zeros_like(product); output[:, torch.as_tensor(selected, device=product.device)] = product[:, torch.as_tensor(selected, device=product.device)]; return output


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source"]).read_text()); arrays = np.load(config["arrays"])
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); token_classes = token_class_from_flags(flags)
    dmeta = {key: arrays[f"discovery_meta_{key}"] for key in ("position", "current_id", "next_id", "loss")}; dmeta["current_class"] = token_classes[dmeta["current_id"]]; dmeta["next_class"] = token_classes[dmeta["next_id"]]; masks = site_strata(dmeta, dmeta["loss"])
    rng = np.random.default_rng(config["seed"]); prepared = {}; observational = {(row["layer"], row["stratum"]): row for row in source["results"] if row["size"] == config["bank_size"]}
    for target in config["targets"]:
        layer = target["layer"]; contributions = arrays[f"discovery_contributions_L{layer}"]
        prepared[layer] = {"local": bank(contributions, masks[target["stratum"]], config["bank_size"]), "global": bank(contributions, masks["all"], config["bank_size"]), "random": [rng.choice(contributions.shape[1], size=config["bank_size"], replace=False) for _ in range(config["random_replicates"])], "loss": dmeta["loss"]}
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"]]
    methods = ["native", "deleted", "local", "global"] + [f"random_{i}" for i in range(config["random_replicates"])]
    metrics = {target["layer"]: {name: empty_metric() for name in methods} for target in config["targets"]}; max_error = 0.0
    for position in config["positions"]:
        length = position + 1
        for start in range(0, len(corpus), config["batch_size"]):
            raw = np.asarray(corpus[start:start + config["batch_size"], :length + 1], dtype=np.int64); ids = torch.as_tensor(raw[:, :length], device="cuda"); targets = torch.as_tensor(raw[:, length], device="cuda")
            reference = full_logits(model, ids); losses = F.cross_entropy(reference, targets, reduction="none").cpu().numpy()
            for target in config["targets"]:
                layer = target["layer"]; selected_examples = runtime_site_mask(target["stratum"], raw, position, losses, token_classes, prepared[layer]["loss"])
                if not selected_examples.any(): continue
                product = native_product(model, ids, layer); conditions = {"native": product, "deleted": torch.zeros_like(product), "local": retain(product, prepared[layer]["local"]), "global": retain(product, prepared[layer]["global"])}
                for index, selected in enumerate(prepared[layer]["random"]): conditions[f"random_{index}"] = retain(product, selected)
                names, logits = intervene_product(model, ids, layer, conditions); outputs = {name: logits[index] for index, name in enumerate(names)}; max_error = max(max_error, float((outputs["native"] - reference).abs().max())); mask = torch.as_tensor(selected_examples, device="cuda")
                for name in methods: update_metric(metrics[layer][name], outputs["native"][mask], outputs["deleted"][mask], outputs[name][mask], targets[mask])
        print(f"finite MLP bank position={position}", flush=True)
    results = {}; logit_advantages = []; ce_advantages = []; observational_advantages = []; local_wins = []; ce_wins = []; random_wins = []; sufficient = []
    for target in config["targets"]:
        layer = target["layer"]; rows = {name: finish(value) for name, value in metrics[layer].items()}; logit_adv = rows["local"]["causal_fidelity"] - rows["global"]["causal_fidelity"]; ce_adv = rows["local"]["ce_effect_fidelity"] - rows["global"]["ce_effect_fidelity"]; random_p95 = float(np.quantile([rows[f"random_{i}"]["causal_fidelity"] for i in range(config["random_replicates"])], .95)); observed = observational[(layer, target["stratum"])]["ce_advantage"]
        results[f"L{layer}:{target['stratum']}"] = {"metrics": rows, "finite_logit_advantage": logit_adv, "finite_ce_advantage": ce_adv, "random_logit_p95": random_p95, "observational_ce_advantage": observed}
        logit_advantages.append(logit_adv); ce_advantages.append(ce_adv); observational_advantages.append(observed); local_wins.append(logit_adv > 0); ce_wins.append(ce_adv > 0); random_wins.append(rows["local"]["causal_fidelity"] > random_p95); sufficient.append(rows["local"]["causal_fidelity"] >= .50)
    correlation = float(np.corrcoef(observational_advantages, ce_advantages)[0, 1]) if np.std(observational_advantages) and np.std(ce_advantages) else 0.0
    hypotheses = {
        "N0_native_replay_is_exact": {"supported": max_error <= .02, "maximum_logit_error": max_error},
        "N1_specialized_banks_improve_finite_logits": {"supported": sum(local_wins) >= 3 and float(np.mean(logit_advantages)) >= .05, "wins": int(sum(local_wins)), "mean_advantage": float(np.mean(logit_advantages))},
        "N2_specialized_banks_improve_finite_ce": {"supported": sum(ce_wins) >= 3 and float(np.mean(ce_advantages)) >= .05, "wins": int(sum(ce_wins)), "mean_advantage": float(np.mean(ce_advantages))},
        "N3_specialized_banks_beat_random_controls": {"supported": all(random_wins), "wins": int(sum(random_wins)), "targets": len(random_wins)},
        "N4_fixed_128_channel_banks_are_often_sufficient": {"supported": sum(sufficient) >= 2, "passing": int(sum(sufficient))},
        "N5_local_linear_advantage_predicts_finite_advantage": {"supported": correlation >= .50, "correlation": correlation},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Finite causality of natural-stratum MLP channel banks", "", "At each selected natural site, all destination-position MLP products outside a frozen 128-channel bank are zeroed and the complete nonlinear suffix recomputes. Specialized, global, and random banks have identical channel budgets.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Target | Examples | Local/global logit | Advantage | Local/global CE | Advantage | Random logit p95 | Local linear advantage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in results.items():
        local = row["metrics"]["local"]; global_row = row["metrics"]["global"]
        lines.append(f"| {name} | {local['examples']} | {local['causal_fidelity']:.3f}/{global_row['causal_fidelity']:.3f} | {row['finite_logit_advantage']:+.3f} | {local['ce_effect_fidelity']:.3f}/{global_row['ce_effect_fidelity']:.3f} | {row['finite_ce_advantage']:+.3f} | {row['random_logit_p95']:.3f} | {row['observational_ce_advantage']:+.3f} |")
    lines += ["", "## Guardrail", "", "The bank keeps each selected channel's native product and deletes all other destination products. This tests sufficiency relative to deleting that position's complete MLP write, not necessity in the intact model. Candidates were selected after the census but frozen before this text panel. Four positive cells cannot establish prevalence across all strata."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
