#!/usr/bin/env python3
"""Find compact, discovery-ranked head programs for semantic construction pulses."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_semantic_construction_head_programs.json")
OUTPUT = TENSOR_ROOT / "bilin18_semantic_construction_head_programs.json"
REPORT = TENSOR_ROOT / "BILIN18_SEMANTIC_CONSTRUCTION_HEAD_PROGRAMS.md"
FIGURE = TENSOR_ROOT / "bilin18_semantic_construction_head_programs.png"


@torch.inference_mode()
def trace_to_site(model, ids, target_layer):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, value_bus = x0, None
    captured = None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), value_bus, model.config, model.tt)
        pre = mixed + details["write"]
        normalized = F.rms_norm(pre, (dimension,))
        mlp = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        x = pre + mlp
        value_bus = details["value_bus"]
        if layer == target_layer:
            captured = {"x0": x0, "mixed": mixed, "attention": details["write"], "mlp": mlp, "output": x, "bus": value_bus, "details": details}
    logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (dimension,))[:, 127]).float() / 30)
    return captured, logits


@torch.inference_mode()
def suffix(model, layer, state, x0, value_bus):
    dimension = model.config.n_embd
    x = state
    for block in model.blocks[layer + 1:]:
        x, value_bus = block(x, value_bus, x0)
    return 30 * torch.tanh(model.lm_head(F.rms_norm(x, (dimension,))[:, 127]).float() / 30)


def head_write_delta(model, layer, details, size):
    heads = model.config.n_head
    head_dim = model.config.n_embd // heads
    destination = 127
    pattern = details["pattern"][:, :, destination]
    values = details["mixed_value"].permute(0, 2, 1, 3)
    attended = torch.einsum("bhs,bhsd->bhd", pattern, values)
    projection = model.blocks[layer].attn.c_proj.weight.T.reshape(heads, head_dim, model.config.n_embd)
    writes = torch.einsum("bhd,hdk->bhk", attended, projection)
    return writes[size:] - writes[:size]


def metrics(target, value):
    target, value = target.double(), value.double()
    energy = target.square().sum().clamp_min(1e-30)
    return {
        "projection_share": float((target * value).sum() / energy),
        "energy_fidelity": float(1 - (target - value).square().sum() / energy),
        "direction_agreement": float((target * value > 0).double().mean()),
        "pearson": float(np.corrcoef(target.numpy(), value.numpy())[0, 1]) if float(value.std()) > 0 else 0.0,
    }


@torch.inference_mode()
def evaluate_cell(model, raw, records, contrast, layer, batch_size, programs):
    targets = []
    effects = {name: [] for name in programs}
    max_state_error = max_logit_error = 0.0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]; size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch])
        changed = original.copy()
        for index, row in enumerate(batch): changed[index, row[3]] = row[5]
        ids = torch.as_tensor(np.concatenate((original, changed)), device="cuda")
        trace, _ = trace_to_site(model, ids, layer)
        direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
        base_state = trace["output"][:size]
        carried = trace["mixed"][size:, 127] - trace["mixed"][:size, 127]
        attention = trace["attention"][size:, 127] - trace["attention"][:size, 127]
        mlp = trace["mlp"][size:, 127] - trace["mlp"][:size, 127]
        head_delta = head_write_delta(model, layer, trace["details"], size)
        state_error = (head_delta.sum(1) - attention).abs()
        max_state_error = max(max_state_error, float(state_error.max()))
        no_attention = base_state.clone()
        no_attention[:, 127] = base_state[:, 127] + carried + mlp
        exact_attention = no_attention.clone(); exact_attention[:, 127] += attention
        states = [no_attention, exact_attention]
        for heads in programs.values():
            state = no_attention.clone()
            if heads: state[:, 127] += head_delta[:, heads].sum(1)
            states.append(state)
        logits = suffix(model, layer, torch.cat(states), torch.cat([trace["x0"][:size]] * len(states)), torch.cat([trace["bus"][:size]] * len(states))).reshape(len(states), size, -1)
        base_score = logits[0] @ contrast
        target = direction * ((logits[1] @ contrast) - base_score)
        targets.append(target.cpu())
        all_heads_index = next((index for index, heads in enumerate(programs.values(), 2) if len(heads) == model.config.n_head), None)
        if all_heads_index is not None: max_logit_error = max(max_logit_error, float((logits[all_heads_index] - logits[1]).abs().max()))
        for index, name in enumerate(programs, 2): effects[name].append((direction * ((logits[index] @ contrast) - base_score)).cpu())
    target = torch.cat(targets)
    return {name: metrics(target, torch.cat(parts)) for name, parts in effects.items()}, {"state_max_abs": max_state_error, "logit_max_abs": max_logit_error, "target_energy": float(target.double().square().mean())}


def cell_data(corpus, base, boundary_flags, counts, cues, config, family_index, fold, split):
    begin, end = base["splits"][split]
    raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
    records = select_records(raw, boundary_flags, cues, counts, 127, config["examples_per_label_per_split"], config["seed"] + 1000 * family_index + 10 * fold + split)
    return raw, records


def main():
    config = json.loads(CONFIG.read_text())
    construction_config = json.loads(Path(config["source_config"]).read_text())
    carrier_config = json.loads(Path(construction_config["source_config"]).read_text())
    base = json.loads(Path(carrier_config["source_config"]).read_text())
    family_source = json.loads(Path(carrier_config["family_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    family_pairs = {"spelling": token_pairs(tokenizer, base["spelling_pairs"])}
    family_pairs.update({name: [tuple(pair) for pair in pairs] for name, pairs in family_source["families"].items()})
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    discovery = {}; maximum_errors = {"state_max_abs": 0.0, "logit_max_abs": 0.0}
    for family_index, family in enumerate(config["sites"]):
        pairs = family_pairs[family]; midpoint = len(pairs) // 2
        folds = ((pairs[:midpoint], pairs[midpoint:]), (pairs[midpoint:], pairs[:midpoint]))
        fold = config["discovery_cell"]["fold"]; split = config["discovery_cell"]["split"]
        targets, cues = folds[fold]
        contrast = make_contrasts(model.lm_head.weight.shape[0], targets, 0, config["seed"] + 100 * family_index + fold, "cuda")[:, 0]
        raw, records = cell_data(corpus, base, boundary_flags, counts, cues, config, family_index, fold, split)
        programs = {f"head_{head}": [head] for head in range(model.config.n_head)}
        programs["all"] = list(range(model.config.n_head))
        result, error = evaluate_cell(model, raw, records, contrast, config["sites"][family], config["batch_size"], programs)
        order = sorted(range(model.config.n_head), key=lambda head: abs(result[f"head_{head}"]["projection_share"]), reverse=True)
        discovery[family] = {"layer": config["sites"][family], "head_order": order, "singletons": result}
        for name in maximum_errors: maximum_errors[name] = max(maximum_errors[name], error[name])
        print(f"head discovery {family} L{config['sites'][family]} order={order}", flush=True)
    results = []
    for family_index, family in enumerate(config["sites"]):
        pairs = family_pairs[family]; midpoint = len(pairs) // 2
        folds = ((pairs[:midpoint], pairs[midpoint:]), (pairs[midpoint:], pairs[:midpoint]))
        order = discovery[family]["head_order"]
        programs = {f"top_{k}": order[:k] for k in range(1, model.config.n_head + 1)}
        for fold, (targets, cues) in enumerate(folds):
            contrast = make_contrasts(model.lm_head.weight.shape[0], targets, 0, config["seed"] + 100 * family_index + fold, "cuda")[:, 0]
            for split in range(len(base["splits"])):
                raw, records = cell_data(corpus, base, boundary_flags, counts, cues, config, family_index, fold, split)
                metrics_by_program, error = evaluate_cell(model, raw, records, contrast, config["sites"][family], config["batch_size"], programs)
                for name in maximum_errors: maximum_errors[name] = max(maximum_errors[name], error[name])
                k90 = next((k for k in range(1, model.config.n_head + 1) if metrics_by_program[f"top_{k}"]["energy_fidelity"] >= .90), None)
                results.append({"family": family, "layer": config["sites"][family], "fold": fold, "split": split, "is_discovery": fold == config["discovery_cell"]["fold"] and split == config["discovery_cell"]["split"], "k90": k90, "programs": metrics_by_program})
                print(f"head eval {family} fold={fold} split={split} k90={k90}", flush=True)
    summary = {}
    for family in config["sites"]:
        rows = [row for row in results if row["family"] == family and not row["is_discovery"]]
        summary[family] = {
            "layer": config["sites"][family],
            "head_order": discovery[family]["head_order"],
            "mean_fidelity": {str(k): float(np.mean([row["programs"][f"top_{k}"]["energy_fidelity"] for row in rows])) for k in range(1, model.config.n_head + 1)},
            "mean_projection": {str(k): float(np.mean([row["programs"][f"top_{k}"]["projection_share"] for row in rows])) for k in range(1, model.config.n_head + 1)},
            "heldout_k90": [row["k90"] for row in rows],
            "median_heldout_k90": float(np.median([row["k90"] for row in rows if row["k90"] is not None])) if any(row["k90"] is not None for row in rows) else None,
        }
    top3_success = {family: row["mean_fidelity"]["3"] >= .80 for family, row in summary.items()}
    compact = {family: row["median_heldout_k90"] is not None and row["median_heldout_k90"] <= 4 for family, row in summary.items()}
    frozen_program_transfer = {}
    for family, row in summary.items():
        discovery_k90 = next((k for k in range(1, model.config.n_head + 1) if results[next(i for i,r in enumerate(results) if r["family"] == family and r["is_discovery"])]["programs"][f"top_{k}"]["energy_fidelity"] >= .90), model.config.n_head)
        row["discovery_k90"] = discovery_k90
        frozen_program_transfer[family] = row["mean_fidelity"][str(discovery_k90)]
    top_heads = {family: row["head_order"][0] for family, row in summary.items()}
    hypotheses = {
        "H0_head_sum_is_exact": {"supported": maximum_errors["logit_max_abs"] <= 1e-4, "observed": maximum_errors},
        "H1_top3_is_often_sufficient": {"supported": sum(top3_success.values()) >= 3, "observed": {family: row["mean_fidelity"]["3"] for family, row in summary.items()}},
        "H2_attention_programs_are_compact": {"supported": sum(compact.values()) >= 3, "observed": {family: row["median_heldout_k90"] for family, row in summary.items()}},
        "H3_programs_transfer_across_lexical_folds": {"supported": all(value >= .65 for value in frozen_program_transfer.values()), "observed": frozen_program_transfer},
        "H4_no_single_universal_construction_head": {"supported": len(set(top_heads.values())) >= 3, "observed": top_heads},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "maximum_errors": maximum_errors, "discovery": discovery, "summary": summary, "results": results}, indent=2, allow_nan=False))
    lines = ["# Compact head programs for semantic construction pulses", "", "Four attention pulses were frozen from the earlier destination-construction experiment. Heads are ranked by absolute finite singleton projection on fold-0/split-0 only. Cumulative programs are then evaluated on the other split and reversed lexical fold through the complete nonlinear suffix.", "", "## Locked decisions", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{row['observed']}`" for name, row in hypotheses.items())
    lines += ["", "![Head programs](bilin18_semantic_construction_head_programs.png)", "", "| Family/site | Discovery head order | Discovery k90 | Held-out k90 | Held-out top-3 fidelity | Frozen-program fidelity |", "|---|---|---:|---:|---:|---:|"]
    for family, row in summary.items():
        lines.append(f"| {family} L{row['layer']} | {', '.join('H'+str(h) for h in row['head_order'])} | {row['discovery_k90']} | {row['median_heldout_k90'] if row['median_heldout_k90'] is not None else '—'} | {row['mean_fidelity']['3']:.3f} | {frozen_program_transfer[family]:.3f} |")
    lines += ["", "## Guardrails", "", "The program reconstructs the finite attention branch at one selected semantic site; it is not a claim that each selected head has one semantic meaning. Ranking uses the declared output contrast on one discovery cell, while the lexical-fold and corpus-split cells are held out. Absolute singleton ranking includes both supportive and suppressive heads, and k90 is reconstruction fidelity rather than a count of independently meaningful modules."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, (family, row) in zip(axes.flat, summary.items()):
        ks = list(range(1, model.config.n_head + 1)); axis.plot(ks, [row["mean_fidelity"][str(k)] for k in ks], marker="o")
        axis.axhline(.9, color="black", linestyle="--", linewidth=.8); axis.axhline(.8, color="gray", linestyle=":", linewidth=.8)
        axis.set(title=f"{family} L{row['layer']}", xlabel="Discovery-ranked heads retained", ylabel="Held-out energy fidelity", xticks=ks); axis.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summary": summary}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
