#!/usr/bin/env python3
"""Test whether compact semantic head programs are semantic or generically dominant."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_semantic_construction_head_programs import head_write_delta, suffix, trace_to_site
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_semantic_head_program_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_semantic_head_program_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_SEMANTIC_HEAD_PROGRAM_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin18_semantic_head_program_redteam.png"


def fidelity(target, value):
    target, value = target.double(), value.double()
    energy = target.square().sum().clamp_min(1e-30)
    return float(1 - (target - value).square().sum() / energy)


def projection(target, value):
    target, value = target.double(), value.double()
    return float((target * value).sum() / target.square().sum().clamp_min(1e-30))


def random_contrast_bank(vocab_size, support, count, seed, device):
    rng = np.random.default_rng(seed)
    bank = []
    for _ in range(count):
        indices = rng.choice(vocab_size, size=2 * support, replace=False)
        value = torch.zeros(vocab_size, dtype=torch.float32, device=device)
        value[torch.as_tensor(indices[:support], device=device)] = 1 / support
        value[torch.as_tensor(indices[support:], device=device)] = -1 / support
        bank.append(value)
    return torch.stack(bank)


@torch.inference_mode()
def evaluate(model, raw, records, contrast, null_bank, layer, program, batch_size):
    true_targets, true_program = [], []
    head_effects = [[] for _ in range(model.config.n_head)]
    null_targets = [[] for _ in range(len(null_bank))]
    null_programs = [[] for _ in range(len(null_bank))]
    state_targets, state_programs = [], []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]; size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch]); changed = original.copy()
        for index, row in enumerate(batch): changed[index, row[3]] = row[5]
        ids = torch.as_tensor(np.concatenate((original, changed)), device="cuda")
        trace, _ = trace_to_site(model, ids, layer)
        direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
        base_state = trace["output"][:size]
        carried = trace["mixed"][size:, 127] - trace["mixed"][:size, 127]
        attention = trace["attention"][size:, 127] - trace["attention"][:size, 127]
        mlp = trace["mlp"][size:, 127] - trace["mlp"][:size, 127]
        heads = head_write_delta(model, layer, trace["details"], size)
        no_attention = base_state.clone(); no_attention[:, 127] = base_state[:, 127] + carried + mlp
        states = [no_attention]
        exact = no_attention.clone(); exact[:, 127] += attention; states.append(exact)
        selected = no_attention.clone(); selected[:, 127] += heads[:, program].sum(1); states.append(selected)
        for head in range(model.config.n_head):
            state = no_attention.clone(); state[:, 127] += heads[:, head]; states.append(state)
        logits = suffix(model, layer, torch.cat(states), torch.cat([trace["x0"][:size]] * len(states)), torch.cat([trace["bus"][:size]] * len(states))).reshape(len(states), size, -1)
        base_logits = logits[0]
        true_targets.append((direction * ((logits[1] @ contrast) - (base_logits @ contrast))).cpu())
        true_program.append((direction * ((logits[2] @ contrast) - (base_logits @ contrast))).cpu())
        for head in range(model.config.n_head):
            head_effects[head].append((direction * ((logits[3 + head] @ contrast) - (base_logits @ contrast))).cpu())
        target_null = (logits[1] - base_logits) @ null_bank.T
        program_null = (logits[2] - base_logits) @ null_bank.T
        for index in range(len(null_bank)):
            null_targets[index].append((direction * target_null[:, index]).cpu())
            null_programs[index].append((direction * program_null[:, index]).cpu())
        state_targets.append(attention.cpu()); state_programs.append(heads[:, program].sum(1).cpu())
    target = torch.cat(true_targets); program_value = torch.cat(true_program)
    state_target = torch.cat(state_targets).flatten(); state_value = torch.cat(state_programs).flatten()
    return {
        "true": {"fidelity": fidelity(target, program_value), "projection": projection(target, program_value)},
        "null_fidelities": [fidelity(torch.cat(t), torch.cat(v)) for t, v in zip(null_targets, null_programs)],
        "head_fidelities": [fidelity(target, torch.cat(parts)) for parts in head_effects],
        "state": {"fidelity": fidelity(state_target, state_value), "projection": projection(state_target, state_value)},
    }


def main():
    config = json.loads(CONFIG.read_text()); source_config = json.loads(Path(config["source_config"]).read_text()); source = json.loads(Path(config["source_results"]).read_text())
    construction_config = json.loads(Path(source_config["source_config"]).read_text()); carrier_config = json.loads(Path(construction_config["source_config"]).read_text()); base = json.loads(Path(carrier_config["source_config"]).read_text()); family_source = json.loads(Path(carrier_config["family_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); corpus = np.load(base["corpus"], mmap_mode="r"); counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    family_pairs = {"spelling": token_pairs(tokenizer, base["spelling_pairs"])}; family_pairs.update({name: [tuple(pair) for pair in pairs] for name, pairs in family_source["families"].items()})
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    results = []
    for family_index, family in enumerate(source_config["sites"]):
        pairs = family_pairs[family]; midpoint = len(pairs) // 2; folds = ((pairs[:midpoint], pairs[midpoint:]), (pairs[midpoint:], pairs[:midpoint]))
        program = source["summary"][family]["head_order"][:source["summary"][family]["discovery_k90"]]
        for fold, (targets, cues) in enumerate(folds):
            contrast = make_contrasts(model.lm_head.weight.shape[0], targets, 0, source_config["seed"] + 100 * family_index + fold, "cuda")[:, 0]
            null_bank = random_contrast_bank(model.lm_head.weight.shape[0], len(targets), config["random_contrasts"], config["seed"] + 100 * family_index + fold, "cuda")
            for split, (begin, end) in enumerate(base["splits"]):
                raw = np.asarray(corpus[begin:end, :128], dtype=np.int64); records = select_records(raw, boundary_flags, cues, counts, 127, source_config["examples_per_label_per_split"], source_config["seed"] + 1000 * family_index + 10 * fold + split)
                result = evaluate(model, raw, records, contrast, null_bank, source_config["sites"][family], program, source_config["batch_size"])
                result.update({"family": family, "layer": source_config["sites"][family], "fold": fold, "split": split, "is_discovery": fold == 0 and split == 0, "program": program})
                results.append(result); print(f"specificity {family} fold={fold} split={split}", flush=True)
    summary = {}
    for family in source_config["sites"]:
        rows = [row for row in results if row["family"] == family and not row["is_discovery"]]
        nulls = np.asarray([row["null_fidelities"] for row in rows])
        head_matrix = np.asarray([row["head_fidelities"] for row in rows])
        summary[family] = {
            "layer": source_config["sites"][family], "program": rows[0]["program"],
            "true_fidelity": float(np.mean([row["true"]["fidelity"] for row in rows])),
            "state_fidelity": float(np.mean([row["state"]["fidelity"] for row in rows])),
            "null_median": float(np.median(nulls.mean(0))), "null_p95": float(np.quantile(nulls.mean(0), .95)),
            "semantic_advantage_over_p95": float(np.mean([row["true"]["fidelity"] for row in rows]) - np.quantile(nulls.mean(0), .95)),
            "mean_head_fidelities": head_matrix.mean(0).tolist(), "best_heldout_head": int(np.argmax(head_matrix.mean(0))),
        }
    layer9 = {family: summary[family]["mean_head_fidelities"] for family in ("spelling", "sentiment")}
    layer9_margins = {"spelling_H8_minus_H7": layer9["spelling"][8] - layer9["spelling"][7], "sentiment_H7_minus_H8": layer9["sentiment"][7] - layer9["sentiment"][8]}
    hypotheses = {
        "R0_program_is_state_dominant": {"supported": all(row["state_fidelity"] >= .80 for row in summary.values()), "observed": {family: row["state_fidelity"] for family, row in summary.items()}},
        "R1_semantic_contrast_is_exceptional": {"supported": sum(row["semantic_advantage_over_p95"] >= .05 for row in summary.values()) >= 2, "observed": {family: row["semantic_advantage_over_p95"] for family, row in summary.items()}},
        "R2_discovery_head_remains_best_heldout": {"supported": all(row["best_heldout_head"] == source["summary"][family]["head_order"][0] for family, row in summary.items()), "observed": {family: {"discovery": source["summary"][family]["head_order"][0], "heldout": row["best_heldout_head"]} for family, row in summary.items()}},
        "R3_layer9_heads_are_family_specific": {"supported": all(value >= .10 for value in layer9_margins.values()), "observed": layer9_margins},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summary": summary, "results": results}, indent=2, allow_nan=False))
    lines = ["# Specificity red team for compact semantic head programs", "", "Frozen construction-head programs are compared with their reconstruction of the complete attention-state change and with 32 random vocabulary contrasts matched for support size and coefficient norm. Every score below uses the three held-out fold/split cells.", "", "## Locked decisions", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{row['observed']}`" for name, row in hypotheses.items())
    lines += ["", "![Specificity red team](bilin18_semantic_head_program_redteam.png)", "", "| Family/site | Program | True output fidelity | State fidelity | Null median | Null p95 | True − null p95 |", "|---|---|---:|---:|---:|---:|---:|"]
    for family, row in summary.items(): lines.append(f"| {family} L{row['layer']} | {', '.join('H'+str(h) for h in row['program'])} | {row['true_fidelity']:.3f} | {row['state_fidelity']:.3f} | {row['null_median']:.3f} | {row['null_p95']:.3f} | {row['semantic_advantage_over_p95']:+.3f} |")
    lines += ["", "## Interpretation boundary", "", "High state fidelity means the program dominates the entire attention-write change, so semantic output fidelity is not evidence that the head is itself semantically specialized. Conversely, low state fidelity with exceptional true-contrast fidelity would indicate output-selective routing. The random contrasts are matched sparse vocabulary objectives, not natural alternative tasks; they diagnose generic output dominance but do not exhaust possible semantic controls."]
    REPORT.write_text("\n".join(lines) + "\n")
    families = list(summary); x = np.arange(len(families)); width = .25
    fig, axis = plt.subplots(figsize=(10, 5), layout="constrained"); axis.bar(x-width, [summary[f]["true_fidelity"] for f in families], width, label="true output"); axis.bar(x, [summary[f]["state_fidelity"] for f in families], width, label="state"); axis.bar(x+width, [summary[f]["null_p95"] for f in families], width, label="random contrast p95"); axis.set(xticks=x, xticklabels=[f"{f} L{summary[f]['layer']}" for f in families], ylabel="Energy fidelity", title="Is the compact program semantic or generically dominant?"); axis.axhline(.9, color="black", linestyle="--", linewidth=.8); axis.legend(); axis.grid(axis="y", alpha=.2); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summary": summary}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
