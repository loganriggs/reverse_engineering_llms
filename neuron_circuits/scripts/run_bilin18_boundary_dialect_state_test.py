#!/usr/bin/env python3
"""Test whether validated L9H8 boundary reads transport a dialect/geography state."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import auc
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_variable_function_atlas import leading_mode


CONFIG = Path("configs/bilin18_boundary_dialect_state_test.json")
OUTPUT = TENSOR_ROOT / "bilin18_boundary_dialect_state_test.json"
REPORT = TENSOR_ROOT / "BILIN18_BOUNDARY_DIALECT_STATE_TEST.md"
FIGURE = TENSOR_ROOT / "bilin18_boundary_dialect_state_test.png"


def matched_rows(raw, boundary_lookup, us_ids, uk_ids, destination, maximum, seed):
    records = []
    for row_index, tokens in enumerate(raw[:, :destination + 1]):
        boundaries = np.flatnonzero(boundary_lookup[tokens])
        if not len(boundaries): continue
        source = int(boundaries[-1]); prefix = set(map(int, tokens[:source]))
        us, uk = bool(prefix & us_ids), bool(prefix & uk_ids)
        if us == uk: continue
        token_kind = int(tokens[source])
        distance = destination - source
        distance_bin = min(int(np.floor(np.log2(max(distance, 1)))), 7)
        records.append((row_index, source, int(us), token_kind, distance_bin))
    groups = {}
    for record in records: groups.setdefault((record[3], record[4]), {0: [], 1: []})[record[2]].append(record)
    rng = np.random.default_rng(seed); selected = []
    for values in groups.values():
        count = min(len(values[0]), len(values[1]))
        if count:
            selected.extend(rng.choice(values[0], count, replace=False).tolist())
            selected.extend(rng.choice(values[1], count, replace=False).tolist())
    rng.shuffle(selected)
    by_label = {label: [row for row in selected if row[2] == label] for label in (0, 1)}
    count = min(maximum, len(by_label[0]), len(by_label[1]))
    selected = by_label[0][:count] + by_label[1][:count]; rng.shuffle(selected)
    return selected


@torch.inference_mode()
def collect(model, raw, records, layer, head, output_mode, contrast, config):
    dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    writes, baseline_scores, delta_scores, labels, address_scores = [], [], [], [], []
    for start in range(0, len(records), config["batch_size"]):
        batch = records[start:start + config["batch_size"]]
        ids = torch.as_tensor(np.stack([raw[row[0], :config["sequence_length"]] for row in batch]), device="cuda")
        source = torch.as_tensor([row[1] for row in batch], device="cuda"); destination = torch.full_like(source, config["destination"])
        rows = torch.arange(len(ids), device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for block_index, block in enumerate(model.blocks):
            if block_index < layer:
                x, bus = block(x, bus, x0); continue
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)[head]
            selected_value = details["pattern"][rows, head, destination, source, None].float() * details["mixed_value"][rows, source, head].float()
            write = selected_value @ projection
            modified = details["write"].float().clone(); modified[rows, destination] -= write
            attention = torch.cat([details["write"].float(), modified], 0)
            x0 = torch.cat([x0.float(), x0.float()], 0); mixed = torch.cat([mixed.float(), mixed.float()], 0)
            bus = torch.cat([details["value_bus"].float(), details["value_bus"].float()], 0)
            state = F.rms_norm(mixed + attention, (dimension,)); x = mixed + attention + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
            for suffix in model.blocks[block_index + 1:]: x, bus = suffix(x, bus, x0)
            rows2 = torch.arange(2 * len(ids), device="cuda"); destination2 = torch.cat([destination, destination])
            logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (dimension,))[rows2, destination2]).float() / 30)).reshape(2, len(ids), -1)
            writes.append(write.cpu())
            baseline_scores.extend((logits[0] @ contrast).cpu().tolist())
            delta_scores.extend(((logits[0] - logits[1]) @ contrast).cpu().tolist())
            labels.extend([row[2] for row in batch])
            address_scores.extend([row[4] + (row[3] % 2) * .01 for row in batch])
            break
    writes = torch.cat(writes)
    return {"writes": writes, "leading_scores": (writes @ output_mode).numpy(), "baseline_scores": np.asarray(baseline_scores), "delta_scores": np.asarray(delta_scores), "labels": np.asarray(labels, dtype=bool), "address_scores": np.asarray(address_scores)}


def orient_auc(labels, discovery_scores, evaluation_labels, evaluation_scores):
    direction = 1 if auc(labels, discovery_scores) >= .5 else -1
    return float(auc(evaluation_labels, direction * evaluation_scores)), direction


def fit_direction(writes, labels):
    normalized = F.normalize(writes.float(), dim=1)
    return F.normalize(normalized[labels].mean(0) - normalized[~labels].mean(0), dim=0)


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(config["corpus"], mmap_mode="r")
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    us_ids = {tokenizer.encode(pair[0], add_special_tokens=False)[0] for pair in config["spelling_pairs"]}
    uk_ids = {tokenizer.encode(pair[1], add_special_tokens=False)[0] for pair in config["spelling_pairs"]}
    contrast = torch.zeros(tokenizer.vocab_size, device="cuda")
    contrast[list(us_ids)] = 1 / len(us_ids); contrast[list(uk_ids)] = -1 / len(uk_ids)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    # The checkpoint pads its output vocabulary beyond the tokenizer's real IDs.
    if contrast.numel() != model.lm_head.weight.shape[0]:
        resized = torch.zeros(model.lm_head.weight.shape[0], device="cuda"); resized[:min(len(resized), len(contrast))] = contrast[:min(len(resized), len(contrast))]; contrast = resized
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    raw_splits, records = {}, {}
    for split, (start, end) in enumerate(config["splits"]):
        raw = np.asarray(corpus[start:end, :config["sequence_length"]], dtype=np.int64); raw_splits[str(split)] = raw
        records[str(split)] = matched_rows(raw, boundary, us_ids, uk_ids, config["destination"], config["maximum_per_label_per_split"], config["seed"] + split)
    collected, results = {}, {}
    for program in config["heads"]:
        block = model.blocks[program["layer"]].attn; heads = model.config.n_head; head_dim = model.config.n_embd // heads; sl = slice(program["head"] * head_dim, (program["head"] + 1) * head_dim)
        output_mode, _, singular = leading_mode(block.c_proj.weight[:, sl].detach().float().cpu(), block.c_v.weight[sl].detach().float().cpu())
        collected[program["name"]] = {}
        for split in ("0", "1"):
            print("dialect", program["name"], split, flush=True)
            collected[program["name"]][split] = collect(model, raw_splits[split], records[split], program["layer"], program["head"], output_mode, contrast, config)
        discovery, evaluation = collected[program["name"]]["0"], collected[program["name"]]["1"]
        leading_auc, orientation = orient_auc(discovery["labels"], discovery["leading_scores"], evaluation["labels"], evaluation["leading_scores"])
        direction = fit_direction(discovery["writes"], discovery["labels"])
        full_auc = float(auc(evaluation["labels"], (F.normalize(evaluation["writes"].float(), dim=1) @ direction).numpy()))
        delta_auc, delta_orientation = orient_auc(discovery["labels"], discovery["delta_scores"], evaluation["labels"], evaluation["delta_scores"])
        results[program["name"]] = {"role": program["role"], "examples_per_split": len(records["0"]), "leading_mode_energy": float(singular[0].square() / singular.square().sum()), "leading_mode_auc": leading_auc, "leading_orientation": orientation, "full_write_auc": full_auc, "delta_style_auc": delta_auc, "delta_orientation": delta_orientation,
            "native_style_auc": float(auc(evaluation["labels"], evaluation["baseline_scores"])), "address_auc": float(auc(evaluation["labels"], evaluation["address_scores"])),
            "mean_aligned_delta_discovery": float(np.mean((2 * discovery["labels"] - 1) * discovery["delta_scores"])), "mean_aligned_delta_evaluation": float(np.mean((2 * evaluation["labels"] - 1) * evaluation["delta_scores"]))}

    predicted = results["boundary_L9H8"]; controls = [results["same_layer_L9H7"], results["L8H1"]]
    d0 = predicted["native_style_auc"] >= .65
    d1 = predicted["leading_mode_auc"] >= .65 and all(predicted["leading_mode_auc"] >= row["leading_mode_auc"] + .05 for row in controls)
    d2 = predicted["full_write_auc"] >= .75 and all(predicted["full_write_auc"] >= row["full_write_auc"] + .05 for row in controls)
    d3 = predicted["delta_style_auc"] >= .60 and all(predicted["delta_style_auc"] >= row["delta_style_auc"] + .05 for row in controls)
    d4 = predicted["address_auc"] <= .55
    hypotheses = {"D0_model_exposes_dialect_state": {"supported": d0}, "D1_boundary_leading_mode_encodes_dialect": {"supported": d1}, "D2_boundary_full_write_encodes_dialect": {"supported": d2}, "D3_boundary_edge_transmits_dialect_to_logits": {"supported": d3}, "D4_address_controls_are_balanced": {"supported": d4}, "D5_dialect_interpretation_is_supported": {"supported": d0 and d2 and d3 and d4}}
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    examples = {}
    for split in ("0", "1"):
        rows = records[split]; raw = raw_splits[split]; examples[split] = []
        for row in rows[:8]: examples[split].append({"label": "US" if row[2] else "UK", "source": row[1], "context": tokenizer.decode(raw[row[0], max(0, row[1] - 24):128].tolist()).replace("\n", "\\n")})
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results, "examples": examples}, indent=2, allow_nan=False))
    lines = ["# Does L9H8 transport dialect state across boundaries?", "", "The weight-only leading mode suggested a US-geography versus British-spelling axis after L9H8 had independently been validated as a boundary reader. Prefixes containing only US or only UK spelling markers are matched on boundary token identity and log-distance. The first 10,000 corpus rows orient/fits directions; the second 10,000 rows are held out.", "", "## Locked results", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Head | Native style AUC | leading weight mode | fitted write direction | finite edge style contribution | address control |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in results.items(): lines.append(f"| {name} | {row['native_style_auc']:.3f} | {row['leading_mode_auc']:.3f} | {row['full_write_auc']:.3f} | {row['delta_style_auc']:.3f} | {row['address_auc']:.3f} |")
    lines += ["", "## Guardrails", "", "The spelling vocabulary was selected after inspecting the L9H8 singular mode, so only the second corpus half tests generalization; this is not an unbiased search over possible semantic axes. Prefix labels are weak supervision: topics, source domains, and geography correlate with spelling. A fitted write probe shows encoded information but not a unique native coordinate. The finite US-minus-UK logit contrast is causal for that declared output axis, not proof that dialect is the head's only function."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axis = plt.subplots(figsize=(8, 4.5), layout="constrained"); names = list(results); x = np.arange(len(names)); metrics = [("leading_mode_auc", "leading mode"), ("full_write_auc", "full write"), ("delta_style_auc", "finite delta"), ("address_auc", "address control")]
    for index, (metric, label) in enumerate(metrics): axis.bar(x + (index - 1.5) * .18, [results[name][metric] for name in names], width=.17, label=label)
    axis.axhline(.5, color="black", linewidth=.8); axis.set(xticks=x, xticklabels=names, ylabel="Held-out dialect AUROC", title="Boundary payload dialect test"); axis.legend(); axis.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
