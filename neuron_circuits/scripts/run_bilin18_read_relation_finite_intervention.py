#!/usr/bin/env python3
"""Finite edge-removal validation of prospectively matched read relations."""

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
from run_bilin18_contextual_read_function_discovery import edge_predicates, edge_template, frequency_quantiles
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_prospective_read_relation_test import candidate_masks, matched_samples, strata
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_vocabulary import token_categories


CONFIG = Path("configs/bilin18_read_relation_finite_intervention.json")
OUTPUT = TENSOR_ROOT / "bilin18_read_relation_finite_intervention.json"
REPORT = TENSOR_ROOT / "BILIN18_READ_RELATION_FINITE_INTERVENTION.md"
FIGURE = TENSOR_ROOT / "bilin18_read_relation_finite_intervention.png"


def kl(reference, prediction):
    logp = F.log_softmax(reference.float(), -1); logq = F.log_softmax(prediction.float(), -1)
    return (logp.exp() * (logp - logq)).sum(-1).clamp_min(0)


@torch.inference_mode()
def native_logits(model, ids, destinations):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
    for block in model.blocks: x, bus = block(x, bus, x0)
    rows = torch.arange(len(ids), device=ids.device)
    raw = model.lm_head(F.rms_norm(x, (dimension,))[rows, destinations]).float()
    return 30 * torch.tanh(raw / 30)


@torch.inference_mode()
def edge_removal_effects(model, ids, sources, destinations, layer, head, probes, targets):
    dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
    rows = torch.arange(len(ids), device=ids.device)
    for block_index, block in enumerate(model.blocks):
        if block_index < layer:
            x, bus = block(x, bus, x0); continue
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
        projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)[head]
        selected_value = details["pattern"][rows, head, destinations, sources, None].float() * details["mixed_value"][rows, sources, head].float()
        selected_write = selected_value @ projection
        modified_attention = details["write"].float().clone()
        modified_attention[rows, destinations] -= selected_write
        attention = torch.cat([details["write"].float(), modified_attention], 0)
        stacked_x0 = torch.cat([x0.float(), x0.float()], 0)
        stacked_mixed = torch.cat([mixed.float(), mixed.float()], 0)
        stacked_bus = torch.cat([details["value_bus"].float(), details["value_bus"].float()], 0)
        pre = stacked_mixed + attention
        state = F.rms_norm(pre, (dimension,))
        destinations2 = torch.cat([destinations, destinations])
        rows2 = torch.arange(2 * len(ids), device=ids.device)
        local = (state[rows2, destinations2] @ probes.T).reshape(2, len(ids), -1)
        x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        bus = stacked_bus; x0 = stacked_x0
        for suffix in model.blocks[block_index + 1:]: x, bus = suffix(x, bus, x0)
        raw = model.lm_head(F.rms_norm(x, (dimension,))[rows2, destinations2]).float()
        logits = (30 * torch.tanh(raw / 30)).reshape(2, len(ids), -1)
        reference = native_logits(model, ids, destinations)
        error = float((logits[0] - reference).abs().max())
        local_effect = (local[0] - local[1]).square().sum(-1)
        downstream_kl = kl(logits[0], logits[1])
        ce_native = -F.log_softmax(logits[0], -1)[rows, targets]
        ce_removed = -F.log_softmax(logits[1], -1)[rows, targets]
        margin_native = logits[0, rows, targets] - logits[0].masked_fill(F.one_hot(targets, logits.shape[-1]).bool(), -torch.inf).max(-1).values
        margin_removed = logits[1, rows, targets] - logits[1].masked_fill(F.one_hot(targets, logits.shape[-1]).bool(), -torch.inf).max(-1).values
        return {
            "local": local_effect.cpu().numpy(), "kl": downstream_kl.cpu().numpy(),
            "ce_delta": (ce_removed - ce_native).cpu().numpy(),
            "margin_delta": (margin_removed - margin_native).cpu().numpy(), "native_error": error,
        }
    raise AssertionError("layer not reached")


def edge_records(flat_indices, positions):
    source_template, destination_template = edge_template(positions); edges = len(source_template)
    flat_indices = np.asarray(flat_indices)
    rows, local = divmod(flat_indices, edges)
    return rows, source_template[local], destination_template[local]


@torch.inference_mode()
def evaluate_indices(model, raw, flat_indices, pair, probes, config):
    row_indices, sources, destinations = edge_records(flat_indices, config["positions"])
    values = {name: [] for name in ("local", "kl", "ce_delta", "margin_delta")}; maximum_error = 0.0
    for start in range(0, len(row_indices), config["batch_size"]):
        selected_rows = row_indices[start:start + config["batch_size"]]
        ids = torch.as_tensor(raw[selected_rows, :config["sequence_length"]], device="cuda")
        source = torch.as_tensor(sources[start:start + config["batch_size"]], device="cuda")
        destination = torch.as_tensor(destinations[start:start + config["batch_size"]], device="cuda")
        target = torch.as_tensor(raw[selected_rows, destinations[start:start + config["batch_size"]] + 1], device="cuda")
        result = edge_removal_effects(model, ids, source, destination, pair["layer"], pair["head"], probes, target)
        maximum_error = max(maximum_error, result.pop("native_error"))
        for name, array in result.items(): values[name].extend(array.tolist())
    return {name: np.asarray(array) for name, array in values.items()}, maximum_error


def compare(candidate, control):
    return {
        "local_auc": float(auc(np.r_[np.ones(len(candidate["local"]), dtype=bool), np.zeros(len(control["local"]), dtype=bool)], np.r_[candidate["local"], control["local"]])),
        "kl_auc": float(auc(np.r_[np.ones(len(candidate["kl"]), dtype=bool), np.zeros(len(control["kl"]), dtype=bool)], np.r_[candidate["kl"], control["kl"]])),
        "local_sum_ratio": float(candidate["local"].sum() / max(control["local"].sum(), 1e-30)),
        "kl_sum_ratio": float(candidate["kl"].sum() / max(control["kl"].sum(), 1e-30)),
        "paired_local_win": float(np.mean(candidate["local"] > control["local"])),
        "paired_kl_win": float(np.mean(candidate["kl"] > control["kl"])),
        "mean_ce_delta_candidate": float(candidate["ce_delta"].mean()),
        "mean_ce_delta_control": float(control["ce_delta"].mean()),
        "mean_abs_margin_delta_candidate": float(np.abs(candidate["margin_delta"]).mean()),
        "mean_abs_margin_delta_control": float(np.abs(control["margin_delta"]).mean()),
    }


def passes(records, candidate, layer, head, local=.60, kl_value=.55):
    rows = [row for row in records if row["candidate"] == candidate and row["layer"] == layer and row["head"] == head]
    return len(rows) == 2 and all(row["local_auc"] >= local and row["kl_auc"] >= kl_value for row in rows)


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(config["corpus"], mmap_mode="r")
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    human = token_categories(flags); frequency_bins, _ = frequency_quantiles(corpus, tokenizer.vocab_size)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    probe_map = {(pair["layer"], pair["head"]): operator_probe_vectors(model.blocks[pair["layer"]], config["operator_probes"], config["seed"] + pair["layer"] * 17 + pair["head"])[0] for pair in config["pairs"]}
    records = []; maximum_error = 0.0
    for split_index, (start, end) in enumerate(config["splits"]):
        raw = np.asarray(corpus[start:end, :config["sequence_length"] + 1], dtype=np.int64)
        predicates, names, _, _ = edge_predicates(raw[:, :config["sequence_length"]], flags, human, frequency_bins, config["positions"])
        candidates = candidate_masks(predicates, names); group = strata(raw[:, :config["sequence_length"]], frequency_bins, config["positions"])
        prepared = {}
        for candidate, mask in candidates.items():
            match = matched_samples(mask, group, config["seed"] + split_index * 100 + len(prepared), 1, 100)
            if match is None: continue
            positive, negative = match["samples"][0]
            generator = np.random.default_rng(config["seed"] + split_index * 1000 + len(prepared))
            count = min(config["matched_edges_per_split"], len(positive), len(negative))
            chosen = generator.choice(len(positive), count, replace=False)
            prepared[candidate] = (positive[chosen], negative[chosen])
        for pair in config["pairs"]:
            print("finite", split_index, pair, flush=True)
            positive, negative = prepared[pair["candidate"]]
            probes = probe_map[(pair["layer"], pair["head"])]
            candidate_values, error_a = evaluate_indices(model, raw, positive, pair, probes, config)
            control_values, error_b = evaluate_indices(model, raw, negative, pair, probes, config)
            maximum_error = max(maximum_error, error_a, error_b)
            records.append({"split": split_index, **pair, "edges_per_class": len(positive), **compare(candidate_values, control_values)})

    boundary = passes(records, "boundary_source", 9, 8)
    function_78 = passes(records, "adjacent_function_word", 7, 8)
    function_97 = passes(records, "adjacent_function_word", 9, 7)
    def specificity(candidate, predicted, control):
        for split in range(2):
            a = next(row for row in records if row["split"] == split and row["candidate"] == candidate and (row["layer"], row["head"]) == predicted)
            b = next(row for row in records if row["split"] == split and row["candidate"] == candidate and (row["layer"], row["head"]) == control)
            if a["local_auc"] < b["local_auc"] + .05: return False
        return True
    name_negative = not passes(records, "adjacent_name_piece", 8, 1)
    hypotheses = {
        "F0_native_branch_is_exact": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
        "F1_boundary_has_finite_local_effect": {"supported": boundary},
        "F2_boundary_reaches_output": {"supported": boundary},
        "F3_L7H8_function_read_is_finite": {"supported": function_78},
        "F4_L9H7_function_read_is_finite": {"supported": function_97},
        "F5_relations_are_head_specific": {"supported": specificity("boundary_source", (9, 8), (8, 1)) and specificity("adjacent_function_word", (7, 8), (8, 1))},
        "F6_failed_name_relation_stays_negative": {"supported": name_negative},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "records": records}, indent=2, allow_nan=False))
    lines = ["# Finite validation of candidate read relations", "", "Each candidate edge is paired with a non-candidate edge matched on destination position, log-distance bin, source frequency, and destination frequency. The native attention edge is then removed exactly, and both the immediate normalized operator sketch and all downstream layers are recomputed. Results use two corpus splits and 256 candidate/control edges per pair per split.", "", f"Maximum unmodified-branch logit error: `{maximum_error:.2e}`.", "", "## Locked results", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Split | Relation / head | Role | Local AUC / ratio | KL AUC / ratio | CE delta candidate/control |", "|---:|---|---|---:|---:|---:|"]
    for row in records:
        lines.append(f"| {row['split']} | {row['candidate']} / L{row['layer']}H{row['head']} | {row['role']} | {row['local_auc']:.3f} / {row['local_sum_ratio']:.2f} | {row['kl_auc']:.3f} / {row['kl_sum_ratio']:.2f} | {row['mean_ce_delta_candidate']:+.4g}/{row['mean_ce_delta_control']:+.4g} |")
    lines += ["", "## Guardrails", "", "This establishes finite causal importance of a relation-conditioned edge population, not a complete semantic description of a head. Candidate and control edges come from different examples, so matching removes declared address/frequency confounds but cannot equalize every latent context. KL measures output-distribution change regardless of whether the removed edge helped the observed target. The local probe vocabulary is weight-derived and fixed before this test. Reusing the prospective corpus for the finite follow-up makes this an intervention validation of frozen candidates, not a second independent semantic discovery split."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    labels = [f"{row['candidate']}\nL{row['layer']}H{row['head']} s{row['split']}" for row in records]
    x = np.arange(len(records)); axes[0].bar(x, [row["local_auc"] for row in records]); axes[1].bar(x, [row["kl_auc"] for row in records])
    for axis, title in zip(axes, ("Immediate operator effect", "Final-logit effect")):
        axis.axhline(.5, color="black", linewidth=.8); axis.set(xticks=x, xticklabels=labels, ylabel="Matched AUROC", title=title); axis.tick_params(axis="x", rotation=75); axis.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "records": records}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
