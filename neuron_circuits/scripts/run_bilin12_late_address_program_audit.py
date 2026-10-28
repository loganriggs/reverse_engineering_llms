#!/usr/bin/env python3
"""Audit whether compact late address operators are understandable causal programs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from discover_bilin18_weight_only_circuits import CATEGORIES, auc, token_flags
from run_bilin12_layer5_address_channel_decomposition import channel_decomposition
from run_bilin12_layer5_weight_conditioned_addresses import (
    derive_cleaned_addresses,
    first_energy_rank,
    output_conditioned_form,
    restricted_operator,
    token_extremes,
)
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer8_restricted_tensor_program import ridge_covariance_transforms
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_late_address_program_audit.json")
OUTPUT = TENSOR_ROOT / "bilin12_late_address_program_audit.json"
REPORT = TENSOR_ROOT / "BILIN12_LATE_ADDRESS_PROGRAM_AUDIT.md"
FIGURE = TENSOR_ROOT / "bilin12_late_address_program_audit.png"


def load_rows(spec):
    source = np.load(spec["path"], mmap_mode="r")
    return np.asarray(source[spec["start"] : spec["start"] + spec["rows"]])


def signed_auc(target, score):
    raw = float(auc(target.astype(bool), score))
    return {"auc": max(raw, 1 - raw), "polarity": "positive" if raw >= .5 else "negative"}


def token_mean_extremes(ids, values, tokenizer, minimum_count, count=5):
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=values) / counts
    keep = counts >= minimum_count
    unique, counts, means = unique[keep], counts[keep], means[keep]
    high = np.argsort(means)[-count:][::-1]; low = np.argsort(means)[:count]
    def rows(indices):
        return [{"token": tokenizer.decode([int(unique[i])]), "mean": float(means[i]), "count": int(counts[i])} for i in indices]
    return {"positive": rows(high), "negative": rows(low)}


def exact_kl(reference, changed):
    logp = F.log_softmax(reference, -1)
    return (logp.exp() * (logp - F.log_softmax(changed, -1))).sum(-1)


def run_suffix(model, x, value_bus, x0, start_layer):
    for block in model.blocks[start_layer:]:
        x, value_bus = block(x, value_bus, x0)
    dimension = model.config.n_embd
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return 30 * torch.tanh(raw / 30)


def build_program(edit, output, down, left, right, square, inverse, prefixes):
    operator = restricted_operator(down, left, right, edit)
    weighted = operator @ square
    u, singular, vh = torch.linalg.svd(weighted, full_matrices=False)
    context_rows = ((left @ edit)[:, None] * right + (right @ edit)[:, None] * left) @ square
    decomposition = channel_decomposition(down, context_rows, prefixes)
    order = decomposition["order"]
    raw_context_rows = (left @ edit)[:, None] * right + (right @ edit)[:, None] * left
    output_role = (output @ weighted).square().sum() / weighted.square().sum().clamp_min(1e-30)
    return {
        "edit": edit, "operator": operator, "u": u, "singular": singular, "v": vh.T,
        "context_read": inverse @ vh.T[:, 0], "order": order,
        "raw_context_rows": raw_context_rows,
        "operator_k90": first_energy_rank(singular, .90),
        "operator_effective_rank": float(singular.square().sum().square() / singular.pow(4).sum().clamp_min(1e-30)),
        "shapley_k90": decomposition["shapley_k90"],
        "output_role_fraction": float(output_role),
        "top_channels": order[:16].cpu().tolist(),
    }


@torch.inference_mode()
def collect_semantics(model, tokenizer, flags, layer, programs, config):
    result = {}
    for corpus, spec in config["semantic_evaluation"].items():
        states, _ = collect_states_writes(model, layer, spec, config["positions"], config["batch_size"])
        rows = load_rows(spec); positions = np.asarray(config["positions"])
        current = rows[:, positions].reshape(-1); nxt = rows[:, positions + 1].reshape(-1)
        corpus_rows = {}
        for index, program in enumerate(programs):
            score = (states @ program["context_read"]).cpu().numpy()
            features = {}
            for prefix, ids in (("current", current), ("next", nxt)):
                selected_flags = flags[ids]
                for category, name in enumerate(CATEGORIES):
                    target = selected_flags[:, category].astype(bool)
                    if min(target.sum(), len(target) - target.sum()) < config["minimum_category_examples"]:
                        continue
                    features[f"{prefix}/{name}"] = signed_auc(target, score)
            corpus_rows[str(index)] = {
                "features": features,
                "current_token_means": token_mean_extremes(current, score, tokenizer, config["minimum_token_count"]),
                "next_token_means": token_mean_extremes(nxt, score, tokenizer, config["minimum_token_count"]),
                "score_quantiles": {str(q): float(np.quantile(score, q)) for q in (.001, .01, .5, .99, .999)},
            }
        result[corpus] = corpus_rows
    return result


def find_recurrent_features(semantics, address_count, threshold=.75):
    result = {}
    for index in range(address_count):
        common = None
        for corpus in semantics:
            strong = {
                (name, row["polarity"]) for name, row in semantics[corpus][str(index)]["features"].items()
                if row["auc"] >= threshold
            }
            common = strong if common is None else common & strong
        result[str(index)] = [f"{name}/{polarity}" for name, polarity in sorted(common or set())]
    return result


@torch.inference_mode()
def causal_audit(model, layer, programs, config):
    block = model.blocks[layer]; dimension = model.config.n_embd
    result = {}
    for corpus, spec in config["causal_evaluation"].items():
        rows = load_rows(spec)
        accum = defaultdict(list)
        for start in range(0, len(rows), config["batch_size"]):
            raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
            ids = torch.as_tensor(raw, device="cuda")
            x0 = F.rms_norm(model.embedding(ids), (dimension,)); x = x0; value_bus = None
            for prefix_block in model.blocks[:layer]: x, value_bus = prefix_block(x, value_bus, x0)
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
            pre = mixed + attention; state = F.rms_norm(pre, (dimension,))
            native_write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
            native_x = pre + native_write
            baseline = run_suffix(model, native_x, value_bus, x0, layer + 1)
            final_state = state[:, -1]
            for index, program in enumerate(programs):
                edit = program["edit"]; operator = program["operator"]
                u = program["u"]; singular = program["singular"]; v = program["v"]
                exact_base = block.mlp.Down(block.mlp.Left(final_state) * block.mlp.Right(final_state)) + block.mlp.Down_bias
                linear_delta = final_state @ operator.T
                coordinate = final_state @ program["context_read"]
                rank1_delta = coordinate[:, None] * singular[0] * u[:, 0][None]
                prefix_delta = {}
                for prefix in config["native_prefixes"]:
                    selected = program["order"][:prefix]
                    prefix_delta[prefix] = (final_state @ program["raw_context_rows"][selected].T) @ block.mlp.Down.weight[:, selected].T
                for scale in config["edit_scales"]:
                    edited_state = final_state + scale * edit
                    exact_write = block.mlp.Down(block.mlp.Left(edited_state) * block.mlp.Right(edited_state)) + block.mlp.Down_bias
                    variants = {
                        "exact": exact_write,
                        "full_linear": exact_base + scale * linear_delta,
                        "rank1": exact_base + scale * rank1_delta,
                    }
                    for prefix, delta in prefix_delta.items(): variants[f"top{prefix}"] = exact_base + scale * delta
                    logits = {}
                    for name, replacement in variants.items():
                        changed_x = native_x.clone(); changed_x[:, -1] = pre[:, -1] + replacement
                        logits[name] = run_suffix(model, changed_x, value_bus, x0, layer + 1)
                    intervention = exact_kl(baseline, logits["exact"])
                    key = f"address{index}/scale{scale:g}"
                    accum[f"{key}/intervention_kl"].extend(intervention.cpu().tolist())
                    for name in variants:
                        if name == "exact": continue
                        prediction = exact_kl(logits["exact"], logits[name])
                        accum[f"{key}/{name}"].extend(prediction.cpu().tolist())
        corpus_rows = {}
        for index in range(len(programs)):
            for scale in config["edit_scales"]:
                key = f"address{index}/scale{scale:g}"
                intervention = np.asarray(accum[f"{key}/intervention_kl"])
                row = {"mean_intervention_kl": float(intervention.mean())}
                for name in ["full_linear", "rank1"] + [f"top{x}" for x in config["native_prefixes"]]:
                    error = np.asarray(accum[f"{key}/{name}"])
                    row[f"mean_{name}_prediction_kl"] = float(error.mean())
                    row[f"relative_prediction_kl/{name}"] = float(error.mean() / max(intervention.mean(), 1e-30))
                corpus_rows[key] = row
        result[corpus] = corpus_rows
    return result


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([token_flags(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)])
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    results = {}; all_semantics = {}; all_causal = {}; recurrent_features = {}
    random_summary = {}
    for layer in config["layers"]:
        print("layer", layer, "program construction", flush=True)
        states, _ = collect_states_writes(model, layer, config["discovery"], config["positions"], config["batch_size"])
        square, inverse, covariance = ridge_covariance_transforms(states, config["covariance_ridge_fraction"])
        block = model.blocks[layer]; down = block.mlp.Down.weight.float(); left = block.mlp.Left.weight.float(); right = block.mlp.Right.weight.float()
        _, outputs = output_map_eigensystem(down, left, right); output = outputs[:, 0]
        form = output_conditioned_form(down, left, right, output)
        _, _, edits = derive_cleaned_addresses(form, square, config["address_count"])
        programs = [build_program(edits[:, i], output, down, left, right, square, inverse, config["native_prefixes"]) for i in range(config["address_count"])]
        random_rows = []
        for _ in range(config["random_direction_count"]):
            edit = square @ torch.randn(len(square), generator=generator, device="cuda")
            edit = edit / edit.norm()
            random_rows.append(build_program(edit, output, down, left, right, square, inverse, config["native_prefixes"])["operator_k90"])
        print("layer", layer, "semantic audit", flush=True)
        semantics = collect_semantics(model, tokenizer, flags, layer, programs, config)
        recurrent = find_recurrent_features(semantics, config["address_count"])
        print("layer", layer, "causal audit", flush=True)
        causal = causal_audit(model, layer, programs, config)
        metadata = []
        for index, program in enumerate(programs):
            row = {key: value for key, value in program.items() if key in ("operator_k90", "operator_effective_rank", "shapley_k90", "output_role_fraction", "top_channels")}
            row["address"] = index
            row["context_embedding_tokens"] = token_extremes(
                (model.embedding.weight.float() @ program["context_read"])[: tokenizer.vocab_size], tokenizer
            )
            row["output_unembedding_tokens"] = token_extremes(
                model.lm_head(program["u"][:, 0][None]).squeeze(0)[: tokenizer.vocab_size], tokenizer
            )
            metadata.append(row)
        results[str(layer)] = {"covariance": covariance, "programs": metadata}
        all_semantics[str(layer)] = semantics; all_causal[str(layer)] = causal; recurrent_features[str(layer)] = recurrent
        random_summary[str(layer)] = {"operator_k90": random_rows, "median_operator_k90": float(np.median(random_rows))}

    compact = [(layer, p["address"]) for layer, row in results.items() for p in row["programs"] if p["operator_k90"] <= 2]
    def causal_values(layer, address, scale, name):
        return [all_causal[layer][corpus][f"address{address}/scale{scale:g}"][f"relative_prediction_kl/{name}"] for corpus in all_causal[layer]]
    small_full = [all_causal[layer][corpus][key]["relative_prediction_kl/full_linear"] for layer in all_causal for corpus in all_causal[layer] for key in all_causal[layer][corpus] if key.endswith("scale0.25")]
    rank_small = [all_causal[layer][corpus][key]["relative_prediction_kl/rank1"] for layer in all_causal for corpus in all_causal[layer] for key in all_causal[layer][corpus] if key.endswith("scale0.25")]
    rank_large = [all_causal[layer][corpus][key]["relative_prediction_kl/rank1"] for layer in all_causal for corpus in all_causal[layer] for key in all_causal[layer][corpus] if key.endswith("scale1")]
    layer_rank_medians = {layer: float(np.median([all_causal[layer][c][k]["relative_prediction_kl/rank1"] for c in all_causal[layer] for k in all_causal[layer][c] if k.endswith("scale0.25")])) for layer in all_causal}
    hypotheses = {
        "L1_late_compactness_replicates": {"supported": len(compact) >= 7, "compact": compact},
        "L2_compact_addresses_beat_random_directions": {"supported": all(np.median([p["operator_k90"] for p in results[layer]["programs"]]) <= .25 * random_summary[layer]["median_operator_k90"] for layer in results)},
        "L3_full_linearization_predicts_small_edits": {"supported": max(small_full) <= .02, "maximum_relative_error": max(small_full)},
        "L4_rank1_predicts_compact_address_edits": {"supported": all(max(causal_values(layer, address, .25, "rank1")) <= .10 for layer, address in compact)},
        "L5_top128_native_channels_predict_compact_edits": {"supported": all(max(causal_values(layer, address, .25, "top128")) <= .10 for layer, address in compact)},
        "L6_compact_programs_have_recurrent_simple_input_semantics": {"supported": sum(bool(features) for layer in recurrent_features.values() for features in layer.values()) >= 4, "recurrent_features": recurrent_features},
        "L7_rank1_behavior_degrades_at_finite_scale": {"supported": np.median(rank_large) >= 2 * np.median(rank_small), "small_median": float(np.median(rank_small)), "large_median": float(np.median(rank_large))},
        "L8_final_layer_is_more_rank1_behavioral": {"supported": layer_rank_medians["11"] <= .5 * layer_rank_medians["10"], "layer_medians": layer_rank_medians},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results, "random_controls": random_summary, "semantics": all_semantics, "recurrent_features": recurrent_features, "causal": all_causal}, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels=[]; op=[]; ch=[]; rel=[]
    for layer in results:
        for p in results[layer]["programs"]:
            labels.append(f"L{layer}A{p['address']}"); op.append(p["operator_k90"]); ch.append(p["shapley_k90"])
            rel.append(np.median(causal_values(layer, p["address"], .25, "rank1")))
    x=np.arange(len(labels)); axes[0].bar(x, op); axes[0].set_title("Operator K90")
    axes[1].bar(x, ch); axes[1].set_title("Native-channel K90")
    axes[2].bar(x, rel); axes[2].axhline(.1,color="black",ls="--",lw=1); axes[2].set_title("Rank-1 relative behavioral KL")
    for ax in axes: ax.set_xticks(x,labels,rotation=45); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURE,dpi=180); plt.close(fig)

    lines=["# Late address-program audit", "", "The all-layer sweep found a compact local-operator phase at Layers 10–11. This audit reconstructs those programs on a separate covariance slice, tests token-level explanations across three corpora, and compares exact finite MLP-input edits with full-linear, rank-one, and native-channel approximations.", "", "## Locked hypotheses", ""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Late programs](bilin12_late_address_program_audit.png)", "", "| Program | Operator K90 | Channel K90 | Output role | Recurrent features | Rank1 rel. KL @0.25 | Top128 rel. KL @0.25 |", "|---|---:|---:|---:|---|---:|---:|"]
    for layer,row in results.items():
        for p in row["programs"]:
            rank=np.median(causal_values(layer,p["address"],.25,"rank1")); top=np.median(causal_values(layer,p["address"],.25,"top128"))
            lines.append(f"| L{layer}A{p['address']} | {p['operator_k90']} | {p['shapley_k90']} | {p['output_role_fraction']:.3f} | {', '.join(recurrent_features[layer][str(p['address'])]) or 'none'} | {rank:.3f} | {top:.3f} |")
    lines += ["", "Low operator rank proves a short basis-invariant local map. Low relative behavioral KL shows that the short map predicts a finite downstream consequence. Native-channel prefixes answer a different question: whether the model's own coordinates give a short implementation. Recurrent token features are descriptive candidates only until an input intervention validates them."]
    REPORT.write_text("\n".join(lines)+"\n"); print(REPORT)


if __name__ == "__main__": main()
