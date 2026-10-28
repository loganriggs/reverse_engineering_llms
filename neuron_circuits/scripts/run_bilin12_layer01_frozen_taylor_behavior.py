#!/usr/bin/env python3
"""Behavioral validation of frozen Taylor-compiled Layer-0/1 coalitions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_conditional_coalition_selector import local_scores
from run_bilin12_layer01_directional_order_frontier import derivative_frontier, taylor_sum
from run_bilin12_layer01_dynamic_alignment_replication import (
    final_product_delta,
    forward_layers_native,
    median,
    selected_delta,
)
from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin12_layer01_frozen_taylor_behavior.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_frozen_taylor_behavior.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_FROZEN_TAYLOR_BEHAVIOR.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_frozen_taylor_behavior.png"


def sample_edits(rows, count, seed, edit_position, sequence_length):
    """Pair distinct natural source rows and natural replacement tokens deterministically."""
    if len(rows) < 2 * count:
        raise ValueError("sampling requires at least twice as many rows as sites")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    sources = order[:count]
    donors = order[count:]
    result = []
    used = set()
    for source_index in sources:
        source = int(rows[source_index, edit_position])
        target = None
        for donor_index in donors:
            candidate = int(rows[donor_index, edit_position])
            if candidate != source:
                target = candidate
                break
        if target is None:
            raise RuntimeError("could not find a distinct natural replacement token")
        # Rotate the donor pool so repeated high-frequency source tokens do not all
        # receive the same target.
        donors = np.roll(donors, -1)
        row = int(source_index)
        if row in used:
            raise AssertionError("duplicate source row")
        used.add(row)
        result.append((row, target))
    return result


def product_from_x0(model, x0, target_layer, position):
    """Differentiable prefix from normalized embeddings to one product vector."""
    dimension = model.config.n_embd
    x, value_bus = x0, None
    for layer, block in enumerate(model.blocks[:target_layer + 1]):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer == target_layer:
            return product[0, position].float()
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    raise AssertionError("unreachable")


def differentiable_suffix(model, layer, x0, value_bus, layer_output, position):
    dimension = model.config.n_embd
    x = layer_output
    for block in model.blocks[layer + 1:]:
        x, value_bus = block(x, value_bus, x0)
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, position]).float()
    return 30 * torch.tanh(raw / 30)


def logits_for_deltas(model, layer, base, deltas, position):
    names = list(deltas)
    decoder = model.blocks[layer].mlp.Down.weight.float()
    outputs, x0s, buses = [], [], []
    for delta in deltas.values():
        output = base["output"].clone()
        output[:, position] += F.linear(delta, decoder).to(output.dtype)
        outputs.append(output)
        x0s.append(base["x0"])
        buses.append(base["value_bus"])
    logits = suffix_from_layer(
        model, layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), position,
    )
    return {name: logits[index] for index, name in enumerate(names)}


def behavior_oracle_scores(model, layer, base, full, reference_logits, position):
    """Endpoint-informed activation*gradient control for matching full-patch behavior."""
    decoder = model.blocks[layer].mlp.Down.weight.float()
    # `forward_layers_native` intentionally returns inference tensors.  Cloning
    # is required before the suffix backward: detach alone preserves the
    # inference-tensor flag and autograd cannot save such values.
    full_value = full.detach().clone()
    x0 = base["x0"].detach().clone()
    value_bus = base["value_bus"].detach().clone()
    layer_output = base["output"].detach().clone()
    # The checkpoint implementation caches rotary tables as plain attributes.
    # A preceding inference-mode forward therefore leaves inference tensors in
    # those caches. Force regeneration under grad mode for this suffix pass.
    for block in model.blocks[layer + 1:]:
        block.attn.rotary.seq_len_cached = None
        block.attn.rotary.cos_cached = None
        block.attn.rotary.sin_cached = None
    variable = torch.zeros_like(full_value, requires_grad=True)
    write = F.linear(variable, decoder)
    mask = torch.zeros_like(layer_output)
    mask[:, position] = write
    logits = differentiable_suffix(
        model, layer, x0, value_bus, layer_output + mask, position,
    )[0]
    target = F.softmax(reference_logits.detach().clone(), -1)
    cross_entropy = -(target * F.log_softmax(logits, -1)).sum()
    gradient = torch.autograd.grad(cross_entropy, variable)[0]
    return (full_value * gradient).abs().detach()


def exact_kl(reference, prediction):
    # Local early-layer patches can induce KLs below 1e-6 over a 50k-token
    # vocabulary.  Float32 summation occasionally made these slightly negative,
    # corrupting normalized recovery ratios.  Accumulate the metric in float64.
    logp = F.log_softmax(reference.double(), -1)
    return float((logp.exp() * (logp - F.log_softmax(prediction.double(), -1))).sum())


def behavior_metrics(base, reference, endpoint, methods, label):
    target_effect = reference - base
    rows = {}
    base_ce = float(F.cross_entropy(base[None], torch.tensor([label], device=base.device)))
    reference_ce = float(F.cross_entropy(reference[None], torch.tensor([label], device=base.device)))
    reference_logit_effect = float(target_effect[label])
    for name, logits in methods.items():
        fidelity, cosine, energy = fisher_fidelity(base, target_effect, logits - base)
        ce = float(F.cross_entropy(logits[None], torch.tensor([label], device=base.device)))
        rows[name] = {
            "kl_error": exact_kl(reference, logits),
            "endpoint_kl_error": exact_kl(endpoint, logits),
            "fisher_fidelity": fidelity,
            "fisher_cosine": cosine,
            "fisher_energy": energy,
            "top1_agreement": int(logits.argmax() == reference.argmax()),
            "endpoint_top1_agreement": int(logits.argmax() == endpoint.argmax()),
            "ce": ce,
            "ce_effect": ce - base_ce,
            "ce_effect_error": (ce - base_ce) - (reference_ce - base_ce),
            "label_logit_effect": float(logits[label] - base[label]),
            "label_logit_effect_error": float(logits[label] - base[label]) - reference_logit_effect,
        }
    return rows


def aggregate(rows, method, rank):
    key = f"{method}_{rank}"
    base_kl = sum(row["base_kl"] for row in rows)
    error = sum(row["metrics"][key]["kl_error"] for row in rows)
    endpoint_base = sum(row["endpoint_base_kl"] for row in rows)
    endpoint_error = sum(row["metrics"][key]["endpoint_kl_error"] for row in rows)
    ce_target = np.asarray([row["reference_ce_effect"] for row in rows], dtype=np.float64)
    ce_error = np.asarray([row["metrics"][key]["ce_effect_error"] for row in rows], dtype=np.float64)
    logit_target = np.asarray([row["reference_label_logit_effect"] for row in rows], dtype=np.float64)
    logit_error = np.asarray([row["metrics"][key]["label_logit_effect_error"] for row in rows], dtype=np.float64)
    return {
        "kl_recovery": float(1 - error / max(base_kl, 1e-30)),
        "endpoint_kl_recovery": float(1 - endpoint_error / max(endpoint_base, 1e-30)),
        "median_fisher_fidelity": median([row["metrics"][key]["fisher_fidelity"] for row in rows]),
        "median_fisher_cosine": median([row["metrics"][key]["fisher_cosine"] for row in rows]),
        "top1_agreement": float(np.mean([row["metrics"][key]["top1_agreement"] for row in rows])),
        "endpoint_top1_agreement": float(np.mean([row["metrics"][key]["endpoint_top1_agreement"] for row in rows])),
        "mean_absolute_ce_error": float(np.mean(np.abs(ce_error))),
        "ce_effect_fidelity": float(1 - np.square(ce_error).sum() / max(np.square(ce_target).sum(), 1e-30)),
        "label_logit_effect_fidelity": float(1 - np.square(logit_error).sum() / max(np.square(logit_target).sum(), 1e-30)),
    }


def bootstrap_difference(rows, left, right, rank, samples, seed):
    left_key, right_key = f"{left}_{rank}", f"{right}_{rank}"
    values = np.asarray([
        row["metrics"][right_key]["kl_error"] - row["metrics"][left_key]["kl_error"]
        for row in rows
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.asarray([np.mean(rng.choice(values, len(values), replace=True)) for _ in range(samples)])
    return float(values.mean()), [float(value) for value in np.quantile(draws, [.025, .975])]


def main():
    config = json.loads(CONFIG.read_text())
    parent = json.loads(Path(config["source_config"]).read_text())
    transfer = json.loads(Path(config["order_transfer_results"]).read_text())
    assert {str(layer): transfer["results"][str(layer)]["fineweb_global_order"] for layer in parent["layers"]} == config["frozen_orders"]
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in parent["evaluation"].items()
    }
    records = {str(layer): [] for layer in parent["layers"]}
    maximum_recomposition_error = 0.0

    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        edits = sample_edits(
            rows, config["sites_per_corpus"], config["sampling_seed"] + corpus_index,
            config["edit_position"], config["sequence_length"],
        )
        for site_index, (row_index, target_token) in enumerate(edits):
            raw_with_label = np.asarray(rows[row_index, :config["sequence_length"] + 1], dtype=np.int64).copy()
            raw = raw_with_label[:-1].copy(); changed = raw.copy()
            source_token = int(raw[config["edit_position"]]); changed[config["edit_position"]] = target_token
            label = int(raw_with_label[-1])
            base_ids = torch.as_tensor(raw, device="cuda")[None]
            endpoint_ids = torch.as_tensor(changed, device="cuda")[None]
            with torch.no_grad():
                base_trace = forward_layers_native(model, base_ids, max(parent["layers"]))
                endpoint_trace = forward_layers_native(model, endpoint_ids, max(parent["layers"]))
                dimension = model.config.n_embd
                base_x0 = F.rms_norm(model.embedding(base_ids), (dimension,)).float()
                endpoint_x0 = F.rms_norm(model.embedding(endpoint_ids), (dimension,)).float()
                endpoint_logits = suffix_from_layer(
                    model, max(parent["layers"]), endpoint_trace[max(parent["layers"])]["x0"],
                    endpoint_trace[max(parent["layers"])]["value_bus"],
                    endpoint_trace[max(parent["layers"])]["output"], config["prediction_position"],
                )[0]
            direction = endpoint_x0 - base_x0

            for layer in parent["layers"]:
                order = config["frozen_orders"][str(layer)]
                function = lambda value, layer=layer: product_from_x0(
                    model, value, layer, config["prediction_position"],
                )
                with torch.enable_grad():
                    derivatives, _ = derivative_frontier(function, base_x0, direction, order)
                derivatives = [value.detach().float() for value in derivatives]
                taylor = taylor_sum(derivatives, .5, integral=False)
                full = final_product_delta(
                    model, layer, base_trace[layer], endpoint_trace[layer], config["prediction_position"],
                )
                decoder = model.blocks[layer].mlp.Down.weight.float()
                normalized = F.rms_norm(base_trace[layer]["pre"][0, config["prediction_position"]].float(), (dimension,))
                base_product = model.blocks[layer].mlp.Left(normalized) * model.blocks[layer].mlp.Right(normalized)

                with torch.no_grad():
                    references = logits_for_deltas(
                        model, layer, base_trace[layer],
                        {"base": torch.zeros_like(full), "full": full}, config["prediction_position"],
                    )
                base_logits, reference_logits = references["base"], references["full"]
                maximum_recomposition_error = max(
                    maximum_recomposition_error,
                    float((reference_logits - logits_for_deltas(
                        model, layer, base_trace[layer], {"check": full},
                        config["prediction_position"],
                    )["check"]).abs().max()),
                )
                with torch.enable_grad():
                    oracle_score = behavior_oracle_scores(
                        model, layer, base_trace[layer], full, reference_logits,
                        config["prediction_position"],
                    )
                generator = torch.Generator(device="cuda").manual_seed(
                    config["random_control_seed"] + corpus_index * 100000 + site_index * 10 + layer
                )
                scores = {
                    "taylor_local": local_scores(taylor, decoder),
                    "taylor_magnitude": taylor.abs(),
                    "base_activation": base_product.abs(),
                    "static_weight": decoder.square().sum(0).sqrt(),
                    "random": torch.rand(len(full), generator=generator, device="cuda"),
                    "endpoint_magnitude": full.abs(),
                    "endpoint_local": local_scores(full, decoder),
                    "behavior_oracle": oracle_score,
                }
                deltas = {}
                for method in config["methods"]:
                    ordering = scores[method].argsort(descending=True)
                    for rank in config["ranks"]:
                        deltas[f"{method}_{rank}"] = selected_delta(full, ordering[:rank])
                with torch.no_grad():
                    method_logits = logits_for_deltas(
                        model, layer, base_trace[layer], deltas, config["prediction_position"],
                    )
                metrics = behavior_metrics(
                    base_logits, reference_logits, endpoint_logits, method_logits, label,
                )
                base_ce = float(F.cross_entropy(base_logits[None], torch.tensor([label], device="cuda")))
                reference_ce = float(F.cross_entropy(reference_logits[None], torch.tensor([label], device="cuda")))
                records[str(layer)].append({
                    "corpus": corpus,
                    "site": site_index,
                    "row": row_index,
                    "source_token": source_token,
                    "target_token": target_token,
                    "label": label,
                    "base_kl": exact_kl(reference_logits, base_logits),
                    "endpoint_base_kl": exact_kl(endpoint_logits, base_logits),
                    "reference_endpoint_kl": exact_kl(endpoint_logits, reference_logits),
                    "reference_ce_effect": reference_ce - base_ce,
                    "reference_label_logit_effect": float(reference_logits[label] - base_logits[label]),
                    "reference_top1_changed": int(reference_logits.argmax() != base_logits.argmax()),
                    "endpoint_top1_changed": int(endpoint_logits.argmax() != base_logits.argmax()),
                    "metrics": metrics,
                })
            if (site_index + 1) % 8 == 0:
                print("behavior", corpus, site_index + 1, "/", len(edits), flush=True)

    summaries = {str(layer): {} for layer in parent["layers"]}
    comparison_rank = config["comparison_rank"]
    for layer in parent["layers"]:
        for corpus_index, corpus in enumerate(arrays):
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            threshold = float(np.quantile([row["base_kl"] for row in selected], .75))
            high = [row for row in selected if row["base_kl"] >= threshold]
            methods = {
                method: {str(rank): aggregate(selected, method, rank) for rank in config["ranks"]}
                for method in config["methods"]
            }
            high_methods = {
                method: {str(rank): aggregate(high, method, rank) for rank in config["ranks"]}
                for method in config["methods"]
            }
            comparisons = {}
            for index, control in enumerate(("random", "static_weight", "endpoint_local", "behavior_oracle")):
                value, interval = bootstrap_difference(
                    selected, "taylor_local", control, comparison_rank,
                    config["bootstrap_samples"], config["bootstrap_seed"] + layer * 100 + corpus_index * 10 + index,
                )
                comparisons[f"taylor_minus_{control}_mean_kl_error_advantage"] = value
                comparisons[f"taylor_minus_{control}_ci"] = interval
            summaries[str(layer)][corpus] = {
                "sites": len(selected),
                "mean_reference_base_kl": float(np.mean([row["base_kl"] for row in selected])),
                "median_reference_base_kl": median([row["base_kl"] for row in selected]),
                "high_effect_threshold": threshold,
                "reference_top1_change_fraction": float(np.mean([row["reference_top1_changed"] for row in selected])),
                "endpoint_top1_change_fraction": float(np.mean([row["endpoint_top1_changed"] for row in selected])),
                "full_patch_endpoint_kl_recovery": float(1 - sum(row["reference_endpoint_kl"] for row in selected) / max(sum(row["endpoint_base_kl"] for row in selected), 1e-30)),
                "methods": methods,
                "high_effect_methods": high_methods,
                "comparisons": comparisons,
            }

    cell = lambda layer, corpus: summaries[str(layer)][corpus]
    metric = lambda layer, corpus, method, field, high=False: cell(layer, corpus)["high_effect_methods" if high else "methods"][method][str(comparison_rank)][field]
    hypotheses = {
        "B0_full_patch_recomposition_is_exact": {"supported": maximum_recomposition_error <= 1e-4},
        "B1_frozen_taylor_preserves_behavior": {"supported": all(
            metric(layer, corpus, "taylor_local", "kl_recovery") >= .50
            and metric(layer, corpus, "taylor_local", "kl_recovery", True) >= .40
            for layer in parent["layers"] for corpus in arrays
        )},
        "B2_taylor_beats_unconditioned_controls": {"supported": all(
            metric(layer, corpus, "taylor_local", "kl_recovery") >= metric(layer, corpus, control, "kl_recovery") + .20
            for layer in parent["layers"] for corpus in arrays for control in ("random", "static_weight")
        )},
        "B3_taylor_approaches_endpoint_local": {"supported": all(
            metric(layer, corpus, "taylor_local", "kl_recovery") >= metric(layer, corpus, "endpoint_local", "kl_recovery") - .15
            for layer in parent["layers"] for corpus in arrays
        )},
        "B4_behavioral_compiler_transfers_ood": {"supported": all(
            abs(metric(layer, "fineweb", "taylor_local", "kl_recovery") - metric(layer, "tinystories", "taylor_local", "kl_recovery")) <= .15
            for layer in parent["layers"]
        )},
        "B5_output_connectivity_matters": {"supported": all(
            metric(layer, corpus, "taylor_local", "kl_recovery") >= metric(layer, corpus, "taylor_magnitude", "kl_recovery") + .03
            for layer in parent["layers"] for corpus in arrays
        )},
        "B6_behavior_oracle_is_an_upper_control": {"supported": all(
            metric(layer, corpus, "behavior_oracle", "kl_recovery") >= metric(layer, corpus, "taylor_local", "kl_recovery") - .02
            for layer in parent["layers"] for corpus in arrays
        )},
        "B7_ce_effect_is_preserved": {"supported": all(
            metric(layer, corpus, "taylor_local", "ce_effect_fidelity") >= .40
            for layer in parent["layers"] for corpus in arrays
        )},
        "B8_high_effect_results_are_not_baseline_triviality": {"supported": all(
            metric(layer, corpus, "taylor_local", "kl_recovery", True) >= metric(layer, corpus, "random", "kl_recovery", True) + .15
            for layer in parent["layers"] for corpus in arrays
        )},
        "B9_orders_and_sampling_are_frozen": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({
        "config": config, "hypotheses": hypotheses,
        "maximum_recomposition_error": maximum_recomposition_error,
        "summaries": summaries, "records": records,
    }, indent=2, allow_nan=False))

    lines = [
        "# Frozen Taylor coalitions preserve downstream behavior", "",
        "Orders learned on the earlier 17-pair semantic panel are frozen (Layer 0 cubic; Layer 1 fifth order), then tested on independent natural-distribution token substitutions. Each prospective score is used only to select native product channels; the exact finite endpoint delta on those selected channels is patched through the complete suffix. KL recovery is normalized between the unpatched base (0) and the full local product patch (1).", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Behavioral rank frontier](bilin12_layer01_frozen_taylor_behavior.png)", "",
              "| Layer | Corpus | Full-patch KL / top-1 changes | Taylor/random/static/endpoint-local/oracle KL recovery @256 | High-effect Taylor/random | Taylor CE-effect fidelity | Full local path recovery to endpoint |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for layer in parent["layers"]:
        for corpus in arrays:
            row = cell(layer, corpus)
            value = lambda method, field="kl_recovery": row["methods"][method][str(comparison_rank)][field]
            high = lambda method: row["high_effect_methods"][method][str(comparison_rank)]["kl_recovery"]
            recoveries = "/".join(f"{value(method):.3f}" for method in ("taylor_local", "random", "static_weight", "endpoint_local", "behavior_oracle"))
            lines.append(
                f"| {layer} | {corpus} | {row['mean_reference_base_kl']:.3e} / {row['reference_top1_change_fraction']:.3f} | "
                f"{recoveries} | {high('taylor_local'):.3f}/{high('random'):.3f} | "
                f"{value('taylor_local', 'ce_effect_fidelity'):.3f} | {row['full_patch_endpoint_kl_recovery']:.3f} |"
            )
    lines += ["", "Top-1 agreement is reported in the JSON but is not treated as decisive because small local effects can leave the argmax unchanged even for the unpatched baseline. The highest-effect quartile and normalized KL/CE effects are the anti-triviality checks."]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for layer in parent["layers"]:
        for corpus in arrays:
            row = cell(layer, corpus)
            for method, style in (("taylor_local", "-"), ("random", ":"), ("static_weight", "--"), ("endpoint_local", "-."), ("behavior_oracle", "-")):
                axes[layer].plot(
                    config["ranks"], [row["methods"][method][str(rank)]["kl_recovery"] for rank in config["ranks"]],
                    linestyle=style, marker="o", label=f"{corpus} {method}",
                )
        axes[layer].set_title(f"Layer {layer}"); axes[layer].set_xlabel("Native product channels")
        axes[layer].set_ylabel("KL recovery"); axes[layer].grid(alpha=.2); axes[layer].legend(fontsize=6, ncol=2)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(REPORT, flush=True)


if __name__ == "__main__":
    main()
