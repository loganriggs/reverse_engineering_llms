#!/usr/bin/env python3
"""Jointly transplant/remove attention writes across layer sets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import trace
from run_bilin12_boundary_carrier_depth_audit import exact_kl, fisher_terms
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_multilayer_attention_routing_audit.json")
OUTPUT = TENSOR_ROOT / "bilin12_multilayer_attention_routing_audit.json"
REPORT = TENSOR_ROOT / "BILIN12_MULTILAYER_ATTENTION_ROUTING_AUDIT.md"
FIGURE = TENSOR_ROOT / "bilin12_multilayer_attention_routing_audit.png"
SCHEMES = ("replacement", "delta")


def routes(ranking):
    return {
        "none": [], "top1": ranking[:1], "top2": ranking[:2], "top4": ranking[:4],
        "top8": ranking[:8], "all_ranked": ranking,
        "early_0_3": [0, 1, 2, 3], "middle_4_7": [4, 5, 6, 7],
        "late_8_11": [8, 9, 10, 11], "all_layers": list(range(12)),
    }


@torch.inference_mode()
def run_intervention(model, ids, base_rows, changed_rows, selected, scheme, direction):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,)); x = x0; value_bus = None
    selected = set(selected)
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        native_attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        attention = native_attention
        if layer in selected:
            base_write = base_rows[layer]["details"]["write"]
            changed_write = changed_rows[layer]["details"]["write"]
            if scheme == "replacement": attention = changed_write if direction == "insertion" else base_write
            elif scheme == "delta":
                delta = changed_write - base_write
                attention = native_attention + delta if direction == "insertion" else native_attention - delta
            else: raise ValueError(scheme)
        pre = mixed + attention
        state = F.rms_norm(pre, (dimension,))
        mlp = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        x = pre + mlp
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return 30 * torch.tanh(raw / 30)


def empty_accumulator(route_names):
    return {
        key: {"target_energy": 0.0, "prediction_energy": 0.0, "cross": 0.0, "error": 0.0,
              "forward_kl": 0.0, "target_error_kl": 0.0}
        for key in route_names
    }


def update_row(row, base, changed, hybrid):
    target = changed - base; prediction = hybrid - base
    target_energy, prediction_energy, cross, error = fisher_terms(base, target, prediction)
    row["target_energy"] += float(target_energy.sum()); row["prediction_energy"] += float(prediction_energy.sum())
    row["cross"] += float(cross.sum()); row["error"] += float(error.sum())
    row["forward_kl"] += float(exact_kl(base, hybrid).sum()); row["target_error_kl"] += float(exact_kl(changed, hybrid).sum())


def finalize(row, full_forward_kl, full_target_error_kl):
    target_energy = max(row["target_energy"], 1e-30)
    return {
        "fisher_fidelity": 1 - row["error"] / target_energy,
        "fisher_cosine": row["cross"] / max(np.sqrt(row["prediction_energy"] * target_energy), 1e-30),
        "target_projection": row["cross"] / target_energy,
        "forward_kl_ratio": row["forward_kl"] / max(full_forward_kl, 1e-30),
        "target_error_kl_ratio": row["target_error_kl"] / max(full_target_error_kl, 1e-30),
    }


@torch.inference_mode()
def evaluate(model, tokenizer, rows, edit_position, ranking, config):
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]; route_map = routes(ranking)
    keys = [f"{direction}/{scheme}/{name}" for direction in ("insertion", "removal") for scheme in SCHEMES for name in route_map]
    accumulator = empty_accumulator(keys); full_forward_kl = 0.0; full_target_error_kl = 0.0; examples = 0; all_removal_base_error = 0.0
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :config["sequence_length"]], dtype=np.int64)
        base_ids = torch.as_tensor(raw, device="cuda"); keep = base_ids[:, edit_position] != newline
        if not bool(keep.any()): continue
        base_ids = base_ids[keep]; changed_ids = base_ids.clone(); changed_ids[:, edit_position] = newline; examples += len(base_ids)
        base_rows, native_base, _ = trace(model, base_ids); changed_rows, native_changed, _ = trace(model, changed_ids)
        full_forward_kl += float(exact_kl(native_base, native_changed).sum()); full_target_error_kl += float(exact_kl(native_changed, native_base).sum())
        for scheme in SCHEMES:
            for name, selected in route_map.items():
                inserted = run_intervention(model, base_ids, base_rows, changed_rows, selected, scheme, "insertion")
                removed = run_intervention(model, changed_ids, base_rows, changed_rows, selected, scheme, "removal")
                update_row(accumulator[f"insertion/{scheme}/{name}"], native_base, native_changed, inserted)
                update_row(accumulator[f"removal/{scheme}/{name}"], native_base, native_changed, removed)
                if scheme == "replacement" and name == "all_layers": all_removal_base_error = max(all_removal_base_error, float((removed - native_base).abs().max()))
    result = {key: finalize(row, full_forward_kl, full_target_error_kl) for key, row in accumulator.items()}
    return {"examples": examples, "mean_full_forward_kl": full_forward_kl / max(examples, 1), "all_replacement_removal_base_error": all_removal_base_error, "routes": result}


def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); model = load_snapshot_model(Path(config["snapshot"]), "cuda"); results = {}
    for corpus, spec in config["evaluation"].items():
        corpus_rows = load_rows(spec); results[corpus] = {}
        for position in config["edit_positions"]:
            print("evaluating", corpus, position, flush=True)
            results[corpus][str(position)] = evaluate(model, tokenizer, corpus_rows, position, config["rankings"][corpus][str(position)], config)
    all_rows = [results[c][str(p)] for c in results for p in config["edit_positions"]]
    def metric(c, p, direction, scheme, route): return results[c][str(p)]["routes"][f"{direction}/{scheme}/{route}"]["fisher_fidelity"]
    hypotheses = {
        "M1_all_attention_replacement_removal_blocks_the_path": {"supported": all(row["all_replacement_removal_base_error"] <= 1e-4 for row in all_rows)},
        "M2_all_attention_insertion_recovers_most_of_the_effect": {"supported": all(metric(c,p,"insertion","replacement","all_layers") >= .50 for c in results for p in config["edit_positions"])},
        "M3_top4_captures_most_available_insertion": {"supported": all(metric(c,p,"insertion","replacement","top4") >= .75 * metric(c,p,"insertion","replacement","all_layers") for c in results for p in config["edit_positions"])},
        "M4_routing_is_distributed_beyond_one_layer": {"supported": all(metric(c,p,"insertion","replacement","all_layers") - metric(c,p,"insertion","replacement","top1") >= .20 for c in results for p in config["edit_positions"])},
        "M5_middle_layers_are_the_main_phase": {"supported": all(metric(c,p,"insertion","replacement","middle_4_7") > max(metric(c,p,"insertion","replacement","early_0_3"), metric(c,p,"insertion","replacement","late_8_11")) for c in results for p in config["edit_positions"])},
        "M6_replacement_and_delta_interventions_agree": {"supported": all(abs(metric(c,p,"insertion","replacement",route) - metric(c,p,"insertion","delta",route)) <= .20 for c in results for p in config["edit_positions"] for route in routes(config["rankings"][c][str(p)]))},
        "M7_topk_frontier_is_monotone": {"supported": all(all(b >= a - 1e-6 for a,b in zip([metric(c,p,"insertion","replacement",route) for route in ("top1","top2","top4","top8","all_layers")][:-1], [metric(c,p,"insertion","replacement",route) for route in ("top1","top2","top4","top8","all_layers")][1:])) for c in results for p in config["edit_positions"])},
        "M8_all_attention_is_structurally_necessary_but_not_a_single_module": {"supported": all(results[c][str(p)]["routes"]["removal/replacement/all_layers"]["target_projection"] <= .01 and metric(c,p,"insertion","replacement","top1") < .50 for c in results for p in config["edit_positions"])},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.7)); route_names = ("top1","top2","top4","top8","all_layers"); x=np.arange(len(route_names))
    for corpus in results:
        values=[np.mean([metric(corpus,p,"insertion","replacement",route) for p in config["edit_positions"]]) for route in route_names]; axes[0].plot(x,values,marker="o",label=corpus)
        phases=[np.mean([metric(corpus,p,"insertion","replacement",route) for p in config["edit_positions"]]) for route in ("early_0_3","middle_4_7","late_8_11")]; axes[1].plot(np.arange(3),phases,marker="o",label=corpus)
        replace=[np.mean([metric(corpus,p,"insertion","replacement",route) for p in config["edit_positions"]]) for route in route_names]; delta=[np.mean([metric(corpus,p,"insertion","delta",route) for p in config["edit_positions"]]) for route in route_names]; axes[2].plot(replace,delta,marker="o",ls="",label=corpus)
    axes[0].set_xticks(x,route_names,rotation=20); axes[0].set_title("Held-out top-k insertion frontier"); axes[1].set_xticks(np.arange(3),("early","middle","late")); axes[1].set_title("Four-layer phase insertion"); axes[2].plot([-1,1],[-1,1],color="black",lw=.8); axes[2].set_xlabel("replacement fidelity"); axes[2].set_ylabel("delta fidelity"); axes[2].set_title("Intervention-definition red team")
    for axis in axes: axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE,dpi=180); plt.close(fig)

    lines=["# Multilayer attention-routing audit", "", "Layer rankings were frozen from the source-distance panel and evaluated on fresh corpus rows. In replacement interventions, selected base-run attention writes are replaced by native changed-run writes (or vice versa for removal). In additive-delta interventions, the native hybrid write is adjusted by the fixed changed-minus-base write. MLPs and all unselected attention layers recompute normally.", "", "## Locked hypotheses", ""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Multilayer routing](bilin12_multilayer_attention_routing_audit.png)", "", "| Corpus | Position | Full KL | Top1 / top4 / all replacement fidelity | Early / middle / late fidelity | All-removal projection | All-removal base error |", "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus in results:
        for position in config["edit_positions"]:
            row=results[corpus][str(position)]; top=" / ".join(f"{metric(corpus,position,'insertion','replacement',r):.3f}" for r in ("top1","top4","all_layers")); phase=" / ".join(f"{metric(corpus,position,'insertion','replacement',r):.3f}" for r in ("early_0_3","middle_4_7","late_8_11")); removal=row['routes']['removal/replacement/all_layers']['target_projection']
            lines.append(f"| {corpus} | {position} | {row['mean_full_forward_kl']:.4f} | {top} | {phase} | {removal:.4f} | {row['all_replacement_removal_base_error']:.2e} |")
    lines += ["", "All-attention removal is a structural path test; all-attention insertion is a sufficiency test. The first can pass while the second fails because changed attention writes were computed jointly with changed source-position residuals, embeddings, and value memory. Top-k curves test compression of the route, not whether the selected layers uniquely own the information."]
    REPORT.write_text("\n".join(lines)+"\n"); print("wrote",REPORT,flush=True)


if __name__=="__main__": main()
