#!/usr/bin/env python3
"""Finite causal test of one source variable routed through different functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_vocabulary import CATEGORIES, token_categories


CONFIG = Path("configs/bilin18_same_variable_function_causality.json")
OUTPUT = TENSOR_ROOT / "bilin18_same_variable_function_causality.json"
REPORT = TENSOR_ROOT / "BILIN18_SAME_VARIABLE_FUNCTION_CAUSALITY.md"


def variant_specs(heads):
    rows = [{"name": "native", "head": None, "scope": "native"}]
    for head in heads:
        for scope in ("all", "near", "far"):
            rows.append({"name": f"H{head}_{scope}", "head": head, "scope": scope})
    return rows


@torch.inference_mode()
def variant_logits(model, ids, source_categories, config, specs):
    variants = len(specs); batch, length = ids.shape; dimension = model.config.n_embd
    x0_base = F.rms_norm(model.embedding(ids), (dimension,)).float()
    x0 = x0_base.unsqueeze(0).repeat(variants, 1, 1, 1).flatten(0, 1)
    x, bus = x0, None
    source_match = source_categories(ids.cpu().numpy()) == CATEGORIES.index(config["source_category"])
    source_match = torch.as_tensor(source_match, device=ids.device)[:, None, None, :]
    destination = torch.arange(length, device=ids.device)[:, None]
    source = torch.arange(length, device=ids.device)[None, :]
    distance = destination - source
    causal = distance >= 0
    near = causal & (distance < config["near_distance_exclusive"])
    far = causal & ~near
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
        bus = details["value_bus"]
        if layer == config["layer"]:
            pattern = details["pattern"].reshape(variants, batch, model.config.n_head, length, length)
            values = details["mixed_value"].reshape(variants, batch, length, model.config.n_head, -1)
            writes = []
            for variant, spec in enumerate(specs):
                if spec["scope"] == "native":
                    writes.append(details["write"].reshape(variants, batch, length, dimension)[variant])
                    continue
                keep = torch.ones_like(pattern[variant], dtype=torch.bool)
                scope = causal if spec["scope"] == "all" else near if spec["scope"] == "near" else far
                remove = source_match & scope[None, None, :, :]
                keep[:, spec["head"]:spec["head"] + 1, :, :] &= ~remove
                attended = torch.einsum("bqhk,bkhd->bqhd", pattern[variant].permute(0, 2, 1, 3) * keep.permute(0, 2, 1, 3), values[variant])
                writes.append(block.attn.c_proj(attended.reshape(batch, length, dimension)).float())
            write = torch.stack(writes).flatten(0, 1)
        else:
            write = details["write"].float()
        pre = mixed + write
        state = F.rms_norm(pre, (dimension,))
        x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    x = x.reshape(variants, batch, length, dimension)[:, :, config["positions"]]
    raw = model.lm_head(F.rms_norm(x, (dimension,))).float()
    return 30 * torch.tanh(raw / 30)


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    static = token_categories(flags)
    source_categories = lambda ids: static[ids]
    corpus = np.load(config["corpus"], mmap_mode="r")
    raw = np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    specs = variant_specs(config["heads"]); names = [row["name"] for row in specs]
    count = len(specs) - 1
    gram = torch.zeros(count, count, dtype=torch.float64, device="cuda")
    kl_sums = torch.zeros(count, dtype=torch.float64, device="cuda")
    site_cosines = {(i, j): [] for i in range(count) for j in range(i + 1, count)}
    sites = 0
    for start in range(0, len(raw), config["evaluation_batch_size"]):
        if start % 16 == 0: print("finite function rows", start, "/", len(raw), flush=True)
        batch = raw[start:start + config["evaluation_batch_size"]]
        ids = torch.as_tensor(batch[:, :config["sequence_length"]], device="cuda")
        logits = variant_logits(model, ids, source_categories, config, specs)
        native = logits[0]
        effects = (native[None] - logits[1:]).float()
        flat = effects.flatten(1)
        gram += torch.einsum("id,jd->ij", flat.double(), flat.double())
        logp = F.log_softmax(native.float(), -1)
        for index in range(count):
            logq = F.log_softmax(logits[index + 1].float(), -1)
            kl_sums[index] += ((logp.exp() * (logp - logq)).sum(-1).clamp_min(0)).double().sum()
        norms = effects.norm(dim=-1)
        for i in range(count):
            for j in range(i + 1, count):
                valid = (norms[i] > 1e-8) & (norms[j] > 1e-8)
                if valid.any():
                    cosine = F.cosine_similarity(effects[i][valid], effects[j][valid], dim=-1)
                    site_cosines[(i, j)].extend(cosine.cpu().tolist())
        sites += native.shape[0] * native.shape[1]
    norm = gram.diag().sqrt()
    cosine = gram / (norm[:, None] * norm[None, :]).clamp_min(1e-30)
    mean_kl = (kl_sums / sites).cpu().numpy()
    index = {name: position - 1 for position, name in enumerate(names) if name != "native"}
    cross = []
    for offset, first in enumerate(config["heads"]):
        for second in config["heads"][offset + 1:]:
            cross.append(float(abs(cosine[index[f"H{first}_all"], index[f"H{second}_all"]])))
    within = [float(abs(cosine[index[f"H{head}_near"], index[f"H{head}_far"]])) for head in config["heads"]]
    median_cross = float(np.median(cross)); median_within = float(np.median(within))
    hypotheses = {
        "D1_each_function_has_finite_downstream_effect": {"supported": all(mean_kl[index[f"H{head}_all"]] > 1e-5 for head in config["heads"]), "all_route_mean_kl": {str(head): float(mean_kl[index[f"H{head}_all"]]) for head in config["heads"]}},
        "D2_same_variable_has_distinct_functions": {"supported": median_cross < .30, "median_abs_cross_head_cosine": median_cross},
        "D3_function_is_more_stable_than_invocation": {"supported": median_within >= median_cross + .10, "median_within_head_near_far_cosine": median_within, "difference": median_within - median_cross},
        "D4_weight_distinction_survives_nonlinear_suffix": {"supported": sum(value < .30 for value in cross) >= 4, "pairs_below_threshold": sum(value < .30 for value in cross)},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    pair_rows = []
    for i in range(count):
        for j in range(i + 1, count):
            values = site_cosines[(i, j)]
            pair_rows.append({"first": names[i + 1], "second": names[j + 1], "global_cosine": float(cosine[i, j]), "median_site_cosine": float(np.median(values)) if values else None})
    interventions = [{"name": names[i + 1], "mean_kl": float(mean_kl[i]), "effect_norm": float(norm[i])} for i in range(count)]
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "interventions": interventions, "pairs": pair_rows}, indent=2, allow_nan=False))
    lines = [
        "# Same variable, different finite functions", "",
        "The source variable `function_word` is held fixed. In Layer 0, all edges carrying that variable are removed separately from Heads 3, 8, 7, and 6. The complete nonlinear suffix is recomputed. Near (`distance < 4`) and far invocations are also removed separately to compare variation within one native function against variation between functions.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Head function | All-route mean KL | Near/far effect cosine |", "|---|---:|---:|"]
    for head, local in zip(config["heads"], within):
        lines.append(f"| L0H{head} | {mean_kl[index[f'H{head}_all']]:.3e} | {local:.3f} |")
    lines += ["", f"Median absolute cross-head all-route cosine: `{median_cross:.3f}`.", "", "Cross-head all-route pairs:", ""]
    for offset, first in enumerate(config["heads"]):
        for second in config["heads"][offset + 1:]:
            value = float(cosine[index[f"H{first}_all"], index[f"H{second}_all"]])
            lines.append(f"- L0H{first} vs L0H{second}: `{value:+.3f}`")
    lines += [
        "", "## Guardrails", "",
        "Removing all matching edges is a broad finite intervention, not a minimal semantic edit. Different downstream effects can arise from different routing weights, source values, or OV transforms; the weight atlas isolates OV distinction while this experiment checks that distinction survives the suffix. Near/far effects need not align even for one head because they carry different token states, so failure of the within-head comparison would not erase the cross-head distinction.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
