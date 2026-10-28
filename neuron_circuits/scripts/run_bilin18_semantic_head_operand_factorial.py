#!/usr/bin/env python3
"""Exact source-role and QK/value factorials for semantic construction heads."""

from __future__ import annotations

import itertools
import json
import math
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
from run_bilin18_semantic_construction_head_programs import suffix, trace_to_site
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_semantic_head_operand_factorial.json")
OUTPUT = TENSOR_ROOT / "bilin18_semantic_head_operand_factorial.json"
REPORT = TENSOR_ROOT / "BILIN18_SEMANTIC_HEAD_OPERAND_FACTORIAL.md"
FIGURE = TENSOR_ROOT / "bilin18_semantic_head_operand_factorial.png"


def shapley(scores, names):
    count = len(names); result = {}
    for factor_index, name in enumerate(names):
        value = torch.zeros_like(next(iter(scores.values())))
        others = [index for index in range(count) if index != factor_index]
        for size in range(count):
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            for subset in itertools.combinations(others, size):
                absent = [0] * count
                for index in subset: absent[index] = 1
                present = absent.copy(); present[factor_index] = 1
                value += weight * (scores[tuple(present)] - scores[tuple(absent)])
        result[name] = value
    return result


def metric(target, value):
    target, value = target.double(), value.double(); energy = target.square().sum().clamp_min(1e-30)
    return {"projection_share": float((target * value).sum() / energy), "energy_fidelity": float(1 - (target - value).square().sum() / energy), "direction_agreement": float((target * value > 0).double().mean())}


def masks(length, cue, boundary, device):
    positions = torch.arange(length, device=device)[None]
    cue, boundary = cue[:, None], boundary[:, None]
    return {
        "before_cue": positions < cue,
        "cue": positions == cue,
        "between": (positions > cue) & (positions < boundary),
        "boundary": positions == boundary,
        "after_boundary": (positions > boundary) & (positions < length - 1),
        "destination": positions == length - 1,
    }


def program_edge_worlds(model, layer, details, size, program, group_masks):
    heads = model.config.n_head; head_dim = model.config.n_embd // heads; destination = 127
    pattern = details["pattern"][:, :, destination]
    values = details["mixed_value"].permute(0, 2, 1, 3)
    projection = model.blocks[layer].attn.c_proj.weight.T.reshape(heads, head_dim, model.config.n_embd)
    p0, p1 = pattern[:size, program], pattern[size:, program]
    v0, v1 = values[:size, program], values[size:, program]
    def edge(p, v): return torch.einsum("bhsd,hdk->bhsk", p[..., None] * v, projection[program])
    base_edge, route_edge, value_edge, full_edge = edge(p0, v0), edge(p1, v0), edge(p0, v1), edge(p1, v1)
    delta = full_edge - base_edge
    groups = {name: (delta * mask[:, None, :, None]).sum((1, 2)) for name, mask in group_masks.items()}
    qkv = {
        (0, 0): torch.zeros_like(groups[next(iter(groups))]),
        (1, 0): (route_edge - base_edge).sum((1, 2)),
        (0, 1): (value_edge - base_edge).sum((1, 2)),
        (1, 1): delta.sum((1, 2)),
    }
    return groups, qkv


@torch.inference_mode()
def collect(model, raw, records, contrast, layer, program, groups, batch_size):
    worlds = tuple(itertools.product((0, 1), repeat=len(groups)))
    targets, allocated = [], {name: [] for name in groups}; qkv_allocated = {name: [] for name in ("routing", "value")}
    max_state_abs = max_output_relative = max_qkv_relative = 0.0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]; size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch]); changed = original.copy()
        for index, row in enumerate(batch): changed[index, row[3]] = row[5]
        ids = torch.as_tensor(np.concatenate((original, changed)), device="cuda")
        trace, _ = trace_to_site(model, ids, layer)
        cue = torch.as_tensor([row[3] for row in batch], device="cuda"); boundary = torch.as_tensor([row[1] for row in batch], device="cuda")
        if not bool((cue < boundary).all()): raise AssertionError("operand partition requires cue before latest boundary")
        direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
        base_state = trace["output"][:size]; carried = trace["mixed"][size:, 127] - trace["mixed"][:size, 127]; mlp = trace["mlp"][size:, 127] - trace["mlp"][:size, 127]
        no_attention = base_state.clone(); no_attention[:, 127] = base_state[:, 127] + carried + mlp
        group_vectors, qkv_vectors = program_edge_worlds(model, layer, trace["details"], size, program, masks(128, cue, boundary, ids.device))
        attention_program = qkv_vectors[(1, 1)]
        grouped = sum(group_vectors.values())
        max_state_abs = max(max_state_abs, float((grouped - attention_program).abs().max()))
        # Chunk the 64 group coalitions to limit suffix memory.
        score_parts = {}
        for chunk_start in range(0, len(worlds), 8):
            chunk = worlds[chunk_start:chunk_start + 8]; states = []
            for world in chunk:
                state = no_attention.clone(); state[:, 127] += sum(world[index] * group_vectors[name] for index, name in enumerate(groups)); states.append(state)
            logits = suffix(model, layer, torch.cat(states), torch.cat([trace["x0"][:size]] * len(states)), torch.cat([trace["bus"][:size]] * len(states))).reshape(len(states), size, -1)
            for index, world in enumerate(chunk): score_parts[world] = logits[index] @ contrast
        phi = shapley(score_parts, groups); target_score = score_parts[(1,) * len(groups)] - score_parts[(0,) * len(groups)]
        max_output_relative = max(max_output_relative, float((sum(phi.values()) - target_score).norm() / target_score.norm().clamp_min(1e-30)))
        target = direction * target_score; targets.append(target.cpu())
        for name in groups: allocated[name].append((direction * phi[name]).cpu())
        qkv_states = []
        for world in ((0, 0), (1, 0), (0, 1), (1, 1)):
            state = no_attention.clone(); state[:, 127] += qkv_vectors[world]; qkv_states.append(state)
        logits = suffix(model, layer, torch.cat(qkv_states), torch.cat([trace["x0"][:size]] * 4), torch.cat([trace["bus"][:size]] * 4)).reshape(4, size, -1)
        qkv_scores = {world: logits[index] @ contrast for index, world in enumerate(((0, 0), (1, 0), (0, 1), (1, 1)))}
        qkv_phi = shapley(qkv_scores, ("routing", "value")); qkv_target = qkv_scores[(1, 1)] - qkv_scores[(0, 0)]
        max_qkv_relative = max(max_qkv_relative, float((sum(qkv_phi.values()) - qkv_target).norm() / qkv_target.norm().clamp_min(1e-30)))
        for name in qkv_allocated: qkv_allocated[name].append((direction * qkv_phi[name]).cpu())
    target = torch.cat(targets)
    return {"examples": len(target), "groups": {name: metric(target, torch.cat(values)) for name, values in allocated.items()}, "qkv": {name: metric(target, torch.cat(values)) for name, values in qkv_allocated.items()}, "errors": {"state_max_abs": max_state_abs, "group_shapley_relative": max_output_relative, "qkv_shapley_relative": max_qkv_relative}}


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b); return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main():
    config = json.loads(CONFIG.read_text()); red_config = json.loads(Path(config["source_config"]).read_text()); head_config = json.loads(Path(red_config["source_config"]).read_text()); source = json.loads(Path(config["source_results"]).read_text())
    construction_config = json.loads(Path(head_config["source_config"]).read_text()); carrier_config = json.loads(Path(construction_config["source_config"]).read_text()); base = json.loads(Path(carrier_config["source_config"]).read_text()); family_source = json.loads(Path(carrier_config["family_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); corpus = np.load(base["corpus"], mmap_mode="r"); counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    family_pairs = {"spelling": token_pairs(tokenizer, base["spelling_pairs"])}; family_pairs.update({name: [tuple(pair) for pair in pairs] for name, pairs in family_source["families"].items()})
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    results = []
    for family_index, family in enumerate(head_config["sites"]):
        pairs = family_pairs[family]; midpoint = len(pairs) // 2; folds = ((pairs[:midpoint], pairs[midpoint:]), (pairs[midpoint:], pairs[:midpoint]))
        program = source["summary"][family]["head_order"][:source["summary"][family]["discovery_k90"]]
        for fold, (targets, cues) in enumerate(folds):
            contrast = make_contrasts(model.lm_head.weight.shape[0], targets, 0, head_config["seed"] + 100 * family_index + fold, "cuda")[:, 0]
            for split, (begin, end) in enumerate(base["splits"]):
                raw = np.asarray(corpus[begin:end, :128], dtype=np.int64); records = select_records(raw, boundary_flags, cues, counts, 127, head_config["examples_per_label_per_split"], head_config["seed"] + 1000 * family_index + 10 * fold + split)
                row = collect(model, raw, records, contrast, head_config["sites"][family], program, config["groups"], head_config["batch_size"]); row.update({"family": family, "layer": head_config["sites"][family], "program": program, "fold": fold, "split": split}); results.append(row)
                print(f"operand {family} fold={fold} split={split}", flush=True)
    summary = {}
    for family in head_config["sites"]:
        rows = [row for row in results if row["family"] == family]
        profiles = [[row["groups"][name]["projection_share"] for name in config["groups"]] for row in rows]
        pairwise = [cosine(profiles[i], profiles[j]) for i in range(len(profiles)) for j in range(i + 1, len(profiles))]
        summary[family] = {"layer": head_config["sites"][family], "program": rows[0]["program"], "groups": {name: float(np.mean([row["groups"][name]["projection_share"] for row in rows])) for name in config["groups"]}, "qkv": {name: float(np.mean([row["qkv"][name]["projection_share"] for row in rows])) for name in ("routing", "value")}, "profile_pairwise_cosine_median": float(np.median(pairwise)), "profile_pairwise_cosines": pairwise}
    max_errors = {name: max(row["errors"][name] for row in results) for name in ("state_max_abs", "group_shapley_relative", "qkv_shapley_relative")}
    dominant = {family: max(abs(value) for value in row["groups"].values()) for family, row in summary.items()}
    boundary_advantage = {family: summary["spelling"]["groups"]["boundary"] - row["groups"]["boundary"] for family, row in summary.items() if family != "spelling"}
    hypotheses = {
        "E0_edge_factorials_are_exact": {"supported": max_errors["group_shapley_relative"] <= 1e-5 and max_errors["qkv_shapley_relative"] <= 1e-5, "observed": max_errors},
        "E1_programs_read_localized_operand_roles": {"supported": sum(value >= .40 for value in dominant.values()) >= 3, "observed": dominant},
        "E2_spelling_is_boundary_exceptional": {"supported": all(value >= .20 for value in boundary_advantage.values()), "observed": boundary_advantage},
        "E3_operand_grammars_transfer": {"supported": all(row["profile_pairwise_cosine_median"] >= .80 for row in summary.values()), "observed": {family: row["profile_pairwise_cosine_median"] for family, row in summary.items()}},
        "E4_contextual_value_change_usually_dominates": {"supported": sum(row["qkv"]["value"] >= .60 for row in summary.values()) >= 3, "observed": {family: row["qkv"] for family, row in summary.items()}},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summary": summary, "results": results}, indent=2, allow_nan=False))
    lines = ["# Operand grammar and QK/value factorial of semantic construction heads", "", "Each frozen head program's exact destination write is split into six disjoint source-position roles using the tensor attention edge product. All 64 role coalitions pass through the nonlinear suffix for exact Shapley allocation. A second 2×2 factorial separates changed QK routing from changed contextual values.", "", "## Locked decisions", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{row['observed']}`" for name, row in hypotheses.items())
    lines += ["", "![Operand factorial](bilin18_semantic_head_operand_factorial.png)", "", "| Family/site | Program | Before cue | Cue | Between | Boundary | After boundary | Destination | Routing | Value | Profile stability |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for family, row in summary.items(): lines.append(f"| {family} L{row['layer']} | {', '.join('H'+str(h) for h in row['program'])} | " + " | ".join(f"{row['groups'][name]:.3f}" for name in config["groups"]) + f" | {row['qkv']['routing']:.3f} | {row['qkv']['value']:.3f} | {row['profile_pairwise_cosine_median']:.3f} |")
    lines += ["", "## Interpretation boundary", "", "A source role is a relational operand in these counterfactual contexts, not a semantic label for every token state the head can read. QK and value changes are endogenous to the same upstream edit; Shapley allocation answers which changed tensor factor expresses the finite program effect, not which parameter independently caused it. Stable role profiles support reusable invocation grammar, while semantic payload still requires held-out behavioral transfer."]
    REPORT.write_text("\n".join(lines) + "\n")
    families = list(summary); x = np.arange(len(families)); bottom = np.zeros(len(families)); fig, axis = plt.subplots(figsize=(12, 5.5), layout="constrained")
    for name in config["groups"]:
        values = np.asarray([summary[family]["groups"][name] for family in families]); axis.bar(x, values, bottom=bottom, label=name); bottom += values
    axis.axhline(1, color="black", linewidth=.8); axis.set(xticks=x, xticklabels=[f"{f} L{summary[f]['layer']}" for f in families], ylabel="Shapley projection share", title="Relational operands of compact semantic head programs"); axis.legend(fontsize=8, ncol=3); axis.grid(axis="y", alpha=.2); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summary": summary}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
