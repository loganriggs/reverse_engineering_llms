#!/usr/bin/env python3
"""Compress semantic head payloads with weight-derived OV singular modes."""

from __future__ import annotations

import itertools
import json
import math
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
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_semantic_construction_head_programs import suffix, trace_to_site
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_semantic_payload_svd.json")
OUTPUT = TENSOR_ROOT / "bilin18_semantic_payload_svd.json"
REPORT = TENSOR_ROOT / "BILIN18_SEMANTIC_PAYLOAD_SVD.md"
FIGURE = TENSOR_ROOT / "bilin18_semantic_payload_svd.png"


def shapley(scores, names):
    count = len(names); result = {}
    for factor_index, name in enumerate(names):
        value = torch.zeros_like(next(iter(scores.values()))); others = [index for index in range(count) if index != factor_index]
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


def ov_modes(model, layer, program, seed):
    block = model.blocks[layer]; heads = model.config.n_head; head_dim = model.config.n_embd // heads
    projection = block.attn.c_proj.weight.detach().T.reshape(heads, head_dim, model.config.n_embd).float(); value_weight = block.attn.c_v.weight.detach().reshape(heads, head_dim, model.config.n_embd).float(); coefficient = float((1 - block.attn.lamb).detach())
    result = {}
    for head in program:
        matrix = coefficient * value_weight[head].T @ projection[head]
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        generator = torch.Generator(device=matrix.device); generator.manual_seed(seed + 1000 * layer + head)
        # The effective OV map has rank at most one head dimension. Random
        # controls choose among those nonzero-capacity modes rather than from
        # the hundreds of structurally zero singular directions.
        permutation = torch.randperm(head_dim, generator=generator, device=matrix.device)
        result[head] = {"u": u, "s": s, "vh": vh, "random_order": permutation}
    return result, projection


def approximate_current(delta_normalized, modes, program, rank, random=False):
    outputs = []
    for head in program:
        row = modes[head]; indices = row["random_order"][:rank] if random else torch.arange(rank, device=delta_normalized.device)
        left = row["u"][:, indices] * row["s"][indices]
        matrix = left @ row["vh"][indices]
        outputs.append(delta_normalized @ matrix)
    return torch.stack(outputs, 1)


@torch.inference_mode()
def collect(model, raw, records, contrast, layer, program, ranks, modes, projection, batch_size):
    names = ["routing", "current", "bus"]; worlds = tuple(itertools.product((0, 1), repeat=3))
    targets = []; top_values = {rank: [] for rank in ranks}; random_values = {rank: [] for rank in ranks}; factor_values = {name: [] for name in names}
    max_logit_error = max_factor_error = 0.0
    block = model.blocks[layer]; coefficient_current = float(1 - block.attn.lamb); coefficient_bus = float(block.attn.lamb)
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]; size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch]); changed = original.copy()
        for index, row in enumerate(batch): changed[index, row[3]] = row[5]
        ids = torch.as_tensor(np.concatenate((original, changed)), device="cuda"); trace, _ = trace_to_site(model, ids, layer)
        direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
        base_state = trace["output"][:size]; carried = trace["mixed"][size:, 127] - trace["mixed"][:size, 127]; mlp = trace["mlp"][size:, 127] - trace["mlp"][:size, 127]
        no_attention = base_state.clone(); no_attention[:, 127] = base_state[:, 127] + carried + mlp
        details = trace["details"]; p0 = details["pattern"][:size, program, 127]; p1 = details["pattern"][size:, program, 127]
        current0 = details["current_value"][:size, :, program].permute(0, 2, 1, 3); current1 = details["current_value"][size:, :, program].permute(0, 2, 1, 3)
        bus0 = details["value_bus"][:size, :, program].permute(0, 2, 1, 3); bus1 = details["value_bus"][size:, :, program].permute(0, 2, 1, 3)
        base_value = coefficient_current * current0 + coefficient_bus * bus0
        delta_current_value = coefficient_current * (current1 - current0); delta_bus_value = coefficient_bus * (bus1 - bus0)
        proj = projection[program]
        def write(pattern, value): return torch.einsum("bhs,bhsd,hdk->bk", pattern, value, proj)
        base_write = write(p0, base_value)
        world_vectors = {}
        for world in worlds:
            pattern = p1 if world[0] else p0; value = base_value + world[1] * delta_current_value + world[2] * delta_bus_value
            world_vectors[world] = write(pattern, value) - base_write
        states = []
        for world in worlds:
            state = no_attention.clone(); state[:, 127] += world_vectors[world]; states.append(state)
        # SVD approximations keep exact changed routing and bus delta while compressing current-value transport.
        normalized = F.rms_norm(trace["mixed"], (model.config.n_embd,)); delta_normalized = normalized[size:] - normalized[:size]
        approximations = []
        for random in (False, True):
            for rank in ranks:
                approx_current_write = approximate_current(delta_normalized, modes, program, rank, random=random)
                current_edge = torch.einsum("bhs,bhsd->bhd", p1, approx_current_write).sum(1)
                route_edge = write(p1, base_value) - base_write; bus_edge = write(p1, delta_bus_value)
                state = no_attention.clone(); state[:, 127] += route_edge + current_edge + bus_edge; approximations.append(state)
        all_states = states + approximations
        logits = suffix(model, layer, torch.cat(all_states), torch.cat([trace["x0"][:size]] * len(all_states)), torch.cat([trace["bus"][:size]] * len(all_states))).reshape(len(all_states), size, -1)
        scores = {world: logits[index] @ contrast for index, world in enumerate(worlds)}; base_score = scores[(0, 0, 0)]; target_score = scores[(1, 1, 1)] - base_score; target = direction * target_score; targets.append(target.cpu())
        phi = shapley(scores, names); max_factor_error = max(max_factor_error, float((sum(phi.values()) - target_score).norm() / target_score.norm().clamp_min(1e-30)))
        for name in names: factor_values[name].append((direction * phi[name]).cpu())
        offset = len(worlds)
        for kind, destination in (("top", top_values), ("random", random_values)):
            for rank in ranks:
                value = direction * ((logits[offset] @ contrast) - base_score); destination[rank].append(value.cpu()); offset += 1
        max_rank = max(ranks); max_index = len(worlds) + ranks.index(max_rank)
        max_logit_error = max(max_logit_error, float((logits[max_index] - logits[worlds.index((1, 1, 1))]).abs().max()))
    target = torch.cat(targets)
    return {"examples": len(target), "top": {str(rank): metric(target, torch.cat(top_values[rank])) for rank in ranks}, "random": {str(rank): metric(target, torch.cat(random_values[rank])) for rank in ranks}, "factors": {name: metric(target, torch.cat(factor_values[name])) for name in names}, "errors": {"max_rank_logit_abs": max_logit_error, "factor_shapley_relative": max_factor_error}}


def main():
    config = json.loads(CONFIG.read_text()); operand_config = json.loads(Path(config["source_config"]).read_text()); head_red_config = json.loads(Path(operand_config["source_config"]).read_text()); head_config = json.loads(Path(head_red_config["source_config"]).read_text()); source = json.loads(Path(config["source_results"]).read_text())
    construction_config = json.loads(Path(head_config["source_config"]).read_text()); carrier_config = json.loads(Path(construction_config["source_config"]).read_text()); base = json.loads(Path(carrier_config["source_config"]).read_text()); family_source = json.loads(Path(carrier_config["family_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); corpus = np.load(base["corpus"], mmap_mode="r"); counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    family_pairs = {"spelling": token_pairs(tokenizer, base["spelling_pairs"])}; family_pairs.update({name: [tuple(pair) for pair in pairs] for name, pairs in family_source["families"].items()})
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval(); results = []
    for family_index, family in enumerate(head_config["sites"]):
        layer = head_config["sites"][family]; program = source["summary"][family]["head_order"][:source["summary"][family]["discovery_k90"]]; modes, projection = ov_modes(model, layer, program, config["seed"])
        pairs = family_pairs[family]; midpoint = len(pairs) // 2; folds = ((pairs[:midpoint], pairs[midpoint:]), (pairs[midpoint:], pairs[:midpoint]))
        for fold, (targets, cues) in enumerate(folds):
            contrast = make_contrasts(model.lm_head.weight.shape[0], targets, 0, head_config["seed"] + 100 * family_index + fold, "cuda")[:, 0]
            for split, (begin, end) in enumerate(base["splits"]):
                raw = np.asarray(corpus[begin:end, :128], dtype=np.int64); records = select_records(raw, boundary_flags, cues, counts, 127, head_config["examples_per_label_per_split"], head_config["seed"] + 1000 * family_index + 10 * fold + split)
                row = collect(model, raw, records, contrast, layer, program, config["ranks"], modes, projection, head_config["batch_size"]); row.update({"family": family, "layer": layer, "program": program, "fold": fold, "split": split}); results.append(row); print(f"payload {family} fold={fold} split={split}", flush=True)
    summary = {}
    for family in head_config["sites"]:
        rows = [row for row in results if row["family"] == family]
        summary[family] = {"layer": rows[0]["layer"], "program": rows[0]["program"], "top_fidelity": {str(rank): float(np.mean([row["top"][str(rank)]["energy_fidelity"] for row in rows])) for rank in config["ranks"]}, "random_fidelity": {str(rank): float(np.mean([row["random"][str(rank)]["energy_fidelity"] for row in rows])) for rank in config["ranks"]}, "factors": {name: float(np.mean([row["factors"][name]["projection_share"] for row in rows])) for name in ("routing", "current", "bus")}}
        summary[family]["k90"] = next((rank for rank in config["ranks"] if summary[family]["top_fidelity"][str(rank)] >= .90), None)
    max_errors = {name: max(row["errors"][name] for row in results) for name in ("max_rank_logit_abs", "factor_shapley_relative")}
    rank4_advantage = {family: row["top_fidelity"]["4"] - row["random_fidelity"]["4"] for family, row in summary.items()}
    hypotheses = {
        "V0_full_rank_recomposes": {"supported": max_errors["max_rank_logit_abs"] <= 1e-4, "observed": max_errors},
        "V1_rank4_payload_is_often_sufficient": {"supported": sum(row["top_fidelity"]["4"] >= .80 for row in summary.values()) >= 3, "observed": {family: row["top_fidelity"]["4"] for family, row in summary.items()}},
        "V2_payload_programs_are_low_rank": {"supported": sum(row["k90"] is not None and row["k90"] <= 8 for row in summary.values()) >= 3, "observed": {family: row["k90"] for family, row in summary.items()}},
        "V3_weight_outliers_beat_random_modes": {"supported": sum(value >= .20 for value in rank4_advantage.values()) >= 3, "observed": rank4_advantage},
        "V4_current_values_not_persistent_bus_dominate": {"supported": sum(row["factors"]["current"] >= .70 for row in summary.values()) >= 3, "observed": {family: row["factors"] for family, row in summary.items()}},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summary": summary, "results": results}, indent=2, allow_nan=False))
    lines = ["# Weight-derived low-rank compression of semantic value payloads", "", "For each frozen semantic construction program, the current-value OV map is decomposed by SVD using weights only. Leading singular modes and matched random singular modes reconstruct the changed current-value payload while retaining exact routing and persistent-bus change. A separate finite factorial allocates routing, current-value, and persistent-bus changes.", "", "## Locked decisions", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{row['observed']}`" for name, row in hypotheses.items())
    lines += ["", "![Payload SVD](bilin18_semantic_payload_svd.png)", "", "| Family/site | Program | k90/head | Rank-4 top | Rank-4 random | Routing | Current value | Bus |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for family, row in summary.items(): lines.append(f"| {family} L{row['layer']} | {', '.join('H'+str(h) for h in row['program'])} | {row['k90'] if row['k90'] is not None else '—'} | {row['top_fidelity']['4']:.3f} | {row['random_fidelity']['4']:.3f} | {row['factors']['routing']:.3f} | {row['factors']['current']:.3f} | {row['factors']['bus']:.3f} |")
    lines += ["", "## Interpretation boundary", "", "Low rank here means that a few large singular modes of the native OV tensor suffice for this finite semantic payload under the original suffix. It does not make those modes uniquely semantic: singular modes are weight-defined capabilities and may carry unrelated payloads elsewhere. Random-mode controls test spectral ordering, while the routing/current/bus factorial distinguishes architectural carriers at the selected invocation."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, (family, row) in zip(axes.flat, summary.items()):
        axis.plot(config["ranks"], [row["top_fidelity"][str(rank)] for rank in config["ranks"]], marker="o", label="leading modes"); axis.plot(config["ranks"], [row["random_fidelity"][str(rank)] for rank in config["ranks"]], marker="o", label="random modes"); axis.set_xscale("log", base=2); axis.axhline(.9, color="black", linestyle="--", linewidth=.8); axis.set(title=f"{family} L{row['layer']}", xlabel="OV rank per head", ylabel="Output energy fidelity", xticks=config["ranks"]); axis.get_xaxis().set_major_formatter(plt.ScalarFormatter()); axis.legend(fontsize=8); axis.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summary": summary}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
