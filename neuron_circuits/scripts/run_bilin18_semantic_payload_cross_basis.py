#!/usr/bin/env python3
"""Causally compare family-specific and shared OV input-mode bases."""

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
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_semantic_construction_head_programs import suffix, trace_to_site
from run_bilin18_semantic_payload_svd import ov_modes
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_semantic_payload_cross_basis.json")
OUTPUT = TENSOR_ROOT / "bilin18_semantic_payload_cross_basis.json"
REPORT = TENSOR_ROOT / "BILIN18_SEMANTIC_PAYLOAD_CROSS_BASIS.md"
FIGURE = TENSOR_ROOT / "bilin18_semantic_payload_cross_basis.png"


def metric(target, value):
    target, value = target.double(), value.double(); energy = target.square().sum().clamp_min(1e-30)
    return {"projection_share": float((target * value).sum() / energy), "energy_fidelity": float(1 - (target - value).square().sum() / energy)}


def family_basis(modes, program, total_rank):
    per_head = total_rank // len(program); extras = total_rank % len(program); columns = []
    for index, head in enumerate(program): columns.append(modes[head]["u"][:, :per_head + int(index < extras)])
    q, _ = torch.linalg.qr(torch.cat(columns, dim=1), mode="reduced")
    return q[:, :total_rank]


def target_matrices(model, layer, program):
    block = model.blocks[layer]; heads = model.config.n_head; head_dim = model.config.n_embd // heads
    projection = block.attn.c_proj.weight.detach().T.reshape(heads, head_dim, model.config.n_embd).float(); value = block.attn.c_v.weight.detach().reshape(heads, head_dim, model.config.n_embd).float(); coefficient = float((1 - block.attn.lamb).detach())
    matrices = torch.stack([coefficient * value[head].T @ projection[head] for head in program])
    return matrices, projection[program], coefficient, float(block.attn.lamb.detach())


@torch.inference_mode()
def collect(model, raw, records, contrast, layer, program, bases, matrices, projection, coefficients, batch_size):
    current_coefficient, bus_coefficient = coefficients; targets = []; values = {name: [] for name in bases}; max_exact_error = 0.0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]; size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch]); changed = original.copy()
        for index, row in enumerate(batch): changed[index, row[3]] = row[5]
        ids = torch.as_tensor(np.concatenate((original, changed)), device="cuda"); trace, _ = trace_to_site(model, ids, layer); details = trace["details"]
        direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
        base_state = trace["output"][:size]; carried = trace["mixed"][size:, 127] - trace["mixed"][:size, 127]; mlp = trace["mlp"][size:, 127] - trace["mlp"][:size, 127]
        no_attention = base_state.clone(); no_attention[:, 127] = base_state[:, 127] + carried + mlp
        p0 = details["pattern"][:size, program, 127]; p1 = details["pattern"][size:, program, 127]
        current0 = details["current_value"][:size, :, program].permute(0, 2, 1, 3); bus0 = details["value_bus"][:size, :, program].permute(0, 2, 1, 3); bus1 = details["value_bus"][size:, :, program].permute(0, 2, 1, 3)
        base_value = current_coefficient * current0 + bus_coefficient * bus0; delta_bus = bus_coefficient * (bus1 - bus0)
        def write(pattern, value): return torch.einsum("bhs,bhsd,hdk->bk", pattern, value, projection)
        base_write = write(p0, base_value); route = write(p1, base_value) - base_write; bus = write(p1, delta_bus)
        normalized = F.rms_norm(trace["mixed"], (model.config.n_embd,)); delta = normalized[size:] - normalized[:size]
        exact_current = torch.einsum("bsd,hdk->bhsk", delta, matrices)
        exact = route + torch.einsum("bhs,bhsd->bd", p1, exact_current) + bus
        states = [no_attention]
        full = no_attention.clone(); full[:, 127] += exact; states.append(full)
        for basis in bases.values():
            projected = (delta @ basis) @ basis.T
            current = torch.einsum("bsd,hdk->bhsk", projected, matrices)
            vector = route + torch.einsum("bhs,bhsd->bd", p1, current) + bus
            state = no_attention.clone(); state[:, 127] += vector; states.append(state)
        logits = suffix(model, layer, torch.cat(states), torch.cat([trace["x0"][:size]] * len(states)), torch.cat([trace["bus"][:size]] * len(states))).reshape(len(states), size, -1)
        base_score = logits[0] @ contrast; target = direction * ((logits[1] @ contrast) - base_score); targets.append(target.cpu())
        for index, name in enumerate(bases, 2): values[name].append((direction * ((logits[index] @ contrast) - base_score)).cpu())
        exact_attention = trace["attention"][size:, 127] - trace["attention"][:size, 127]
        selected_exact = torch.einsum("bhs,bhsd,hdk->bk", p1, current_coefficient * (details["current_value"][size:, :, program].permute(0,2,1,3) - current0) + delta_bus, projection) + route
        max_exact_error = max(max_exact_error, float((selected_exact - exact).abs().max()))
    target = torch.cat(targets)
    return {name: metric(target, torch.cat(parts)) for name, parts in values.items()}, max_exact_error


def overlap(first, second): return float((first.T @ second).square().sum() / min(first.shape[1], second.shape[1]))


def main():
    config = json.loads(CONFIG.read_text()); svd_config = json.loads(Path(config["source_config"]).read_text()); operand_config = json.loads(Path(svd_config["source_config"]).read_text()); head_red_config = json.loads(Path(operand_config["source_config"]).read_text()); head_config = json.loads(Path(head_red_config["source_config"]).read_text()); source = json.loads(Path(config["source_results"]).read_text())
    construction_config = json.loads(Path(head_config["source_config"]).read_text()); carrier_config = json.loads(Path(construction_config["source_config"]).read_text()); base = json.loads(Path(carrier_config["source_config"]).read_text()); family_source = json.loads(Path(carrier_config["family_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); corpus = np.load(base["corpus"], mmap_mode="r"); counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    family_pairs = {"spelling": token_pairs(tokenizer, base["spelling_pairs"])}; family_pairs.update({name: [tuple(pair) for pair in pairs] for name, pairs in family_source["families"].items()})
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval(); families = list(head_config["sites"])
    bases = {}; target_specs = {}
    for family in families:
        layer = head_config["sites"][family]; program = source["summary"][family]["head_order"][:source["summary"][family]["discovery_k90"]]; modes, _ = ov_modes(model, layer, program, config["seed"]); bases[family] = family_basis(modes, program, config["total_input_rank"]); matrices, projection, current, bus = target_matrices(model, layer, program); target_specs[family] = (layer, program, matrices, projection, current, bus)
    generator = torch.Generator(device="cuda"); generator.manual_seed(config["seed"]); random_raw = torch.randn(model.config.n_embd, config["total_input_rank"], generator=generator, device="cuda"); random_basis, _ = torch.linalg.qr(random_raw, mode="reduced"); all_bases = {**bases, "random": random_basis}
    results = []; max_error = 0.0
    for family_index, family in enumerate(families):
        layer, program, matrices, projection, current, bus = target_specs[family]; pairs = family_pairs[family]; midpoint = len(pairs) // 2; folds = ((pairs[:midpoint], pairs[midpoint:]), (pairs[midpoint:], pairs[:midpoint]))
        for fold, (targets, cues) in enumerate(folds):
            contrast = make_contrasts(model.lm_head.weight.shape[0], targets, 0, head_config["seed"] + 100 * family_index + fold, "cuda")[:, 0]
            for split, (begin, end) in enumerate(base["splits"]):
                raw = np.asarray(corpus[begin:end, :128], dtype=np.int64); records = select_records(raw, boundary_flags, cues, counts, 127, head_config["examples_per_label_per_split"], head_config["seed"] + 1000 * family_index + 10 * fold + split)
                row, error = collect(model, raw, records, contrast, layer, program, all_bases, matrices, projection, (current, bus), head_config["batch_size"]); max_error = max(max_error, error); results.append({"family": family, "fold": fold, "split": split, "metrics": row}); print(f"cross basis {family} fold={fold} split={split}", flush=True)
    matrix = {target: {donor: float(np.mean([row["metrics"][donor]["energy_fidelity"] for row in results if row["family"] == target])) for donor in all_bases} for target in families}
    overlaps = {first: {second: overlap(bases[first], bases[second]) for second in families} for first in families}
    margins = {target: matrix[target][target] - max(matrix[target][donor] for donor in families if donor != target) for target in families}
    off_overlap, off_fidelity = [], []
    for target in families:
        for donor in families:
            if target != donor: off_overlap.append(overlaps[target][donor]); off_fidelity.append(matrix[target][donor])
    correlation = float(np.corrcoef(off_overlap, off_fidelity)[0,1])
    hypotheses = {
        "B0_own_weight_basis_is_sufficient": {"supported": all(matrix[f][f] >= .85 for f in families), "observed": {f: matrix[f][f] for f in families}},
        "B1_payload_bases_are_family_specific": {"supported": sum(value >= .20 for value in margins.values()) >= 3, "observed": margins},
        "B2_payload_input_subspaces_are_geometrically_distinct": {"supported": all(overlaps[a][b] < .30 for i,a in enumerate(families) for b in families[i+1:]), "observed": overlaps},
        "B3_geometric_overlap_predicts_causal_cross_capture": {"supported": correlation >= .50, "observed": correlation},
        "B4_own_bases_beat_random": {"supported": all(matrix[f][f] >= matrix[f]["random"] + .30 for f in families), "observed": {f: matrix[f][f] - matrix[f]["random"] for f in families}},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "maximum_exact_current_error": max_error, "causal_fidelity": matrix, "subspace_overlap": overlaps, "results": results}, indent=2, allow_nan=False))
    lines = ["# Cross-family reuse of weight-derived semantic payload bases", "", "Each family supplies an eight-dimensional residual-input basis from the leading left singular modes of its frozen OV program. Every basis is applied to every target family's changed residual states while retaining that target's heads, routing, output map, bus change, and nonlinear suffix. This tests coordinate reuse rather than swapping whole circuits.", "", "## Locked decisions", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{row['observed']}`" for name, row in hypotheses.items())
    lines += ["", "![Cross-basis fidelity](bilin18_semantic_payload_cross_basis.png)", "", "## Causal cross-capture", "", "| Target \\ donor | " + " | ".join(families + ["random"]) + " |", "|---|" + "---:|" * (len(families)+1)]
    for target in families: lines.append("| " + target + " | " + " | ".join(f"{matrix[target][donor]:.3f}" for donor in families + ["random"]) + " |")
    lines += ["", "## Geometric projector overlap", "", "| Basis | " + " | ".join(families) + " |", "|---|" + "---:|"*len(families)]
    for first in families: lines.append("| " + first + " | " + " | ".join(f"{overlaps[first][second]:.3f}" for second in families) + " |")
    lines += ["", "## Interpretation boundary", "", "Diagonal superiority would support family-specific residual payload coordinates; broad cross-capture would support a shared generic payload basis. These are weight-derived input modes of selected OV maps, not globally semantic directions. Cross-layer comparison is meaningful because all blocks share the residual coordinate system, but layerwise normalization and state distributions can still change how the same direction is occupied."]
    REPORT.write_text("\n".join(lines) + "\n")
    array = np.asarray([[matrix[target][donor] for donor in families + ["random"]] for target in families]); fig, axis = plt.subplots(figsize=(8,6), layout="constrained"); image = axis.imshow(array, vmin=min(0,float(array.min())), vmax=1, cmap="viridis"); axis.set(xticks=np.arange(len(families)+1), xticklabels=families+["random"], yticks=np.arange(len(families)), yticklabels=families, xlabel="Donor input basis", ylabel="Target semantic program", title="Finite causal cross-basis fidelity");
    for i in range(len(families)):
        for j in range(len(families)+1): axis.text(j,i,f"{array[i,j]:.2f}",ha="center",va="center",color="white" if array[i,j]<.5 else "black")
    fig.colorbar(image, ax=axis); fig.savefig(FIGURE,dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "matrix": matrix}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
