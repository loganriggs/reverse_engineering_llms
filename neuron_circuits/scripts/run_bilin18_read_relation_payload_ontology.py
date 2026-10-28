#!/usr/bin/env python3
"""Determine whether validated read relations share or diverge in payload function."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_contextual_read_function_discovery import edge_predicates, edge_template, frequency_quantiles
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_prospective_read_relation_test import candidate_masks, matched_samples, strata
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_vocabulary import token_categories
from run_bilin18_variable_function_atlas import decode_direction, factored_frobenius_cosines, leading_mode


CONFIG = Path("configs/bilin18_read_relation_payload_ontology.json")
OUTPUT = TENSOR_ROOT / "bilin18_read_relation_payload_ontology.json"
REPORT = TENSOR_ROOT / "BILIN18_READ_RELATION_PAYLOAD_ONTOLOGY.md"
FIGURE = TENSOR_ROOT / "bilin18_read_relation_payload_ontology.png"


def edge_records(flat_indices, positions):
    source_template, destination_template = edge_template(positions); edges = len(source_template)
    rows, local = divmod(np.asarray(flat_indices), edges)
    return rows, source_template[local], destination_template[local]


@torch.inference_mode()
def payload_effects(model, raw, flat_indices, layer, head, config):
    row_indices, sources, destinations = edge_records(flat_indices, config["positions"])
    dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    writes, deltas = [], []
    for start in range(0, len(row_indices), config["batch_size"]):
        selected_rows = row_indices[start:start + config["batch_size"]]
        ids = torch.as_tensor(raw[selected_rows, :config["sequence_length"]], device="cuda")
        source = torch.as_tensor(sources[start:start + config["batch_size"]], device="cuda")
        destination = torch.as_tensor(destinations[start:start + config["batch_size"]], device="cuda")
        rows = torch.arange(len(ids), device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for block_index, block in enumerate(model.blocks):
            if block_index < layer:
                x, bus = block(x, bus, x0); continue
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)[head]
            selected_value = details["pattern"][rows, head, destination, source, None].float() * details["mixed_value"][rows, source, head].float()
            selected_write = selected_value @ projection
            modified_attention = details["write"].float().clone(); modified_attention[rows, destination] -= selected_write
            attention = torch.cat([details["write"].float(), modified_attention], 0)
            x0 = torch.cat([x0.float(), x0.float()], 0); mixed = torch.cat([mixed.float(), mixed.float()], 0)
            bus = torch.cat([details["value_bus"].float(), details["value_bus"].float()], 0)
            pre = mixed + attention; state = F.rms_norm(pre, (dimension,))
            x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
            for suffix in model.blocks[block_index + 1:]: x, bus = suffix(x, bus, x0)
            rows2 = torch.arange(2 * len(ids), device="cuda"); destination2 = torch.cat([destination, destination])
            raw_logits = model.lm_head(F.rms_norm(x, (dimension,))[rows2, destination2]).float()
            logits = (30 * torch.tanh(raw_logits / 30)).reshape(2, len(ids), -1)
            writes.append(selected_write.cpu()); deltas.append((logits[0] - logits[1]).cpu())
            break
    return torch.cat(writes), torch.cat(deltas)


def normalized_svd(matrix):
    matrix = F.normalize(matrix.float(), dim=1)
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    energy = singular.square(); probability = energy / energy.sum().clamp_min(1e-30)
    effective_rank = float(1 / probability.square().sum())
    return singular, vh, effective_rank


def cross_capture(train, test, rank):
    _, basis, _ = normalized_svd(train)
    test = F.normalize(test.float(), dim=1)
    return float((test @ basis[:rank].T).square().sum() / test.square().sum().clamp_min(1e-30))


def subspace_overlap(first, second, rank):
    _, basis_a, _ = normalized_svd(first); _, basis_b, _ = normalized_svd(second)
    singular = torch.linalg.svdvals(basis_a[:rank] @ basis_b[:rank].T)
    return float(singular.square().mean())


def cosine(first, second):
    return float(F.cosine_similarity(first.float().reshape(1, -1), second.float().reshape(1, -1)).item())


def decode_vector(vector, tokenizer, count):
    positive = torch.topk(vector, count).indices.tolist(); negative = torch.topk(-vector, count).indices.tolist()
    clean = lambda index: tokenizer.decode([index]).replace("\n", "\\n").replace("\t", "\\t")
    return [clean(index) for index in positive], [clean(index) for index in negative]


def matrix_summary(matrix, ranks):
    singular, _, effective_rank = normalized_svd(matrix)
    energy = singular.square(); total = energy.sum().clamp_min(1e-30)
    return {"effective_rank": effective_rank, "energy": {str(rank): float(energy[:rank].sum() / total) for rank in ranks}}


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(config["corpus"], mmap_mode="r")
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    human = token_categories(flags); frequency_bins, _ = frequency_quantiles(corpus, tokenizer.vocab_size)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    collected = {program["name"]: {} for program in config["programs"]}
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
            count = min(config["edges_per_split"], len(positive), len(negative)); chosen = generator.choice(len(positive), count, replace=False)
            prepared[candidate] = (positive[chosen], negative[chosen])
        for program in config["programs"]:
            print("payload", split_index, program, flush=True)
            positive, negative = prepared[program["candidate"]]
            candidate_write, candidate_delta = payload_effects(model, raw, positive, program["layer"], program["head"], config)
            control_write, control_delta = payload_effects(model, raw, negative, program["layer"], program["head"], config)
            collected[program["name"]][str(split_index)] = {"candidate_write": candidate_write, "candidate_delta": candidate_delta, "control_write": control_write, "control_delta": control_delta}

    results = {}
    for program in config["programs"]:
        name = program["name"]; result = {"splits": {}}
        for split in ("0", "1"):
            values = collected[name][split]
            mean = values["candidate_delta"].mean(0); positive, negative = decode_vector(mean, tokenizer, config["decode_tokens"])
            result["splits"][split] = {
                "candidate_write": matrix_summary(values["candidate_write"], config["subspace_ranks"]),
                "control_write": matrix_summary(values["control_write"], config["subspace_ranks"]),
                "candidate_output": matrix_summary(values["candidate_delta"], config["subspace_ranks"]),
                "control_output": matrix_summary(values["control_delta"], config["subspace_ranks"]),
                "mean_output_positive": positive, "mean_output_negative": negative,
            }
        result["write_cross_capture_16"] = [cross_capture(collected[name]["0"]["candidate_write"], collected[name]["1"]["candidate_write"], 16), cross_capture(collected[name]["1"]["candidate_write"], collected[name]["0"]["candidate_write"], 16)]
        result["output_cross_capture_16"] = [cross_capture(collected[name]["0"]["candidate_delta"], collected[name]["1"]["candidate_delta"], 16), cross_capture(collected[name]["1"]["candidate_delta"], collected[name]["0"]["candidate_delta"], 16)]
        result["mean_output_split_cosine"] = cosine(collected[name]["0"]["candidate_delta"].mean(0), collected[name]["1"]["candidate_delta"].mean(0))
        results[name] = result

    first, second = "function_L7H8", "function_L9H7"
    pair_metrics = {}
    for split in ("0", "1"):
        a, b = collected[first][split], collected[second][split]
        pair_metrics[split] = {
            "write_subspace_overlap_16": subspace_overlap(a["candidate_write"], b["candidate_write"], 16),
            "output_subspace_overlap_16": subspace_overlap(a["candidate_delta"], b["candidate_delta"], 16),
            "mean_paired_write_cosine": float(F.cosine_similarity(a["candidate_write"].float(), b["candidate_write"].float(), dim=1).mean()),
            "mean_paired_output_cosine": float(F.cosine_similarity(a["candidate_delta"].float(), b["candidate_delta"].float(), dim=1).mean()),
        }
    factors = []
    for program in config["programs"][:2]:
        block = model.blocks[program["layer"]].attn; heads = model.config.n_head; head_dim = model.config.n_embd // heads; sl = slice(program["head"] * head_dim, (program["head"] + 1) * head_dim)
        factors.append((block.c_proj.weight[:, sl].detach().float().cpu(), block.c_v.weight[sl].detach().float().cpu()))
    ov_cosine, _ = factored_frobenius_cosines(factors); pair_metrics["exact_ov_cosine"] = float(ov_cosine[0, 1])

    # Weight-only leading modes give a separate, context-free description of each native transform.
    embedding = model.embedding.weight.detach().float().cpu()[:tokenizer.vocab_size]; unembedding = model.lm_head.weight.detach().float().cpu()[:tokenizer.vocab_size]
    weight_modes = {}
    for program in config["programs"]:
        block = model.blocks[program["layer"]].attn; heads = model.config.n_head; head_dim = model.config.n_embd // heads; sl = slice(program["head"] * head_dim, (program["head"] + 1) * head_dim)
        output = block.c_proj.weight[:, sl].detach().float().cpu(); value = block.c_v.weight[sl].detach().float().cpu()
        output_mode, input_mode, singular = leading_mode(output, value)
        input_pos, input_neg, _, _ = decode_direction(input_mode, embedding, tokenizer, config["decode_tokens"])
        output_pos, output_neg, _, _ = decode_direction(output_mode, unembedding, tokenizer, config["decode_tokens"])
        weight_modes[program["name"]] = {"leading_energy_fraction": float(singular[0].square() / singular.square().sum()), "input_positive": input_pos, "input_negative": input_neg, "output_positive": output_pos, "output_negative": output_neg}

    write_replicates = all(min(results[p["name"]]["write_cross_capture_16"]) >= .60 for p in config["programs"])
    output_replicates = all(min(results[p["name"]]["output_cross_capture_16"]) >= .40 for p in config["programs"])
    mean_output_cosines = [pair_metrics[split]["mean_paired_output_cosine"] for split in ("0", "1")]
    overlaps = [pair_metrics[split]["output_subspace_overlap_16"] for split in ("0", "1")]
    distinct = abs(pair_metrics["exact_ov_cosine"]) < .50 and max(overlaps) < .50 and max(mean_output_cosines) < .30
    same = min(overlaps) >= .70 and min(mean_output_cosines) >= .50
    compressible = 0
    for program in config["programs"]:
        if all(results[program["name"]]["splits"][split]["candidate_output"]["effective_rank"] < results[program["name"]]["splits"][split]["control_output"]["effective_rank"] for split in ("0", "1")): compressible += 1
    hypotheses = {
        "O1_payload_write_subspaces_replicate": {"supported": write_replicates},
        "O2_output_effect_subspaces_replicate": {"supported": output_replicates},
        "O3_shared_read_has_distinct_functions": {"supported": distinct},
        "O4_shared_read_has_same_function": {"supported": same},
        "O5_candidate_payload_is_more_compressible": {"supported": compressible >= 2, "count": compressible},
        "O6_boundary_payload_is_stable": {"supported": results["boundary_L9H8"]["mean_output_split_cosine"] >= .50, "cosine": results["boundary_L9H8"]["mean_output_split_cosine"]},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results, "shared_function_pair": pair_metrics, "weight_modes": weight_modes}, indent=2, allow_nan=False))
    lines = ["# Payload ontology of validated read relations", "", "This separates the validated address variable from its payload function. For each candidate edge, the exact native edge write and the finite final-logit difference caused by removing that edge are collected. Subspaces are learned independently on two corpus splits. L7H8 and L9H7 are evaluated on the same adjacent-function-word edges.", "", "## Locked results", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "## Program summaries", "", "| Program | write cross-capture | output cross-capture | mean-output split cosine | candidate/control output effective rank |", "|---|---:|---:|---:|---:|"]
    for program in config["programs"]:
        row = results[program["name"]]; effective = "/".join(f"{row['splits'][split]['candidate_output']['effective_rank']:.1f}/{row['splits'][split]['control_output']['effective_rank']:.1f}" for split in ("0", "1"))
        lines.append(f"| {program['name']} | {min(row['write_cross_capture_16']):.3f} | {min(row['output_cross_capture_16']):.3f} | {row['mean_output_split_cosine']:.3f} | {effective} |")
    lines += ["", "## Same address variable, two functions", "", f"Exact L7H8/L9H7 OV cosine: `{pair_metrics['exact_ov_cosine']:.3f}`."]
    for split in ("0", "1"):
        row = pair_metrics[split]; lines.append(f"- Split {split}: write-subspace overlap `{row['write_subspace_overlap_16']:.3f}`, output-subspace overlap `{row['output_subspace_overlap_16']:.3f}`, paired write cosine `{row['mean_paired_write_cosine']:.3f}`, paired output cosine `{row['mean_paired_output_cosine']:.3f}`.")
    lines += ["", "## Empirical output poles", ""]
    for program in config["programs"]:
        row = results[program["name"]]["splits"]["0"]
        lines.append(f"- **{program['name']}** promotes `{row['mean_output_positive'][:8]}` and suppresses `{row['mean_output_negative'][:8]}` in its split-0 mean finite effect.")
    lines += ["", "## Guardrails", "", "A low-dimensional effect subspace is a compact behavior family, not necessarily a human concept. Mean logit poles can cancel context-dependent roles, while singular subspaces omit signs and may combine distinct uses. Exact OV cosine is weight-only but ignores QK addressing, normalization, and downstream nonlinear computation. Cross-split capture tests reusable geometry; it does not establish that the decoded token poles are a complete semantic explanation."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    names = [p["name"] for p in config["programs"]]; x = np.arange(len(names))
    axes[0].bar(x - .18, [min(results[n]["write_cross_capture_16"]) for n in names], width=.36, label="write"); axes[0].bar(x + .18, [min(results[n]["output_cross_capture_16"]) for n in names], width=.36, label="output"); axes[0].set(xticks=x, xticklabels=names, ylabel="Cross-split rank-16 capture", title="Reusable payload geometry"); axes[0].tick_params(axis="x", rotation=25); axes[0].legend()
    labels = ["OV cosine", "write overlap s0", "write overlap s1", "output overlap s0", "output overlap s1", "paired output s0", "paired output s1"]
    values = [pair_metrics["exact_ov_cosine"], pair_metrics["0"]["write_subspace_overlap_16"], pair_metrics["1"]["write_subspace_overlap_16"], pair_metrics["0"]["output_subspace_overlap_16"], pair_metrics["1"]["output_subspace_overlap_16"], pair_metrics["0"]["mean_paired_output_cosine"], pair_metrics["1"]["mean_paired_output_cosine"]]
    axes[1].bar(np.arange(len(values)), values); axes[1].axhline(0, color="black", linewidth=.8); axes[1].set(xticks=np.arange(len(values)), xticklabels=labels, ylabel="Similarity", title="L7H8 versus L9H7"); axes[1].tick_params(axis="x", rotation=55)
    for axis in axes: axis.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "pair": pair_metrics}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
