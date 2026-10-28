#!/usr/bin/env python3
"""Separate source variables, read invocations, and native OV functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_vocabulary import CATEGORIES, discovery_scores, token_categories


CONFIG = Path("configs/bilin18_variable_function_atlas.json")
OUTPUT = TENSOR_ROOT / "bilin18_variable_function_atlas.json"
REPORT = TENSOR_ROOT / "BILIN18_VARIABLE_FUNCTION_ATLAS.md"


def effective_support(weights):
    weights = np.asarray(weights, dtype=np.float64)
    return float(weights.sum() ** 2 / max(np.square(weights).sum(), 1e-30))


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(values_a, values_b):
    return float(np.corrcoef(rankdata(values_a), rankdata(values_b))[0, 1])


def factored_frobenius_cosines(factors):
    count = len(factors)
    gram = np.eye(count, dtype=np.float64)
    norms = []
    for output, value in factors:
        oo = output.T @ output
        vv = value @ value.T
        norms.append(float(torch.sum(oo * vv.T).sqrt()))
    for i in range(count):
        oi, vi = factors[i]
        for j in range(i + 1, count):
            oj, vj = factors[j]
            numerator = float(torch.sum((oi.T @ oj) * (vj @ vi.T).T))
            gram[i, j] = gram[j, i] = numerator / max(norms[i] * norms[j], 1e-30)
    return gram, np.asarray(norms)


def leading_mode(output, value):
    qo, ro = torch.linalg.qr(output, mode="reduced")
    qv, rv = torch.linalg.qr(value.T, mode="reduced")
    u, singular, vh = torch.linalg.svd(ro @ rv.T, full_matrices=False)
    return qo @ u[:, 0], qv @ vh.T[:, 0], singular


def decode_direction(direction, matrix, tokenizer, count):
    score = F.normalize(matrix.float(), dim=1) @ F.normalize(direction.float(), dim=0)
    positive = torch.topk(score, count).indices.tolist()
    negative = torch.topk(-score, count).indices.tolist()
    clean = lambda index: tokenizer.decode([index]).replace("\n", "\\n").replace("\t", "\\t")
    return [clean(index) for index in positive], [clean(index) for index in negative], positive, negative


def pole_purity(indices, categories):
    counts = np.bincount(categories[np.asarray(indices)], minlength=len(CATEGORIES))
    return float(counts.max() / len(indices))


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    categories = token_categories(flags)
    corpus = np.load(config["corpus"], mmap_mode="r")
    discovery = np.asarray(corpus[config["selection_start"]:config["selection_start"] + config["selection_rows"], :config["sequence_length"]], dtype=np.int64)
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    heads = model.config.n_head; distances = len(config["distance_bins"]) - 1
    function_count = len(config["layers"]) * heads
    route_profiles = np.zeros((function_count, distances * len(CATEGORIES)), dtype=np.float64)
    selected_mass = np.zeros((len(CATEGORIES), function_count), dtype=np.float64)
    selected_presence = np.zeros_like(selected_mass, dtype=bool)
    selected_rows = []
    factors = []
    for layer in config["layers"]:
        print("variable/function route scores layer", layer, flush=True)
        probes, _, _ = operator_probe_vectors(model.blocks[layer], config["operator_probes"], config["seed"] + layer)
        local = {**config, "batch_size": config["selection_batch_size"], "category_count": len(CATEGORIES)}
        scores, _ = discovery_scores(model, layer, discovery, categories, local, probes)
        scores = scores.cpu().numpy().reshape(heads, distances, len(CATEGORIES))
        root = np.sqrt(np.maximum(scores, 0))
        route_profiles[layer * heads:(layer + 1) * heads] = root.reshape(heads, -1)
        flat_order = np.argsort(scores.reshape(-1))[::-1][:config["route_budget"]]
        for index in flat_order:
            head, remainder = divmod(int(index), distances * len(CATEGORIES))
            distance, category = divmod(remainder, len(CATEGORIES))
            function = layer * heads + head
            selected_mass[category, function] += scores[head, distance, category]
            selected_presence[category, function] = True
            selected_rows.append({"layer": layer, "head": head, "distance": distance, "category": CATEGORIES[category], "score": float(scores[head, distance, category])})
        block = model.blocks[layer].attn
        head_dim = model.config.n_embd // heads
        for head in range(heads):
            sl = slice(head * head_dim, (head + 1) * head_dim)
            factors.append((block.c_proj.weight[:, sl].detach().float().cpu(), block.c_v.weight[sl].detach().float().cpu()))

    print("factorized OV similarities", flush=True)
    ov_cosine, ov_norm = factored_frobenius_cosines(factors)
    route_norm = np.linalg.norm(route_profiles, axis=1, keepdims=True)
    route_cosine = (route_profiles @ route_profiles.T) / np.maximum(route_norm @ route_norm.T, 1e-30)
    triangle = np.triu_indices(function_count, 1)
    correlation = spearman(route_cosine[triangle], np.abs(ov_cosine[triangle]))
    layer_ids = np.arange(function_count) // heads
    same_layer = layer_ids[triangle[0]] == layer_ids[triangle[1]]
    within_layer_correlation = spearman(
        route_cosine[triangle][same_layer], np.abs(ov_cosine[triangle])[same_layer],
    )
    cross_layer_correlation = spearman(
        route_cosine[triangle][~same_layer], np.abs(ov_cosine[triangle])[~same_layer],
    )
    stratum_correlations = {}
    for first in config["layers"]:
        for second in config["layers"]:
            if second < first:
                continue
            mask = (
                ((layer_ids[triangle[0]] == first) & (layer_ids[triangle[1]] == second))
                | ((layer_ids[triangle[0]] == second) & (layer_ids[triangle[1]] == first))
            )
            if mask.sum() >= 8:
                stratum_correlations[f"{first}-{second}"] = spearman(
                    route_cosine[triangle][mask], np.abs(ov_cosine[triangle])[mask],
                )
    variable_rows = []
    pair_cosines = []
    for category, name in enumerate(CATEGORIES):
        functions = np.flatnonzero(selected_presence[category])
        local_pairs = [abs(ov_cosine[i, j]) for offset, i in enumerate(functions) for j in functions[offset + 1:]]
        pair_cosines.extend(local_pairs)
        variable_rows.append({
            "variable": name,
            "native_functions": int(len(functions)),
            "effective_functions": effective_support(selected_mass[category, functions]),
            "median_abs_ov_cosine": float(np.median(local_pairs)) if local_pairs else None,
            "functions": [{"layer": int(index // heads), "head": int(index % heads)} for index in functions],
        })
    function_rows = []
    mode_purities = []
    embedding = model.embedding.weight.detach().float().cpu()[:tokenizer.vocab_size]
    unembedding = model.lm_head.weight.detach().float().cpu()[:tokenizer.vocab_size]
    function_route_mass = selected_mass.sum(0)
    representative = np.argsort(function_route_mass)[::-1][:12]
    for index, (output, value) in enumerate(factors):
        output_mode, input_mode, singular = leading_mode(output, value)
        input_pos, input_neg, input_pos_ids, input_neg_ids = decode_direction(input_mode, embedding, tokenizer, config["decode_tokens"])
        output_pos, output_neg, output_pos_ids, output_neg_ids = decode_direction(output_mode, unembedding, tokenizer, config["decode_tokens"])
        purities = [pole_purity(ids, categories) for ids in (input_pos_ids, input_neg_ids, output_pos_ids, output_neg_ids)]
        mode_purities.extend(purities)
        cats = np.flatnonzero(selected_presence[:, index])
        function_rows.append({
            "layer": int(index // heads), "head": int(index % heads),
            "selected_variables": [CATEGORIES[value] for value in cats],
            "selected_variable_count": int(len(cats)),
            "route_mass": float(function_route_mass[index]),
            "ov_norm": float(ov_norm[index]),
            "leading_singular_fraction": float(singular[0].square() / singular.square().sum()),
            "input_positive": input_pos, "input_negative": input_neg,
            "output_positive": output_pos, "output_negative": output_neg,
            "pole_purities": purities,
        })
    rng = np.random.default_rng(config["seed"])
    random_purities = []
    for _ in range(2000):
        ids = rng.choice(len(categories), config["decode_tokens"], replace=False)
        random_purities.append(pole_purity(ids, categories))
    random_p95 = float(np.quantile(random_purities, .95))
    median_effective = float(np.median([row["effective_functions"] for row in variable_rows]))
    used_functions = [row["selected_variable_count"] for row in function_rows if row["selected_variable_count"]]
    median_variables = float(np.median(used_functions))
    median_pair_cosine = float(np.median(pair_cosines))
    median_mode_purity = float(np.median(mode_purities))
    hypotheses = {
        "F1_variables_are_reused_across_functions": {"supported": median_effective >= 4, "median_effective_functions": median_effective},
        "F2_functions_are_reused_across_variables": {"supported": median_variables >= 5, "median_variables": median_variables},
        "F3_same_variable_uses_distinct_transforms": {"supported": median_pair_cosine < .20, "median_abs_ov_cosine": median_pair_cosine},
        "F4_routing_similarity_predicts_transform_similarity": {"supported": correlation >= .20, "spearman": correlation},
        "F5_native_head_modes_are_lexically_readable": {"supported": median_mode_purity >= random_p95 + .10, "median_purity": median_mode_purity, "random_p95": random_p95},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    redteam = {
        "within_layer_spearman": within_layer_correlation,
        "cross_layer_spearman": cross_layer_correlation,
        "layer_pair_spearman": stratum_correlations,
        "median_layer_pair_spearman": float(np.median(list(stratum_correlations.values()))),
    }
    payload = {"config": config, "hypotheses": hypotheses, "redteam": redteam, "variables": variable_rows, "functions": function_rows, "selected_routes": selected_rows}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = [
        "# Variable/function atlas for early attention", "",
        "The human token class is treated as a source variable. `(layer, head)` is treated as a native value/write function, while distance and category determine its route invocations. Each function is represented weight-only by its exact current-value OV map `W_O,h W_V,h`; route selection uses the frozen operator-effect score on FineWeb.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "", "## Main measurements", "",
        f"- Median effective functions per variable: `{median_effective:.2f}`.",
        f"- Median variables per used function: `{median_variables:.1f}`.",
        f"- Median within-variable absolute OV cosine: `{median_pair_cosine:.4f}`.",
        f"- Route-profile/OV-similarity Spearman correlation: `{correlation:.3f}`.",
        f"- Post-hoc layer-control red team: within-layer `{within_layer_correlation:.3f}`, cross-layer `{cross_layer_correlation:.3f}`, median layer-pair `{redteam['median_layer_pair_spearman']:.3f}`.",
        f"- Leading-mode pole category purity: median `{median_mode_purity:.3f}`, random p95 `{random_p95:.3f}`.",
        "", "## Representative high-mass native functions", "",
    ]
    for index in representative:
        row = function_rows[int(index)]
        lines.append(f"- **L{row['layer']}H{row['head']}** reads `{', '.join(row['selected_variables'])}`; input + `{row['input_positive'][:5]}`, input − `{row['input_negative'][:5]}`; output + `{row['output_positive'][:5]}`, output − `{row['output_negative'][:5]}`.")
    lines += [
        "", "## Guardrails", "",
        "A native head is a convenient exact function boundary, not necessarily an MDL-optimal function. The OV map omits both squared-QK routing factors, RMS normalization, the persistent Layer-0 value bus, and downstream nonlinear computation. Token decoding of singular vectors is descriptive and can miss a distributed semantic direction. Low OV cosine establishes distinct linear transforms, not distinct human meanings. The layer-controlled correlations are post-hoc red teams; if they collapse, the preregistered pooled correlation was a layer-identity confound. Causal output-vector interventions are the next validation step.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "representative": [f"L{function_rows[int(i)]['layer']}H{function_rows[int(i)]['head']}" for i in representative]}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
