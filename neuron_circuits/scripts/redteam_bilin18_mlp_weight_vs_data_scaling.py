#!/usr/bin/env python3
"""Test whether static Down-mode advantage is a small-data PCA artifact."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_taskblind_mlp_subspace_finite import (
    affine_pca, intervene_write, output_weight_basis, project, random_basis,
    target_product_and_write,
)
from run_bilin18_unseeded_route_end_to_end_causality import empty_metric, finish, full_logits, update_metric


CONFIG = Path("configs/bilin18_mlp_weight_vs_data_scaling_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_mlp_weight_vs_data_scaling_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_MLP_WEIGHT_VS_DATA_SCALING_REDTEAM.md"


def origin_pca(values: np.ndarray, rank: int):
    values = np.asarray(values, dtype=np.float64); gram = values @ values.T
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T)); order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order][:rank]; eigenvectors = eigenvectors[:, order[:rank]]
    return ((eigenvectors.T @ values) / np.sqrt(eigenvalues[:, None].clip(1e-30))).astype(np.float32)


@torch.inference_mode()
def collect_writes(model, corpus, config):
    dimension = model.config.n_embd; wanted = set(config["layers"]); output = {layer: [] for layer in wanted}
    for position in config["positions"]:
        length = position + 1
        for start in range(0, len(corpus), config["batch_size"]):
            raw = np.asarray(corpus[start:start + config["batch_size"], :length], dtype=np.int64); ids = torch.as_tensor(raw, device="cuda")
            x0 = F.rms_norm(model.embedding(ids), (dimension,)); x = x0; bus = None
            for layer, block in enumerate(model.blocks):
                mixed = block.lambdas[0] * x + block.lambdas[1] * x0; details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = details["value_bus"]
                pre = mixed + details["write"]; normalized = F.rms_norm(pre, (dimension,)); product = block.mlp.Left(normalized) * block.mlp.Right(normalized); write = block.mlp.Down(product)
                if layer in wanted: output[layer].append(write[:, -1].float().cpu().numpy())
                x = pre + write + block.mlp.Down_bias
        print(f"write discovery position={position}", flush=True)
    return {layer: np.concatenate(rows) for layer, rows in output.items()}


def main():
    config = json.loads(CONFIG.read_text()); rng = np.random.default_rng(config["seed"]); corpus = np.load(config["corpus"], mmap_mode="r")
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    discovery = collect_writes(model, corpus[config["discovery_start"]:config["discovery_start"] + config["discovery_rows"]], config)
    prepared = {}
    for layer in config["layers"]:
        permutation = rng.permutation(len(discovery[layer])); values = discovery[layer][permutation]; down = model.blocks[layer].mlp.Down.weight.detach().float().cpu().numpy()
        layer_row = {"weight": torch.as_tensor(output_weight_basis(down, max(config["ranks"])), device="cuda"), "random": [torch.as_tensor(random_basis(down.shape[0], 64, rng), device="cuda") for _ in range(config["random_replicates"])]}
        for count in config["discovery_site_counts"]:
            mean, centered = affine_pca(values[:count], max(config["ranks"])); origin = origin_pca(values[:count], max(config["ranks"]))
            layer_row[count] = {"mean": torch.as_tensor(mean, device="cuda"), "centered": torch.as_tensor(centered, device="cuda"), "origin": torch.as_tensor(origin, device="cuda")}
        prepared[layer] = layer_row
    methods = ["native", "deleted"]
    for count in config["discovery_site_counts"]:
        for rank in config["ranks"]: methods += [f"centered_{count}_{rank}", f"origin_{count}_{rank}"]
    for rank in config["ranks"]: methods.append(f"weight_{rank}")
    methods += [f"random64_{index}" for index in range(config["random_replicates"])]
    metrics = {layer: {name: empty_metric() for name in methods} for layer in config["layers"]}; maximum_error = 0.0
    validation = corpus[config["validation_start"]:config["validation_start"] + config["validation_rows"]]
    for position in config["positions"]:
        length = position + 1
        for start in range(0, len(validation), config["batch_size"]):
            raw = np.asarray(validation[start:start + config["batch_size"], :length + 1], dtype=np.int64); ids = torch.as_tensor(raw[:, :length], device="cuda"); targets = torch.as_tensor(raw[:, length], device="cuda"); reference = full_logits(model, ids)
            for layer in config["layers"]:
                _, write = target_product_and_write(model, ids, layer); row = prepared[layer]; conditions = {"native": write, "deleted": torch.zeros_like(write)}
                for count in config["discovery_site_counts"]:
                    fit = row[count]
                    for rank in config["ranks"]:
                        conditions[f"centered_{count}_{rank}"] = project(write, fit["mean"], fit["centered"][:rank])
                        basis = fit["origin"][:rank]; conditions[f"origin_{count}_{rank}"] = (write @ basis.T) @ basis
                for rank in config["ranks"]:
                    basis = row["weight"][:rank]; conditions[f"weight_{rank}"] = (write @ basis.T) @ basis
                for index, basis in enumerate(row["random"]): conditions[f"random64_{index}"] = (write @ basis.T) @ basis
                names, logits = intervene_write(model, ids, layer, conditions); outputs = {name: logits[index] for index, name in enumerate(names)}; maximum_error = max(maximum_error, float((outputs["native"] - reference).abs().max()))
                for name in methods: update_metric(metrics[layer][name], outputs["native"], outputs["deleted"], outputs[name], targets)
        print(f"weight/data scaling validation position={position}", flush=True)
    results = {str(layer): {name: finish(value) for name, value in rows.items()} for layer, rows in metrics.items()}
    gains = []; data_wins = []; mean_close = []; early_compact = []; random_wins = []
    for layer in config["layers"]:
        row = results[str(layer)]; gains.append(row["centered_1024_64"]["causal_fidelity"] - row["centered_64_64"]["causal_fidelity"] >= .10); data_wins.append(row["centered_1024_64"]["causal_fidelity"] - row["weight_64"]["causal_fidelity"] >= .05); mean_close.append(abs(row["centered_1024_64"]["causal_fidelity"] - row["origin_1024_64"]["causal_fidelity"]) < .05)
        if layer in (1, 8, 12): early_compact.append(row["centered_1024_128"]["causal_fidelity"] >= .50)
        p95 = float(np.quantile([row[f"random64_{index}"]["causal_fidelity"] for index in range(config["random_replicates"])], .95)); row["random64_p95"] = p95; random_wins.append(row["centered_1024_64"]["causal_fidelity"] > p95)
    hypotheses = {
        "G0_native_replay_is_exact": {"supported": maximum_error <= .02, "maximum_logit_error": maximum_error},
        "G1_data_pca_improves_with_sample_size": {"supported": sum(gains) >= 3, "passing": int(sum(gains))},
        "G2_large_sample_data_geometry_beats_static_weights": {"supported": sum(data_wins) >= 3, "passing": int(sum(data_wins))},
        "G3_affine_mean_does_not_explain_weight_advantage": {"supported": sum(mean_close) >= 3, "passing": int(sum(mean_close))},
        "G4_early_mid_write_is_moderately_compressible": {"supported": sum(early_compact) >= 2, "passing": int(sum(early_compact))},
        "G5_late_compactness_replicates": {"supported": results["17"]["centered_1024_16"]["causal_fidelity"] >= .95, "fidelity": results["17"]["centered_1024_16"]["causal_fidelity"]},
        "G6_large_sample_data_pca_beats_random": {"supported": all(random_wins), "passing": int(sum(random_wins))},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Data-scaling red team for MLP write geometry", "", "The earlier task-blind finite test used 256 discovery sites and found leading static Down-output modes competitive with or better than data PCA before the final layer. This red team fits centered and through-origin PCA on 64, 256, and 1,024 new discovery sites, then evaluates all bases on a separate 256-site FineWeb panel.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Layer | Centered64 by data size 64/256/1024 | Origin64 64/256/1024 | Centered1024 rank16/64/128 | Weight rank16/64/128 | Random64 p95 |", "|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]; series = lambda prefix, counts: "/".join(f"{row[f'{prefix}_{count}_64']['causal_fidelity']:.3f}" for count in counts); ranks = lambda prefix: "/".join(f"{row[f'{prefix}_{rank}']['causal_fidelity']:.3f}" for rank in config["ranks"])
        lines.append(f"| {layer} | {series('centered', config['discovery_site_counts'])} | {series('origin', config['discovery_site_counts'])} | {ranks('centered_1024')} | {ranks('weight')} | {row['random64_p95']:.3f} |")
    lines += ["", "## Guardrail", "", "Improvement with sample size diagnoses estimation error, not semantic clarity. Static Down modes describe the layer's available output gain; data PCA describes its natural write distribution. Both can be causally useful without being a unique feature basis. All tests preserve the model's nonlinear suffix and are sufficiency relative to deleting the audited write."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
