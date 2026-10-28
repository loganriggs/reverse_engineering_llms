#!/usr/bin/env python3
"""Finite causal comparison of task-blind MLP coordinate systems."""

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
from run_bilin18_stratified_natural_mlp_census import bank
from run_bilin18_taskblind_mlp_multiview_atlas import canonical_channel_scale
from run_bilin18_unseeded_route_end_to_end_causality import empty_metric, finish, full_logits, update_metric


CONFIG = Path("configs/bilin18_taskblind_mlp_subspace_finite.json")
OUTPUT = TENSOR_ROOT / "bilin18_taskblind_mlp_subspace_finite.json"
REPORT = TENSOR_ROOT / "BILIN18_TASKBLIND_MLP_SUBSPACE_FINITE.md"


def affine_pca(values: np.ndarray, rank: int):
    values = np.asarray(values, dtype=np.float64); mean = values.mean(0); centered = values - mean
    gram = centered @ centered.T; eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T)); order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]; eigenvectors = eigenvectors[:, order]
    keep = min(rank, int((eigenvalues > 1e-10 * max(eigenvalues[0], 1e-30)).sum()))
    eigenvalues = eigenvalues[:keep]; eigenvectors = eigenvectors[:, :keep]
    basis = (eigenvectors.T @ centered) / np.sqrt(eigenvalues[:, None].clip(1e-30))
    return mean.astype(np.float32), basis.astype(np.float32)


def output_weight_basis(down: np.ndarray, rank: int):
    gram = np.asarray(down, dtype=np.float64) @ np.asarray(down, dtype=np.float64).T
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T)); return vectors[:, np.argsort(values)[::-1][:rank]].T.astype(np.float32)


def project(values: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor):
    centered = values - mean; return mean + (centered @ basis.T) @ basis


def random_basis(dimension: int, rank: int, rng: np.random.Generator):
    values = rng.standard_normal((dimension, rank)); q, _ = np.linalg.qr(values); return q.T.astype(np.float32)


@torch.inference_mode()
def intervene_write(model, ids, target_layer, writes_by_condition):
    dimension = model.config.n_embd; x0 = F.rms_norm(model.embedding(ids), (dimension,)); x = x0; bus = None; destination = ids.shape[1] - 1
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0; details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = details["value_bus"]
        pre = mixed + details["write"]; normalized = F.rms_norm(pre, (dimension,)); product = block.mlp.Left(normalized) * block.mlp.Right(normalized); mlp_write = block.mlp.Down(product)
        if layer != target_layer:
            x = pre + mlp_write + block.mlp.Down_bias; continue
        names = list(writes_by_condition); states = []
        for name in names:
            modified = mlp_write.clone(); modified[:, destination] = writes_by_condition[name].to(modified.dtype)
            states.append(pre + modified + block.mlp.Down_bias)
        count = len(names); x = torch.cat(states); x0 = x0.repeat(count, 1, 1); bus = bus.repeat(count, 1, 1, 1) if bus.ndim == 4 else bus.repeat(count, 1, 1)
        for later in model.blocks[layer + 1:]:
            mixed = later.lambdas[0] * x + later.lambdas[1] * x0; later_details = attention_details(later.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = later_details["value_bus"]
            pre = mixed + later_details["write"]; normalized = F.rms_norm(pre, (dimension,)); x = pre + later.mlp.Down(later.mlp.Left(normalized) * later.mlp.Right(normalized)) + later.mlp.Down_bias
        raw = model.lm_head(F.rms_norm(x, (dimension,))[:, destination]).float(); logits = 30 * torch.tanh(raw / 30)
        return names, logits.reshape(count, len(ids), -1)
    raise ValueError(target_layer)


@torch.inference_mode()
def target_product_and_write(model, ids, target_layer):
    dimension = model.config.n_embd; x0 = F.rms_norm(model.embedding(ids), (dimension,)); x = x0; bus = None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0; details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = details["value_bus"]
        pre = mixed + details["write"]; normalized = F.rms_norm(pre, (dimension,)); product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer == target_layer: return product[:, -1].float(), block.mlp.Down(product[:, -1]).float()
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    raise ValueError(target_layer)


def main():
    config = json.loads(CONFIG.read_text()); source = np.load(config["source"]); rng = np.random.default_rng(config["seed"])
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    prepared = {}
    for layer in config["layers"]:
        products = source[f"discovery_products_L{layer}"].astype(np.float32); contributions = source[f"discovery_contributions_L{layer}"].astype(np.float32)
        down = model.blocks[layer].mlp.Down.weight.detach().float().cpu().numpy(); scale = canonical_channel_scale(down).astype(np.float32); allocation = products * scale; writes = products @ down.T
        alloc_mean, alloc_max = affine_pca(allocation, max(config["ranks"])); write_mean, write_max = affine_pca(writes, max(config["ranks"])); weight_max = output_weight_basis(down, max(config["ranks"]))
        prepared[layer] = {
            "scale": torch.as_tensor(scale, device="cuda"), "alloc_mean": torch.as_tensor(alloc_mean, device="cuda"), "alloc_basis": torch.as_tensor(alloc_max, device="cuda"),
            "write_mean": torch.as_tensor(write_mean, device="cuda"), "write_basis": torch.as_tensor(write_max, device="cuda"), "weight_basis": torch.as_tensor(weight_max, device="cuda"),
            "bank": torch.as_tensor(bank(contributions, np.ones(len(contributions), dtype=bool), config["channel_bank_size"]), device="cuda"),
            "random": {rank: [torch.as_tensor(random_basis(down.shape[0], rank, rng), device="cuda") for _ in range(config["random_replicates"])] for rank in config["random_ranks"]},
        }
    methods = ["native", "deleted", "mean_write", "channel_bank"]
    for rank in config["ranks"]: methods += [f"write_pca_{rank}", f"allocation_pca_{rank}", f"weight_output_{rank}"]
    for rank in config["random_ranks"]:
        methods += [f"random_write_{rank}_{index}" for index in range(config["random_replicates"])]
    metrics = {layer: {name: empty_metric() for name in methods} for layer in config["layers"]}; maximum_error = 0.0
    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"]]
    for position in config["positions"]:
        length = position + 1
        for start in range(0, len(corpus), config["batch_size"]):
            raw = np.asarray(corpus[start:start + config["batch_size"], :length + 1], dtype=np.int64); ids = torch.as_tensor(raw[:, :length], device="cuda"); targets = torch.as_tensor(raw[:, length], device="cuda"); reference = full_logits(model, ids)
            for layer in config["layers"]:
                product, write = target_product_and_write(model, ids, layer); row = prepared[layer]; allocation = product * row["scale"]
                conditions = {"native": write, "deleted": torch.zeros_like(write), "mean_write": row["write_mean"].expand_as(write)}
                kept = torch.zeros_like(product); kept[:, row["bank"]] = product[:, row["bank"]]; conditions["channel_bank"] = model.blocks[layer].mlp.Down(kept)
                for rank in config["ranks"]:
                    conditions[f"write_pca_{rank}"] = project(write, row["write_mean"], row["write_basis"][:rank])
                    alloc_projected = project(allocation, row["alloc_mean"], row["alloc_basis"][:rank]); conditions[f"allocation_pca_{rank}"] = model.blocks[layer].mlp.Down(alloc_projected / row["scale"])
                    conditions[f"weight_output_{rank}"] = (write @ row["weight_basis"][:rank].T) @ row["weight_basis"][:rank]
                for rank in config["random_ranks"]:
                    for index, basis in enumerate(row["random"][rank]): conditions[f"random_write_{rank}_{index}"] = project(write, row["write_mean"], basis)
                names, logits = intervene_write(model, ids, layer, conditions); outputs = {name: logits[index] for index, name in enumerate(names)}; maximum_error = max(maximum_error, float((outputs["native"] - reference).abs().max()))
                for name in methods: update_metric(metrics[layer][name], outputs["native"], outputs["deleted"], outputs[name], targets)
        print(f"finite task-blind MLP subspaces position={position}", flush=True)
    results = {str(layer): {name: finish(value) for name, value in rows.items()} for layer, rows in metrics.items()}
    bank_wins = []; random_wins = []; distinct = []; data_wins = []; monotone = []
    for layer in config["layers"]:
        row = results[str(layer)]; bank_wins.append(row["write_pca_64"]["causal_fidelity"] - row["channel_bank"]["causal_fidelity"] >= .15)
        random_p95 = float(np.quantile([row[f"random_write_64_{index}"]["causal_fidelity"] for index in range(config["random_replicates"])], .95)); row["random_write_64_p95"] = random_p95; random_wins.append(row["write_pca_64"]["causal_fidelity"] > random_p95)
        distinct.append(abs(row["write_pca_64"]["causal_fidelity"] - row["allocation_pca_64"]["causal_fidelity"]) >= .10)
        data_wins.append(row["write_pca_64"]["causal_fidelity"] - row["weight_output_64"]["causal_fidelity"] >= .05)
        values = [row[f"write_pca_{rank}"]["causal_fidelity"] for rank in config["ranks"]]; monotone.append(all(second + 1e-6 >= first for first, second in zip(values, values[1:])))
    hypotheses = {
        "F0_native_replay_is_exact": {"supported": maximum_error <= .02, "maximum_logit_error": maximum_error},
        "F1_dense_write_subspaces_beat_native_banks": {"supported": sum(bank_wins) >= 3, "passing": int(sum(bank_wins))},
        "F2_write_subspaces_beat_random_controls": {"supported": all(random_wins), "passing": int(sum(random_wins))},
        "F3_write_and_allocation_are_distinct_causal_ontologies": {"supported": sum(distinct) >= 2, "passing": int(sum(distinct))},
        "F4_late_write_is_compact": {"supported": results["17"]["write_pca_16"]["causal_fidelity"] >= .80, "fidelity": results["17"]["write_pca_16"]["causal_fidelity"]},
        "F5_data_geometry_beats_static_weight_geometry": {"supported": sum(data_wins) >= 3, "passing": int(sum(data_wins))},
        "F6_rank_curves_are_monotone": {"supported": all(monotone), "passing": int(sum(monotone))},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Finite causal comparison of task-blind MLP coordinate systems", "", "At fresh FineWeb sites, the destination MLP write is replaced by four kinds of equal-rank approximation and the complete nonlinear suffix recomputes: dense PCA of exact writes, PCA of symmetry-canonical native allocation followed through Down, leading output modes of the static Down weight, and matched random residual subspaces. A frozen 128-native-channel bank is an additional sparse-coordinate baseline.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Layer | Bank128 | Exact-write PCA 4/16/64/128 | Allocation PCA 4/16/64/128 | Weight modes 4/16/64/128 | Random64 p95 |", "|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]; fmt = lambda prefix: "/".join(f"{row[f'{prefix}_{rank}']['causal_fidelity']:.3f}" for rank in config["ranks"])
        lines.append(f"| {layer} | {row['channel_bank']['causal_fidelity']:.3f} | {fmt('write_pca')} | {fmt('allocation_pca')} | {fmt('weight_output')} | {row['random_write_64_p95']:.3f} |")
    lines += ["", "## Guardrail", "", "These are sufficiency tests relative to deleting one destination-position MLP write, not necessity tests in the intact model. Exact-write PCA is an external communication ontology; allocation PCA is an internal code ontology; Down singular modes are a static capability ontology. A winner is useful for its stated purpose and need not be the unique or most semantic basis."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
