#!/usr/bin/env python3
"""Does a sparse output basis also sparsify upstream causal computation?"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin12_layer01_frozen_taylor_behavior import logits_for_deltas
from run_bilin18_layer1_cross_position_role_redteam import cross_gradient, native_logits
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, mlp_vjp, rms_vjp
from run_bilin18_layer1_supervised_coalition_causality import product_at
from run_bilin18_layer17_tangent_output_semantics import vocabulary_table
from run_bilin18_output_objective_sparse_discovery import fit_sparse_bases
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_output_basis_upstream_compactness.json")
OUTPUT = TENSOR_ROOT / "bilin18_output_basis_upstream_compactness.json"
REPORT = TENSOR_ROOT / "BILIN18_OUTPUT_BASIS_UPSTREAM_COMPACTNESS.md"
FIGURE = TENSOR_ROOT / "bilin18_output_basis_upstream_compactness.png"


def batched_final_adjoint(model, final_state, position, contrasts):
    """One exact final-state adjoint per column of ``contrasts``."""
    dimension = final_state.shape[-1]
    normalized = F.rms_norm(final_state.float(), (dimension,))
    weight = model.lm_head.weight.float()[:contrasts.shape[0]]
    raw = normalized[:, position] @ weight.T
    cap = 1 - torch.tanh(raw / 30).square()
    local = (contrasts.T * cap[0][None]) @ weight
    adjoint = torch.zeros(contrasts.shape[1], *final_state.shape, device=final_state.device)
    adjoint[:, 0, position] = local
    return rms_vjp(final_state.float(), adjoint)


def batched_source_gradients(model, trace, position, contrasts, source_layers):
    """Compose exact MLP+attention VJPs once for a complete objective basis."""
    current = batched_final_adjoint(model, trace[-1]["output"], position, contrasts)
    return propagate_source_adjoints(model, trace, position, current, source_layers)


def propagate_source_adjoints(model, trace, position, current, source_layers):
    """Transport an already constructed batch of final-state adjoints."""
    wanted = set(source_layers); results = {}
    for layer in range(len(model.blocks) - 1, 0, -1):
        block = model.blocks[layer]
        updated = current + mlp_vjp(block, trace[layer]["pre"].float(), current)
        previous = trace[layer - 1]["output"].float(); x0 = trace[layer]["x0"].float()
        mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
        updated += attention_vjp(block, mixed, trace[layer - 1]["value_bus"], updated)
        current = block.lambdas[0].float() * updated
        source = layer - 1
        if source in wanted:
            results[source] = current[:, 0, position] @ model.blocks[source].mlp.Down.weight.float()
    return results


def top_fraction(values, rank, squared=False):
    values = values.abs()
    if squared: values = values.square()
    return values.topk(rank, dim=-1).values.sum(-1) / values.sum(-1).clamp_min(1e-30)


def effective_support(values):
    values = values.abs()
    return values.sum(-1).square() / values.square().sum(-1).clamp_min(1e-30)


def pearson(first, second):
    first = np.asarray(first, dtype=np.float64); second = np.asarray(second, dtype=np.float64)
    if len(first) < 2 or first.std() < 1e-30 or second.std() < 1e-30: return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def aggregate(records, layer, basis):
    rows = [row["metrics"][basis] for row in records if row["layer"] == layer]
    keys = rows[0]
    return {key: float(np.median([row[key] for row in rows])) for key in keys}


def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False)
    axes = core["output"][:, :config["rank"]].float().cuda()
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    weight_scores = (model.lm_head.weight.float()[:tokenizer.vocab_size] @ axes).detach().cpu()
    fit_mask = np.arange(tokenizer.vocab_size) % 2 == 0
    generator = torch.Generator().manual_seed(config["seed"])
    sparse, optimization = fit_sparse_bases(weight_scores[fit_mask], generator, {**config, "ica_starts": 5, "ica_iterations": 1000})
    random_qs = [torch.linalg.qr(torch.randn(config["rank"], config["rank"], generator=generator)).Q for _ in range(config["random_rotations"])]
    basis_q = {"native": torch.eye(config["rank"]), "ica": sparse["ica"], "varimax": sparse["varimax"], **{f"random_{index}": q for index, q in enumerate(random_qs)}}
    basis_q_cuda = {name: q.cuda() for name, q in basis_q.items()}
    vocabulary_contrasts = model.lm_head.weight.float()[:tokenizer.vocab_size] @ axes
    # The checkpoint pads its output matrix past the tokenizer vocabulary.
    # These rows are never valid token objectives, but differentiable suffixes
    # return them. Pad the declared contrasts with exact zeros so analytic and
    # independent-autograd paths address the same logit vector.
    contrasts = torch.zeros(model.lm_head.weight.shape[0], config["rank"], device="cuda")
    contrasts[:tokenizer.vocab_size] = vocabulary_contrasts
    corpus = np.load(config["corpus"], mmap_mode="r")
    rows = corpus[config["start_row"]:config["start_row"] + config["rows"]]
    records, finite_records, maximum_autograd_error = [], [], 0.0
    for row_index, raw_row in enumerate(rows):
        raw = np.asarray(raw_row[:128], dtype=np.int64); ids = torch.as_tensor(raw, device="cuda")[None]
        with torch.no_grad(): trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        for position in config["positions"]:
            native_gradients = batched_source_gradients(model, trace, position, contrasts, config["source_layers"])
            for layer in config["source_layers"]:
                native = native_gradients[layer]; product = product_at(model, trace, layer, position)
                metrics = {}
                for name, q in basis_q_cuda.items():
                    gradient = q.T @ native; contribution = gradient * product[None]
                    metrics[name] = {
                        "gradient_top64_energy": float(top_fraction(gradient, config["support_rank"], squared=True).median()),
                        "gradient_effective_support": float(effective_support(gradient).median()),
                        "contribution_top64_mass": float(top_fraction(contribution, config["support_rank"]).median()),
                        "contribution_effective_support": float(effective_support(contribution).median()),
                    }
                records.append({"row": row_index, "position": position, "layer": layer, "metrics": metrics})
                if row_index < config["autograd_check_sites"] and position == config["positions"][0]:
                    for axis_index in (0, config["rank"] - 1):
                        reference = cross_gradient(model, trace, layer, position, position, contrasts[:, axis_index])
                        maximum_autograd_error = max(maximum_autograd_error, float((native[axis_index] - reference).norm() / reference.norm().clamp_min(1e-30)))
                if row_index < config["finite_rows"] and position in config["finite_positions"] and layer in config["finite_layers"]:
                    finite_basis = {name: basis_q_cuda[name] for name in ("native", "ica", "varimax", *[f"random_{i}" for i in range(config["finite_random_rotations"])])}
                    deltas, metadata = {"base": torch.zeros_like(product)}, []
                    for name, q in finite_basis.items():
                        gradient = q.T @ native; rotated_contrasts = contrasts @ q
                        for axis_index in range(config["rank"]):
                            contribution = gradient[axis_index] * product
                            support = contribution.abs().topk(config["support_rank"]).indices
                            delta = torch.zeros_like(product); delta[support] = -config["finite_scale"] * product[support]
                            key = f"{name}:{axis_index}"; deltas[key] = delta
                            metadata.append((key, name, axis_index, float(gradient[axis_index] @ delta), rotated_contrasts[:, axis_index]))
                    with torch.no_grad(): changed = logits_for_deltas(model, layer, trace[layer], deltas, position)
                    base_logits = changed["base"].float()
                    for key, name, axis_index, predicted, contrast in metadata:
                        actual = float((changed[key].float() - base_logits) @ contrast)
                        finite_records.append({"row": row_index, "position": position, "layer": layer, "basis": name, "axis": axis_index, "predicted": predicted, "actual": actual})
        print(f"upstream compactness row {row_index + 1}/{len(rows)}", flush=True)
    summaries = {str(layer): {basis: aggregate(records, layer, basis) for basis in basis_q} for layer in config["source_layers"]}
    random_summary = {}
    for layer in config["source_layers"]:
        random_summary[str(layer)] = {}
        for metric in summaries[str(layer)]["native"]:
            values = [summaries[str(layer)][f"random_{index}"][metric] for index in range(config["random_rotations"])]
            random_summary[str(layer)][metric] = {"median": float(np.median(values)), "p95": float(np.quantile(values, .95)), "maximum": float(np.max(values))}
    finite_summary = {}
    for layer in config["finite_layers"]:
        finite_summary[str(layer)] = {}
        for basis in ("native", "ica", "varimax", *[f"random_{i}" for i in range(config["finite_random_rotations"])]):
            values = [row for row in finite_records if row["layer"] == layer and row["basis"] == basis]
            predicted = [row["predicted"] for row in values]; actual = [row["actual"] for row in values]
            finite_summary[str(layer)][basis] = {"correlation": pearson(predicted, actual), "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(actual))), "median_relative_error": float(np.median([abs(a - p) / max(abs(p), 1e-30) for p, a in zip(predicted, actual)]))}
    best_by_layer = {str(layer): max(("native", "ica", "varimax"), key=lambda basis: summaries[str(layer)][basis]["contribution_top64_mass"]) for layer in config["source_layers"]}
    hypotheses = {
        "C0_exact_batched_recurrence": {"supported": maximum_autograd_error <= 1e-5},
        "C1_varimax_sparsifies_upstream_gradients": {"supported": all(summaries[str(layer)]["varimax"]["gradient_top64_energy"] >= random_summary[str(layer)]["gradient_top64_energy"]["p95"] + .02 for layer in config["source_layers"])},
        "C2_varimax_sparsifies_native_contributions": {"supported": all(summaries[str(layer)]["varimax"]["contribution_top64_mass"] >= random_summary[str(layer)]["contribution_top64_mass"]["p95"] + .02 for layer in config["source_layers"])},
        "C3_varimax_beats_native_upstream": {"supported": all(summaries[str(layer)]["varimax"][metric] >= summaries[str(layer)]["native"][metric] + .02 for layer in config["source_layers"] for metric in ("gradient_top64_energy", "contribution_top64_mass"))},
        "C4_sparse_basis_is_depth_consistent": {"supported": len(set(best_by_layer.values())) == 1 and next(iter(set(best_by_layer.values()))) != "native"},
        "C5_local_compactness_survives_finite_edits": {"supported": all(finite_summary[str(layer)]["varimax"]["correlation"] >= .90 and finite_summary[str(layer)]["varimax"]["sign_agreement"] >= .95 for layer in config["finite_layers"])},
    }
    result = {"config": config, "hypotheses": hypotheses, "maximum_autograd_error": maximum_autograd_error, "optimization": optimization, "best_nonrandom_basis_by_layer": best_by_layer, "summaries": summaries, "random_summary": random_summary, "finite_summary": finite_summary, "records": records, "finite_records": finite_records}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ["# Does the sparse output basis simplify upstream computation?", "", "Native HOSVD, label-free ICA/varimax, and 32 matched rotations span exactly the same rank-16 late output space. Exact batched tensor VJPs transport all sixteen output objectives to MLP product channels in source Layers 0--3. Gradient energy and native zero-ablation contribution mass measure coordinate compactness; scale-0.1 finite interventions validate the local metric.", "", f"Maximum batched recurrence error against independent autograd: `{maximum_autograd_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Upstream compactness](bilin18_output_basis_upstream_compactness.png)", "", "| Layer | Basis | Gradient top-64 energy | Gradient support | Contribution top-64 mass | Contribution support |", "|---:|---|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        for basis in ("native", "ica", "varimax"):
            row = summaries[str(layer)][basis]; lines.append(f"| {layer} | {basis} | {row['gradient_top64_energy']:.3f} | {row['gradient_effective_support']:.1f} | {row['contribution_top64_mass']:.3f} | {row['contribution_effective_support']:.1f} |")
        null = random_summary[str(layer)]; lines.append(f"| {layer} | random p95 | {null['gradient_top64_energy']['p95']:.3f} | — | {null['contribution_top64_mass']['p95']:.3f} | — |")
    lines += ["", "## Finite validation", "", "| Layer | Basis | Predicted/finite correlation | Sign agreement | Median relative error |", "|---:|---|---:|---:|---:|"]
    for layer in config["finite_layers"]:
        for basis in ("native", "ica", "varimax"):
            row = finite_summary[str(layer)][basis]; lines.append(f"| {layer} | {basis} | {row['correlation']:.3f} | {row['sign_agreement']:.3f} | {row['median_relative_error']:.3f} |")
    lines += ["", "## Guardrails", "", "All bases occupy the identical physical output subspace, so differences concern coordinate choice, not information content. Activation-weighted contribution uses the native product relative to zero and is an objective-specific local ablation statistic. The finite scale-0.1 check tests its predictive validity; it does not prove that a full zero ablation is linear or that the reported axes are semantic."]
    REPORT.write_text("\n".join(lines) + "\n")
    x = np.arange(len(config["source_layers"])); fig, axes_plot = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for basis in ("native", "ica", "varimax"):
        axes_plot[0].plot(x, [summaries[str(layer)][basis]["gradient_top64_energy"] for layer in config["source_layers"]], marker="o", label=basis)
        axes_plot[1].plot(x, [summaries[str(layer)][basis]["contribution_top64_mass"] for layer in config["source_layers"]], marker="o", label=basis)
    axes_plot[0].plot(x, [random_summary[str(layer)]["gradient_top64_energy"]["p95"] for layer in config["source_layers"]], "k--", label="random p95")
    axes_plot[1].plot(x, [random_summary[str(layer)]["contribution_top64_mass"]["p95"] for layer in config["source_layers"]], "k--", label="random p95")
    for axis in axes_plot: axis.set_xticks(x, config["source_layers"]); axis.set_xlabel("Source layer"); axis.grid(alpha=.2); axis.legend()
    axes_plot[0].set(ylabel="Top-64 squared-gradient fraction", title="Upstream sensitivity compactness")
    axes_plot[1].set(ylabel="Top-64 |activation × gradient| fraction", title="Native ablation-contribution compactness")
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "maximum_autograd_error": maximum_autograd_error, "best_nonrandom_basis_by_layer": best_by_layer, "summaries": {layer: {basis: summaries[layer][basis] for basis in ("native", "ica", "varimax")} for layer in summaries}, "random_summary": random_summary, "finite_summary": finite_summary}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
