#!/usr/bin/env python3
"""Validate layerwise signed margin-constraint networks in grammar models."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import GrammarTransformer


SCALES = (.25, 1.0, 4.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--panel-size", type=int, default=48)
    parser.add_argument("--source-count", type=int, default=8)
    parser.add_argument("--panel-json", type=Path,
                        help="Optional fixed fact/competitor panel shared across model conditions.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-underfit", action="store_true",
                        help="Run exploratory audits even when the preregistered fidelity gate fails.")
    return parser.parse_args()


def load_model(path: Path, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GrammarTransformer(
        VOCAB.size, checkpoint["width"], checkpoint["heads"], checkpoint["depth"],
        mlp_kind=checkpoint["mlp_kind"],
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def answer_logits(model, tokens):
    return model(tokens)[:, -1, list(VOCAB.answers)].float()


def strongest_competitors(logits, targets):
    wrong = logits.clone(); wrong[torch.arange(len(targets), device=logits.device), targets] = -torch.inf
    return wrong.argmax(1)


def margins(logits, targets, competitors):
    rows = torch.arange(len(targets), device=logits.device)
    return logits[rows, targets] - logits[rows, competitors]


def parameter_groups(model):
    grouped = defaultdict(list)
    for name, parameter in model.named_parameters():
        if name == "position" or name.startswith("embedding."):
            group = "input"
        elif name.startswith("blocks."):
            group = f"block_{name.split('.')[1]}"
        else:
            group = "output"
        grouped[group].append((name, parameter))
    order = ["input", *(f"block_{i}" for i in range(model.layers_count)), "output"]
    return {name: grouped[name] for name in order}


def gradient_matrices(model, tokens, targets, competitors, groups):
    all_parameters = [p for entries in groups.values() for _, p in entries]
    parameter_slices, cursor = {}, 0
    for name, entries in groups.items():
        parameter_slices[name] = slice(cursor, cursor + len(entries)); cursor += len(entries)
    matrices = {
        name: torch.empty(len(tokens), sum(p.numel() for _, p in entries), dtype=torch.float32)
        for name, entries in groups.items()
    }
    for row in range(len(tokens)):
        logits = answer_logits(model, tokens[row:row + 1])
        margin = logits[0, targets[row]] - logits[0, competitors[row]]
        gradients = torch.autograd.grad(margin, all_parameters)
        for name, entries in groups.items():
            offset = 0
            for gradient in gradients[parameter_slices[name]]:
                flat = gradient.detach().float().reshape(-1).cpu()
                matrices[name][row, offset:offset + len(flat)] = flat; offset += len(flat)
    return matrices


def corr(left, right):
    return float(np.corrcoef(left, right)[0, 1]) if np.std(left) and np.std(right) else float("nan")


def origin_r2(actual, predicted):
    denominator = np.sum(actual ** 2)
    return float(1 - np.sum((actual - predicted) ** 2) / denominator) if denominator else float("nan")


def effective_support_rows(weights):
    numerator = weights.sum(1) ** 2
    denominator = (weights ** 2).sum(1)
    valid = denominator > 1e-30
    return float(np.mean(numerator[valid] / denominator[valid])) if np.any(valid) else 0.0


def kernel_summary(cosine, metadata, null_cosine=None):
    n = len(cosine); offdiag = ~np.eye(n, dtype=bool)
    supports = []
    for row in range(n):
        edges = np.abs(np.delete(cosine[row], row))
        supports.append(edges.sum() ** 2 / max(np.sum(edges ** 2), 1e-30))
    ii, jj = np.triu_indices(n, 1)
    shared_subject_digits = (metadata["subject_digits"][ii] == metadata["subject_digits"][jj]).mean(1)
    shared_object_digits = (metadata["object_digits"][ii] == metadata["object_digits"][jj]).mean(1)
    design = np.column_stack((
        np.ones(len(ii)), metadata["relation"][ii] == metadata["relation"][jj],
        metadata["answer"][ii] == metadata["answer"][jj],
        metadata["competitor"][ii] == metadata["competitor"][jj],
        metadata["selected"][ii] == metadata["selected"][jj],
        metadata["subject"][ii] == metadata["subject"][jj],
        metadata["object"][ii] == metadata["object"][jj],
        shared_subject_digits, shared_object_digits,
    )).astype(float)
    values = cosine[ii, jj]
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design @ coefficients
    output = {
        "mean_absolute_cosine": float(np.mean(np.abs(cosine[offdiag]))),
        "positive_fraction": float(np.mean(cosine[offdiag] > 0)),
        "negative_fraction": float(np.mean(cosine[offdiag] < 0)),
        "mean_effective_support": float(np.mean(supports)),
        "control_r2": float(1 - np.sum(residual ** 2) / max(np.sum((values - values.mean()) ** 2), 1e-30)),
        "mean_absolute_residual_cosine": float(np.mean(np.abs(residual))),
    }
    if null_cosine is not None:
        null_absolute = np.abs(null_cosine[offdiag])
        threshold = float(np.quantile(null_absolute, .95))
        excess = np.maximum(np.abs(cosine) - threshold, 0)
        np.fill_diagonal(excess, 0)
        output["sign_null_mean_absolute_cosine"] = float(np.mean(null_absolute))
        output["absolute_cosine_to_sign_null_ratio"] = float(
            output["mean_absolute_cosine"] / max(np.mean(null_absolute), 1e-30)
        )
        output["sign_null_q95"] = threshold
        output["fraction_edges_above_sign_null_q95"] = float(
            np.mean(np.abs(cosine[offdiag]) > threshold)
        )
        output["mean_effective_support_above_sign_null"] = effective_support_rows(excess)
    return output


def kkt_stationarity_fit(matrix: torch.Tensor, parameter: np.ndarray) -> dict:
    """Fit theta ~= sum_i lambda_i grad margin_i with lambda_i >= 0."""
    gradients = matrix.numpy().astype(np.float64, copy=False)
    theta = parameter.astype(np.float64, copy=False)
    gram = gradients @ gradients.T
    target = gradients @ theta
    eigenmax = float(np.linalg.eigvalsh(gram).max())
    coefficients = np.zeros(len(gradients), dtype=np.float64)
    momentum = coefficients.copy(); accelerated = 1.0
    if eigenmax > 1e-30:
        for _ in range(5000):
            updated = np.maximum(0, momentum - (gram @ momentum - target) / eigenmax)
            next_accelerated = (1 + np.sqrt(1 + 4 * accelerated ** 2)) / 2
            next_momentum = updated + (accelerated - 1) / next_accelerated * (updated - coefficients)
            if np.linalg.norm(updated - coefficients) <= 1e-10 * (1 + np.linalg.norm(coefficients)):
                coefficients = updated
                break
            coefficients, momentum, accelerated = updated, next_momentum, next_accelerated
    fitted = coefficients @ gradients
    theta_norm = np.linalg.norm(theta)
    fitted_norm = np.linalg.norm(fitted)
    cosine = float(np.dot(theta, fitted) / max(theta_norm * fitted_norm, 1e-30))
    relative_r2 = float(1 - np.sum((theta - fitted) ** 2) / max(np.sum(theta ** 2), 1e-30))
    ridge = 1e-8 * max(float(np.trace(gram)) / max(len(gram), 1), 1e-30)
    unconstrained = np.linalg.solve(gram + ridge * np.eye(len(gram)), target)
    unconstrained_fit = unconstrained @ gradients
    unconstrained_cosine = float(
        np.dot(theta, unconstrained_fit)
        / max(theta_norm * np.linalg.norm(unconstrained_fit), 1e-30)
    )
    active = coefficients > max(float(coefficients.max()) * 1e-6, 1e-12)
    return {
        "nonnegative_cosine": cosine,
        "nonnegative_relative_r2": relative_r2,
        "unconstrained_cosine": unconstrained_cosine,
        "active_constraints": int(active.sum()),
        "panel_constraints": int(len(coefficients)),
    }


def finite_edits(model, tokens, targets, competitors, baseline, groups, grams, sources):
    baseline_margin = margins(baseline, targets, competitors).detach().cpu().numpy()
    all_parameters = [p for entries in groups.values() for _, p in entries]
    group_slices, cursor = {}, 0
    for name, entries in groups.items():
        group_slices[name] = slice(cursor, cursor + len(entries)); cursor += len(entries)
    records = []
    for source_position, source in enumerate(sources):
        logits = answer_logits(model, tokens[source:source + 1])
        margin = logits[0, targets[source]] - logits[0, competitors[source]]
        gradients = torch.autograd.grad(margin, all_parameters)
        for group_name, entries in groups.items():
            selected_gradients = gradients[group_slices[group_name]]
            norm_sq = float(sum((g.detach().float() ** 2).sum() for g in selected_gradients))
            if norm_sq <= 1e-20: continue
            originals = [p.detach().clone() for _, p in entries]
            parameter_norm = float(sum((p.detach().float() ** 2).sum() for _, p in entries).sqrt())
            for scale in SCALES:
                with torch.no_grad():
                    for (_, parameter), gradient in zip(entries, selected_gradients):
                        parameter.add_(gradient, alpha=scale / norm_sq)
                    edited = answer_logits(model, tokens)
                    for (_, parameter), original in zip(entries, originals): parameter.copy_(original)
                actual = margins(edited, targets, competitors).cpu().numpy() - baseline_margin
                gram = grams[group_name]
                predicted = scale * gram[:, source] / gram[source, source]
                null_source = int(sources[(source_position + 1) % len(sources)])
                null = scale * gram[:, null_source] / gram[null_source, null_source]
                keep = np.arange(len(tokens)) != source
                records.append({
                    "source": int(source), "group": group_name, "scale": scale,
                    "actual": actual.tolist(), "predicted": predicted.tolist(), "null_predicted": null.tolist(),
                    "offdiag_pearson": corr(actual[keep], predicted[keep]),
                    "offdiag_origin_r2": origin_r2(actual[keep], predicted[keep]),
                    "null_pearson": corr(actual[keep], null[keep]),
                    "relative_parameter_change": float((scale / np.sqrt(norm_sq)) / max(parameter_norm, 1e-30)),
                })
    return records


def pooled(records, groups, scale):
    actual, predicted, null = [], [], []
    for row in records:
        if row["group"] not in groups or row["scale"] != scale: continue
        keep = np.arange(len(row["actual"])) != row["source"]
        actual.extend(np.asarray(row["actual"])[keep]); predicted.extend(np.asarray(row["predicted"])[keep])
        null.extend(np.asarray(row["null_predicted"])[keep])
    actual, predicted, null = map(np.asarray, (actual, predicted, null))
    return {"count": len(actual), "pearson": corr(actual, predicted),
            "origin_r2": origin_r2(actual, predicted), "null_pearson": corr(actual, null),
            "mean_absolute_collateral": float(np.mean(np.abs(actual)))}


def choose_panel(root, run, manifest, size, panel_json=None):
    labels, selected, changed = labels_for_rho(manifest, "train", run["rho"])
    with np.load(root / f"rho_{run['rho']:g}" / f"depth_{run['depth']}" / "predictions.npz") as stored:
        eligible = ((stored["train_template0_prediction"] == labels) &
                    (stored["train_template6_prediction"] == labels))
    if panel_json is not None:
        specification = json.loads(panel_json.read_text())
        lookup = {tuple(map(int, fact)): index for index, fact in enumerate(np.asarray(manifest["train_facts"]))}
        try:
            panel = np.asarray([lookup[tuple(map(int, fact))] for fact in specification["facts"]], dtype=int)
        except KeyError as error:
            raise ValueError(f"fixed panel fact missing from {root}: {error.args[0]}") from error
        if not np.all(eligible[panel]):
            raise ValueError(f"fixed panel contains facts not correct on both templates in {root}")
        if not np.array_equal(labels[panel], np.asarray(specification["targets"], dtype=np.uint8)):
            raise ValueError(f"fixed panel target mismatch in {root}")
        return panel, np.asarray(specification["competitors"], dtype=np.uint8)
    rng = np.random.default_rng(81000 + run["depth"] * 10 + round(10 * run["rho"]))
    indices = np.flatnonzero(eligible)
    if run["rho"] == .5:
        first = rng.choice(indices[changed[indices]], size // 2, replace=False)
        second = rng.choice(indices[~selected[indices]], size - size // 2, replace=False)
        panel = np.concatenate((first, second)); rng.shuffle(panel)
    else:
        panel = rng.choice(indices, size, replace=False)
    return panel, None


def run_condition(root, run, manifest, args):
    model = load_model(root / f"rho_{run['rho']:g}" / f"depth_{run['depth']}" / "checkpoint.pt", args.device)
    torch.manual_seed(run["seed"] + run["depth"] * 1000)
    initial_model = GrammarTransformer(
        VOCAB.size, run["width"], run["heads"], run["depth"], mlp_kind=run["mlp_kind"],
    )
    initial_state = {name: value.detach().float().cpu().numpy()
                     for name, value in initial_model.named_parameters()}
    panel, fixed_competitors = choose_panel(root, run, manifest, args.panel_size, args.panel_json)
    facts = np.asarray(manifest["train_facts"])[panel]
    labels, selected, changed = labels_for_rho(manifest, "train", run["rho"])
    targets = torch.from_numpy(labels[panel].astype(np.int64)).to(args.device)
    tokens = torch.from_numpy(encode_prompts(manifest, facts, 0).astype(np.int64)).to(args.device)
    baseline = answer_logits(model, tokens)
    competitors = (torch.from_numpy(fixed_competitors.astype(np.int64)).to(args.device)
                   if fixed_competitors is not None else strongest_competitors(baseline, targets))
    baseline_margin = margins(baseline, targets, competitors).detach().cpu().numpy()
    digits = np.asarray(manifest["digits"])
    metadata = {
        "relation": facts[:, 2], "answer": targets.cpu().numpy(), "competitor": competitors.cpu().numpy(),
        "selected": selected[panel], "changed": changed[panel], "subject": facts[:, 0], "object": facts[:, 1],
        "subject_digits": digits[facts[:, 0]], "object_digits": digits[facts[:, 1]],
    }
    groups = parameter_groups(model)
    matrices = gradient_matrices(model, tokens, targets, competitors, groups)
    grams, kernels, gradient_norms, kkt = {}, {}, {}, {}
    for group_index, (name, matrix) in enumerate(matrices.items()):
        gram = (matrix @ matrix.T).numpy(); norm = np.sqrt(np.maximum(np.diag(gram), 1e-30))
        cosine = gram / norm[:, None] / norm[None, :]
        null_rng = np.random.default_rng(73000 + run["seed"] * 100 + group_index)
        signs = null_rng.integers(0, 2, size=matrix.shape, dtype=np.int8) * 2 - 1
        signed = matrix.numpy() * signs
        null_gram = signed @ signed.T
        null_norm = np.sqrt(np.maximum(np.diag(null_gram), 1e-30))
        null_cosine = null_gram / null_norm[:, None] / null_norm[None, :]
        grams[name] = gram; kernels[name] = kernel_summary(cosine, metadata, null_cosine)
        gradient_norms[name] = {
            "median": float(np.median(norm)),
            "q05": float(np.quantile(norm, .05)),
            "q95": float(np.quantile(norm, .95)),
        }
        entries = groups[name]
        parameter = np.concatenate([value.detach().float().cpu().numpy().reshape(-1)
                                    for _, value in entries])
        initial_parameter = np.concatenate([initial_state[parameter_name].reshape(-1)
                                            for parameter_name, _ in entries])
        current_fit = kkt_stationarity_fit(matrix, parameter)
        displacement_fit = kkt_stationarity_fit(matrix, parameter - initial_parameter)
        kkt[name] = {
            **current_fit,
            **{f"displacement_{key}": value for key, value in displacement_fit.items()},
        }
        if name.startswith("block_"):
            mlp_mask = np.concatenate([
                np.full(value.numel(), any(part in parameter_name for part in
                                           (".mlp_norm.", ".left.", ".right.", ".down.")), dtype=bool)
                for parameter_name, value in entries
            ])
            current_fit = kkt_stationarity_fit(matrix[:, mlp_mask], parameter[mlp_mask])
            displacement_fit = kkt_stationarity_fit(
                matrix[:, mlp_mask], (parameter - initial_parameter)[mlp_mask],
            )
            kkt[f"{name}_mlp"] = {
                **current_fit,
                **{f"displacement_{key}": value for key, value in displacement_fit.items()},
            }
    del matrices
    sources = np.linspace(0, len(panel) - 1, args.source_count, dtype=int)
    edits = finite_edits(model, tokens, targets, competitors, baseline, groups, grams, sources)
    block_names = [f"block_{index}" for index in range(model.layers_count)]
    return {
        "rho": run["rho"], "depth": run["depth"], "mlp_kind": run["mlp_kind"],
        "fidelity_gate_passed": bool(run["best"]["fixed_accuracy"] >= .97 and run["best"]["transfer_accuracy"] >= .95),
        "baseline_margin": {
            "median": float(np.median(baseline_margin)),
            "q05": float(np.quantile(baseline_margin, .05)),
            "q95": float(np.quantile(baseline_margin, .95)),
        },
        "gradient_norms": gradient_norms,
        "kkt_stationarity": kkt,
        "panel_indices": panel.tolist(), "sources": sources.tolist(), "kernels": kernels,
        "panel_facts": facts.tolist(), "fixed_panel": fixed_competitors is not None,
        "pooled_block_edits": {str(scale): pooled(edits, block_names, scale) for scale in SCALES},
        "edits": edits,
    }


def main():
    args = parse_args(); manifest = load_manifest(args.root / "manifest.npz")
    runs = json.loads((args.root / "pilot_results.json").read_text())["runs"]
    for run in runs:
        path = args.root / f"rho_{run['rho']:g}" / f"depth_{run['depth']}" / "constraint_audit.json"
        if path.exists() and not args.force:
            print(f"SKIP {path}", flush=True); continue
        if (not args.include_underfit and
                (run["best"]["fixed_accuracy"] < .97 or run["best"]["transfer_accuracy"] < .95)):
            print(f"UNDERFIT rho={run['rho']:g} depth={run['depth']}", flush=True); continue
        print(f"START rho={run['rho']:g} depth={run['depth']}", flush=True)
        result = run_condition(args.root, run, manifest, args)
        path.write_text(json.dumps(result), encoding="utf-8")
        metric = result["pooled_block_edits"]["1.0"]
        print(f"DONE rho={run['rho']:g} depth={run['depth']} pearson={metric['pearson']:.3f}", flush=True)


if __name__ == "__main__":
    main()
