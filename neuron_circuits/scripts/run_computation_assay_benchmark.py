#!/usr/bin/env python3
"""Qualify computation-organization assays on planted ground-truth mechanisms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SIGNAL_FRACTIONS = (1.0, .9, .75, .5, .25, .125, .0625)
RANKS = (4, 8, 16, 32, 64)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/computation_assay_benchmark"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--examples", type=int, default=4096)
    return parser.parse_args()


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def median_fidelity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.median(row_cosine(left, right)))


def orthogonal(rng: np.random.Generator, size: int) -> np.ndarray:
    matrix = rng.normal(size=(size, size))
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))[None, :]
    return q.astype(np.float32)


def normalized_rows(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    values = rng.normal(size=(rows, columns)).astype(np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def effects(hidden: np.ndarray, decoder: np.ndarray) -> np.ndarray:
    return hidden @ decoder


def contributions(hidden: np.ndarray, decoder: np.ndarray) -> np.ndarray:
    return hidden[:, :, None] * decoder[None, :, :]


def s90_counts(hidden: np.ndarray, decoder: np.ndarray, threshold: float = .9) -> np.ndarray:
    terms = contributions(hidden, decoder)
    full = terms.sum(1)
    order = np.argsort(np.linalg.norm(terms, axis=2), axis=1)[:, ::-1]
    ordered = np.take_along_axis(terms, order[:, :, None], axis=1)
    partial = np.cumsum(ordered, axis=1)
    fidelity = row_cosine(partial, full[:, None, :])
    crossed = fidelity >= threshold
    first = np.argmax(crossed, axis=1) + 1
    first[~np.any(crossed, axis=1)] = hidden.shape[1]
    return first


def fit_basis(hidden: np.ndarray, rank: int) -> np.ndarray:
    _, _, vh = np.linalg.svd(hidden, full_matrices=False)
    return vh[:rank].T.astype(np.float32)


def project(hidden: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (hidden @ basis) @ basis.T


def operation_subspace_fidelity(
    discovery_hidden: np.ndarray,
    discovery_operations: np.ndarray,
    test_hidden: np.ndarray,
    test_operations: np.ndarray,
    decoder: np.ndarray,
    rank: int,
) -> float:
    reconstructed = np.zeros_like(test_hidden)
    for operation in np.unique(discovery_operations):
        basis = fit_basis(discovery_hidden[discovery_operations == operation], rank)
        keep = test_operations == operation
        reconstructed[keep] = project(test_hidden[keep], basis)
    return median_fidelity(effects(reconstructed, decoder), effects(test_hidden, decoder))


def make_axis_family(rng: np.random.Generator, count: int):
    operations = rng.integers(0, 8, size=count)
    hidden = np.zeros((count, 128), dtype=np.float32)
    for operation in range(8):
        keep = operations == operation
        hidden[keep, operation * 4:(operation + 1) * 4] = (
            1 + .25 * rng.normal(size=(int(keep.sum()), 4))
        )
    decoder = np.zeros((128, 32), dtype=np.float32)
    decoder[:32] = normalized_rows(rng, 32, 32)
    # Non-planted channels must still be able to carry the dense injection.
    # Leaving these rows at zero would make the apparent background axis-clean.
    decoder[32:] = normalized_rows(rng, 96, 32)
    return hidden, operations, decoder


def rotate_representation(hidden: np.ndarray, decoder: np.ndarray, rotation: np.ndarray):
    return hidden @ rotation, rotation.T @ decoder


def make_dense_background(rng: np.random.Generator, count: int, decoder: np.ndarray):
    state = rng.normal(size=(count, 64)).astype(np.float32)
    map_ = rng.normal(size=(64, 128)).astype(np.float32) / 8
    hidden = np.tanh(state @ map_)
    current = np.median(np.linalg.norm(effects(hidden, decoder), axis=1))
    return hidden / max(current, 1e-12)


def match_background(hidden: np.ndarray, background: np.ndarray, decoder: np.ndarray) -> np.ndarray:
    signal = np.median(np.linalg.norm(effects(hidden, decoder), axis=1))
    noise = np.median(np.linalg.norm(effects(background, decoder), axis=1))
    return background * signal / max(noise, 1e-12)


def mix(hidden: np.ndarray, background: np.ndarray, fraction: float) -> np.ndarray:
    return fraction * hidden + math.sqrt(max(0, 1 - fraction ** 2)) * background


def axis_and_rotated(seed: int, count: int) -> dict:
    rng = np.random.default_rng(seed)
    discovery, discovery_ops, decoder = make_axis_family(rng, count)
    test, test_ops, _ = make_axis_family(np.random.default_rng(seed + 10000), count)
    # Keep the planted decoder fixed across splits.
    test_decoder_rng = np.random.default_rng(seed + 10000)
    _ = test_decoder_rng.integers(0, 8, size=count)
    # make_axis_family sampled a new decoder; replace it deliberately.
    rotation = orthogonal(rng, 128)
    rotated_discovery, rotated_decoder = rotate_representation(discovery, decoder, rotation)
    rotated_test, _ = rotate_representation(test, decoder, rotation)
    background = match_background(test, make_dense_background(rng, count, decoder), decoder)
    rotated_background, _ = rotate_representation(background, decoder, rotation)
    random_rotation = orthogonal(rng, 128)
    rows = []
    for fraction in SIGNAL_FRACTIONS:
        native = mix(test, background, fraction)
        random_hidden, random_decoder = rotate_representation(native, decoder, random_rotation)
        rotated = mix(rotated_test, rotated_background, fraction)
        native_s90 = float(np.median(s90_counts(native, decoder)))
        random_s90 = float(np.median(s90_counts(random_hidden, random_decoder)))
        rotated_s90 = float(np.median(s90_counts(rotated, rotated_decoder)))
        axis_advantage = math.log2((random_s90 + 1) / (native_s90 + 1))
        rotated_advantage = math.log2((random_s90 + 1) / (rotated_s90 + 1))
        learned = operation_subspace_fidelity(
            rotated_discovery, discovery_ops, rotated, test_ops, rotated_decoder, 4,
        )
        rows.append({
            "signal_fraction": fraction,
            "native_s90": native_s90,
            "random_rotation_s90": random_s90,
            "native_axis_advantage_bits": axis_advantage,
            "rotated_s90": rotated_s90,
            "rotated_axis_advantage_bits": rotated_advantage,
            "rotated_operation_subspace_fidelity": learned,
        })
    return {"rows": rows}


def softplus(values: np.ndarray) -> np.ndarray:
    return np.logaddexp(values, 0)


def top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    return np.argpartition(scores, -count, axis=1)[:, -count:]


def routed_hidden(scores: np.ndarray, selected: np.ndarray, channels: int = 128) -> np.ndarray:
    hidden = np.zeros((len(scores), channels), dtype=np.float32)
    rows = np.arange(len(scores))[:, None]
    hidden[rows, selected] = softplus(scores[rows, selected]) + .25
    return hidden


def router_family(seed: int, count: int) -> dict:
    rng = np.random.default_rng(seed + 20000)
    route = rng.normal(size=(32, 64)).astype(np.float32) / math.sqrt(32)
    decoder = np.zeros((128, 32), dtype=np.float32)
    decoder[:64] = normalized_rows(rng, 64, 32)
    discovery_state = rng.normal(size=(count, 64)).astype(np.float32)
    test_state = rng.normal(size=(count, 64)).astype(np.float32)
    discovery_scores = discovery_state[:, :32] @ route
    test_scores = test_state[:, :32] @ route
    discovery_selected = top_indices(discovery_scores, 4)
    test_selected = top_indices(test_scores, 4)
    discovery = routed_hidden(discovery_scores, discovery_selected)
    test = routed_hidden(test_scores, test_selected)
    background = match_background(test, make_dense_background(rng, count, decoder), decoder)
    mean_activity = np.mean(np.abs(discovery), axis=0)
    fixed = np.argsort(mean_activity)[-4:]
    rows = []
    for fraction in SIGNAL_FRACTIONS:
        target_hidden = mix(test, background, fraction)
        target_effect = effects(target_hidden, decoder)
        oracle = np.zeros_like(test)
        sample_rows = np.arange(count)[:, None]
        oracle[sample_rows, test_selected] = test[sample_rows, test_selected]
        fixed_hidden = np.zeros_like(test)
        fixed_hidden[:, fixed] = test[:, fixed]
        router_metrics = {}
        for dimension in (2, 4, 8, 16, 24, 32):
            design = discovery_state[:, :dimension]
            coefficient = np.linalg.lstsq(design, discovery_scores, rcond=None)[0]
            predicted_scores = test_state[:, :dimension] @ coefficient
            predicted_selected = top_indices(predicted_scores, 4)
            predicted = routed_hidden(predicted_scores, predicted_selected)
            recall = np.mean([
                len(set(left).intersection(right)) / 4
                for left, right in zip(predicted_selected.tolist(), test_selected.tolist())
            ])
            router_metrics[str(dimension)] = {
                "effect_fidelity": median_fidelity(
                    effects(fraction * predicted, decoder), target_effect,
                ),
                "support_recall": float(recall),
            }
        rows.append({
            "signal_fraction": fraction,
            "oracle_top4_fidelity": median_fidelity(effects(fraction * oracle, decoder), target_effect),
            "fixed_top4_fidelity": median_fidelity(effects(fraction * fixed_hidden, decoder), target_effect),
            "routers": router_metrics,
        })
    return {"rows": rows}


def local_project(hidden: np.ndarray, local_basis: np.ndarray) -> np.ndarray:
    reconstructed = np.empty_like(hidden)
    for index in range(len(hidden)):
        q, _ = np.linalg.qr(local_basis[index])
        reconstructed[index] = q @ (q.T @ hidden[index])
    return reconstructed


def atlas_family(seed: int, count: int) -> dict:
    rng = np.random.default_rng(seed + 30000)
    controls, channels, local_rank = 32, 128, 4
    # Keep only a small common chart so the planted tangent orientation is
    # genuinely state-dependent rather than dominated by a global rank-4 base.
    base = .1 * rng.normal(size=(channels, local_rank)).astype(np.float32)
    generators = rng.normal(size=(controls, channels, local_rank)).astype(np.float32) / math.sqrt(controls)
    decoder = normalized_rows(rng, channels, 32)

    def sample():
        state = rng.normal(size=(count, 64)).astype(np.float32)
        coordinates = rng.normal(size=(count, local_rank)).astype(np.float32)
        basis = base[None] + np.einsum("nk,kcr->ncr", np.tanh(state[:, :controls]), generators)
        hidden = np.einsum("ncr,nr->nc", basis, coordinates)
        return state, basis.astype(np.float32), hidden.astype(np.float32)

    _, discovery_basis, discovery = sample()
    _, test_basis, test = sample()
    background = match_background(test, make_dense_background(rng, count, decoder), decoder)
    local = local_project(test, test_basis)
    global_bases = {rank: fit_basis(discovery, rank) for rank in RANKS}
    rows = []
    for fraction in SIGNAL_FRACTIONS:
        target_hidden = mix(test, background, fraction)
        target_effect = effects(target_hidden, decoder)
        globals_ = {
            str(rank): median_fidelity(
                effects(fraction * project(test, basis), decoder), target_effect,
            )
            for rank, basis in global_bases.items()
        }
        rows.append({
            "signal_fraction": fraction,
            "local_rank4_fidelity": median_fidelity(effects(fraction * local, decoder), target_effect),
            "global_subspace_fidelity": globals_,
        })
    return {"rows": rows}


def dense_family(seed: int, count: int) -> dict:
    rng = np.random.default_rng(seed + 40000)
    decoder = normalized_rows(rng, 128, 32)
    discovery = make_dense_background(rng, count, decoder)
    test = make_dense_background(rng, count, decoder)
    basis = fit_basis(discovery, 4)
    target = effects(test, decoder)
    pca = effects(project(test, basis), decoder)
    terms = contributions(test, decoder)
    order = np.argsort(np.linalg.norm(terms, axis=2), axis=1)[:, -4:]
    selected = np.take_along_axis(terms, order[:, :, None], axis=1).sum(1)
    fixed = np.argsort(np.mean(np.linalg.norm(terms, axis=2), axis=0))[-4:]
    return {
        "identity_fidelity": median_fidelity(target, target),
        "pca_rank4_fidelity": median_fidelity(pca, target),
        "oracle_top4_fidelity": median_fidelity(selected, target),
        "fixed_top4_fidelity": median_fidelity(terms[:, fixed].sum(1), target),
    }


def effective_support(matrix: np.ndarray) -> float:
    absolute = np.abs(matrix.copy())
    np.fill_diagonal(absolute, 0)
    numerator = np.sum(absolute, axis=1) ** 2
    denominator = np.sum(absolute ** 2, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def constraint_system(seed: int) -> dict:
    rng = np.random.default_rng(seed + 50000)
    facts, parameters, groups = 128, 256, 8
    common = np.eye(parameters, dtype=np.float32)[:groups]
    unique = np.eye(parameters, dtype=np.float32)[groups:groups + facts]
    taut = np.empty((facts, parameters), dtype=np.float32)
    for index in range(facts):
        taut[index] = math.sqrt(.8) * common[index % groups] + math.sqrt(.2) * unique[index]
    shared = np.eye(parameters, dtype=np.float32)[groups + facts:groups + facts + 16]
    shared_coefficients = normalized_rows(rng, facts, 16)
    diffuse = math.sqrt(.95) * unique + math.sqrt(.05) * (shared_coefficients @ shared)

    def audit(gradients: np.ndarray):
        gram = gradients @ gradients.T
        norms = np.sqrt(np.diag(gram))
        cosine = gram / norms[:, None] / norms[None, :]
        offdiag = ~np.eye(facts, dtype=bool)
        # One source from each planted taut cluster; a stride of 16 would
        # accidentally choose the same cluster every time.
        sources = np.arange(groups)
        actual, predicted, null = [], [], []
        for position, source in enumerate(sources):
            delta = gradients[source] / gram[source, source]
            observed = gradients @ delta
            estimate = gram[:, source] / gram[source, source]
            null_source = sources[(position + 1) % len(sources)]
            control = gram[:, null_source] / gram[null_source, null_source]
            keep = np.arange(facts) != source
            actual.extend(observed[keep]); predicted.extend(estimate[keep]); null.extend(control[keep])
        actual, predicted, null = map(np.asarray, (actual, predicted, null))
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
        null_correlation = float(np.corrcoef(actual, null)[0, 1])
        r2 = float(1 - np.sum((actual - predicted) ** 2) / np.sum(actual ** 2))

        # Mildly nonlinear margins validate the boundary between an exact
        # tangent kernel and a finite edit:
        # m_i(theta)=g_i theta + beta/2 (q_i theta)^2.
        queries = normalized_rows(rng, facts, parameters)
        theta = .1 * rng.normal(size=parameters).astype(np.float32)
        beta, scale = .3, .25
        query_state = queries @ theta
        local_gradients = gradients + beta * query_state[:, None] * queries
        local_gram = local_gradients @ local_gradients.T
        nonlinear_actual, nonlinear_predicted = [], []
        baseline = gradients @ theta + .5 * beta * query_state ** 2
        for source in sources:
            delta = scale * local_gradients[source] / local_gram[source, source]
            changed_state = queries @ (theta + delta)
            changed = gradients @ (theta + delta) + .5 * beta * changed_state ** 2
            observed = changed - baseline
            estimate = scale * local_gram[:, source] / local_gram[source, source]
            keep = np.arange(facts) != source
            nonlinear_actual.extend(observed[keep])
            nonlinear_predicted.extend(estimate[keep])
        nonlinear_actual = np.asarray(nonlinear_actual)
        nonlinear_predicted = np.asarray(nonlinear_predicted)
        nonlinear_correlation = float(np.corrcoef(nonlinear_actual, nonlinear_predicted)[0, 1])
        nonlinear_r2 = float(
            1 - np.sum((nonlinear_actual - nonlinear_predicted) ** 2)
            / np.sum(nonlinear_actual ** 2)
        )
        return {
            "mean_absolute_cosine": float(np.mean(np.abs(cosine[offdiag]))),
            "effective_support": effective_support(cosine),
            "mean_absolute_collateral": float(np.mean(np.abs(actual))),
            "finite_edit_pearson": correlation,
            "finite_edit_origin_r2": r2,
            "permuted_source_pearson": null_correlation,
            "nonlinear_small_scale": {
                "scale": scale,
                "pearson": nonlinear_correlation,
                "origin_r2": nonlinear_r2,
            },
        }

    return {"taut": audit(taut), "diffuse": audit(diffuse)}


def evaluate_seed(seed: int, examples: int) -> dict:
    return {
        "seed": seed,
        "axis_and_rotated": axis_and_rotated(seed, examples),
        "router": router_family(seed, examples),
        "atlas": atlas_family(seed, examples),
        "dense": dense_family(seed, examples),
        "constraints": constraint_system(seed),
    }


def pure_checks(run: dict) -> dict[str, bool]:
    axis = run["axis_and_rotated"]["rows"][0]
    router = run["router"]["rows"][0]
    atlas = run["atlas"]["rows"][0]
    dense = run["dense"]
    constraint = run["constraints"]
    return {
        "native_axis_detected": axis["native_axis_advantage_bits"] > 1,
        "rotated_axis_not_privileged": abs(axis["rotated_axis_advantage_bits"]) < .5,
        "rotated_subspace_recovered": axis["rotated_operation_subspace_fidelity"] > .95,
        "oracle_router_payload_recovered": router["oracle_top4_fidelity"] > .95,
        "fixed_router_payload_fails": router["fixed_top4_fidelity"] < .60,
        "full_router_recovered": router["routers"]["32"]["effect_fidelity"] > .90,
        "local_atlas_recovered": atlas["local_rank4_fidelity"] > .95,
        "global_rank4_fails_atlas": atlas["global_subspace_fidelity"]["4"] < .60,
        "dense_identity_exact": dense["identity_fidelity"] > .99999,
        "dense_compact_methods_fail": max(
            dense["pca_rank4_fidelity"], dense["oracle_top4_fidelity"], dense["fixed_top4_fidelity"],
        ) < .60,
        "constraint_kernel_exact": min(
            constraint["taut"]["finite_edit_pearson"], constraint["taut"]["finite_edit_origin_r2"],
            constraint["diffuse"]["finite_edit_pearson"], constraint["diffuse"]["finite_edit_origin_r2"],
        ) > .98,
        # An anticorrelated wrong-source predictor is a successful null. The
        # failure condition is positive predictive correlation, not magnitude.
        "constraint_null_low": max(
            constraint["taut"]["permuted_source_pearson"],
            constraint["diffuse"]["permuted_source_pearson"],
        ) < .10,
        "constraint_nonlinear_small_scale": min(
            constraint["taut"]["nonlinear_small_scale"]["pearson"],
            constraint["taut"]["nonlinear_small_scale"]["origin_r2"],
            constraint["diffuse"]["nonlinear_small_scale"]["pearson"],
            constraint["diffuse"]["nonlinear_small_scale"]["origin_r2"],
        ) > .90,
        "taut_is_stronger": (
            constraint["taut"]["mean_absolute_cosine"] > constraint["diffuse"]["mean_absolute_cosine"]
            and constraint["taut"]["mean_absolute_collateral"] > constraint["diffuse"]["mean_absolute_collateral"]
        ),
        "diffuse_is_broader": constraint["diffuse"]["effective_support"] > constraint["taut"]["effective_support"],
    }


def aggregate(runs: list[dict]) -> dict:
    check_names = pure_checks(runs[0]).keys()
    checks = {name: sum(pure_checks(run)[name] for run in runs) for name in check_names}
    detection = {}
    for family, metric, threshold, comparison in (
        ("native_axis", "native_axis_advantage_bits", 1.0, "greater"),
        ("rotated_subspace", "rotated_operation_subspace_fidelity", .95, "greater"),
    ):
        fractions = {}
        for position, fraction in enumerate(SIGNAL_FRACTIONS):
            values = [run["axis_and_rotated"]["rows"][position][metric] for run in runs]
            hits = sum(value > threshold for value in values) if comparison == "greater" else sum(value < threshold for value in values)
            fractions[str(fraction)] = {"hits": hits, "median": float(np.median(values))}
        detection[family] = fractions
    for family, getter, threshold in (
        ("routed_payload", lambda run, p: run["router"]["rows"][p]["routers"]["32"]["effect_fidelity"], .90),
        ("local_atlas", lambda run, p: run["atlas"]["rows"][p]["local_rank4_fidelity"], .95),
    ):
        fractions = {}
        for position, fraction in enumerate(SIGNAL_FRACTIONS):
            values = [getter(run, position) for run in runs]
            fractions[str(fraction)] = {"hits": sum(value > threshold for value in values), "median": float(np.median(values))}
        detection[family] = fractions
    return {"pure_check_passes_out_of_seeds": checks, "detection_curves": detection}


def plot_detection(summary: dict, path: Path):
    fig, axis = plt.subplots(figsize=(7.4, 4.6), layout="constrained")
    for name, rows in summary["detection_curves"].items():
        values = [rows[str(fraction)]["median"] for fraction in SIGNAL_FRACTIONS]
        axis.plot(SIGNAL_FRACTIONS, values, marker="o", label=name.replace("_", " "))
    axis.set_xscale("log", base=2)
    axis.invert_xaxis()
    axis.set_xlabel("planted signal fraction (dense background increases →)")
    axis.set_ylabel("locked assay statistic / fidelity")
    axis.axhline(.9, color="black", linestyle="--", linewidth=1, alpha=.5)
    axis.grid(alpha=.25)
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(output: Path, runs: list[dict], summary: dict):
    checks = summary["pure_check_passes_out_of_seeds"]
    lines = [
        "# Ground-truth computation-assay benchmark", "",
        "This benchmark asks whether the planned interpretability assays recover mechanisms that are known to exist before their absence is interpreted in a real model.", "",
        "## Pure-mechanism qualification", "",
        "| qualification gate | seeds passing / 5 |", "|---|---:|",
    ]
    lines.extend(f"| {name.replace('_', ' ')} | {value}/5 |" for name, value in checks.items())
    lines += ["", "## Dense-background sufficiency curves", "",
              "A hit means the original locked pure-mechanism threshold still passes after adding a norm-matched dense background. This asks whether the planted mechanism remains sufficient for the whole effect; it is not a test of whether a weaker component remains present.", "",
              "| family | " + " | ".join(f"signal {fraction:g}" for fraction in SIGNAL_FRACTIONS) + " |",
              "|---|" + "---:|" * len(SIGNAL_FRACTIONS)]
    for family, rows in summary["detection_curves"].items():
        cells = [f"{rows[str(fraction)]['hits']}/5 ({rows[str(fraction)]['median']:.3f})" for fraction in SIGNAL_FRACTIONS]
        lines.append(f"| {family.replace('_', ' ')} | " + " | ".join(cells) + " |")
    all_pass = all(value == len(runs) for value in checks.values())
    lines += ["", "## Verdict", "",
              f"All pure-mechanism gates passed in every seed: **{all_pass}**.", "",
              "A failed pure gate is an assay failure, not a scientific null. The mixture curves define when a planted component remains nearly sufficient. They cannot by themselves exclude a weaker component. Full per-seed values are in `results.json`.", "",
              "![Detection curves](detection_curves.png)"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        print(f"START seed={seed}", flush=True)
        run = evaluate_seed(seed, args.examples)
        runs.append(run)
        print(f"DONE seed={seed} passes={sum(pure_checks(run).values())}/{len(pure_checks(run))}", flush=True)
    summary = aggregate(runs)
    payload = {"config": {"seeds": args.seeds, "examples": args.examples,
                           "signal_fractions": SIGNAL_FRACTIONS}, "runs": runs, "summary": summary}
    (args.output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_detection(summary, args.output / "detection_curves.png")
    write_report(args.output, runs, summary)
    print(json.dumps(summary["pure_check_passes_out_of_seeds"], indent=2), flush=True)


if __name__ == "__main__":
    main()
