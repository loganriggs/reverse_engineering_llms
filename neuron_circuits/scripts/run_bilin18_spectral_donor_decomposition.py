#!/usr/bin/env python3
"""Dose-response and SAE decomposition red-team for spectral donor transfer."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from audit_bilin18_mlp_discriminators import SNAPSHOT
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_companion_paths import sparse_code
from run_bilin18_spectral_fixed_companions import OUTPUT as CORE_OUTPUT
from run_bilin18_spectral_gate_validation import CONFIG as GATE_CONFIG, OUTPUT as GATE_OUTPUT
from run_bilin18_spectral_spike_pilot import (
    ARRAYS as PILOT_ARRAYS,
    OUTPUT as PILOT_OUTPUT,
    SPARSE,
    TENSOR_ROOT,
    fisher_fidelity,
    safe_name,
)
from train_bilin18_causal_cp import prefix_to_layer1, suffix_from_layer1


CONFIG = Path("configs/bilin18_spectral_donor_decomposition.json")
OUTPUT = TENSOR_ROOT / "spectral_donor_decomposition.json"
REPORT = TENSOR_ROOT / "SPECTRAL_DONOR_DECOMPOSITION.md"
FIGURE = TENSOR_ROOT / "spectral_donor_decomposition.png"


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    gate_config = json.loads(GATE_CONFIG.read_text())
    gate = json.loads(GATE_OUTPUT.read_text())
    cores = json.loads(CORE_OUTPUT.read_text())["cores"]
    pilot = json.loads(PILOT_OUTPUT.read_text())
    arrays = np.load(PILOT_ARRAYS)
    categories = config["categories"]

    archive = np.load(SPARSE / "natural_large.npz")
    train = torch.from_numpy(archive["fineweb_train__h1"].astype(np.float32))
    mean = train.mean(0).cuda()
    std = train.std(0).clamp_min(1e-4).cuda()
    del train, archive

    sae = torch.load(
        SPARSE / "activation_sae_large_seed101.pt",
        map_location="cpu",
        weights_only=True,
    )
    encoder_weight = sae["encoder.weight"].float().cuda()
    encoder_bias = sae["encoder.bias"].float().cuda()
    decoder_original = std[:, None] * sae["decoder.weight"].float().cuda()
    del sae

    features = {
        category: pilot["spectra"][f"semantic/seed101/{category}"]["metadata"]["feature"]
        for category in categories
    }
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    all_rows = np.load(DEFAULT_CORPUS, mmap_mode="r")
    panel = gate_config["panel"]
    rows = all_rows[panel["start"] : panel["start"] + panel["rows"]]

    def state_at(site: dict) -> dict:
        prefix = np.array(
            rows[site["row"], : site["position"] + 1],
            dtype=np.int64,
            copy=True,
        )
        ids = torch.as_tensor(prefix, device="cuda")[None]
        x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
        product = model.blocks[1].mlp.Left(normalized) * model.blocks[1].mlp.Right(normalized)
        native_write = model.blocks[1].mlp.Down(product) + model.blocks[1].mlp.Down_bias
        native_logits = suffix_from_layer1(model, x0, value_bus, pre + native_write)[0, -1]
        return {
            "x0": x0,
            "value_bus": value_bus,
            "pre": pre,
            "normalized": normalized,
            "product": product,
            "native_logits": native_logits,
        }

    def logits_for_product(state: dict, product: torch.Tensor) -> torch.Tensor:
        write = model.blocks[1].mlp.Down(product) + model.blocks[1].mlp.Down_bias
        return suffix_from_layer1(
            model,
            state["x0"],
            state["value_bus"],
            state["pre"] + write,
        )[0, -1]

    records = []
    for category_index, category in enumerate(categories):
        name = f"semantic/seed101/{category}"
        key = safe_name(name + "_raw")
        all_vectors = torch.from_numpy(arrays[f"{key}_eigenvectors"][:, :16]).cuda()
        top_vectors = all_vectors[:, : config["rank"]]
        low_vectors = all_vectors[:, -config["rank"] :]
        wrong_category = categories[(category_index + 1) % len(categories)]
        wrong_key = safe_name(f"semantic/seed101/{wrong_category}_raw")
        wrong_vectors = torch.from_numpy(
            arrays[f"{wrong_key}_eigenvectors"][:, : config["rank"]]
        ).cuda()
        donors = [
            site for site in gate["sites"][category]
            if site["stratum"] == config["donor_stratum"]
        ]
        recipients = [
            site for site in gate["sites"][category]
            if site["stratum"] == config["recipient_stratum"]
        ]
        feature = features[category]

        for donor_site, recipient_site in zip(donors, recipients):
            donor = state_at(donor_site)
            recipient = state_at(recipient_site)
            donor_x = donor["normalized"][0, -1].float()
            recipient_x = recipient["normalized"][0, -1].float()
            donor_pre = float(
                F.linear(
                    (donor["product"][0, -1].float() - mean) / std,
                    encoder_weight,
                    encoder_bias,
                )[feature]
            )
            recipient_pre = float(
                F.linear(
                    (recipient["product"][0, -1].float() - mean) / std,
                    encoder_weight,
                    encoder_bias,
                )[feature]
            )
            denominator = donor_pre - recipient_pre

            def coordinate_delta(vectors: torch.Tensor) -> torch.Tensor:
                return ((donor_x @ vectors) - (recipient_x @ vectors)) @ vectors.T

            raw_deltas = {
                "top": coordinate_delta(top_vectors),
                "low": coordinate_delta(low_vectors),
                "wrong_category": coordinate_delta(wrong_vectors),
            }

            for fraction in config["norm_fractions"]:
                cap = fraction * recipient_x.norm()
                top_delta = raw_deltas["top"].clone()
                top_delta *= min(1.0, float(cap / top_delta.norm().clamp_min(1e-20)))
                target_norm = top_delta.norm()
                matched = {"top": top_delta}
                for variant in ("low", "wrong_category"):
                    delta = raw_deltas[variant].clone()
                    delta *= target_norm / delta.norm().clamp_min(1e-20)
                    matched[variant] = delta

                variants = {}
                for variant, delta in matched.items():
                    patched_product = recipient["product"].clone()
                    patched_x = recipient_x + delta
                    patched_product[0, -1] = (
                        model.blocks[1].mlp.Left(patched_x)
                        * model.blocks[1].mlp.Right(patched_x)
                    )
                    patched_logits = logits_for_product(recipient, patched_product)
                    patched_pre = float(
                        F.linear(
                            (patched_product[0, -1].float() - mean) / std,
                            encoder_weight,
                            encoder_bias,
                        )[feature]
                    )
                    transfer = (
                        (patched_pre - recipient_pre) / denominator
                        if abs(denominator) > 1e-12 else float("nan")
                    )
                    row = {
                        "input_delta_norm": float(delta.norm()),
                        "feature_transfer_fraction": transfer,
                        "feature_preactivation": patched_pre,
                    }

                    if variant == "top" and fraction == max(config["norm_fractions"]):
                        before = sparse_code(
                            recipient["product"][0, -1],
                            mean,
                            std,
                            encoder_weight,
                            encoder_bias,
                        )
                        after = sparse_code(
                            patched_product[0, -1],
                            mean,
                            std,
                            encoder_weight,
                            encoder_bias,
                        )
                        delta_code = after - before
                        actual_effect = patched_logits - recipient["native_logits"]
                        product_delta = patched_product[0, -1] - recipient["product"][0, -1]
                        contribution = delta_code.abs() * (
                            model.blocks[1].mlp.Down.weight.float().cuda()
                            @ decoder_original
                        ).square().sum(0).sqrt()
                        adaptive_order = torch.argsort(contribution, descending=True)
                        nonzero = int((delta_code != 0).sum())

                        def reconstruct(indices: torch.Tensor | list[int]) -> dict:
                            indices = torch.as_tensor(indices, device="cuda", dtype=torch.long)
                            reconstruction = decoder_original[:, indices] @ delta_code[indices]
                            predicted_product = recipient["product"].clone()
                            predicted_product[0, -1] += reconstruction
                            predicted_logits = logits_for_product(recipient, predicted_product)
                            fidelity, cosine, _ = fisher_fidelity(
                                patched_logits,
                                actual_effect,
                                predicted_logits - recipient["native_logits"],
                            )
                            product_cosine = float(
                                F.cosine_similarity(product_delta, reconstruction, dim=0)
                            )
                            relative_product_error = float(
                                (product_delta - reconstruction).norm()
                                / product_delta.norm().clamp_min(1e-20)
                            )
                            return {
                                "downstream_fidelity": fidelity,
                                "downstream_cosine": cosine,
                                "product_cosine": product_cosine,
                                "relative_product_error": relative_product_error,
                            }

                        reconstructions = {}
                        for budget in config["adaptive_budgets"]:
                            reconstructions[f"adaptive_{budget}"] = reconstruct(
                                adaptive_order[: min(budget, nonzero)]
                            )
                        reconstructions["fixed_32"] = reconstruct(
                            cores[category][: config["fixed_budget"]]
                        )
                        reconstructions["full_code"] = reconstruct(
                            torch.nonzero(delta_code).flatten()
                        )
                        row["reconstructions"] = reconstructions
                        row["changed_features"] = nonzero
                    variants[variant] = row

                records.append(
                    {
                        "category": category,
                        "wrong_category": wrong_category,
                        "fraction": fraction,
                        "donor": donor_site,
                        "recipient": recipient_site,
                        "donor_feature_preactivation": donor_pre,
                        "recipient_feature_preactivation": recipient_pre,
                        "variants": variants,
                    }
                )

    transfer_summary = {}
    for fraction in config["norm_fractions"]:
        transfer_summary[str(fraction)] = {}
        selected = [row for row in records if row["fraction"] == fraction]
        for variant in ("top", "low", "wrong_category"):
            transfer_summary[str(fraction)][variant] = float(
                np.nanmedian(
                    [row["variants"][variant]["feature_transfer_fraction"] for row in selected]
                )
            )

    max_fraction = max(config["norm_fractions"])
    decomposition_rows = [
        row["variants"]["top"]["reconstructions"]
        for row in records if row["fraction"] == max_fraction
    ]
    reconstruction_summary = {}
    names = [f"adaptive_{budget}" for budget in config["adaptive_budgets"]]
    names += ["fixed_32", "full_code"]
    for name in names:
        reconstruction_summary[name] = {
            metric: float(np.median([row[name][metric] for row in decomposition_rows]))
            for metric in (
                "downstream_fidelity",
                "downstream_cosine",
                "product_cosine",
                "relative_product_error",
            )
        }

    top_doses = [
        transfer_summary[str(fraction)]["top"]
        for fraction in config["norm_fractions"]
    ]
    top_at_max = transfer_summary[str(max_fraction)]["top"]
    low_at_max = transfer_summary[str(max_fraction)]["low"]
    wrong_at_max = transfer_summary[str(max_fraction)]["wrong_category"]
    full_fidelity = reconstruction_summary["full_code"]["downstream_fidelity"]
    adaptive_32 = reconstruction_summary["adaptive_32"]["downstream_fidelity"]
    fixed_32 = reconstruction_summary["fixed_32"]["downstream_fidelity"]
    hypotheses = {
        "R1_dose_response": {
            "supported": all(a < b for a, b in zip(top_doses, top_doses[1:])),
            "top_doses": top_doses,
        },
        "R2_spectral_specificity": {
            "supported": top_at_max - max(low_at_max, wrong_at_max) >= 0.10,
            "top": top_at_max,
            "low": low_at_max,
            "wrong_category": wrong_at_max,
        },
        "R3_full_sae_coverage": {
            "supported": full_fidelity >= 0.50,
            "fidelity": full_fidelity,
        },
        "R4_adaptive_compactness": {
            "supported": adaptive_32 >= 0.8 * full_fidelity,
            "adaptive_32": adaptive_32,
            "full_code": full_fidelity,
            "fraction": adaptive_32 / max(full_fidelity, 1e-30),
        },
        "R5_fixed_core_transfer": {
            "supported": fixed_32 >= 0.75 * adaptive_32,
            "fixed_32": fixed_32,
            "adaptive_32": adaptive_32,
            "fraction": fixed_32 / max(adaptive_32, 1e-30),
        },
    }

    output = {
        "config": config,
        "transfer_summary": transfer_summary,
        "reconstruction_summary": reconstruction_summary,
        "hypotheses": hypotheses,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Spectral donor-transfer dose and decomposition red-team",
        "",
        "This tests whether donor-coordinate transfer is dose-responsive and whether its downstream effect is carried by a fixed category core, a context-adaptive sparse ensemble, or changes outside the SAE code.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Dose and decomposition](spectral_donor_decomposition.png)",
        "",
        "| Input-norm dose | Top modes | Low modes | Wrong category |",
        "|---:|---:|---:|---:|",
    ]
    for fraction in config["norm_fractions"]:
        row = transfer_summary[str(fraction)]
        lines.append(
            f"| {fraction:.0%} | {row['top']:.3f} | {row['low']:.3f} | "
            f"{row['wrong_category']:.3f} |"
        )
    lines += [
        "",
        "| Reconstruction | Downstream fidelity | Product cosine | Relative product error |",
        "|---|---:|---:|---:|",
    ]
    for name, row in reconstruction_summary.items():
        lines.append(
            f"| {name} | {row['downstream_fidelity']:.3f} | "
            f"{row['product_cosine']:.3f} | {row['relative_product_error']:.3f} |"
        )
    lines += [
        "",
        "Raw paired examples are in `spectral_donor_decomposition.json`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for variant in ("top", "low", "wrong_category"):
        axes[0].plot(
            config["norm_fractions"],
            [transfer_summary[str(f)][variant] for f in config["norm_fractions"]],
            marker="o",
            label=variant,
        )
    axes[0].set_xlabel("input edit / recipient norm")
    axes[0].set_ylabel("median donor-feature transfer")
    axes[0].legend()
    plot_names = ["adaptive_1", "adaptive_4", "adaptive_16", "adaptive_32", "adaptive_64", "adaptive_128", "adaptive_256", "full_code", "fixed_32"]
    axes[1].plot(
        range(len(plot_names)),
        [reconstruction_summary[name]["downstream_fidelity"] for name in plot_names],
        marker="o",
    )
    axes[1].set_xticks(range(len(plot_names)), plot_names, rotation=45, ha="right")
    axes[1].set_ylabel("median downstream fidelity")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
