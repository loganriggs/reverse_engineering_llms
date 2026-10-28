#!/usr/bin/env python3
"""Separate SAE encoder, support-selection, and dictionary-span failures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_companion_paths import sparse_code
from run_bilin18_spectral_fixed_companions import OUTPUT as CORE_OUTPUT
from run_bilin18_spectral_gate_validation import CONFIG as GATE_CONFIG, OUTPUT as GATE_OUTPUT
from run_bilin18_spectral_input_interventions import euclidean_fidelity
from run_bilin18_spectral_spike_pilot import (
    ARRAYS as PILOT_ARRAYS,
    SPARSE,
    TENSOR_ROOT,
    fisher_fidelity,
    safe_name,
)
from train_bilin18_causal_cp import prefix_to_layer1, suffix_from_layer1


CONFIG = Path("configs/bilin18_spectral_decoder_refit.json")
OUTPUT = TENSOR_ROOT / "spectral_decoder_refit.json"
REPORT = TENSOR_ROOT / "SPECTRAL_DECODER_REFIT.md"
FIGURE = TENSOR_ROOT / "spectral_decoder_refit.png"


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    gate_config = json.loads(GATE_CONFIG.read_text())
    gate = json.loads(GATE_OUTPUT.read_text())
    cores = json.loads(CORE_OUTPUT.read_text())["cores"]
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

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    down = model.blocks[1].mlp.Down.weight.float()
    write_dictionary = down @ decoder_original
    unit_write_dictionary = write_dictionary / write_dictionary.norm(dim=0).clamp_min(1e-20)
    gram = write_dictionary @ write_dictionary.T
    ridge = config["ridge_fraction"] * float(torch.trace(gram) / gram.shape[0])
    chol = torch.linalg.cholesky(
        gram + ridge * torch.eye(gram.shape[0], device="cuda")
    )

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
            "native_write": native_write,
            "native_logits": native_logits,
        }

    def logits_with_write_delta(state: dict, delta: torch.Tensor) -> torch.Tensor:
        write = state["native_write"].clone()
        write[0, -1] += delta
        return suffix_from_layer1(
            model,
            state["x0"],
            state["value_bus"],
            state["pre"] + write,
        )[0, -1]

    records = []
    for category_index, category in enumerate(categories):
        key = safe_name(f"semantic/seed101/{category}_raw")
        vectors = torch.from_numpy(
            arrays[f"{key}_eigenvectors"][:, : config["rank"]]
        ).cuda()
        wrong_category = categories[(category_index + 1) % len(categories)]
        donors = [
            site for site in gate["sites"][category]
            if site["stratum"] == config["donor_stratum"]
        ]
        recipients = [
            site for site in gate["sites"][category]
            if site["stratum"] == config["recipient_stratum"]
        ]

        for donor_site, recipient_site in zip(donors, recipients):
            donor = state_at(donor_site)
            recipient = state_at(recipient_site)
            donor_x = donor["normalized"][0, -1].float()
            recipient_x = recipient["normalized"][0, -1].float()
            input_delta = ((donor_x @ vectors) - (recipient_x @ vectors)) @ vectors.T
            cap = config["input_norm_fraction"] * recipient_x.norm()
            input_delta *= min(1.0, float(cap / input_delta.norm().clamp_min(1e-20)))
            patched_product = recipient["product"].clone()
            patched_x = recipient_x + input_delta
            patched_product[0, -1] = (
                model.blocks[1].mlp.Left(patched_x)
                * model.blocks[1].mlp.Right(patched_x)
            )
            product_delta = patched_product[0, -1] - recipient["product"][0, -1]
            target_write = down @ product_delta
            actual_logits = logits_with_write_delta(recipient, target_write)
            actual_effect = actual_logits - recipient["native_logits"]

            before = sparse_code(
                recipient["product"][0, -1], mean, std, encoder_weight, encoder_bias
            )
            after = sparse_code(
                patched_product[0, -1], mean, std, encoder_weight, encoder_bias
            )
            delta_code = after - before
            changed_support = torch.nonzero(delta_code).flatten()

            def evaluate_write(predicted_write: torch.Tensor) -> dict:
                predicted_logits = logits_with_write_delta(recipient, predicted_write)
                downstream, downstream_cosine, _ = fisher_fidelity(
                    actual_logits,
                    actual_effect,
                    predicted_logits - recipient["native_logits"],
                )
                direct, direct_cosine, _ = euclidean_fidelity(
                    target_write, predicted_write
                )
                return {
                    "downstream_fidelity": downstream,
                    "downstream_cosine": downstream_cosine,
                    "residual_fidelity": direct,
                    "residual_cosine": direct_cosine,
                }

            def least_squares_support(indices: torch.Tensor | list[int]) -> dict:
                indices = torch.as_tensor(indices, device="cuda", dtype=torch.long)
                basis = write_dictionary[:, indices]
                coefficients = torch.linalg.lstsq(basis, target_write).solution
                return evaluate_write(basis @ coefficients)

            observed_write = write_dictionary @ delta_code
            methods = {"observed_full_code": evaluate_write(observed_write)}
            methods["encoder_support_refit"] = least_squares_support(changed_support)
            methods["fixed_own_32_refit"] = least_squares_support(
                cores[category][: config["fixed_budget"]]
            )
            methods["fixed_wrong_32_refit"] = least_squares_support(
                cores[wrong_category][: config["fixed_budget"]]
            )
            correlation = (unit_write_dictionary.T @ target_write).abs()
            order = torch.argsort(correlation, descending=True)
            for budget in config["oracle_budgets"]:
                methods[f"oracle_correlation_{budget}"] = least_squares_support(
                    order[:budget]
                )
            alpha = torch.cholesky_solve(target_write[:, None], chol)[:, 0]
            methods["full_dictionary_ridge"] = evaluate_write(gram @ alpha)
            records.append(
                {
                    "category": category,
                    "wrong_category": wrong_category,
                    "donor": donor_site,
                    "recipient": recipient_site,
                    "changed_support_size": int(changed_support.numel()),
                    "methods": methods,
                }
            )

    method_names = list(records[0]["methods"])
    summary = {
        method: {
            metric: float(np.median([row["methods"][method][metric] for row in records]))
            for metric in (
                "downstream_fidelity",
                "downstream_cosine",
                "residual_fidelity",
                "residual_cosine",
            )
        }
        for method in method_names
    }
    full = summary["full_dictionary_ridge"]
    observed = summary["observed_full_code"]["downstream_fidelity"]
    support_refit = summary["encoder_support_refit"]["downstream_fidelity"]
    oracle32 = summary["oracle_correlation_32"]["downstream_fidelity"]
    own = summary["fixed_own_32_refit"]["downstream_fidelity"]
    wrong = summary["fixed_wrong_32_refit"]["downstream_fidelity"]
    hypotheses = {
        "E1_dictionary_span": {
            "supported": full["residual_fidelity"] >= 0.95
            and full["downstream_fidelity"] >= 0.95,
            **full,
        },
        "E2_encoder_coefficient_failure": {
            "supported": support_refit - observed >= 0.20,
            "observed": observed,
            "support_refit": support_refit,
            "gain": support_refit - observed,
        },
        "E3_oracle_compactness": {
            "supported": oracle32 >= 0.70,
            "fidelity": oracle32,
        },
        "E4_fixed_membership": {
            "supported": own >= 0.50,
            "fidelity": own,
        },
        "E5_category_specificity": {
            "supported": own - wrong >= 0.15,
            "own": own,
            "wrong": wrong,
            "gap": own - wrong,
        },
    }
    output = {
        "config": config,
        "ridge_absolute": ridge,
        "summary": summary,
        "hypotheses": hypotheses,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Decoder-refit red-team",
        "",
        "The same donor spectral intervention is evaluated after removing the SAE encoder from coefficient estimation. Supports receive least-squares-optimal coefficients for the exact native MLP residual-write change. This distinguishes dictionary span, support membership, and encoder coefficient errors.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Decoder refit](spectral_decoder_refit.png)",
        "",
        "| Method | Residual fidelity | Downstream fidelity |",
        "|---|---:|---:|",
    ]
    for method, row in summary.items():
        lines.append(
            f"| {method} | {row['residual_fidelity']:.3f} | "
            f"{row['downstream_fidelity']:.3f} |"
        )
    lines += ["", "Per-example fits are in `spectral_decoder_refit.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    plot_methods = [
        "observed_full_code",
        "encoder_support_refit",
        "fixed_own_32_refit",
        "fixed_wrong_32_refit",
        "oracle_correlation_4",
        "oracle_correlation_16",
        "oracle_correlation_32",
        "oracle_correlation_64",
        "oracle_correlation_128",
        "full_dictionary_ridge",
    ]
    x = np.arange(len(plot_methods))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        x,
        [summary[name]["residual_fidelity"] for name in plot_methods],
        marker="o",
        label="residual",
    )
    ax.plot(
        x,
        [summary[name]["downstream_fidelity"] for name in plot_methods],
        marker="o",
        label="downstream",
    )
    ax.set_xticks(x, plot_methods, rotation=45, ha="right")
    ax.set_ylabel("median fidelity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
