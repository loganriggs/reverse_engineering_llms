#!/usr/bin/env python3
"""Evaluate an actually contracted and factored Hybrid-64 final MLP."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_input_basis_tradeoffs import ARRAYS as BASIS_ARRAYS
from run_bilin18_query_conditioned_interpreter import CONFIG as INTERPRETER_CONFIG
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_hybrid64_efficient_factorization.json")
OUTPUT = TENSOR_ROOT / "hybrid64_efficient_factorization.json"
ARRAYS = TENSOR_ROOT / "hybrid64_efficient_factorization.npz"
REPORT = TENSOR_ROOT / "HYBRID64_EFFICIENT_FACTORIZATION.md"
FIGURE = TENSOR_ROOT / "hybrid64_efficient_factorization.png"


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    interpreter_config = json.loads(INTERPRETER_CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[17]
    model_dimension = model.config.n_embd
    channels = block.mlp.Left.weight.shape[0]
    basis = torch.from_numpy(np.load(BASIS_ARRAYS)[config["basis"]]).cuda()
    basis_dimension = basis.shape[1]
    left_basis = block.mlp.Left.weight.float() @ basis
    right_basis = block.mlp.Right.weight.float() @ basis
    tensor = torch.einsum("dh,hi,hj->dij", block.mlp.Down.weight.float(), left_basis, right_basis)
    tensor = 0.5 * (tensor + tensor.transpose(1, 2))
    tensor_flat = tensor.flatten(1)
    output_u, singular, matrix_vh = torch.linalg.svd(tensor_flat, full_matrices=False)
    cumulative_energy = torch.cumsum(singular.square(), 0) / singular.square().sum()

    native_multiplications = 3 * model_dimension * channels
    compute = {}
    for rank in config["ranks"]:
        factor_multiplications = (
            model_dimension * basis_dimension
            + basis_dimension * basis_dimension
            + rank * basis_dimension * basis_dimension
            + rank * model_dimension
        )
        factor_parameters = (
            model_dimension * basis_dimension
            + rank * basis_dimension * basis_dimension
            + rank * model_dimension
            + model_dimension
        )
        compute[str(rank)] = {
            "multiplications_per_token": factor_multiplications,
            "speedup_over_native_dense_multiplications": native_multiplications / factor_multiplications,
            "parameters_including_basis_and_bias": factor_parameters,
            "tensor_energy": float(cumulative_energy[rank - 1]),
        }

    results = {}
    exact_relative_errors = []
    positions = interpreter_config["positions"]
    for corpus_name, corpus_config in interpreter_config["evaluation"].items():
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]]
        accumulators = {
            str(rank): {"native_kl": [], "hybrid_kl": [], "ce": [], "agreement": []}
            for rank in config["ranks"]
        }
        native_ce = []
        exact_hybrid_ce = []
        exact_native_kl = []
        for start in range(0, len(rows), interpreter_config["batch_size"]):
            raw = np.asarray(rows[start : start + interpreter_config["batch_size"], :129], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :128], device="cuda")
            pre, x = prefix_to_layer17(model, ids)
            selected_pre = pre[:, positions].float().flatten(0, 1)
            selected_x = x[:, positions].float().flatten(0, 1)
            target = torch.as_tensor(raw[:, np.asarray(positions) + 1].reshape(-1), device="cuda")
            native_write = block.mlp.Down(block.mlp.Left(selected_x) * block.mlp.Right(selected_x)) + block.mlp.Down_bias
            native_logits = 30 * torch.tanh(
                model.lm_head(F.rms_norm(selected_pre + native_write, (model_dimension,))).float() / 30
            )
            native_log_prob = native_logits.log_softmax(-1)
            native_probability = native_log_prob.exp()
            native_top = native_logits.argmax(-1)
            native_ce.extend(F.cross_entropy(native_logits, target, reduction="none").cpu().tolist())

            z = selected_x @ basis
            phi = torch.einsum("ni,nj->nij", z, z).flatten(1)
            exact_write = phi @ tensor_flat.T + block.mlp.Down_bias
            projected_x = z @ basis.T
            projected_native_write = block.mlp.Down(
                block.mlp.Left(projected_x) * block.mlp.Right(projected_x)
            ) + block.mlp.Down_bias
            exact_relative_errors.append(float(
                (exact_write - projected_native_write).norm()
                / projected_native_write.norm().clamp_min(1e-20)
            ))
            exact_logits = 30 * torch.tanh(
                model.lm_head(F.rms_norm(selected_pre + exact_write, (model_dimension,))).float() / 30
            )
            exact_log_prob = exact_logits.log_softmax(-1)
            exact_probability = exact_log_prob.exp()
            exact_hybrid_ce.extend(F.cross_entropy(exact_logits, target, reduction="none").cpu().tolist())
            exact_native_kl.extend(
                (native_probability * (native_log_prob - exact_log_prob)).sum(-1).cpu().tolist()
            )

            for rank in config["ranks"]:
                features = phi @ matrix_vh[:rank].T
                write = (features * singular[:rank]) @ output_u[:, :rank].T + block.mlp.Down_bias
                logits = 30 * torch.tanh(
                    model.lm_head(F.rms_norm(selected_pre + write, (model_dimension,))).float() / 30
                )
                log_prob = logits.log_softmax(-1)
                values = accumulators[str(rank)]
                values["native_kl"].extend((native_probability * (native_log_prob - log_prob)).sum(-1).cpu().tolist())
                values["hybrid_kl"].extend((exact_probability * (exact_log_prob - log_prob)).sum(-1).cpu().tolist())
                values["ce"].extend(F.cross_entropy(logits, target, reduction="none").cpu().tolist())
                values["agreement"].extend((logits.argmax(-1) == native_top).cpu().tolist())

        native_ce_mean = float(np.mean(native_ce))
        results[corpus_name] = {
            "native_ce": native_ce_mean,
            "exact_hybrid": {
                "median_native_kl": float(np.median(exact_native_kl)),
                "ce_increase": float(np.mean(exact_hybrid_ce) - native_ce_mean),
            },
            "ranks": {},
        }
        for rank, values in accumulators.items():
            results[corpus_name]["ranks"][rank] = {
                "median_native_kl": float(np.median(values["native_kl"])),
                "median_additional_hybrid_kl": float(np.median(values["hybrid_kl"])),
                "ce_increase": float(np.mean(values["ce"]) - native_ce_mean),
                "native_top1_agreement": float(np.mean(values["agreement"])),
            }

    maximum_exact_error = max(exact_relative_errors)
    hypotheses = {
        "E1_exact_contraction": {
            "supported": maximum_exact_error < 1e-5,
            "maximum_relative_error": maximum_exact_error,
        },
        "E2_rank128_preserves_hybrid": {
            "supported": all(row["ranks"]["128"]["median_additional_hybrid_kl"] <= 0.005 for row in results.values()),
            "per_corpus": {corpus: row["ranks"]["128"]["median_additional_hybrid_kl"] for corpus, row in results.items()},
        },
        "E3_rank128_native_fidelity": {
            "supported": all(row["ranks"]["128"]["median_native_kl"] <= 0.05 for row in results.values()),
            "per_corpus": {corpus: row["ranks"]["128"]["median_native_kl"] for corpus, row in results.items()},
        },
        "E4_rank128_compute_reduction": {
            "supported": compute["128"]["speedup_over_native_dense_multiplications"] >= 10,
            **compute["128"],
            "native_multiplications_per_token": native_multiplications,
        },
        "E5_rank64_useful": {
            "supported": all(row["ranks"]["64"]["median_additional_hybrid_kl"] <= 0.01 for row in results.values()),
            "per_corpus": {corpus: row["ranks"]["64"]["median_additional_hybrid_kl"] for corpus, row in results.items()},
        },
    }
    output = {
        "config": config,
        "compute": compute,
        "native_multiplications_per_token": native_multiplications,
        "results": results,
        "hypotheses": hypotheses,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))
    np.savez_compressed(
        ARRAYS,
        basis=basis.cpu().numpy(),
        singular_values=singular.cpu().numpy(),
        output_factors=output_u[:, :512].cpu().numpy(),
        quadratic_factors=matrix_vh[:512].reshape(512, basis_dimension, basis_dimension).cpu().numpy(),
    )

    lines = [
        "# Efficient factorization of the Hybrid-64 final MLP",
        "",
        "The native 4,608-channel final MLP is first contracted exactly into an output×64×64 tensor, then SVD-factorized into shared scalar quadratic features and output directions. Evaluation uses the factored program directly; native channels are not scored.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Efficient factorization](hybrid64_efficient_factorization.png)",
        "",
        "| Rank | Tensor energy | Counted multiplications | Dense speedup | FineWeb native KL | WikiText native KL | TinyStories native KL |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank in config["ranks"]:
        key = str(rank)
        lines.append(
            f"| {rank} | {compute[key]['tensor_energy']:.1%} | {compute[key]['multiplications_per_token']:,} | "
            f"{compute[key]['speedup_over_native_dense_multiplications']:.1f}× | "
            f"{results['fineweb']['ranks'][key]['median_native_kl']:.4f} | "
            f"{results['wikitext']['ranks'][key]['median_native_kl']:.4f} | "
            f"{results['tinystories']['ranks'][key]['median_native_kl']:.4f} |"
        )
    lines += [
        "",
        "Multiply counts include input projection, explicit quadratic monomials, quadratic-feature scoring, and output decoding. They omit RMSNorm, unembedding, memory traffic, and kernel-launch overhead, so they are algorithmic—not measured wall-clock—speedups.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(11, 4.2))
    for corpus, row in results.items():
        plots[0].plot(config["ranks"], [row["ranks"][str(rank)]["median_native_kl"] for rank in config["ranks"]], marker="o", label=corpus)
        plots[1].plot(config["ranks"], [row["ranks"][str(rank)]["median_additional_hybrid_kl"] for rank in config["ranks"]], marker="o", label=corpus)
    plots[0].axhline(0.05, color="black", linestyle=":", linewidth=1)
    plots[0].set_ylabel("median KL from native model")
    plots[1].axhline(0.005, color="black", linestyle=":", linewidth=1)
    plots[1].set_ylabel("additional KL from exact Hybrid-64")
    for plot in plots:
        plot.set_xscale("log", base=2)
        plot.set_yscale("log")
        plot.set_xlabel("tensor factor rank")
        plot.legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
