#!/usr/bin/env python3
"""Build and validate the exact readout-to-interaction-matrix interpreter."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, token_flags
from run_bilin18_joint_axis_atlas import ARRAYS as ATLAS_ARRAYS
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_query_conditioned_interpreter.json")
OUTPUT = TENSOR_ROOT / "query_conditioned_interpreter.json"
ARRAYS = TENSOR_ROOT / "query_conditioned_interpreter.npz"
REPORT = TENSOR_ROOT / "QUERY_CONDITIONED_INTERPRETER.md"
FIGURE = TENSOR_ROOT / "query_conditioned_interpreter.png"


def normalize(rows: torch.Tensor) -> torch.Tensor:
    return rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-20)


def score_metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = target - target.mean(0, keepdims=True)
    prediction = prediction - prediction.mean(0, keepdims=True)
    covariance = (target * prediction).sum(0)
    correlation = covariance / np.maximum(
        np.sqrt((target * target).sum(0) * (prediction * prediction).sum(0)), 1e-30
    )
    r2 = 1 - ((prediction - target) ** 2).sum(0) / np.maximum((target ** 2).sum(0), 1e-30)
    return correlation, r2


@torch.inference_mode()
def collect_states_and_writes(model, corpus_config: dict, positions: list[int], batch_size: int):
    source = np.load(corpus_config["path"], mmap_mode="r")
    rows = source[corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]]
    total = len(rows) * len(positions)
    states = np.empty((total, 1152), dtype=np.float32)
    writes = np.empty((total, 1152), dtype=np.float32)
    next_ids = np.empty(total, dtype=np.int64)
    offset = 0
    block = model.blocks[17]
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start : start + batch_size, :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        _, x = prefix_to_layer17(model, ids)
        selected = x[:, positions].float().flatten(0, 1)
        product = block.mlp.Left(selected) * block.mlp.Right(selected)
        write = block.mlp.Down(product)
        count = len(selected)
        states[offset : offset + count] = selected.cpu().numpy()
        writes[offset : offset + count] = write.cpu().numpy()
        next_ids[offset : offset + count] = raw[:, np.asarray(positions) + 1].reshape(-1)
        offset += count
    return states, writes, next_ids


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[17]
    axes = torch.from_numpy(np.load(ATLAS_ARRAYS)["layer17_axes"]).cuda()
    left_axes = block.mlp.Left.weight.float() @ axes
    right_axes = block.mlp.Right.weight.float() @ axes
    down = block.mlp.Down.weight.float()

    # T[d,i,j] is the exact linear map from a residual readout y[d] to the
    # symmetric 32D quadratic interaction matrix A_y[i,j].
    query_tensor = torch.einsum("dh,hi,hj->dij", down, left_axes, right_axes)
    query_tensor = 0.5 * (query_tensor + query_tensor.transpose(1, 2))
    query_flat = query_tensor.flatten(1)
    map_u, map_s, map_vh = torch.linalg.svd(query_flat, full_matrices=False)

    discovery_states, discovery_writes, discovery_next = collect_states_and_writes(
        model, config["discovery"], config["positions"], config["batch_size"]
    )
    discovery_write_tensor = torch.from_numpy(discovery_writes).cuda()
    centered_write = discovery_write_tensor - discovery_write_tensor.mean(0, keepdim=True)
    _, _, natural_vh = torch.pca_lowrank(
        centered_write,
        q=config["queries_per_family"],
        center=False,
        niter=4,
    )
    natural_queries = natural_vh.T

    unembedding = model.lm_head.weight.float()[: tokenizer.vocab_size]
    flags = np.stack([
        token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)
    ])
    semantic_rows = []
    semantic_names = []
    for concept in config["semantic_concepts"]:
        positive = torch.as_tensor(
            flags[:, CATEGORIES.index(concept)].astype(bool), device="cuda"
        )
        semantic_rows.append(unembedding[positive].mean(0) - unembedding[~positive].mean(0))
        semantic_names.append(concept)
    semantic_queries = normalize(torch.stack(semantic_rows))

    counts = np.bincount(discovery_next, minlength=tokenizer.vocab_size)
    frequent_ids = np.argsort(counts)[::-1][: config["queries_per_family"]].copy()
    frequent_queries = normalize(unembedding[torch.as_tensor(frequent_ids, device="cuda")])
    rng = np.random.default_rng(config["seed"])
    random_ids = rng.choice(tokenizer.vocab_size, size=config["queries_per_family"], replace=False)
    random_token_queries = normalize(unembedding[torch.as_tensor(random_ids, device="cuda")])
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    gaussian_queries = normalize(torch.randn(
        config["queries_per_family"], 1152, generator=generator, device="cuda"
    ))
    query_families = {
        "semantic": semantic_queries,
        "frequent_token": frequent_queries,
        "random_token": random_token_queries,
        "gaussian": gaussian_queries,
        "natural_write_pca": natural_queries,
    }

    # Algebraic exactness check, independent of corpus evaluation.
    test_y = gaussian_queries[:8]
    test_z = torch.randn(16, 32, generator=generator, device="cuda")
    projected_write = down @ (
        (test_z @ left_axes.T) * (test_z @ right_axes.T)
    ).T
    direct_scores = projected_write.T @ test_y.T
    mapped_forms = (test_y @ query_flat).reshape(-1, 32, 32)
    mapped_scores = torch.einsum("ni,qij,nj->nq", test_z, mapped_forms, test_z)
    exact_relative_error = float(
        (direct_scores - mapped_scores).abs().max()
        / direct_scores.abs().max().clamp_min(1e-20)
    )

    results = {}
    rank_results = {}
    for corpus_name, corpus_config in config["evaluation"].items():
        states, writes, _ = collect_states_and_writes(
            model, corpus_config, config["positions"], config["batch_size"]
        )
        x = torch.from_numpy(states).cuda()
        z = x @ axes
        phi = torch.einsum("ni,nj->nij", z, z).flatten(1)
        write = torch.from_numpy(writes).cuda()
        results[corpus_name] = {}
        rank_results[corpus_name] = {}
        for family, queries in query_families.items():
            target = (write @ queries.T).cpu().numpy()
            full_forms = queries @ query_flat
            full_prediction = (phi @ full_forms.T).cpu().numpy()
            correlation, r2 = score_metrics(target, full_prediction)
            results[corpus_name][family] = {
                "median_correlation": float(np.median(correlation)),
                "median_centered_r2": float(np.median(r2)),
                "q10_correlation": float(np.quantile(correlation, 0.1)),
                "q90_correlation": float(np.quantile(correlation, 0.9)),
            }
            rank_results[corpus_name][family] = {}
            for rank in config["map_ranks"]:
                compressed_forms = (
                    (queries @ map_u[:, :rank]) * map_s[:rank]
                ) @ map_vh[:rank]
                prediction = (phi @ compressed_forms.T).cpu().numpy()
                rank_correlation, rank_r2 = score_metrics(target, prediction)
                rank_results[corpus_name][family][str(rank)] = {
                    "median_correlation": float(np.median(rank_correlation)),
                    "median_centered_r2": float(np.median(rank_r2)),
                }

    energy = map_s.square()
    cumulative_energy = torch.cumsum(energy, 0) / energy.sum()
    bulk_start, bulk_end = 32, 512
    ranks = np.arange(bulk_start + 1, bulk_end + 1)
    bulk_values = map_s[bulk_start:bulk_end].cpu().numpy()
    slope, intercept = np.polyfit(np.log(ranks), np.log(np.maximum(bulk_values, 1e-30)), 1)
    fitted = intercept + slope * np.log(ranks)
    bulk_r2 = float(1 - np.square(np.log(bulk_values) - fitted).sum() / np.square(np.log(bulk_values) - np.log(bulk_values).mean()).sum())
    full = results
    rank128 = rank_results
    hypotheses = {
        "Q1_exact_query_map": {
            "supported": exact_relative_error < 1e-5,
            "maximum_relative_error": exact_relative_error,
        },
        "Q2_semantic_query_transfer": {
            "supported": all(full[c]["semantic"]["median_correlation"] >= 0.75 for c in full),
            "per_corpus": {c: full[c]["semantic"]["median_correlation"] for c in full},
        },
        "Q3_frequent_token_transfer": {
            "supported": all(full[c]["frequent_token"]["median_correlation"] >= 0.50 for c in full),
            "per_corpus": {c: full[c]["frequent_token"]["median_correlation"] for c in full},
        },
        "Q4_language_query_advantage": {
            "supported": all(
                full[c]["frequent_token"]["median_correlation"]
                - full[c]["gaussian"]["median_correlation"] >= 0.15 for c in full
            ),
            "gaps": {
                c: full[c]["frequent_token"]["median_correlation"]
                - full[c]["gaussian"]["median_correlation"] for c in full
            },
        },
        "Q5_rank128_map_compression": {
            "supported": all(
                full[c]["semantic"]["median_correlation"] - rank128[c]["semantic"]["128"]["median_correlation"] <= 0.03
                and full[c]["frequent_token"]["median_correlation"] - rank128[c]["frequent_token"]["128"]["median_correlation"] <= 0.05
                for c in full
            ),
            "losses": {
                c: {
                    family: full[c][family]["median_correlation"]
                    - rank128[c][family]["128"]["median_correlation"]
                    for family in ("semantic", "frequent_token")
                }
                for c in full
            },
        },
        "Q6_natural_write_queries": {
            "supported": all(full[c]["natural_write_pca"]["median_correlation"] >= 0.70 for c in full),
            "per_corpus": {c: full[c]["natural_write_pca"]["median_correlation"] for c in full},
        },
    }
    output = {
        "config": config,
        "exact_relative_error": exact_relative_error,
        "map_spectrum": {
            "top_singular_values": map_s[:32].cpu().tolist(),
            "energy_by_rank": {
                str(rank): float(cumulative_energy[rank - 1]) for rank in config["map_ranks"]
            },
            "bulk_power_law_exponent": float(-slope),
            "bulk_loglog_r2": bulk_r2,
        },
        "queries": {
            "semantic_names": semantic_names,
            "frequent_tokens": [tokenizer.decode([int(index)]) for index in frequent_ids],
            "random_tokens": [tokenizer.decode([int(index)]) for index in random_ids],
        },
        "results": results,
        "rank_results": rank_results,
        "hypotheses": hypotheses,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))
    np.savez_compressed(
        ARRAYS,
        singular_values=map_s.cpu().numpy(),
        output_components=map_u[:, :256].cpu().numpy(),
        matrix_components=map_vh[:256].reshape(256, 32, 32).cpu().numpy(),
        frequent_ids=frequent_ids,
        random_ids=random_ids,
    )

    lines = [
        "# Query-conditioned interaction interpreter",
        "",
        "For the frozen layer-17 32D input subspace, the bilinear weights define an exact linear map from any residual readout direction to a 32×32 symmetric interaction matrix. This report tests whether those mapped scalar computations reproduce the full MLP on natural states for readout families not used to construct the subspace.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Query interpreter](query_conditioned_interpreter.png)",
        "",
        "| Corpus | Semantic | Frequent token | Random token | Gaussian | Natural-write PCA |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus_name, families in results.items():
        lines.append(
            f"| {corpus_name} | {families['semantic']['median_correlation']:.3f} | "
            f"{families['frequent_token']['median_correlation']:.3f} | "
            f"{families['random_token']['median_correlation']:.3f} | "
            f"{families['gaussian']['median_correlation']:.3f} | "
            f"{families['natural_write_pca']['median_correlation']:.3f} |"
        )
    lines += [
        "",
        f"The exact map check has maximum relative error `{exact_relative_error:.2e}`. The map spectrum has fitted bulk exponent {(-slope):.3f} with log-log R² {bulk_r2:.3f}; rank-128 contains {float(cumulative_energy[127]):.1%} of global map energy.",
        "",
        "This interpreter answers a declared scalar readout query. It is not a replacement for the entire MLP, and high correlation does not imply that the 32D state is sufficient for all logits simultaneously.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(11, 4.2))
    plots[0].loglog(np.arange(1, len(map_s) + 1), map_s.cpu().numpy())
    plots[0].set_xlabel("query-map singular rank")
    plots[0].set_ylabel("singular value")
    plots[0].set_title("Readout→interaction map spectrum")
    families = ["semantic", "frequent_token", "random_token", "gaussian", "natural_write_pca"]
    x_positions = np.arange(len(families))
    for corpus_name in results:
        plots[1].plot(
            x_positions,
            [results[corpus_name][family]["median_correlation"] for family in families],
            marker="o",
            label=corpus_name,
        )
    plots[1].set_xticks(x_positions, families, rotation=25, ha="right")
    plots[1].set_ylabel("median natural-score correlation")
    plots[1].set_ylim(-0.1, 1.02)
    plots[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
