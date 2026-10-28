#!/usr/bin/env python3
"""Compare full versus block-diagonal use of the same Layer-17 tail span."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer17_bottleneck_tail_ontology import corpus_rows
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_layer17_tail_cleaning_sweep import covariance_transforms, trace_clean
from run_bilin18_layer17_tail_continuous_fisher import continuous_prediction, score_summary, summarize
from run_bilin18_layer17_tail_tensor_canonicalization import common_input
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_tail_cross_term_redteam.json")
SOURCE_ARRAYS = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.npz"
FISHER_ARRAYS = TENSOR_ROOT / "layer17_tail_continuous_fisher.npz"
OUTPUT = TENSOR_ROOT / "layer17_tail_cross_term_redteam.json"
ARRAYS = TENSOR_ROOT / "layer17_tail_cross_term_redteam.npz"
REPORT = TENSOR_ROOT / "LAYER17_TAIL_CROSS_TERM_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer17_tail_cross_term_redteam.png"


def program_prediction(coordinates, forms, blocks):
    return sum(continuous_prediction(coordinates, forms, quotient) for quotient in blocks)


@torch.inference_mode()
def evaluate_programs(model, spec, config, output_basis, inverse, candidate_basis, forms, traces, programs):
    rows = corpus_rows(spec); positions = config["positions"]
    block = model.blocks[config["layer"]]; dimension = model.config.n_embd
    records = {name: {field: [] for field in ("exact_kl", "ce_delta", "top1")} for name in programs}
    cross_records = []
    score_target, score_predictions = [], {name: [] for name in programs}
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        pre, normalized = prefix_to_layer17(model, ids)
        pre = pre[:, positions].float().flatten(0, 1)
        x = normalized[:, positions].float().flatten(0, 1)
        raw_write = block.mlp.Down(block.mlp.Left(x) * block.mlp.Right(x)).float()
        native_write = raw_write + block.mlp.Down_bias.float()
        native_score = raw_write @ output_basis
        z = x @ inverse; coordinates = z @ candidate_basis
        isotropic = z.square().sum(1, keepdim=True) * traces[None]
        predictions = {
            name: program_prediction(coordinates, forms, blocks) + isotropic
            for name, blocks in programs.items()
        }
        score_target.append(native_score.cpu())
        for name, prediction in predictions.items(): score_predictions[name].append(prediction.cpu())
        native_logits = 30 * torch.tanh(model.lm_head(F.rms_norm(pre + native_write, (dimension,))).float() / 30)
        native_logprob = native_logits.log_softmax(-1); native_probability = native_logprob.exp()
        targets = torch.as_tensor(rows[start:start + len(raw), np.asarray(positions) + 1].reshape(-1), device="cuda", dtype=torch.long)
        native_ce = -native_logprob.gather(1, targets[:, None]).squeeze(1); native_top1 = native_logits.argmax(-1)
        changed_logprobs = {}
        for name, prediction in predictions.items():
            changed_write = native_write + (prediction - native_score) @ output_basis.T
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(pre + changed_write, (dimension,))).float() / 30)
            logprob = logits.log_softmax(-1); changed_logprobs[name] = logprob
            exact_kl = (native_probability * (native_logprob - logprob)).sum(1)
            ce = -logprob.gather(1, targets[:, None]).squeeze(1)
            records[name]["exact_kl"].extend(exact_kl.cpu().tolist())
            records[name]["ce_delta"].extend((ce - native_ce).cpu().tolist())
            records[name]["top1"].extend((logits.argmax(-1) == native_top1).cpu().tolist())
        full_probability = changed_logprobs["full_union16"].exp()
        cross_kl = (full_probability * (changed_logprobs["full_union16"] - changed_logprobs["block_union16"])).sum(1)
        cross_records.extend(cross_kl.cpu().tolist())
    target = torch.cat(score_target)
    return {
        "scores": {name: score_summary(target, torch.cat(values)) for name, values in score_predictions.items()},
        "behavioral": {name: summarize(row) for name, row in records.items()},
        "cross_terms": {"mean_full_to_block_kl": float(np.mean(cross_records)), "median_full_to_block_kl": float(np.median(cross_records))},
    }


def main():
    config = json.loads(CONFIG.read_text()); source = np.load(SOURCE_ARRAYS); fisher = np.load(FISHER_ARRAYS)
    output_basis = torch.from_numpy(source["tail_output_basis"][:, :4]).cuda().float()
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    native = model.blocks[config["layer"]].mlp
    down, left, right = native.Down.weight.float().detach(), native.Left.weight.float().detach(), native.Right.weight.float().detach()
    discovery_x, _ = collect(model, config["layer"], config["discovery"], config)
    square, inverse, covariance = covariance_transforms(discovery_x, config["covariance_floor_fraction"])
    cleaned_forms, traces = [], []
    for component in range(4):
        coefficients = down.T @ output_basis[:, component]
        raw = left.T @ (coefficients[:, None] * right); raw = 0.5 * (raw + raw.T)
        clean, trace = trace_clean(square @ raw @ square); cleaned_forms.append(clean); traces.append(trace)
    traces_tensor = torch.stack(traces); _, common_basis = common_input(cleaned_forms)
    candidate_basis = common_basis[:, :config["candidate_pool_rank"]]
    forms = torch.einsum("di,kde,ej->kij", candidate_basis, torch.stack(cleaned_forms), candidate_basis).detach()
    fisher8 = torch.from_numpy(fisher["continuous_a"]).cuda().float()
    frobenius8 = torch.eye(config["candidate_pool_rank"], device="cuda")[:, :8]
    residual = frobenius8 - fisher8 @ (fisher8.T @ frobenius8)
    complement8 = torch.linalg.qr(residual, mode="reduced").Q
    full_union16 = torch.linalg.qr(torch.cat((fisher8, complement8), dim=1), mode="reduced").Q
    programs = {
        "fisher8": [fisher8],
        "frobenius8": [frobenius8],
        "full_union16": [full_union16],
        "block_union16": [fisher8, complement8],
        "complement8": [complement8],
    }
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("external evaluation", corpus, flush=True)
        results[corpus] = evaluate_programs(model, spec, config, output_basis, inverse, candidate_basis, forms, traces_tensor, programs)
    hypotheses = {
        "X1_block_union_beats_full_union": {"supported": all(r["behavioral"]["block_union16"]["mean_exact_kl"] <= .8 * r["behavioral"]["full_union16"]["mean_exact_kl"] for r in results.values())},
        "X2_block_union_improves_fisher8": {"supported": all(r["behavioral"]["block_union16"]["mean_exact_kl"] < r["behavioral"]["fisher8"]["mean_exact_kl"] for r in results.values())},
        "X3_block_union_reaches_median_gate": {"supported": all(r["behavioral"]["block_union16"]["median_exact_kl"] <= .005 for r in results.values())},
        "X4_cross_terms_are_behaviorally_material": {"supported": all(r["cross_terms"]["mean_full_to_block_kl"] >= .001 for r in results.values())},
        "X5_block_union_preserves_task_ce": {"supported": all(r["behavioral"]["block_union16"]["ce_delta"] <= .01 for r in results.values())},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "covariance": covariance, "results": results}, indent=2, allow_nan=False))
    np.savez_compressed(ARRAYS, fisher8=fisher8.cpu().numpy(), complement8=complement8.cpu().numpy(), full_union16=full_union16.cpu().numpy())
    lines = ["# Cross-term red team of the Layer-17 dual-metric union", "", "The same 16D span is evaluated either as one full bilinear block or as two additive 8D blocks. Only cross-terms differ.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Cross-term red team](layer17_tail_cross_term_redteam.png)", "", "| Corpus | Program | Score R² | Mean KL | Median KL | CE Δ | Top1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        for name in programs:
            score, behavior = row["scores"][name], row["behavioral"][name]
            lines.append(f"| {corpus} | {name} | {score['centered_vector_r2']:.3f} | {behavior['mean_exact_kl']:.5f} | {behavior['median_exact_kl']:.5f} | {behavior['ce_delta']:.5f} | {behavior['top1_agreement']:.3f} |")
        lines.append(f"| {corpus} | full→block cross-term divergence | — | {row['cross_terms']['mean_full_to_block_kl']:.5f} | {row['cross_terms']['median_full_to_block_kl']:.5f} | — | — |")
    lines += ["", "A block advantage would mean the added directions are useful but their induced pairwise interactions are not. Failure means the extra variance directions themselves are behaviorally misaligned, beyond the cross-term issue."]
    REPORT.write_text("\n".join(lines) + "\n")
    figure, plots = plt.subplots(1, 3, figsize=(15, 4.5)); names = list(programs)
    for corpus, row in results.items():
        plots[0].plot(names, [row["scores"][n]["centered_vector_r2"] for n in names], marker="o", label=corpus)
        plots[1].plot(names, [row["behavioral"][n]["mean_exact_kl"] for n in names], marker="o", label=corpus)
        plots[2].plot(names, [row["behavioral"][n]["ce_delta"] for n in names], marker="o", label=corpus)
    for plot, title in zip(plots, ("Score fidelity", "Native distribution", "Next-token utility")):
        plot.set_title(title); plot.tick_params(axis="x", rotation=30); plot.grid(alpha=.2); plot.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__": main()
