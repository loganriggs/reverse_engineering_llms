#!/usr/bin/env python3
"""Compare local causal-Jacobian families across rule and memorization regimes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import ActivationPatch
from run_grammar_final_interface_redteam import load_model


CONFIG = Path("configs/grammar_jacobian_regime_sweep.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_jacobian_regime_sweep.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_JACOBIAN_REGIME_SWEEP.md")


def forward_capture(model, tokens):
    hidden = model.embedding(tokens) + model.position[:tokens.shape[1]]; activations = []
    for block in model.blocks:
        normalized = block.attn_norm(hidden)
        query, key, value = block.qkv(normalized).chunk(3, dim=-1)
        def heads(tensor): return tensor.view(len(tokens), tokens.shape[1], block.heads, block.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(heads(query), heads(key), heads(value), is_causal=True)
        attended = attended.transpose(1, 2).reshape_as(hidden)
        hidden = hidden + block.attn_out(attended)
        normalized = block.mlp_norm(hidden); left, right = block.left(normalized), block.right(normalized)
        activation = left * right
        activations.append(activation)
        hidden = hidden + block.down(activation)
    logits = model.unembedding(model.final_norm(hidden)).float()
    return logits, activations


def collect_jacobians(model, prompts, batch_size, device):
    answer_ids = torch.as_tensor(list(VOCAB.answers), device=device); stores = [[] for _ in model.blocks]
    activation_stores = [[] for _ in model.blocks]; logit_stores = []; maximum_forward_error = 0.0
    for start in range(0, len(prompts), batch_size):
        tokens = torch.as_tensor(prompts[start:start + batch_size].astype(np.int64), device=device)
        logits, activations = forward_capture(model, tokens)
        native_logits = model(tokens).float()
        maximum_forward_error = max(
            maximum_forward_error, float((logits - native_logits).abs().max().detach()),
        )
        answer = logits[:, -1, answer_ids]
        layer_rows = [[None] * 8 for _ in model.blocks]
        for output in range(8):
            gradients = torch.autograd.grad(answer[:, output].sum(), activations,
                                            retain_graph=output < 7, create_graph=False)
            for layer, gradient in enumerate(gradients): layer_rows[layer][output] = gradient[:, -1].detach().cpu()
        for layer in range(model.layers_count):
            jacobian = torch.stack(layer_rows[layer], 1); jacobian -= jacobian.mean(1, keepdim=True)
            stores[layer].append(jacobian)
            activation_stores[layer].append(activations[layer][:, -1].detach().cpu())
        answer = answer.detach().cpu(); answer -= answer.mean(1, keepdim=True); logit_stores.append(answer)
    return ([torch.cat(rows) for rows in stores], [torch.cat(rows) for rows in activation_stores],
            torch.cat(logit_stores), maximum_forward_error)


def effective_rank(values):
    singular = torch.linalg.svdvals(values - values.mean(0)); energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-30)
    return float(torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum()))


def variance_energy(values): return float((values - values.mean(0)).square().mean())


def reconstruction_fidelity(train, evaluation, rank):
    mean = train.mean(0); centered = train - mean
    if rank == 0: prediction = mean.expand_as(evaluation)
    else:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False); basis = vh[:rank]
        prediction = mean + ((evaluation - mean) @ basis.T) @ basis
    return 1 - float((evaluation - prediction).square().sum() / evaluation.square().sum().clamp_min(1e-30))


def shuffled_top(values, generator):
    centered = values - values.mean(0); rows, columns = centered.shape
    order = torch.rand(rows, columns, generator=generator).argsort(0)
    shuffled = centered.gather(0, order)
    _, singular, _ = torch.pca_lowrank(shuffled, q=4, center=False, niter=3)
    return float(singular[0])


@torch.no_grad()
def correct(model, prompts, labels, device):
    tokens = torch.as_tensor(prompts.astype(np.int64), device=device)
    logits = model(tokens)[:, -1, list(VOCAB.answers)]
    return (logits.argmax(-1).cpu().numpy() == labels)


@torch.no_grad()
def finite_fidelity(model, prompts, base_logits, activations, jacobians, layer, relative_norm, seed, device):
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(activations.shape, generator=generator)
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    delta = relative_norm * activations.norm(dim=-1, keepdim=True) * direction
    tokens = torch.as_tensor(prompts.astype(np.int64), device=device)
    changed = model(tokens, patch=ActivationPatch(layer, delta.to(device), additive=True))[:, -1, list(VOCAB.answers)].cpu()
    changed -= changed.mean(1, keepdim=True)
    target = changed - base_logits
    prediction = torch.einsum("boc,bc->bo", jacobians, delta); prediction -= prediction.mean(1, keepdim=True)
    return 1 - float((target - prediction).square().sum() / target.square().sum().clamp_min(1e-30))


def ranks(values):
    order = np.argsort(values, kind="mergesort"); result = np.empty(len(values)); result[order] = np.arange(len(values)); return result


def spearman(first, second): return float(np.corrcoef(ranks(first), ranks(second))[0, 1])


def audit_run(spec, config, device):
    root = Path(spec["root"]); rho = spec["rho"]; manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
    labels, selected, _ = labels_for_rho(manifest, "train", rho); facts = np.asarray(manifest["train_facts"])
    prompt0 = encode_prompts(manifest, facts, config["discovery_template"])
    prompt6 = encode_prompts(manifest, facts, config["evaluation_template"])
    eligible = correct(model, prompt0, labels, device) & correct(model, prompt6, labels, device)
    rng = np.random.default_rng(config["seed"] + spec["seed"] * 100 + round(rho * 10)); discovery = []; evaluation = []
    for answer in range(8):
        indices = np.flatnonzero(eligible & (labels == answer)); rng.shuffle(indices)
        need = config["examples_per_answer"]
        if len(indices) < need: raise RuntimeError(f"{spec['name']} answer {answer}: {len(indices)} < {need}")
        split = config["discovery_per_answer"]; discovery.extend(indices[:split]); evaluation.extend(indices[split:need])
    discovery = np.asarray(discovery); evaluation = np.asarray(evaluation)
    all_indices = np.concatenate([discovery, evaluation]); prompts = prompt6[all_indices]
    jacobians, activations, logits, maximum_forward_error = collect_jacobians(
        model, prompts, config["batch_size"], device,
    )
    split = len(discovery); generator = torch.Generator().manual_seed(config["seed"] + spec["seed"] * 1000 + round(rho * 100))
    layers = []
    for layer in range(model.layers_count):
        train = jacobians[layer][:split].flatten(1); test = jacobians[layer][split:].flatten(1)
        singular = torch.linalg.svdvals(train - train.mean(0)); nulls = [shuffled_top(train, generator) for _ in range(config["null_replicates"])]
        test_selected = selected[evaluation]; groups = {}
        for name, mask in (("selected", test_selected), ("unselected", ~test_selected)):
            if int(mask.sum()) >= 16:
                values = test[torch.from_numpy(mask)]
                groups[name] = {"sites": len(values), "effective_rank": effective_rank(values),
                                "variance_energy": variance_energy(values)}
        layers.append({
            "centered_effective_rank": effective_rank(train),
            "leading_to_null_ratio": float(singular[0]) / max(nulls),
            "rank_fidelity": {str(rank): reconstruction_fidelity(train, test, rank) for rank in config["ranks"]},
            "finite_fidelity": {
                str(norm): finite_fidelity(
                    model, prompt6[evaluation], logits[split:], activations[layer][split:],
                    jacobians[layer][split:], layer, norm,
                    config["seed"] + layer + spec["seed"] * 100, device,
                )
                for norm in config["finite_relative_norms"]
            },
            "groups": groups,
        })
    return {"name": spec["name"], "rho": rho, "seed": spec["seed"], "sites": len(all_indices),
            "maximum_forward_error": maximum_forward_error, "layers": layers}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"; runs = []
    for spec in config["runs"]:
        print("audit", spec["name"], flush=True); runs.append(audit_run(spec, config, device))
    summary = {}
    for rho in (0.0, .5, 1.0):
        selected = [row for row in runs if row["rho"] == rho]; layers = []
        for layer in range(4):
            layers.append({key: float(np.mean([row["layers"][layer][key] for row in selected]))
                           for key in ("centered_effective_rank", "leading_to_null_ratio")} |
                          {"finite_fidelity": {
                              str(norm): float(np.mean([
                                  row["layers"][layer]["finite_fidelity"][str(norm)] for row in selected
                              ])) for norm in config["finite_relative_norms"]
                          }} |
                          {"rank_fidelity": {str(rank): float(np.mean([row["layers"][layer]["rank_fidelity"][str(rank)] for row in selected]))
                                             for rank in config["ranks"]}})
        summary[str(rho)] = {"runs": len(selected), "layers": layers}
    paired = {rho: [row for row in runs if row["rho"] == rho and row["seed"] in (0, 1)] for rho in (0.0, 1.0)}
    g0 = (all(row["maximum_forward_error"] <= 1e-5 for row in runs)
          and all(layer["finite_fidelity"]["0.01"] >= .99
                  and layer["finite_fidelity"]["0.05"] >= .95
                  for row in runs for layer in row["layers"]))
    g1 = all(paired[0.0][seed]["layers"][0]["centered_effective_rank"] <= .8 * paired[1.0][seed]["layers"][0]["centered_effective_rank"] for seed in range(2))
    g2 = all(paired[0.0][seed]["layers"][0]["rank_fidelity"]["16"] >= paired[1.0][seed]["layers"][0]["rank_fidelity"]["16"] + .05 for seed in range(2))
    g3 = True
    for seed in range(2):
        pure, arbitrary = paired[0.0][seed]["layers"][3], paired[1.0][seed]["layers"][3]
        g3 &= abs(pure["centered_effective_rank"] - arbitrary["centered_effective_rank"]) <= .1 * arbitrary["centered_effective_rank"]
        g3 &= abs(pure["rank_fidelity"]["16"] - arbitrary["rank_fidelity"]["16"]) <= .05
    mixed = [row for row in runs if row["rho"] == .5]
    g4 = all(any(layer["groups"].get("selected", {}).get("variance_energy", 0) >=
                     1.2 * layer["groups"].get("unselected", {}).get("variance_energy", float("inf"))
                     for layer in row["layers"][:2]) for row in mixed)
    g5 = all(paired[0.0][seed]["layers"][0]["leading_to_null_ratio"] >= 1.25 * paired[1.0][seed]["layers"][0]["leading_to_null_ratio"] for seed in range(2))
    correlations = {}
    g6 = True
    for rho in (0.0, .5, 1.0):
        selected = [row for row in runs if row["rho"] == rho]; values = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                correlation = spearman([x["centered_effective_rank"] for x in selected[i]["layers"]],
                                       [x["centered_effective_rank"] for x in selected[j]["layers"]])
                values.append(correlation); g6 &= correlation >= .80
        correlations[str(rho)] = values
    hypotheses = {name: {"supported": bool(value)} for name, value in {
        "G0_jacobians_pass_finite_validation": g0, "G1_reusable_rules_have_lower_early_chart_rank": g1,
        "G2_rules_have_a_more_compact_early_dictionary": g2, "G3_late_answer_interfaces_converge": g3,
        "G4_randomized_facts_increase_within_model_chart_variation": g4,
        "G5_rule_charts_have_stronger_spectral_outliers": g5,
        "G6_depth_profiles_replicate_across_seeds": g6}.items()}
    hypotheses["G6_depth_profiles_replicate_across_seeds"]["correlations"] = correlations
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summary": summary, "runs": runs}, indent=2, allow_nan=False))
    lines = ["# Local Jacobian charts across rule and memorization regimes", "",
             "For balanced, correctly learned facts, the centered answer-logit Jacobian with respect to the final-position post-product MLP channels is measured at every layer. Discovery examples define a PCA dictionary; disjoint unseen-template examples test reconstruction. One-sided activation edits validate that the Jacobians predict finite effects. Rho controls the fraction of fixed arbitrary assignments.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| rho | Layer | Effective rank | Leading/null | Mean/r16/r32 fidelity | Finite 1%/5% fidelity |", "|---:|---:|---:|---:|---:|---:|"]
    for rho in ("0.0", "0.5", "1.0"):
        for layer, row in enumerate(summary[rho]["layers"]):
            rank = row["rank_fidelity"]; finite = row["finite_fidelity"]
            lines.append(f"| {rho} | {layer} | {row['centered_effective_rank']:.1f} | {row['leading_to_null_ratio']:.2f} | "
                         f"{rank['0']:.3f}/{rank['16']:.3f}/{rank['32']:.3f} | "
                         f"{finite['0.01']:.3f}/{finite['0.05']:.3f} |")
    lines += ["", "A lower-rank rule regime would support the hypothesis that reuse compresses the family of local computations. Convergence only at the final layer would instead separate upstream rule/address organization from a shared answer interface. Null or reversed load effects would falsify Jacobian rank as a simple memorization axis."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
