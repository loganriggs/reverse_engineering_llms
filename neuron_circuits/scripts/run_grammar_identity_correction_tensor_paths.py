#!/usr/bin/env python3
"""Exact integrated Layer-1--3 tensor paths for a finite identity correction."""

from __future__ import annotations

from itertools import product
import json
import math
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import ActivationPatch
from run_grammar_exact_factor_chain import rms_jacobian
from run_grammar_final_interface_redteam import load_model
from run_grammar_final_gate_code_dimension import eligible_mask
from run_grammar_layer0_qk_value_identity_redteam import candidate_gates
from run_grammar_mixed_address_router import fisher_rows, select_panel


CONFIG = Path("configs/grammar_identity_correction_tensor_paths.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_identity_correction_tensor_paths.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_IDENTITY_CORRECTION_TENSOR_PATHS.md")
TERMS = ("I", "Q", "V", "ML", "MR", "MLQ", "MLV", "MRQ", "MRV")
PATHS = tuple(product(TERMS, repeat=3))
BILINEAR_TERMS = frozenset(("ML", "MR", "MLQ", "MLV", "MRQ", "MRV"))


@torch.inference_mode()
def stable_local_components(block, hidden):
    """Final-position block Jacobian without the fused SDPA kernel.

    Thousands of tiny fused-SDPA calls can crash the CUDA process on this
    workstation.  This algebra is identical to local_block_components, but the
    natural attention write is evaluated explicitly with softmax/einsum.
    """
    width = hidden.shape[-1]; final = hidden[-1]
    norm_jacobian = rms_jacobian(final, block.attn_norm.weight, block.attn_norm.eps)
    normalized = block.attn_norm(hidden); query, key, value = block.qkv(normalized).chunk(3, -1)
    query = query.view(len(hidden), block.heads, block.head_dim)
    key = key.view(len(hidden), block.heads, block.head_dim)
    value = value.view(len(hidden), block.heads, block.head_dim)
    qkv_weight = block.qkv.weight.view(3, block.heads, block.head_dim, width)
    pattern_derivatives, value_derivatives, attended = [], [], []
    for head in range(block.heads):
        q = query[-1, head]; keys = key[:, head]; values = value[:, head]
        attention = torch.softmax(keys @ q / math.sqrt(block.head_dim), 0)
        wq, wk, wv = qkv_weight[0, head], qkv_weight[1, head], qkv_weight[2, head]
        score_derivative = keys @ wq / math.sqrt(block.head_dim)
        score_derivative[-1] += q @ wk / math.sqrt(block.head_dim)
        softmax_derivative = torch.diag(attention) - torch.outer(attention, attention)
        pattern_derivatives.append(values.T @ (softmax_derivative @ score_derivative))
        value_derivatives.append(attention[-1] * wv); attended.append(attention @ values)
    qk = block.attn_out.weight @ torch.cat(pattern_derivatives) @ norm_jacobian
    value_component = block.attn_out.weight @ torch.cat(value_derivatives) @ norm_jacobian
    after = final + block.attn_out(torch.cat(attended))
    mlp_norm_jacobian = rms_jacobian(after, block.mlp_norm.weight, block.mlp_norm.eps)
    normalized_mlp = block.mlp_norm(after); left, right = block.left(normalized_mlp), block.right(normalized_mlp)
    left_arm = block.down.weight @ (right[:, None] * block.left.weight) @ mlp_norm_jacobian
    right_arm = block.down.weight @ (left[:, None] * block.right.weight) @ mlp_norm_jacobian
    return {"Q": qk, "V": value_component, "ML": left_arm, "MR": right_arm}


def component_terms(components: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    identity = torch.eye(components["Q"].shape[0], device=components["Q"].device)
    return {
        "I": identity, "Q": components["Q"], "V": components["V"],
        "ML": components["ML"], "MR": components["MR"],
        "MLQ": components["ML"] @ components["Q"],
        "MLV": components["ML"] @ components["V"],
        "MRQ": components["MR"] @ components["Q"],
        "MRV": components["MR"] @ components["V"],
    }


@torch.inference_mode()
def integrated_block(block, corrupt: torch.Tensor, native: torch.Tensor, nodes, weights):
    delta = native[-1] - corrupt[-1]
    integrated = {term: torch.zeros(block.width, block.width, device=delta.device) for term in TERMS}
    for node, weight in zip(nodes, weights):
        hidden = corrupt.clone(); hidden[-1] += float(node) * delta
        components = stable_local_components(block, hidden); terms = component_terms(components)
        for term in TERMS: integrated[term] += float(weight) * terms[term]
    return integrated


@torch.inference_mode()
def trace_one(model, prompt: np.ndarray, corrupt_gate: torch.Tensor, nodes, weights, device: str):
    tokens = torch.as_tensor(prompt.astype(np.int64), device=device)[None]
    initial = model.embedding(tokens) + model.position[:tokens.shape[1]]
    native, _ = model.blocks[0](initial)
    corrupt, _ = model.blocks[0](initial, ActivationPatch(0, corrupt_gate.to(device)))
    native, corrupt = native[0], corrupt[0]; delta0 = native[-1] - corrupt[-1]
    integrated_layers = []; state_errors = []
    maximum_prefix_error = float((native[:-1] - corrupt[:-1]).abs().max())
    for block in model.blocks[1:]:
        integrated = integrated_block(block, corrupt, native, nodes, weights); integrated_layers.append(integrated)
        native_out, _ = block(native[None]); corrupt_out, _ = block(corrupt[None])
        predicted = sum(integrated.values()) @ (native[-1] - corrupt[-1])
        actual = native_out[0, -1] - corrupt_out[0, -1]
        state_errors.append(float((predicted - actual).norm() / actual.norm().clamp_min(1e-30)))
        maximum_prefix_error = max(maximum_prefix_error, float((native_out[0, :-1] - corrupt_out[0, :-1]).abs().max()))
        native, corrupt = native_out[0], corrupt_out[0]
    answer_weight = model.unembedding.weight[list(VOCAB.answers)]
    center = torch.eye(8, device=device) - torch.ones(8, 8, device=device) / 8
    final_delta = native[-1] - corrupt[-1]; readout = torch.zeros(8, model.width, device=device)
    for node, weight in zip(nodes, weights):
        state = corrupt[-1] + float(node) * final_delta
        readout += float(weight) * (center @ answer_weight @ rms_jacobian(state, model.final_norm.weight, model.final_norm.eps))
    vectors = {(): delta0}
    for integrated in integrated_layers:
        vectors = {prefix + (term,): matrix @ vector for prefix, vector in vectors.items() for term, matrix in integrated.items()}
    path_effects = torch.stack([readout @ vectors[path] for path in PATHS])
    native_logits = model.unembedding(model.final_norm(native[-1]))[list(VOCAB.answers)].float()
    corrupt_logits = model.unembedding(model.final_norm(corrupt[-1]))[list(VOCAB.answers)].float()
    actual = center @ (native_logits - corrupt_logits); predicted = path_effects.sum(0)
    logit_error = float((predicted - actual).norm() / actual.norm().clamp_min(1e-30))
    return path_effects.cpu(), native_logits.cpu(), corrupt_logits.cpu(), {
        "maximum_prefix_error": maximum_prefix_error,
        "maximum_state_relative_error": max(state_errors),
        "logit_relative_error": logit_error,
    }


def collect_paths(model, prompts, corrupt_gates, config, device):
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(config["quadrature_points"])
    nodes, weights = (raw_nodes + 1) / 2, raw_weights / 2
    paths, native, corrupt = [], [], []
    errors = {key: 0.0 for key in ("maximum_prefix_error", "maximum_state_relative_error", "logit_relative_error")}
    for prompt, gate in zip(prompts, corrupt_gates):
        effect, native_logits, corrupt_logits, current = trace_one(model, prompt, gate, nodes, weights, device)
        paths.append(effect); native.append(native_logits); corrupt.append(corrupt_logits)
        for key in errors: errors[key] = max(errors[key], current[key])
    return {"paths": torch.stack(paths), "native_logits": torch.stack(native), "corrupt_logits": torch.stack(corrupt)}, errors


def path_energy(data):
    probability = torch.softmax(data["corrupt_logits"], -1)[:, None]
    paths = data["paths"]; centered = paths - (probability * paths).sum(-1, keepdim=True)
    return (probability * centered.square()).sum(-1).mean(0)


def assess(data, chosen, threshold):
    target = data["native_logits"] - data["corrupt_logits"]
    predicted = data["paths"][:, chosen].sum(1) if len(chosen) else torch.zeros_like(target)
    energy, cosine, fidelity = fisher_rows(data["corrupt_logits"], target, predicted); keep = energy >= threshold
    return {"effectful_examples": int(keep.sum()), "median_fidelity": float(fidelity[keep].median()),
            "median_cosine": float(cosine[keep].median())}


def grouped_projection(data, mask):
    base = data["corrupt_logits"]; target = data["native_logits"] - base; predicted = data["paths"][:, mask].sum(1)
    probability = torch.softmax(base, -1)
    target = target - (probability * target).sum(-1, keepdim=True)
    predicted = predicted - (probability * predicted).sum(-1, keepdim=True)
    denominator = (probability * target.square()).sum().clamp_min(1e-30)
    return {"signed_projection": float((probability * target * predicted).sum() / denominator),
            "norm_ratio": float(torch.sqrt((probability * predicted.square()).sum() / denominator))}


def analyze(spec, config, device):
    root = Path(spec["root"]); manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / f"rho_{config['rho']:g}/depth_4/checkpoint.pt", device)
    facts = np.asarray(manifest["train_facts"]); labels, selected, changed = labels_for_rho(manifest, "train", config["rho"])
    eligible = eligible_mask(model, manifest, facts, labels, config["evaluation_templates"], device)
    panel = select_panel(labels, selected, changed, eligible, facts, config, config["seed"] + spec["seed"])
    output = {}; maximum_errors = {key: 0.0 for key in ("maximum_prefix_error", "maximum_state_relative_error", "logit_relative_error")}
    for flag, status in ((False, "unselected"), (True, "selected")):
        discovery_idx, evaluation_idx = panel[flag]["discovery"], panel[flag]["evaluation"]
        discovery_prompts = encode_prompts(manifest, facts[discovery_idx], config["discovery_template"])
        discovery_gates, _ = candidate_gates(model, discovery_prompts, config["discovery_template"], facts[discovery_idx, 2], device)
        discovery, errors = collect_paths(model, discovery_prompts, discovery_gates["irrelevant_mean_value"], config, device)
        for key in maximum_errors: maximum_errors[key] = max(maximum_errors[key], errors[key])
        energies = path_energy(discovery); order = energies.argsort(descending=True)
        output[status] = {"path_order": [{"layer1": PATHS[int(index)][0], "layer2": PATHS[int(index)][1], "layer3": PATHS[int(index)][2], "energy": float(energies[index])} for index in order]}
        for template in config["evaluation_templates"]:
            prompts = encode_prompts(manifest, facts[evaluation_idx], template)
            gates, _ = candidate_gates(model, prompts, template, facts[evaluation_idx, 2], device)
            output[status][str(template)] = {}
            for corruption, gate_name in (("mean", "irrelevant_mean_value"), ("roll", "irrelevant_roll_value")):
                data, errors = collect_paths(model, prompts, gates[gate_name], config, device)
                for key in maximum_errors: maximum_errors[key] = max(maximum_errors[key], errors[key])
                methods = {f"top_{count}": assess(data, order[:count], config["minimum_fisher_effect_norm"]) for count in config["path_counts"]}
                identity_index = PATHS.index(("I", "I", "I"))
                methods["identity_only"] = assess(data, torch.as_tensor([identity_index]), config["minimum_fisher_effect_norm"])
                methods["path_k90"] = next((count for count in config["path_counts"] if methods[f"top_{count}"]["median_fidelity"] >= .90), 730)
                bilinear_mask = torch.as_tensor([any(term in BILINEAR_TERMS for term in path) for path in PATHS])
                methods["contains_bilinear"] = grouped_projection(data, bilinear_mask)
                methods["no_bilinear"] = grouped_projection(data, ~bilinear_mask)
                for layer in range(3):
                    for term in TERMS:
                        mask = torch.as_tensor([path[layer] == term for path in PATHS])
                        methods[f"layer{layer + 1}_{term}"] = grouped_projection(data, mask)
                generator = np.random.default_rng(config["seed"] + spec["seed"] * 100 + template * 10 + int(flag) + (corruption == "roll"))
                random = []
                for _ in range(config["random_path_banks"]):
                    chosen = torch.as_tensor(generator.choice(729, 128, replace=False))
                    random.append(assess(data, chosen, config["minimum_fisher_effect_norm"])["median_fidelity"])
                methods["random_128"] = {"median_fidelity_mean": float(np.mean(random)), "median_fidelity_p95": float(np.quantile(random, .95)), "replicates": len(random)}
                output[status][str(template)][corruption] = methods
    return {"seed": spec["seed"], "maximum_errors": maximum_errors, "results": output}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = []
    for spec in config["models"]:
        print("trace", spec["seed"], flush=True); runs.append(analyze(spec, config, device))
    first, held = map(str, config["evaluation_templates"])
    def val(run, status, template, corruption, method, key="median_fidelity"):
        return run["results"][status][template][corruption][method][key]
    p0 = all(max(run["maximum_errors"].values()) < 1e-4 and val(run, status, template, corruption, "top_729") >= .9999 for run in runs for status in ("unselected", "selected") for template in (first, held) for corruption in ("mean", "roll"))
    p1 = all(val(run, "selected", held, "mean", "identity_only") < .50 for run in runs)
    p2 = all(val(run, "selected", held, "mean", "top_128") < .90 for run in runs)
    p3 = all(val(run, "selected", held, "mean", "top_512") >= .90 for run in runs)
    p4 = all(run["results"]["selected"][held]["mean"]["path_k90"] >=
             run["results"]["unselected"][held]["mean"]["path_k90"] + 32 for run in runs)
    p5 = all(val(run, "selected", held, "mean", "contains_bilinear", "signed_projection") >= .60 for run in runs)
    p6 = all(val(run, "selected", held, "roll", "top_256") >= .75 for run in runs)
    p7 = all(val(run, "selected", held, "mean", "top_128") >= run["results"]["selected"][held]["mean"]["random_128"]["median_fidelity_p95"] + .10 for run in runs)
    p8 = all(abs(val(run, "selected", first, "mean", "top_256") - val(run, "selected", held, "mean", "top_256")) <= .10 for run in runs)
    p9 = all(row["effectful_examples"] >= 64 for run in runs for status in ("unselected", "selected") for template in (first, held) for corruption in ("mean", "roll") for row in run["results"][status][template][corruption].values() if isinstance(row, dict) and "effectful_examples" in row)
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9))}
    OUTPUT.write_text(json.dumps({"config": config, "terms": TERMS, "hypotheses": hypotheses, "runs": runs}, indent=2, allow_nan=False))
    lines = ["# Eight-point downstream tensor paths for the Layer-0 identity correction", "",
             "**Numerical warning:** the locked exactness gate failed on rare large interventions. Use the retained 32-point convergence red team for mechanistic conclusions.", "",
             "Rule-irrelevant digit values are erased at the Layer-0 final-position gate, then restored. Causal masking leaves every prefix state identical between worlds. Eight-point Gauss--Legendre integration approximates the finite Jacobian of each downstream block along its realized state chord. Each block is expanded into `I,Q,V,ML,MR,MLQ,MLV,MRQ,MRV`; their three-layer product gives 729 named paths, followed by an integrated final RMS/unembedding readout. Rankings use only trained-template mean-erasure discovery effects.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Seed | Status | Corruption | top 32/128/256/512 | K90 | identity | bilinear projection/norm | random128 p95 |",
              "|---:|---|---|---:|---:|---:|---:|---:|"]
    for run in runs:
        for status in ("unselected", "selected"):
            for corruption in ("mean", "roll"):
                row = run["results"][status][held][corruption]; bilinear = row["contains_bilinear"]
                lines.append(f"| {run['seed']} | {status} | {corruption} | {row['top_32']['median_fidelity']:.3f}/{row['top_128']['median_fidelity']:.3f}/{row['top_256']['median_fidelity']:.3f}/{row['top_512']['median_fidelity']:.3f} | {row['path_k90']} | {row['identity_only']['median_fidelity']:.3f} | {bilinear['signed_projection']:.3f}/{bilinear['norm_ratio']:.3f} | {row['random_128']['median_fidelity_p95']:.3f} |")
    lines += ["", "Individual path terms are an exact algebraic decomposition of the integrated local operator along this one finite correction. Their signed contributions can cancel, so a high path count need not mean 729 independent semantic mechanisms. Conversely, a low-rank sum would describe this intervention family only; it would not prove a globally small model circuit."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__": main()
