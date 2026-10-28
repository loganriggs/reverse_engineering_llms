#!/usr/bin/env python3
"""Do output-conditioned Jacobian subspaces recur for the same known feature?"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin18_dynamic_mlp_jacobian_spectrum import analytic_mlp_jacobian, exact_incoming_adjoints
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_suffix_degree_pilot import mlp_vjp
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_dynamic_mlp_feature_recurrence.json")
OUTPUT = TENSOR_ROOT / "bilin18_dynamic_mlp_feature_recurrence.json"
REPORT = TENSOR_ROOT / "BILIN18_DYNAMIC_MLP_FEATURE_RECURRENCE.md"
FIGURE = TENSOR_ROOT / "bilin18_dynamic_mlp_feature_recurrence.png"


def projection_energy(vector, basis):
    return float((vector @ basis).square().sum() / vector.square().sum().clamp_min(1e-30))


def summarize(values):
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "q10": float(np.quantile(values, .1)), "q90": float(np.quantile(values, .9))}


def contrast_vectors(tokenizer):
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    return torch.as_tensor(flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None], device="cuda", dtype=torch.float32)


def selected_sites(causal, old, count):
    sites = []
    for feature in causal["features"]:
        rows = [row for row in old["records"] if row["feature"] == feature]
        for selection in ("extreme", "uniform"):
            sites.extend([row for row in rows if row["selection"] == selection][:count])
    return sites


def group_scores(matrix, labels, positions):
    same = []; different = []; nearest_position = []
    for target in range(len(labels)):
        same_indices = [donor for donor in range(len(labels)) if donor != target and labels[donor] == labels[target]]
        different_indices = [donor for donor in range(len(labels)) if labels[donor] != labels[target]]
        distance = min(abs(positions[donor] - positions[target]) for donor in different_indices)
        nearest = [donor for donor in different_indices if abs(positions[donor] - positions[target]) == distance]
        same.append(float(matrix[target, same_indices].mean()))
        different.append(float(matrix[target, different_indices].mean()))
        nearest_position.append(float(matrix[target, nearest].mean()))
    return np.asarray(same), np.asarray(different), np.asarray(nearest_position)


def permutation_pvalue(matrix, labels, positions, replicates, rng):
    observed = float((group_scores(matrix, labels, positions)[0] - group_scores(matrix, labels, positions)[1]).mean())
    null = []
    for _ in range(replicates):
        shuffled = rng.permutation(labels)
        same, different, _ = group_scores(matrix, shuffled, positions)
        null.append(float((same - different).mean()))
    return observed, (1 + sum(value >= observed for value in null)) / (replicates + 1), summarize(null)


def direct_similarity_summary(items, labels, positions, replicates, rng):
    unit = torch.stack([item.float() / item.float().norm().clamp_min(1e-30) for item in items])
    matrix = (unit @ unit.T).square().cpu().numpy()
    same, different, nearest = group_scores(matrix, labels, positions)
    observed, pvalue, null = permutation_pvalue(matrix, labels, positions, replicates, rng)
    return {"same_feature": summarize(same), "different_feature": summarize(different), "nearest_position_different_feature": summarize(nearest), "same_to_different_ratio": float(same.mean() / max(different.mean(), 1e-30)), "same_to_nearest_position_ratio": float(same.mean() / max(nearest.mean(), 1e-30)), "paired_difference_mean": observed, "permutation_pvalue": pvalue, "permutation_null": null}


def main():
    config = json.loads(CONFIG.read_text()); causal = json.loads(Path(config["source_config"]).read_text()); old = json.loads(Path(config["source_results"]).read_text()); source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); contrasts = contrast_vectors(tokenizer)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); heldout_start = source["start_row"] + source["discovery_rows"]
    sites = selected_sites(causal, old, config["sites_per_selection_per_feature"]); labels = np.asarray([row["feature"] for row in sites]); positions = np.asarray([row["recipient_position"] for row in sites])
    store = {str(block): {str(rank): {family: [] for family in ("task_v", "spectral_v", "task_u")} for rank in config["ranks"]} for block in config["blocks"]}
    vectors = {str(block): {"response": [], "incoming": []} for block in config["blocks"]}
    records = []
    for site_index, row in enumerate(sites):
        feature = FEATURES.index(row["feature"]); contrast = contrasts[:, feature]; position = row["recipient_position"]
        raw = np.asarray(corpus[heldout_start + row["recipient_row"], :source["sequence_length"]], dtype=np.int64); ids = torch.as_tensor(raw, device="cuda")[None]
        with torch.no_grad(): trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        incoming = exact_incoming_adjoints(model, trace, position, contrast); site_record = {"site": site_index, "feature": row["feature"], "selection": row["selection"], "position": position, "token": tokenizer.decode([int(raw[position])]), "blocks": {}}
        for layer in config["blocks"]:
            matrix = analytic_mlp_jacobian(model.blocks[layer], trace[layer]["pre"][0, position].float()); u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
            vector = incoming[layer][0, 0, position]; response = mlp_vjp(model.blocks[layer], trace[layer]["pre"].float(), incoming[layer])[0, 0, position]
            coefficients = (vector @ u) * singular; order = coefficients.abs().argsort(descending=True)
            vectors[str(layer)]["response"].append(response.cpu()); vectors[str(layer)]["incoming"].append(vector.cpu())
            for rank in config["ranks"]:
                chosen = order[:rank]
                store[str(layer)][str(rank)]["task_v"].append(vh[chosen].T.cpu())
                store[str(layer)][str(rank)]["spectral_v"].append(vh[:rank].T.cpu())
                store[str(layer)][str(rank)]["task_u"].append(u[:, chosen].cpu())
            site_record["blocks"][str(layer)] = {"leading_singular": float(singular[0]), "task_optimal_k90": int(torch.searchsorted(coefficients[order].square().cumsum(0), .9 * coefficients.square().sum()).item()) + 1}
        records.append(site_record); print(f"feature recurrence {site_index + 1}/{len(sites)} {row['feature']} {row['selection']}", flush=True)
    rng = np.random.default_rng(config["seed"]); summaries = {}
    for layer in config["blocks"]:
        summaries[str(layer)] = {}
        for rank in config["ranks"]:
            summaries[str(layer)][str(rank)] = {}
            for family in ("task_v", "spectral_v", "task_u"):
                target_vectors = vectors[str(layer)]["incoming" if family == "task_u" else "response"]
                bases = store[str(layer)][str(rank)][family]; matrix = np.zeros((len(sites), len(sites)))
                for target, vector in enumerate(target_vectors):
                    vector = vector.float()
                    for donor, basis in enumerate(bases): matrix[target, donor] = projection_energy(vector, basis.float())
                same, different, nearest = group_scores(matrix, labels, positions); observed, pvalue, null = permutation_pvalue(matrix, labels, positions, config["permutation_replicates"], rng)
                summaries[str(layer)][str(rank)][family] = {"same_feature": summarize(same), "different_feature": summarize(different), "nearest_position_different_feature": summarize(nearest), "same_to_different_ratio": float(same.mean() / max(different.mean(), 1e-30)), "same_to_nearest_position_ratio": float(same.mean() / max(nearest.mean(), 1e-30)), "paired_difference_mean": observed, "permutation_pvalue": pvalue, "permutation_null": null}
        summaries[str(layer)]["direct"] = {
            "response": direct_similarity_summary(vectors[str(layer)]["response"], labels, positions, config["permutation_replicates"], rng),
            "incoming": direct_similarity_summary(vectors[str(layer)]["incoming"], labels, positions, config["permutation_replicates"], rng),
        }
    rank = "64"
    hypotheses = {
        "R0_same_feature_task_subspaces_recur": {"supported": all(summaries[str(block)][rank]["task_v"]["same_to_different_ratio"] >= 1.25 for block in config["blocks"])},
        "R1_recurrence_is_not_only_state_spectrum": {"supported": sum(summaries[str(block)][rank]["task_v"]["same_to_different_ratio"] > summaries[str(block)][rank]["spectral_v"]["same_to_different_ratio"] for block in config["blocks"]) >= 6},
        "R2_output_role_recurrence": {"supported": all(summaries[str(block)][rank]["task_u"]["same_to_different_ratio"] >= 1.25 for block in config["blocks"])},
        "R3_recurrence_survives_position_control": {"supported": sum(summaries[str(block)][rank]["task_v"]["same_to_nearest_position_ratio"] > 1 for block in config["blocks"]) >= 6},
        "R4_feature_labels_beat_permutation": {"supported": sum(summaries[str(block)][rank]["task_v"]["permutation_pvalue"] <= .05 for block in config["blocks"]) >= 6},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "site_count": len(sites), "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Dynamic MLP feature-subspace recurrence", "", f"This audit uses {len(sites)} held-out sites: two extreme and two uniform examples for each of eight supervised token features. At each sampled suffix block it asks whether a target response lies preferentially in task-selected singular subspaces discovered on other examples of the same feature.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Feature recurrence](bilin18_dynamic_mlp_feature_recurrence.png)", "", "| Block | Task V same/different | Task V same/position-control | Task V p | Spectral V same/different | Task U same/different | Direct response/incoming similarity ratio |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for block in config["blocks"]:
        row = summaries[str(block)][rank]; direct = summaries[str(block)]["direct"]; lines.append(f"| {block} | {row['task_v']['same_to_different_ratio']:.2f} | {row['task_v']['same_to_nearest_position_ratio']:.2f} | {row['task_v']['permutation_pvalue']:.3f} | {row['spectral_v']['same_to_different_ratio']:.2f} | {row['task_u']['same_to_different_ratio']:.2f} | {direct['response']['same_to_different_ratio']:.2f}/{direct['incoming']['same_to_different_ratio']:.2f} |")
    lines += ["", "## Interpretation guardrails", "", "The eight labels are deliberately simple supervised probes, not a complete ontology. A positive recurrence result would show a reusable subspace for these output effects, not prove that it is the model's only or preferred feature basis. A null result can mean that the feature is implemented by context-conditioned coalitions. The nearest-position control matches absolute position only; it does not match the full lexical or syntactic context. Task-selected modes are chosen using the target semantic adjoint and therefore diagnose output-conditioned computation rather than an unsupervised basis. Direct squared-cosine recurrence of the incoming adjoints and local responses is reported as a post-hoc control: if it already explains the task-U separation, the latter should be read as an output-objective effect rather than a newly discovered latent ontology."]
    REPORT.write_text("\n".join(lines) + "\n")
    blocks = config["blocks"]; x = np.arange(len(blocks)); fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for family, label in (("task_v", "task-selected V"), ("spectral_v", "top-spectral V"), ("task_u", "task-selected U")): axes[0].plot(x, [summaries[str(block)][rank][family]["same_to_different_ratio"] for block in blocks], marker="o", label=label)
    axes[0].axhline(1, color="black", linestyle="--", linewidth=1); axes[0].set_xticks(x, blocks); axes[0].set(xlabel="Block", ylabel="Same/different projection ratio", title="Does a feature reuse a subspace?"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].bar(x, [summaries[str(block)][rank]["task_v"]["same_to_nearest_position_ratio"] for block in blocks]); axes[1].axhline(1, color="black", linestyle="--", linewidth=1); axes[1].set_xticks(x, blocks); axes[1].set(xlabel="Block", ylabel="Same/position-matched different ratio", title="Position-controlled task-V recurrence"); axes[1].grid(axis="y", alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summaries_rank64": {block: summaries[block][rank] for block in summaries}}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
