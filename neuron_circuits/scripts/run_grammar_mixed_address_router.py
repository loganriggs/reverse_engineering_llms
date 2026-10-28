#!/usr/bin/env python3
"""Locate and causally test the address program for arbitrary grammar facts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model


CONFIG = Path("configs/grammar_mixed_address_router.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_mixed_address_router.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_MIXED_ADDRESS_ROUTER.md")


@torch.inference_mode()
def collect(model, prompts, device, batch_size=512):
    stores = {key: [] for key in ("raw", "boundary2_full", "boundary3_final", "qk", "attn_write",
                                   "premlp", "write", "after", "logits", "zero_logits")}
    maximum_forward_error = 0.0; maximum_tensor_error = 0.0
    block = model.blocks[3]; answer_ids = list(VOCAB.answers)
    for start in range(0, len(prompts), batch_size):
        tokens = torch.as_tensor(prompts[start:start + batch_size].astype(np.int64), device=device)
        hidden = model.embedding(tokens) + model.position[:tokens.shape[1]]
        for earlier in model.blocks[:3]: hidden, _ = earlier(hidden)
        boundary2 = hidden
        normalized = block.attn_norm(hidden); query, key, value = block.qkv(normalized).chunk(3, -1)
        def heads(values): return values.view(len(tokens), tokens.shape[1], block.heads, block.head_dim).transpose(1, 2)
        qh, kh, vh = heads(query), heads(key), heads(value)
        pattern = torch.softmax(torch.einsum("bhd,bhtd->bht", qh[:, :, -1], kh) / math.sqrt(block.head_dim), -1)
        attended = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        attention_write = block.attn_out(attended.transpose(1, 2).reshape_as(hidden))
        after = hidden + attention_write
        premlp = block.mlp_norm(after[:, -1])
        activation = block.left(premlp) * block.right(premlp)
        write = activation @ block.down.weight.T
        # Equivalent explicit contraction of the fixed CP-form bilinear tensor.
        left = premlp @ block.left.weight.T; right = premlp @ block.right.weight.T
        tensor_write = (left * right) @ block.down.weight.T
        maximum_tensor_error = max(maximum_tensor_error, float((tensor_write - write).abs().max()))
        final = after[:, -1] + write
        logits = model.unembedding(model.final_norm(final))[:, answer_ids].float()
        zero_logits = model.unembedding(model.final_norm(after[:, -1]))[:, answer_ids].float()
        native = model(tokens)[:, -1, answer_ids].float()
        maximum_forward_error = max(maximum_forward_error, float((native - logits).abs().max()))
        raw = F.one_hot(tokens, VOCAB.size).flatten(1).float()
        for key_name, values in (("raw", raw), ("boundary2_full", boundary2.flatten(1)),
                                 ("boundary3_final", boundary2[:, -1]), ("qk", pattern.flatten(1)),
                                 ("attn_write", attention_write[:, -1]), ("premlp", premlp),
                                 ("write", write), ("after", after[:, -1]),
                                 ("logits", logits), ("zero_logits", zero_logits)):
            stores[key_name].append(values.cpu())
    return {key: torch.cat(values) for key, values in stores.items()}, maximum_forward_error, maximum_tensor_error


def metadata_features(manifest, facts):
    digits = np.asarray(manifest["digits"]); values = np.concatenate((digits[facts[:, 0]], digits[facts[:, 1]]), 1)
    parts = [F.one_hot(torch.as_tensor(values[:, position], dtype=torch.long), 8) for position in range(6)]
    parts.append(F.one_hot(torch.as_tensor(facts[:, 2], dtype=torch.long), 8))
    return torch.cat(parts, 1).float()


def fit_pca(train, others, rank):
    mean = train.mean(0); centered = train - mean
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    chosen = min(rank, len(vh)); basis = vh[:chosen]
    scale = (singular[:chosen] / max(np.sqrt(len(train) - 1), 1)).clamp_min(1e-8)
    return [((values - mean) @ basis.T) / scale for values in (train, *others)]


def quadratic(values, rank):
    selected = values[:, :rank]; upper = torch.triu_indices(rank, rank)
    return torch.cat((selected, selected[:, upper[0]] * selected[:, upper[1]]), 1)


def fit_ridge(train_x, train_y, validation_x, validation_y, penalties):
    train_x = train_x.double(); train_y = train_y.double(); validation_x = validation_x.double()
    xmean, ymean = train_x.mean(0), train_y.mean(0); x, y = train_x - xmean, train_y - ymean
    gram = x.T @ x; scale = float(torch.trace(gram) / max(gram.shape[0], 1))
    best = None
    for penalty in penalties:
        weight = torch.linalg.solve(gram + penalty * max(scale, 1e-30) * torch.eye(gram.shape[0], dtype=torch.double), x.T @ y)
        prediction = (validation_x - xmean) @ weight + ymean
        denominator = (validation_y.double() - validation_y.double().mean(0)).square().sum().clamp_min(1e-30)
        r2 = float(1 - (prediction - validation_y.double()).square().sum() / denominator)
        if best is None or r2 > best[0]: best = (r2, penalty, xmean.float(), ymean.float(), weight.float())
    return {"validation_r2": best[0], "penalty": best[1], "xmean": best[2], "ymean": best[3], "weight": best[4]}


def predict_ridge(values, fitted):
    return (values - fitted["xmean"]) @ fitted["weight"] + fitted["ymean"]


def prototypes(train_write, train_relation, train_label, by_relation):
    output = {}
    for relation in (range(8) if by_relation else (None,)):
        for label in range(8):
            keep = train_label == label
            if relation is not None: keep &= train_relation == relation
            output[(relation, label)] = train_write[torch.as_tensor(keep)].mean(0)
    return output


def apply_prototypes(table, relation, labels, by_relation):
    return torch.stack([table[(int(r) if by_relation else None, int(y))] for r, y in zip(relation, labels)])


def knn(train_x, train_y, test_x, neighbors):
    mean = train_x.mean(0); train = train_x - mean; test = test_x - mean
    train = train / train.norm(dim=1, keepdim=True).clamp_min(1e-30)
    test = test / test.norm(dim=1, keepdim=True).clamp_min(1e-30)
    indices = (test @ train.T).topk(neighbors, 1).indices
    return train_y[indices].mean(1)


def fisher_rows(base, target_effect, predicted_effect):
    probability = torch.softmax(base, -1)
    target = target_effect - (probability * target_effect).sum(-1, keepdim=True)
    predicted = predicted_effect - (probability * predicted_effect).sum(-1, keepdim=True)
    target_energy = (probability * target.square()).sum(-1).clamp_min(1e-30)
    predicted_energy = (probability * predicted.square()).sum(-1).clamp_min(1e-30)
    cosine = (probability * target * predicted).sum(-1) / torch.sqrt(target_energy * predicted_energy).clamp_min(1e-30)
    fidelity = 1 - (probability * (predicted - target).square()).sum(-1) / target_energy
    return target_energy.sqrt(), cosine, fidelity


@torch.inference_mode()
def score(predicted, target, after, zero_logits, logits, labels, prototype, model, threshold):
    predicted_logits = model.unembedding(model.final_norm((after + predicted).to(next(model.parameters()).device)))[:, list(VOCAB.answers)].float().cpu()
    target_effect = logits - zero_logits; predicted_effect = predicted_logits - zero_logits
    energy, cosine, fidelity = fisher_rows(zero_logits, target_effect, predicted_effect)
    keep = energy >= threshold
    denominator = (target - target.mean(0)).square().sum().clamp_min(1e-30)
    residual_denominator = (target - prototype).square().sum().clamp_min(1e-30)
    return {"write_r2": float(1 - (predicted - target).square().sum() / denominator),
            "oracle_relation_answer_residual_r2": float(1 - (predicted - target).square().sum() / residual_denominator),
            "median_write_cosine": float(F.cosine_similarity(predicted, target, 1).median()),
            "effectful_examples": int(keep.sum()),
            "median_effect_cosine": float(cosine[keep].median()) if bool(keep.any()) else None,
            "median_effect_fidelity": float(fidelity[keep].median()) if bool(keep.any()) else None,
            "answer_accuracy": float((predicted_logits[keep].argmax(-1) == torch.as_tensor(labels)[keep]).float().mean()) if bool(keep.any()) else None}


def select_panel(labels, selected, changed, eligible, facts, config, seed):
    rng = np.random.default_rng(seed); output = {False: {"discovery": [], "validation": [], "evaluation": []},
                                                 True: {"discovery": [], "validation": [], "evaluation": []}}
    for status in (False, True):
        mask_status = (selected & changed) if status else ~selected
        for relation in range(8):
            for answer in range(8):
                indices = np.flatnonzero(eligible & mask_status & (facts[:, 2] == relation) & (labels == answer))
                rng.shuffle(indices); a = config["discovery_per_relation_answer_status"]
                b = a + config["validation_per_relation_answer_status"]
                c = b + config["evaluation_per_relation_answer_status"]
                if len(indices) < c: raise RuntimeError((status, relation, answer, len(indices), c))
                output[status]["discovery"].extend(indices[:a]); output[status]["validation"].extend(indices[a:b]); output[status]["evaluation"].extend(indices[b:c])
    return {status: {name: np.asarray(rows) for name, rows in splits.items()} for status, splits in output.items()}


def analyze_model(spec, config, device):
    root = Path(spec["root"]); manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / f"rho_{config['rho']:g}/depth_4/checkpoint.pt", device)
    facts = np.asarray(manifest["train_facts"]); labels, selected, changed = labels_for_rho(manifest, "train", config["rho"])
    eligible = np.ones(len(facts), bool)
    for template in config["evaluation_templates"]:
        prompts = encode_prompts(manifest, facts, template)
        for start in range(0, len(prompts), 1024):
            tokens = torch.as_tensor(prompts[start:start + 1024].astype(np.int64), device=device)
            prediction = model(tokens)[:, -1, list(VOCAB.answers)].argmax(-1).cpu().numpy()
            eligible[start:start + len(prediction)] &= prediction == labels[start:start + len(prediction)]
    panel = select_panel(labels, selected, changed, eligible, facts, config, config["seed"] + spec["seed"])
    data = {}
    maximum_forward_error = 0.0; maximum_tensor_error = 0.0
    for status in (False, True):
        data[status] = {}
        for split, indices in panel[status].items():
            data[status][split] = {}
            for template in ([config["discovery_template"]] if split != "evaluation" else config["evaluation_templates"]):
                collected, forward_error, tensor_error = collect(model, encode_prompts(manifest, facts[indices], template), device)
                collected["metadata"] = metadata_features(manifest, facts[indices])
                data[status][split][template] = collected
                maximum_forward_error = max(maximum_forward_error, forward_error); maximum_tensor_error = max(maximum_tensor_error, tensor_error)
    results = {False: {}, True: {}}
    for status in (False, True):
        train = data[status]["discovery"][config["discovery_template"]]
        validation = data[status]["validation"][config["discovery_template"]]
        train_indices, validation_indices, evaluation_indices = (panel[status][name] for name in ("discovery", "validation", "evaluation"))
        train_labels, validation_labels, evaluation_labels = labels[train_indices], labels[validation_indices], labels[evaluation_indices]
        train_relation, evaluation_relation = facts[train_indices, 2], facts[evaluation_indices, 2]
        label_proto = prototypes(train["write"], train_relation, train_labels, False)
        relation_label_proto = prototypes(train["write"], train_relation, train_labels, True)
        pca_sources = {}
        for source in ("boundary2_full", "premlp"):
            eval_values = [data[status]["evaluation"][template][source] for template in config["evaluation_templates"]]
            pca_sources[source] = fit_pca(train[source], [validation[source], *eval_values], config["pca_rank"])
        feature_sets = {
            "metadata": (train["metadata"], validation["metadata"], *[data[status]["evaluation"][t]["metadata"] for t in config["evaluation_templates"]]),
            "raw_tokens": (train["raw"], validation["raw"], *[data[status]["evaluation"][t]["raw"] for t in config["evaluation_templates"]]),
            "boundary2_full_pca64": tuple(pca_sources["boundary2_full"]),
            "boundary3_final": (train["boundary3_final"], validation["boundary3_final"], *[data[status]["evaluation"][t]["boundary3_final"] for t in config["evaluation_templates"]]),
            "qk_pattern": (train["qk"], validation["qk"], *[data[status]["evaluation"][t]["qk"] for t in config["evaluation_templates"]]),
            "attention_write": (train["attn_write"], validation["attn_write"], *[data[status]["evaluation"][t]["attn_write"] for t in config["evaluation_templates"]]),
            "premlp_linear": (train["premlp"], validation["premlp"], *[data[status]["evaluation"][t]["premlp"] for t in config["evaluation_templates"]]),
        }
        premlp_pca = pca_sources["premlp"]
        for rank in config["quadratic_ranks"]:
            feature_sets[f"premlp_quadratic_{rank}"] = tuple(quadratic(values, rank) for values in premlp_pca)
        generator = torch.Generator().manual_seed(config["seed"] + spec["seed"] * 10 + int(status))
        shuffled = train["write"][torch.randperm(len(train["write"]), generator=generator)]
        fitted = {name: fit_ridge(values[0], train["write"], values[1], validation["write"], config["ridge_relative"])
                  for name, values in feature_sets.items()}
        fitted["premlp_quadratic_32_shuffled"] = fit_ridge(feature_sets["premlp_quadratic_32"][0], shuffled,
                                                            feature_sets["premlp_quadratic_32"][1], validation["write"], config["ridge_relative"])
        results[status] = {}
        for template_position, template in enumerate(config["evaluation_templates"]):
            evaluation = data[status]["evaluation"][template]
            prototype = apply_prototypes(relation_label_proto, evaluation_relation, evaluation_labels, True)
            predictions = {name: predict_ridge(values[2 + template_position], fitted[name]) for name, values in feature_sets.items()}
            predictions["premlp_quadratic_32_shuffled"] = predict_ridge(feature_sets["premlp_quadratic_32"][2 + template_position], fitted["premlp_quadratic_32_shuffled"])
            predictions["answer_prototype_oracle"] = apply_prototypes(label_proto, evaluation_relation, evaluation_labels, False)
            predictions["relation_answer_prototype_oracle"] = prototype
            for neighbors in config["knn_neighbors"]:
                predictions[f"premlp_knn_{neighbors}"] = knn(train["premlp"], train["write"], evaluation["premlp"], neighbors)
            # The explicit fixed tensor is the exact, non-compressed ceiling.
            block = model.blocks[3]
            predictions["exact_bilinear_tensor"] = ((evaluation["premlp"] @ block.left.weight.detach().cpu().T)
                                                     * (evaluation["premlp"] @ block.right.weight.detach().cpu().T)) @ block.down.weight.detach().cpu().T
            methods = {name: score(values, evaluation["write"], evaluation["after"], evaluation["zero_logits"],
                                    evaluation["logits"], evaluation_labels, prototype, model,
                                    config["minimum_fisher_effect_norm"])
                       for name, values in predictions.items()}
            results[status][str(template)] = methods
    return {"seed": spec["seed"], "maximum_forward_error": maximum_forward_error,
            "maximum_tensor_error": maximum_tensor_error,
            "panel_sizes": {str(status): {name: len(rows) for name, rows in splits.items()} for status, splits in panel.items()},
            "results": {"unselected": results[False], "selected": results[True]}}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"; runs = []
    for spec in config["models"]:
        print("audit", spec["seed"], flush=True); runs.append(analyze_model(spec, config, device))
    template = str(config["evaluation_templates"][-1])
    def metric(run, status, method, key): return run["results"][status][template][method][key]
    nonoracle = ("metadata", "raw_tokens", "boundary2_full_pca64", "boundary3_final", "qk_pattern",
                 "attention_write", "premlp_linear", "premlp_quadratic_16", "premlp_quadratic_32",
                 "premlp_knn_1", "premlp_knn_8")
    a0 = all(run["maximum_forward_error"] <= 1e-5 and run["maximum_tensor_error"] <= 1e-5 for run in runs)
    a1 = all(metric(run, "selected", "metadata", "write_r2") < .1 for run in runs)
    a2 = all(metric(run, "selected", "premlp_linear", "median_effect_fidelity") >= .7 for run in runs)
    a3 = all(metric(run, "selected", "premlp_quadratic_32", "median_effect_fidelity") >= .8 and
             metric(run, "selected", "premlp_quadratic_32", "median_effect_fidelity") >= metric(run, "selected", "premlp_linear", "median_effect_fidelity") + .1 for run in runs)
    a4 = all(metric(run, "selected", "boundary2_full_pca64", "median_effect_fidelity") >= .7 for run in runs)
    a5 = all(metric(run, "selected", "qk_pattern", "median_effect_fidelity") >= .5 for run in runs)
    a6 = all(metric(run, "selected", "premlp_knn_8", "median_effect_fidelity") >= .7 for run in runs)
    a7 = all(max(metric(run, "selected", method, "median_effect_fidelity") for method in nonoracle)
             >= max(metric(run, "unselected", method, "median_effect_fidelity") for method in nonoracle) - .15 for run in runs)
    a8 = all(max(metric(run, "selected", method, "oracle_relation_answer_residual_r2") for method in nonoracle) >= .5 for run in runs)
    a9 = all(metric(run, "selected", "relation_answer_prototype_oracle", "median_effect_fidelity") >= .8 for run in runs)
    a10 = all(metric(run, "selected", "premlp_quadratic_32", "median_effect_fidelity") >= metric(run, "selected", "premlp_quadratic_32_shuffled", "median_effect_fidelity") + .2 for run in runs)
    a11 = all(row["effectful_examples"] >= 64 for run in runs for status in ("unselected", "selected")
              for template_rows in run["results"][status].values() for row in template_rows.values())
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11))}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "runs": runs}, indent=2, allow_nan=False))
    lines = ["# Within-model address router for mixed arbitrary facts", "",
             "Predictors are fit on disjoint learned facts in template 0 and evaluated on new facts in both template 0 and held-out template 6. The target is the exact final bilinear MLP residual write. Every predicted write is executed through the target model's final RMSNorm and unembedding; effect fidelity is relative to zeroing that natural write. Raw and state predictors never receive the arbitrary answer label. Answer prototypes are explicit leaky diagnostics.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    display = ("metadata", "raw_tokens", "boundary2_full_pca64", "boundary3_final", "qk_pattern", "attention_write", "premlp_linear", "premlp_quadratic_16", "premlp_quadratic_32", "premlp_knn_8", "relation_answer_prototype_oracle", "exact_bilinear_tensor")
    lines += ["", "| Seed | Status | Template | " + " | ".join(display) + " |", "|---:|---|---:|" + "---:|" * len(display)]
    for run in runs:
        for status in ("unselected", "selected"):
            for template_key in map(str, config["evaluation_templates"]):
                rows = run["results"][status][template_key]
                values = [rows[name]["median_effect_fidelity"] for name in display]
                lines.append(f"| {run['seed']} | {status} | {template_key} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += ["", "Write R2 asks for coordinate reconstruction; oracle-residual R2 subtracts the relation+answer prototype and asks whether fact-specific variation remains predictable. Finite effect fidelity asks whether the predicted write causes the same answer-distribution change. Accuracy asks whether that change preserves the learned answer. These are deliberately reported separately."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
