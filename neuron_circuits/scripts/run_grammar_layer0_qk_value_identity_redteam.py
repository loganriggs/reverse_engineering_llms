#!/usr/bin/env python3
"""Separate raw digit identity in Layer-0 QK keys from value content."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_final_gate_code_dimension import eligible_mask, pure_panel
from run_grammar_gate_bottleneck_depth_telescope import native_data, patched_data, metric
from run_grammar_final_gate_answer_quotient import answer_table, centroid_basis
from run_grammar_layer0_token_value_pair_program import (
    DIGIT_COMPONENTS, canonical_source_positions, relevant_digit_components,
)
from run_grammar_mixed_address_router import select_panel


CONFIG = Path("configs/grammar_layer0_qk_value_identity_redteam.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_layer0_qk_value_identity_redteam.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_LAYER0_QK_VALUE_IDENTITY_REDTEAM.md")


def digit_projection_table(model, physical_position: int):
    block = model.blocks[0]; device = next(model.parameters()).device
    ids = torch.as_tensor(VOCAB.digits, device=device)
    states = model.embedding(ids) + model.position[physical_position]
    _, keys, values = block.qkv(block.attn_norm(states)).chunk(3, -1)
    return keys.view(8, block.heads, block.head_dim), values.view(8, block.heads, block.head_dim)


@torch.inference_mode()
def candidate_gates(model, prompts: np.ndarray, template: int, relations: np.ndarray, device: str, batch_size: int = 256):
    block = model.blocks[0]; positions = canonical_source_positions(template)
    digit_roles = tuple(sorted(DIGIT_COMPONENTS)); physical = {component: positions[component - 1] for component in digit_roles}
    tables = {component: digit_projection_table(model, position) for component, position in physical.items()}
    outputs = {name: [] for name in (
        "native", "irrelevant_mean_key", "irrelevant_mean_value", "irrelevant_mean_both",
        "irrelevant_roll_key", "irrelevant_roll_value", "irrelevant_roll_both",
        "all_mean_key", "all_mean_value", "all_mean_both",
    )}
    maximum_attention_error = maximum_gate_error = 0.0
    for start in range(0, len(prompts), batch_size):
        tokens = torch.as_tensor(prompts[start:start + batch_size].astype(np.int64), device=device)
        relation = relations[start:start + len(tokens)]
        hidden = model.embedding(tokens) + model.position[:tokens.shape[1]]
        normalized = block.attn_norm(hidden); q, native_k, native_v = block.qkv(normalized).chunk(3, -1)
        batch, length, width = hidden.shape; heads, dimension = block.heads, block.head_dim
        q = q.view(batch, length, heads, dimension).transpose(1, 2)
        native_k = native_k.view(batch, length, heads, dimension).transpose(1, 2)
        native_v = native_v.view(batch, length, heads, dimension).transpose(1, 2)
        def gate(keys, values):
            pattern = torch.softmax(torch.einsum("bhd,bhsd->bhs", q[:, :, -1], keys) / math.sqrt(dimension), -1)
            attended = torch.einsum("bhs,bhsd->bhd", pattern, values).reshape(batch, width)
            write = block.attn_out(attended); after = hidden[:, -1] + write
            return block.left(block.mlp_norm(after)) * block.right(block.mlp_norm(after)), write
        native_gate, native_write = gate(native_k, native_v)
        native_output, native_activations = model(tokens, return_activations=True)
        maximum_gate_error = max(maximum_gate_error, float((native_gate - native_activations[0]).abs().max()))
        explicit_logits, _ = patched_data(model, prompts[start:start + len(tokens)], 0, native_gate.cpu(), device)
        maximum_attention_error = max(maximum_attention_error,
                                      float((explicit_logits - native_output[:, -1, list(VOCAB.answers)].float().cpu()).abs().max()))
        outputs["native"].append(native_gate.float().cpu())
        for scope in ("irrelevant", "all"):
            for replacement in ("mean", "roll") if scope == "irrelevant" else ("mean",):
                for branch in ("key", "value", "both"):
                    keys, values = native_k.clone(), native_v.clone()
                    for row in range(batch):
                        allowed = DIGIT_COMPONENTS if scope == "all" else DIGIT_COMPONENTS - relevant_digit_components(int(relation[row]))
                        for component in allowed:
                            position = physical[component]; key_table, value_table = tables[component]
                            if replacement == "mean": key_value, value_value = key_table.mean(0), value_table.mean(0)
                            else:
                                digit_id = int(tokens[row, position].item() - VOCAB.digits[0]); shifted = (digit_id + 1) % 8
                                key_value, value_value = key_table[shifted], value_table[shifted]
                            if branch in ("key", "both"): keys[row, :, position] = key_value
                            if branch in ("value", "both"): values[row, :, position] = value_value
                    candidate, _ = gate(keys, values)
                    outputs[f"{scope}_{replacement}_{branch}"].append(candidate.float().cpu())
    return {name: torch.cat(values) for name, values in outputs.items()}, {
        "maximum_native_gate_error": maximum_gate_error,
        "maximum_native_logit_error": maximum_attention_error,
    }


def analyze(spec, regime: str, rho: float, config: dict, device: str):
    root = Path(spec["root"]); manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / f"rho_{rho:g}/depth_4/checkpoint.pt", device)
    facts = np.asarray(manifest["train_facts"]); labels, selected, changed = labels_for_rho(manifest, "train", rho)
    eligible = eligible_mask(model, manifest, facts, labels, config["evaluation_templates"], device)
    panel_config = dict(config)
    for field in ("discovery_per_relation_answer_status", "validation_per_relation_answer_status", "evaluation_per_relation_answer_status"):
        panel_config[field] = config.get(f"{regime}_{field}", config[field])
    if regime == "pure": panels = {"all": pure_panel(labels, eligible, facts, panel_config, config["seed"] + spec["seed"])}
    else:
        raw = select_panel(labels, selected, changed, eligible, facts, panel_config, config["seed"] + spec["seed"])
        panels = {"unselected": raw[False], "selected": raw[True]}
    output = {}; errors = {"maximum_native_gate_error": 0.0, "maximum_native_logit_error": 0.0}
    for status, panel in panels.items():
        train_idx, eval_idx = panel["discovery"], panel["evaluation"]
        train = native_data(model, encode_prompts(manifest, facts[train_idx], config["discovery_template"]), [0, 3], device)
        answer3_mean, answer3_basis, _ = centroid_basis(answer_table(train["gates"][:, 3], labels[train_idx]))
        output[status] = {}
        for template in config["evaluation_templates"]:
            prompts = encode_prompts(manifest, facts[eval_idx], template)
            native = native_data(model, prompts, [0, 3], device)
            candidates, current_errors = candidate_gates(model, prompts, template, facts[eval_idx, 2], device)
            for key in errors: errors[key] = max(errors[key], current_errors[key])
            methods = {}
            for name, values in candidates.items():
                logits, gate3 = patched_data(model, prompts, 0, values, device)
                methods[name] = metric(logits, gate3, native, 0, labels[eval_idx], answer3_mean, answer3_basis,
                                       config["minimum_fisher_effect_norm"])
            output[status][str(template)] = methods
    return {"regime": regime, "seed": spec["seed"], "errors": errors, "results": output}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    pure = [analyze(spec, "pure", config["pure_rho"], config, device) for spec in config["pure_models"]]
    mixed = [analyze(spec, "mixed", config["mixed_rho"], config, device) for spec in config["mixed_models"]]
    runs = pure + mixed; first, held = map(str, config["evaluation_templates"])
    def val(run, status, template, method, key="median_effect_fidelity"): return run["results"][status][template][method][key]
    q0 = all(max(run["errors"].values()) < 1e-4 and val(run, status, template, "native") >= .9999 for run in runs for status in run["results"] for template in (first, held))
    q1 = all(min(val(run, "all", held, "irrelevant_mean_both"), val(run, "all", held, "irrelevant_roll_both")) >= .85 for run in pure)
    q2 = all(val(p, "all", held, "irrelevant_mean_both") >= val(m, "selected", held, "irrelevant_mean_both") + .10 for p, m in zip(pure, mixed))
    q3 = all(val(p, "all", held, "irrelevant_roll_both") >= val(m, "selected", held, "irrelevant_roll_both") + .10 for p, m in zip(pure, mixed))
    q4 = all(val(run, "selected", held, f"irrelevant_{kind}_key") >= val(run, "selected", held, f"irrelevant_{kind}_value") + .10 for run in mixed for kind in ("mean", "roll"))
    q5 = all(min(val(run, "selected", held, "irrelevant_mean_key"), val(run, "selected", held, "irrelevant_roll_key")) < .95 for run in mixed)
    q6 = all(val(p, "all", held, "irrelevant_mean_both") >= val(m, "unselected", held, "irrelevant_mean_both") + .05 for p, m in zip(pure, mixed))
    q7 = all(abs(val(run, status, held, "irrelevant_mean_both") - val(run, status, held, "irrelevant_roll_both")) <= .15 for run in runs for status in run["results"])
    q8 = all(abs(val(run, status, first, "irrelevant_mean_both") - val(run, status, held, "irrelevant_mean_both")) <= .10 for run in runs for status in run["results"])
    q9 = all(row["effectful_examples"] >= 64 for run in runs for status in run["results"] for template in (first, held) for row in run["results"][status][template].values())
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (q0,q1,q2,q3,q4,q5,q6,q7,q8,q9))}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "runs": runs}, indent=2, allow_nan=False))
    lines = ["# Layer-0 QK-key versus value-identity red team", "",
             "For each raw digit position, its Layer-0 key and value are replaced independently before attention. The mean control uses the exact average projection of all eight digit tokens at that physical position. The cyclic control substitutes the next digit's projection, preserving a natural projection norm distribution while making identity wrong. Only rule-irrelevant digits are altered unless the method begins with `all`. Candidate attention states are decoded into a bilinear gate, patched into the native Layer-0 execution, and followed by the exact suffix; the direct attention residual remains native so this isolates how QK/value information controls the MLP gate.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Regime | Seed | Status | irrelevant mean K/V/both | irrelevant roll K/V/both | all mean K/V/both |",
              "|---|---:|---|---:|---:|---:|"]
    for run in runs:
        for status in run["results"]:
            row = run["results"][status][held]
            triplet = lambda prefix: "/".join(f"{row[f'{prefix}_{branch}']['median_effect_fidelity']:.3f}" for branch in ("key", "value", "both"))
            lines.append(f"| {run['regime']} | {run['seed']} | {status} | {triplet('irrelevant_mean')} | {triplet('irrelevant_roll')} | {triplet('all_mean')} |")
    lines += ["", "Key replacement changes the attention pattern and therefore every source coefficient; value replacement changes transported content while retaining native routing. Neither intervention removes the same digit's influence through later layers or non-final token states. The result is specific to the Layer-0 final-position MLP path, not a claim about the model's total use of a token."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__": main()
