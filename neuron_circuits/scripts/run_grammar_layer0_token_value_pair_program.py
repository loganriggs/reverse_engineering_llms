#!/usr/bin/env python3
"""Exact raw-token value-route pair program of the Layer-0 bilinear MLP.

At Layer 0, each attention value is a linear transform of one raw token state.
The final-position attention write is therefore an exact sum over source-token
roles.  After applying the one native RMS scale, the bilinear gate is an exact
14 x 14 ordered-pair expansion: the final-token boundary plus thirteen
canonical attention source roles.  Ranked and semantic subsets are validated
by literal gate patches followed by complete suffix recomputation.
"""

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
from run_grammar_mixed_address_router import select_panel


CONFIG = Path("configs/grammar_layer0_token_value_pair_program.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_layer0_token_value_pair_program.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_LAYER0_TOKEN_VALUE_PAIR_PROGRAM.md")

SOURCE_ROLES = (
    "bos", "sentence_marker", "subject_marker",
    "subject_digit0", "subject_digit1", "subject_digit2",
    "object_marker", "object_digit0", "object_digit1", "object_digit2",
    "query_marker", "relation", "answer_marker",
)
COMPONENTS = ("final_boundary", *SOURCE_ROLES)
DIGIT_COMPONENTS = frozenset((4, 5, 6, 8, 9, 10))
RELATION_COMPONENT = 12


def canonical_source_positions(template: int) -> tuple[int, ...]:
    """Physical sequence positions in SOURCE_ROLES order."""
    specifications = (
        (0, "SOQ"), (1, "SOQ"), (0, "OSQ"), (1, "OSQ"),
        (0, "SQO"), (1, "OQS"), (1, "SQO"), (0, "OQS"),
    )
    _, order = specifications[template]
    starts = {}; position = 2
    for field in order:
        starts[field] = position
        position += 4 if field in "SO" else 2
    if position != 12:
        raise AssertionError(position)
    return (
        0, 1,
        starts["S"], starts["S"] + 1, starts["S"] + 2, starts["S"] + 3,
        starts["O"], starts["O"] + 1, starts["O"] + 2, starts["O"] + 3,
        starts["Q"], starts["Q"] + 1,
        12,
    )


def relevant_digit_components(relation: int) -> frozenset[int]:
    # Component 0 is the final boundary and source components are offset by 1.
    table = (
        (4,), (5,), (8,), (9,),
        (6, 4, 8), (10, 5, 9), (4, 10, 6), (8, 6, 10),
    )
    return frozenset(table[relation])


@torch.inference_mode()
def token_role_pairs(model, prompts: np.ndarray, template: int, device: str, batch_size: int = 128):
    block = model.blocks[0]; positions = torch.as_tensor(canonical_source_positions(template), device=device)
    stores = []; errors = {"attention": 0.0, "after_attention": 0.0, "gate": 0.0}
    for start in range(0, len(prompts), batch_size):
        tokens = torch.as_tensor(prompts[start:start + batch_size].astype(np.int64), device=device)
        hidden = model.embedding(tokens) + model.position[:tokens.shape[1]]
        normalized = block.attn_norm(hidden); q, k, v = block.qkv(normalized).chunk(3, -1)
        batch, length, width = hidden.shape; heads, dimension = block.heads, block.head_dim
        q = q.view(batch, length, heads, dimension).transpose(1, 2)
        k = k.view(batch, length, heads, dimension).transpose(1, 2)
        v = v.view(batch, length, heads, dimension).transpose(1, 2)
        pattern = torch.softmax(torch.einsum("bhd,bhsd->bhs", q[:, :, -1], k) / math.sqrt(dimension), -1)
        weighted = pattern[..., None] * v
        output_weight = block.attn_out.weight.view(width, heads, dimension)
        source_writes = torch.einsum("bhsd,whd->bsw", weighted, output_weight)
        native_attention = block.attn_out(torch.einsum("bhs,bhsd->bhd", pattern, v).reshape(batch, width))
        errors["attention"] = max(errors["attention"], float((source_writes.sum(1) - native_attention).abs().max()))
        canonical = source_writes[:, positions]
        components = torch.cat((hidden[:, -1, None], canonical), 1)
        after = components.sum(1); native_after = hidden[:, -1] + native_attention
        errors["after_attention"] = max(errors["after_attention"], float((after - native_after).abs().max()))
        epsilon = block.mlp_norm.eps
        if epsilon is None: epsilon = torch.finfo(after.dtype).eps
        scale = block.mlp_norm.weight / torch.sqrt(after.square().mean(-1, keepdim=True) + epsilon)
        normalized_components = components * scale[:, None]
        left = torch.einsum("bcw,hw->bch", normalized_components, block.left.weight)
        right = torch.einsum("bcw,hw->bch", normalized_components, block.right.weight)
        pairs = left[:, :, None] * right[:, None, :]
        native_gate = block.left(block.mlp_norm(native_after)) * block.right(block.mlp_norm(native_after))
        errors["gate"] = max(errors["gate"], float((pairs.sum((1, 2)) - native_gate).abs().max()))
        stores.append(pairs.float().cpu())
    return torch.cat(stores), errors


def rule_relevant_mask(relations: np.ndarray) -> torch.Tensor:
    masks = []
    for relation in relations:
        irrelevant = DIGIT_COMPONENTS - relevant_digit_components(int(relation))
        masks.append(torch.as_tensor([
            not (left in irrelevant or right in irrelevant)
            for left in range(14) for right in range(14)
        ], dtype=torch.bool))
    return torch.stack(masks)


def fixed_mask(predicate) -> torch.Tensor:
    return torch.as_tensor([predicate(left, right) for left in range(14) for right in range(14)], dtype=torch.bool)


def pair_description(index: int) -> dict:
    return {"left": COMPONENTS[index // 14], "right": COMPONENTS[index % 14]}


def analyze(spec, regime: str, rho: float, config: dict, device: str):
    root = Path(spec["root"]); manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / f"rho_{rho:g}/depth_4/checkpoint.pt", device)
    facts = np.asarray(manifest["train_facts"]); labels, selected, changed = labels_for_rho(manifest, "train", rho)
    eligible = eligible_mask(model, manifest, facts, labels, config["evaluation_templates"], device)
    panel_config = dict(config)
    for field in ("discovery_per_relation_answer_status", "validation_per_relation_answer_status",
                  "evaluation_per_relation_answer_status"):
        panel_config[field] = config.get(f"{regime}_{field}", config[field])
    if regime == "pure":
        panels = {"all": pure_panel(labels, eligible, facts, panel_config, config["seed"] + spec["seed"])}
    else:
        raw = select_panel(labels, selected, changed, eligible, facts, panel_config, config["seed"] + spec["seed"])
        panels = {"unselected": raw[False], "selected": raw[True]}
    down = model.blocks[0].down.weight.detach().cpu(); output = {}; maximum_errors = {key: 0.0 for key in ("attention", "after_attention", "gate")}
    for status, panel in panels.items():
        train_idx, eval_idx = panel["discovery"], panel["evaluation"]
        train_prompts = encode_prompts(manifest, facts[train_idx], config["discovery_template"])
        train_native = native_data(model, train_prompts, [0, 3], device)
        train_pairs, errors = token_role_pairs(model, train_prompts, config["discovery_template"], device)
        for key in maximum_errors: maximum_errors[key] = max(maximum_errors[key], errors[key])
        pair_mean = train_pairs.mean(0); gate_mean = pair_mean.sum((0, 1)); centered = train_pairs - pair_mean
        # Rank exact tensor terms only by their variable Layer-0 residual write norm.
        write_variation = torch.einsum("bijh,wh->bijw", centered, down)
        energy = write_variation.square().mean((0, 3)); order = energy.flatten().argsort(descending=True)
        answer3_mean, answer3_basis, _ = centroid_basis(answer_table(train_native["gates"][:, 3], labels[train_idx]))
        output[status] = {"pair_order": [{**pair_description(int(index)), "write_energy": float(energy.flatten()[index])} for index in order]}
        for template in config["evaluation_templates"]:
            prompts = encode_prompts(manifest, facts[eval_idx], template)
            native = native_data(model, prompts, [0, 3], device)
            pairs, errors = token_role_pairs(model, prompts, template, device)
            for key in maximum_errors: maximum_errors[key] = max(maximum_errors[key], errors[key])
            flat = (pairs - pair_mean).flatten(1, 2); methods = {}
            def assess(name: str, mask: torch.Tensor):
                if mask.ndim == 1:
                    candidate = gate_mean + flat[:, mask].sum(1)
                else:
                    candidate = gate_mean + (flat * mask[:, :, None]).sum(1)
                logits, gate3 = patched_data(model, prompts, 0, candidate, device)
                methods[name] = metric(logits, gate3, native, 0, labels[eval_idx], answer3_mean, answer3_basis,
                                       config["minimum_fisher_effect_norm"])
            for count in config["pair_counts"]: assess(f"top_{count}", torch.isin(torch.arange(196), order[:count]))
            assess("rule_relevant_only", rule_relevant_mask(facts[eval_idx, 2]))
            assess("no_identity_variation", fixed_mask(lambda left, right: left not in DIGIT_COMPONENTS and right not in DIGIT_COMPONENTS))
            assess("identity_involving_only", fixed_mask(lambda left, right: left in DIGIT_COMPONENTS or right in DIGIT_COMPONENTS))
            assess("relation_identity_only", fixed_mask(lambda left, right: (left == RELATION_COMPONENT and right in DIGIT_COMPONENTS) or (right == RELATION_COMPONENT and left in DIGIT_COMPONENTS)))
            assess("no_cross_source_variation", fixed_mask(lambda left, right: not (left > 0 and right > 0 and left != right)))
            generator = np.random.default_rng(config["seed"] + spec["seed"] * 100 + template * 10 + (status == "selected"))
            random_fidelities = []
            for _ in range(config["random_pair_banks"]):
                chosen = generator.choice(196, 64, replace=False)
                mask = torch.zeros(196, dtype=torch.bool); mask[chosen] = True
                candidate = gate_mean + flat[:, mask].sum(1)
                logits, gate3 = patched_data(model, prompts, 0, candidate, device)
                row = metric(logits, gate3, native, 0, labels[eval_idx], answer3_mean, answer3_basis,
                             config["minimum_fisher_effect_norm"])
                random_fidelities.append(row["median_effect_fidelity"])
            methods["random_64"] = {"median_fidelity_mean": float(np.mean(random_fidelities)),
                                    "median_fidelity_p95": float(np.quantile(random_fidelities, .95)),
                                    "replicates": len(random_fidelities)}
            output[status][str(template)] = methods
    return {"regime": regime, "seed": spec["seed"], "maximum_errors": maximum_errors, "results": output}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    pure = [analyze(spec, "pure", config["pure_rho"], config, device) for spec in config["pure_models"]]
    mixed = [analyze(spec, "mixed", config["mixed_rho"], config, device) for spec in config["mixed_models"]]
    runs = pure + mixed; first, held = map(str, config["evaluation_templates"])
    def val(run, status, template, method, key="median_effect_fidelity"):
        return run["results"][status][template][method][key]
    v0 = all(max(run["maximum_errors"].values()) < 1e-4 and val(run, status, template, "top_196") >= .9999
             for run in runs for status in run["results"] for template in (first, held))
    v1 = all(val(run, "all", held, "top_64") >= .90 for run in pure)
    v2 = all(val(p, "all", held, "top_32") >= val(m, "unselected", held, "top_32") + .05 for p, m in zip(pure, mixed))
    v3 = all(val(run, "all", held, "rule_relevant_only") >= .85 for run in pure)
    v4 = all(val(p, "all", held, "rule_relevant_only") >= val(m, "selected", held, "rule_relevant_only") + .10 for p, m in zip(pure, mixed))
    v5 = all(val(p, "all", held, "rule_relevant_only") >= val(m, "unselected", held, "rule_relevant_only") + .05 for p, m in zip(pure, mixed))
    v6 = all(val(run, "selected", held, "no_cross_source_variation") < .90 for run in mixed)
    v7 = all(val(run, status, held, "top_64") >= run["results"][status][held]["random_64"]["median_fidelity_p95"] + .10
             for run in runs for status in run["results"])
    v8 = all(abs(val(run, status, first, "top_64") - val(run, status, held, "top_64")) <= .10
             for run in runs for status in run["results"])
    v9 = all(row["effectful_examples"] >= 64 for run in runs for status in run["results"]
             for template in (first, held) for row in run["results"][status][template].values()
             if isinstance(row, dict) and "effectful_examples" in row)
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (v0,v1,v2,v3,v4,v5,v6,v7,v8,v9))}
    OUTPUT.write_text(json.dumps({"config": config, "components": COMPONENTS, "hypotheses": hypotheses, "runs": runs}, indent=2, allow_nan=False))
    lines = ["# Exact Layer-0 raw-token value-route pair program", "",
             "At the first block, every attention value is still a linear transform of one raw token state. The final-position attention write is exactly regrouped into thirteen canonical source roles, independent of template order. Together with the final-token boundary this gives 14 additive normalized-input components and 196 ordered bilinear gate terms. Pair means and rankings are fit on a disjoint trained-template panel; all reported cells are literal Layer-0 gate patches with the full suffix recomputed.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Regime | Seed | Status | top 16/32/64 | rule-relevant | no identity | identity-only | relation×identity | no cross-source | random64 p95 |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        for status in run["results"]:
            row = run["results"][status][held]
            lines.append(f"| {run['regime']} | {run['seed']} | {status} | {row['top_16']['median_effect_fidelity']:.3f}/{row['top_32']['median_effect_fidelity']:.3f}/{row['top_64']['median_effect_fidelity']:.3f} | {row['rule_relevant_only']['median_effect_fidelity']:.3f} | {row['no_identity_variation']['median_effect_fidelity']:.3f} | {row['identity_involving_only']['median_effect_fidelity']:.3f} | {row['relation_identity_only']['median_effect_fidelity']:.3f} | {row['no_cross_source_variation']['median_effect_fidelity']:.3f} | {row['random_64']['median_fidelity_p95']:.3f} |")
    lines += ["", "The semantic masks isolate value/write routes while retaining the native QK pattern. In particular, rule-irrelevant tokens can still alter every retained source contribution through the softmax denominator. Therefore success of `rule-relevant-only` does not establish complete token irrelevance; a separate QK intervention is required. Conversely, failure can reflect distributed value transport, normalization coupling, or an off-manifold mean fill rather than semantic polysemanticity."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__": main()
